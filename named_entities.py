# Lisans notu: Bu dosya zemberek-nlp (https://github.com/ahmetaa/zemberek-nlp;
# Copyright 2018 Ahmet A. Akın, Mehmet D. Akın; Apache License 2.0) kaynak kodundan
# Python'a uyarlanmış ve değiştirilmiştir. Ayrıntılar için NOTICE dosyasına bakın.
"""
Türkçe adlandırılmış varlık tanıma (NER): averaged perceptron ile eğitim, tahmin ve değerlendirme.

  NePosition, NerToken, NamedEntity, NerSentence, NerDataSet -> veri yapıları; BRACKET, ENAMEX
                                                                ve OpenNLP anotasyon formatları
  PerceptronNer                                              -> tahmin + öznitelikler
  PerceptronNerTrainer                                       -> eğitim + değerlendirme
  strip_entity_suffixes                                      -> varlıktan ekleri temizleme
                                                                ("Ankara'da" -> "Ankara")

WikiANN-tr ile eğitilmiş model ./ner_model altındadır (train_ner_wikiann.py ile yeniden eğitilebilir).
Orijinal Java uygulamasıyla eğitilmiş modeller de (text veya compressed) `PerceptronNer.load_model`
ile okunabilir.

Ek olarak model gerektirmeyen `ProperNounExtractor` var: morfolojinin özel isim/kısaltma
etiketlerinden ve büyük harf bilgisinden varlık adayları çıkarır (LLM değerlendirmesinde
"kaynakta olmayan varlık" kontrolü için yeterli bir taban çizgisi).

Örnek (eğitim):
    from zemberek import TurkishMorphology
    from named_entities import NerDataSet, AnnotationStyle, PerceptronNerTrainer
    m = TurkishMorphology.create_with_defaults()
    train = NerDataSet.load("train.txt", AnnotationStyle.BRACKET)
    test  = NerDataSet.load("test.txt",  AnnotationStyle.BRACKET)
    ner = PerceptronNerTrainer(m).train(train, test, iteration_count=7, learning_rate=0.1)
    ner.save_model("ner-model")
    print(ner.find_named_entities("Ahmet Bey dün Ankara'ya gitti.").named_entities())
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from zemberek import TurkishMorphology, TurkishTokenizer
from zemberek.core.text.text_util import TextUtil
from zemberek.core.turkish import SecondaryPos, TurkishAlphabet
from zemberek.tokenization.token import Token

from turkish_utils import get_lemmas, get_pos

log = logging.getLogger("named_entities")


def _tr_lower(s: str) -> str:
    return s.translate(TurkishAlphabet.lower_map).lower()


def _tr_upper_first(s: str) -> str:
    if not s:
        return s
    up = {"i": "İ", "ı": "I"}.get(s[0], s[0].upper())
    return up + s[1:]


_SPACES = re.compile(r"[\s ​]+")


def _normalize_line(line: str) -> str:
    line = TextUtil.normalize_apostrophes(line)
    line = TextUtil.normalize_quotes_hyphens(line)
    return _SPACES.sub(" ", line.replace("­", "")).strip()


# ---------------------------------------------------------------------------
# Veri yapıları
# ---------------------------------------------------------------------------

class NePosition(Enum):
    """BILOU konum bilgisi."""
    BEGIN = "B"
    INSIDE = "I"
    LAST = "L"
    OUTSIDE = "O"
    UNIT = "U"

    @staticmethod
    def from_string(s: str) -> "NePosition":
        return NePosition(s)


OUT_TOKEN_TYPE = "OUT"


@dataclass
class NerToken:
    index: int
    word: str
    normalized: str
    type: str
    position: NePosition

    @property
    def token_id(self) -> str:
        return "O" if self.position == NePosition.OUTSIDE else f"{self.type}_{self.position.value}"

    @staticmethod
    def from_type_position_string(index: int, word: str, normalized: str, id_: str) -> "NerToken":
        if id_ == "O":
            return NerToken(index, word, normalized, OUT_TOKEN_TYPE, NePosition.OUTSIDE)
        if "_" not in id_:
            raise ValueError(f"Id '_' içermeli: {id_}")
        p = id_.index("_")
        return NerToken(index, word, normalized, id_[:p], NePosition.from_string(id_[p + 1:]))

    def __eq__(self, o):
        return (isinstance(o, NerToken) and self.index == o.index and self.word == o.word
                and self.normalized == o.normalized and self.token_id == o.token_id)

    def __hash__(self):
        return hash((self.index, self.word, self.normalized, self.token_id))


@dataclass
class NamedEntity:
    type: str
    tokens: List[NerToken]

    def content(self) -> str:
        return " ".join(t.word for t in self.tokens)

    def indexes(self) -> List[int]:
        return [t.index for t in self.tokens]

    def __eq__(self, o):
        return isinstance(o, NamedEntity) and self.type == o.type and self.tokens == o.tokens

    def __hash__(self):
        return hash((self.type, tuple(self.tokens)))

    def __repr__(self):
        return f"[{self.type} {self.content()}]"


class AnnotationStyle(Enum):
    ENAMEX = "ENAMEX"
    BRACKET = "BRACKET"
    OPEN_NLP = "OPEN_NLP"


@dataclass
class NerSentence:
    content: str
    tokens: List[NerToken]

    def all_entities(self) -> List[NamedEntity]:
        out: List[NamedEntity] = []
        buf: List[NerToken] = []
        for t in self.tokens:
            buf.append(t)
            if t.position in (NePosition.UNIT, NePosition.LAST, NePosition.OUTSIDE):
                out.append(NamedEntity(t.type, buf))
                buf = []
        return out

    def named_entities(self) -> List[NamedEntity]:
        return [e for e in self.all_entities() if e.type != OUT_TOKEN_TYPE]

    def as_training_sentence(self, style: AnnotationStyle) -> str:
        parts: List[str] = []
        for ne in self.all_entities():
            if ne.type == OUT_TOKEN_TYPE:
                parts.append(ne.tokens[0].word)
                continue
            open_, close = {
                AnnotationStyle.ENAMEX: (f'<b_enamex TYPE="{ne.type}">', "<e_enamex>"),
                AnnotationStyle.OPEN_NLP: (f"<START:{ne.type}>", "<END>"),
                AnnotationStyle.BRACKET: (f"[{ne.type}", "]"),
            }[style]
            parts.append(open_)
            parts.extend(t.word for t in ne.tokens)
            parts.append(close)
        return " ".join(parts)

    def matching_nes(self, nes: Sequence[NamedEntity]) -> List[NamedEntity]:
        mine = set(self.named_entities())
        return [ne for ne in nes if ne in mine]


_PATTERNS = {
    AnnotationStyle.ENAMEX: (
        r'((<b_enamex TYPE=")(?P<TYPE>.+?)(">)(?P<CONTENT>.+?)(<e_enamex>))',
        r'((<b_enamex TYPE=")(?P<TYPE>.+?)(">)(?P<CONTENT>.+?)(<e_enamex>))|([^ ]+)',
    ),
    AnnotationStyle.BRACKET: (
        r"((\[)(?P<TYPE>[^ ]+)( )(?P<CONTENT>.+?)(]))",
        r"((\[)(?P<TYPE>[^ ]+)( )(?P<CONTENT>.+?)(]))|([^ ]+)",
    ),
    AnnotationStyle.OPEN_NLP: (
        r"((<START:)(?P<TYPE>.+?)(>)(?P<CONTENT>.+?)(<END>))",
        r"((<START:)(?P<TYPE>.+?)(>)(?P<CONTENT>.+?)(<END>))|([^ ]+)",
    ),
}

_NUMERIC_TYPES = {Token.Type.Date, Token.Type.Number, Token.Type.Time}


def normalize_for_ner(word: str) -> str:
    """NerDataSet.normalizeForNer: küçük harf + sayı/tarih/saatlerde rakamları 'D' yap."""
    out = []
    for t in TurkishTokenizer.DEFAULT.tokenize(_tr_lower(word)):
        s = t.content
        if t.type_ in _NUMERIC_TYPES:
            s = "*" + re.sub(r"[0-9]", "D", s) + "*"
        out.append(s)
    return "".join(out)


class NerDataSet:
    _rnd = random.Random(0xCAFE)

    def __init__(self, sentences: List[NerSentence]):
        self.sentences = sentences
        self.types = {t.type for s in sentences for t in s.tokens}
        self.type_ids = {t.token_id for s in sentences for t in s.tokens}

    def shuffle(self):
        self._rnd.shuffle(self.sentences)

    @staticmethod
    def load(path: str, style: AnnotationStyle) -> "NerDataSet":
        with open(path, encoding="utf-8") as f:
            return NerDataSet.from_lines(f.read().splitlines(), style)

    @staticmethod
    def from_lines(lines: Iterable[str], style: AnnotationStyle) -> "NerDataSet":
        ne_pat, split_pat = (re.compile(p, re.IGNORECASE) for p in _PATTERNS[style])
        sentences: List[NerSentence] = []
        for line in lines:
            line = _normalize_line(line)
            if len(line.strip()) < 2:
                continue
            tokens = [m.group(0) for m in split_pat.finditer(line)]
            ner_tokens: List[NerToken] = []
            index = 0
            for tok in tokens:
                if index > 0 and tok.startswith("'") and not tok.endswith("'"):
                    ner_tokens[index - 1].word += tok
                    ner_tokens[index - 1].normalized += tok
                    continue
                m = ne_pat.fullmatch(tok)
                if not m:
                    ner_tokens.append(NerToken(index, tok, normalize_for_ner(tok), OUT_TOKEN_TYPE, NePosition.OUTSIDE))
                    index += 1
                    continue
                type_, words = m.group("TYPE"), m.group("CONTENT").split()
                for i, w in enumerate(words):
                    if len(words) == 1:
                        pos = NePosition.UNIT
                    elif i == 0:
                        pos = NePosition.BEGIN
                    elif i == len(words) - 1:
                        pos = NePosition.LAST
                    else:
                        pos = NePosition.INSIDE
                    ner_tokens.append(NerToken(index, w, normalize_for_ner(w), type_, pos))
                    index += 1
            sentences.append(NerSentence(line, ner_tokens))
        return NerDataSet(sentences)

    def add_set(self, other: "NerDataSet"):
        self.sentences.extend(other.sentences)
        self.types |= other.types
        self.type_ids |= other.type_ids

    def sub_set(self, start: int, end: int) -> "NerDataSet":
        return NerDataSet(list(self.sentences[start:end]))

    def info(self) -> str:
        type_hist: Dict[str, int] = defaultdict(int)
        tok_hist: Dict[str, int] = defaultdict(int)
        n_tok = 0
        for s in self.sentences:
            n_tok += len(s.tokens)
            for t in s.tokens:
                tok_hist[t.type] += 1
                if t.position in (NePosition.OUTSIDE, NePosition.BEGIN, NePosition.UNIT):
                    type_hist[t.type] += 1
        lines = [f"Number of sentences      = {len(self.sentences)}",
                 f"Number of tokens         = {n_tok}"]
        for t, c in sorted(type_hist.items(), key=lambda kv: -kv[1]):
            lines.append(f"Type = {t} (Count = {c}, Token Count = {tok_hist[t]} Av. Token = {tok_hist[t] / c:.2f} )")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Öznitelikler (PerceptronNer.FeatureData)
# ---------------------------------------------------------------------------

def _j(x) -> str:
    """Java string birleştirmesindeki null/boolean davranışı."""
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "true" if x else "false"
    return str(x)


class _FeatureExtractor:
    def __init__(self, morphology: TurkishMorphology):
        self.morphology = morphology
        self._morph_cache: Dict[str, List[str]] = {}

    def _morph_features(self, word: Optional[str], prefix: str, out: List[str]):
        if word is None:
            return
        key = prefix + "\t" + word
        cached = self._morph_cache.get(key)
        if cached is None:
            cached = self._compute_morph(word, prefix)
            if len(self._morph_cache) < 300_000:
                self._morph_cache[key] = cached
        out.extend(cached)

    def _compute_morph(self, word: str, prefix: str) -> List[str]:
        analyses = list(self.morphology.analyze(word))
        if not analyses:
            return []  # Java: SingleAnalysis.unknown -> isUnknown -> return
        longest = analyses[-1]
        for a in analyses:
            if a.is_unknown():
                return []
            if a is longest:
                continue
            if len(get_lemmas(a)[-1]) > len(get_lemmas(longest)[-1]):
                longest = a
        feats = [f"{prefix}Stem:{longest.get_stem()}"]
        ending = longest.get_ending()
        if ending:
            feats.append(f"{prefix}Ending:{ending}")
        feats.append(f"{prefix}LongLemma:{get_lemmas(longest)[-1]}")
        feats.append(f"{prefix}POS:{get_pos(longest).name}")
        last = longest.get_group(int(longest.group_boundaries.shape[0]) - 1)
        feats.append(f"{prefix}LastIg:{last.lexical_form()}")
        return feats

    @staticmethod
    def _word_features(word: Optional[str], prefix: str, out: List[str]):
        if word is None:
            return
        out.append(f"{prefix}Upper:{_j(word[0].isupper())}")
        out.append(f"{prefix}Punct:{_j(len(word) == 1)}")
        out.append(f"{prefix}AllCap:{_j(all(c.isupper() for c in word))}")
        s = TextUtil.normalize_apostrophes(word)
        ai = s.find("'")
        out.append(f"{prefix}Apost:{_j(ai >= 0)}")
        if ai >= 0:
            out.append(f"{prefix}Stem:{word[:ai]}")
            out.append(f"{prefix}Ending:{word[ai + 1:]}")

    def features(self, tokens: Sequence[NerToken], i: int) -> List[str]:
        n = len(tokens)
        cw, cwo = tokens[i].normalized, tokens[i].word
        nw = nw2 = nwo = pw = pw2 = pwo = None
        # Java'daki gibi (nextWord2Orig/previousWord2Orig hiç kullanılmıyor)
        if i < n - 1:
            nw, nwo = tokens[i + 1].normalized, tokens[i + 1].word
            if i < n - 2:
                nw2 = tokens[i + 2].normalized
        if i > 0:
            pw, pwo = tokens[i - 1].normalized, tokens[i - 1].word
            if i > 1:
                pw2 = tokens[i - 2].normalized

        f = [
            "CW:" + _j(cw),
            "NW:" + _j(nw),
            "CW-NW:" + _j(cw) + _j(nw),
            "2NW:" + _j(nw2),
            "CW+NW+2NW:" + _j(cw) + _j(nw) + _j(nw2),
            "PW:" + _j(pw),
            "2PW:" + _j(pw2),
            "CW-PW:" + _j(cw) + _j(pw),
            "CW-PW-2PW:" + _j(cw) + _j(pw) + _j(pw),  # Java'daki hali korunuyor (model uyumu)
            "PW-CW-NW:" + _j(pw) + _j(cw) + _j(nw),
        ]
        self._word_features(cwo, "CW", f)
        self._word_features(pwo, "PW", f)
        self._word_features(nwo, "NW", f)
        self._morph_features(cwo, "CW", f)
        self._morph_features(pwo, "PW", f)
        self._morph_features(nwo, "NW", f)
        if len(cw) > 2:
            f.append("CWLast2:" + cw[-2:])
            f.append("CWFirst2:" + cw[:2])
        if len(cw) > 3:
            f.append("CWLast3:" + cw[-3:])
            f.append("CWFirst3:" + cw[:3])
        return f


def _history_features(pred_ids: Sequence[str], i: int) -> List[str]:
    f = []
    if i > 0:
        f.append("PreType=" + pred_ids[i - 1])
    if i > 1:
        f.append("2PreType=" + pred_ids[i - 2])
    if i > 2:
        f.append("3PreType=" + pred_ids[i - 3])
    return f


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ClassModel:
    """Bir sınıfın (ör. PER_B) seyrek ağırlıkları."""

    def __init__(self, id_: str, weights: Optional[Dict[str, float]] = None, lookup=None):
        self.id = id_
        self.weights: Dict[str, float] = weights if weights is not None else defaultdict(float)
        self._lookup = lookup  # Java compressed model (salt okunur)

    def get(self, key: str) -> float:
        if self._lookup is not None:
            return float(self._lookup.get_(key))
        return self.weights.get(key, 0.0)

    def update(self, keys: Iterable[str], value: float):
        if self._lookup is not None:
            raise RuntimeError("Sıkıştırılmış model güncellenemez")
        w = self.weights
        for k in keys:
            w[k] = w.get(k, 0.0) + value

    def copy(self) -> "ClassModel":
        return ClassModel(self.id, defaultdict(float, self.weights))


class PerceptronNer:
    def __init__(self, model: Dict[str, ClassModel], morphology: TurkishMorphology):
        self.model = model
        self.morphology = morphology
        self._fx = _FeatureExtractor(morphology)

    # -- tahmin ----------------------------------------------------------

    @staticmethod
    def predict_type_and_position(model: Dict[str, ClassModel], keys: Sequence[str]) -> Tuple[str, float]:
        best, best_score = None, float("-inf")
        for out, cm in model.items():
            if cm._lookup is None:
                w = cm.weights
                score = sum(w.get(k, 0.0) for k in keys)
            else:
                score = sum(cm.get(k) for k in keys)
            if score > best_score:
                best, best_score = out, score
        return best, best_score

    def _tag(self, sentence: NerSentence) -> NerSentence:
        pred: List[NerToken] = []
        ids: List[str] = []
        for i, tok in enumerate(sentence.tokens):
            keys = self._fx.features(sentence.tokens, i) + _history_features(ids, i)
            pid, _ = self.predict_type_and_position(self.model, keys)
            ids.append(pid)
            pred.append(NerToken.from_type_position_string(tok.index, tok.word, tok.normalized, pid))
        return NerSentence(sentence.content, pred)

    def evaluate(self, data: NerDataSet) -> NerDataSet:
        return NerDataSet([self._tag(s) for s in data.sentences])

    def find_named_entities(self, sentence: str, words: Optional[List[str]] = None) -> NerSentence:
        if words is None:
            words = [t.content for t in TurkishTokenizer.DEFAULT.tokenize(sentence)]
        toks = [NerToken(i, w, normalize_for_ner(w), OUT_TOKEN_TYPE, NePosition.OUTSIDE) for i, w in enumerate(words)]
        return self._tag(NerSentence(sentence, toks))

    # -- kaydet / yükle --------------------------------------------------

    def save_model(self, model_root: str):
        """Java'nın text formatında kaydeder ('%.3f anahtar' satırları + types dosyası)."""
        os.makedirs(model_root, exist_ok=True)
        for id_, cm in self.model.items():
            with open(os.path.join(model_root, f"{id_}.ner.model"), "w", encoding="utf-8") as f:
                for k, v in cm.weights.items():
                    if v != 0.0:
                        f.write(f"{v:.3f} {k}\n")
        with open(os.path.join(model_root, "types"), "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(self.model)) + "\n")

    @staticmethod
    def load_model(model_root: str, morphology: TurkishMorphology) -> "PerceptronNer":
        """Java veya bu modülle eğitilmiş (text ya da compressed) modeli yükler."""
        model: Dict[str, ClassModel] = {}
        for fn in os.listdir(model_root):
            if not fn.endswith(".ner.model"):
                continue
            id_ = fn[: -len(".ner.model")]
            path = os.path.join(model_root, fn)
            cm = _load_class_model(id_, path)
            model[id_] = cm
        if not model:
            raise FileNotFoundError(f"{model_root} içinde .ner.model dosyası yok")
        return PerceptronNer(model, morphology)


def _load_class_model(id_: str, path: str) -> ClassModel:
    try:  # Java CompressedWeights mı?
        from zemberek.core.compression.lossy_int_lookup import LossyIntLookup
        from zemberek.core.data.compressed_weights import CompressedWeights
        with open(path, "rb") as f:
            magic = f.read(4)
        if magic == b"\x00\x00\xca\xfe" or magic[:2] == b"\xca\xfe":
            return ClassModel(id_, lookup=CompressedWeights.deserialize(path))
    except Exception:
        pass
    w: Dict[str, float] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            sp = line.find(" ")
            w[line[sp + 1:]] = float(line[:sp])
    return ClassModel(id_, w)


# ---------------------------------------------------------------------------
# Eğitim + değerlendirme (PerceptronNerTrainer)
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    error_count: int = 0
    token_count: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    test_ne_count: int = 0
    correct_ne_count: int = 0

    def token_error_ratio(self):
        return self.error_count / max(self.token_count, 1)

    def token_precision(self):
        return self.true_positives / max(self.true_positives + self.false_positives, 1)

    def token_recall(self):
        return self.true_positives / max(self.true_positives + self.false_negatives, 1)

    def exact_match(self):
        return self.correct_ne_count / max(self.test_ne_count, 1)

    def dump(self) -> str:
        return (f"Token Error ratio   = {self.token_error_ratio():.6f}\n"
                f"NE Token Precision  = {self.token_precision():.6f}\n"
                f"NE Token Recall     = {self.token_recall():.6f}\n"
                f"Exact NE match      = {self.exact_match():.6f}")


def collect_evaluation_data(reference: NerDataSet, prediction: NerDataSet) -> TestResult:
    r = TestResult()
    for ts, ps in zip(reference.sentences, prediction.sentences):
        for tt, pt in zip(ts.tokens, ps.tokens):
            if tt.token_id != pt.token_id:
                r.error_count += 1
                if tt.position == NePosition.OUTSIDE:
                    r.false_positives += 1
                if pt.position == NePosition.OUTSIDE:
                    r.false_negatives += 1
            elif tt.position != NePosition.OUTSIDE:
                r.true_positives += 1
            r.token_count += 1
        nes = ts.named_entities()
        r.test_ne_count += len(nes)
        r.correct_ne_count += len(ps.matching_nes(nes))
    return r


class PerceptronNerTrainer:
    def __init__(self, morphology: TurkishMorphology):
        self.morphology = morphology

    def train(self, training_set: NerDataSet, dev_set: Optional[NerDataSet] = None,
              iteration_count: int = 7, learning_rate: float = 0.1, verbose: bool = True) -> PerceptronNer:
        model = {t: ClassModel(t) for t in training_set.type_ids}
        averages = {t: ClassModel(t) for t in training_set.type_ids}
        counts: Dict[str, int] = defaultdict(int)
        fx = _FeatureExtractor(self.morphology)

        # öznitelikler iterasyonlar arasında değişmiyor (gold geçmiş kullanılıyor) -> önbellek
        feats_cache: Dict[int, List[List[str]]] = {}

        for it in range(iteration_count):
            errors = tokens = 0
            training_set.shuffle()
            for sent in training_set.sentences:
                key = id(sent)
                feats = feats_cache.get(key)
                if feats is None:
                    gold_ids = [t.token_id for t in sent.tokens]
                    feats = [fx.features(sent.tokens, i) + _history_features(gold_ids, i)
                             for i in range(len(sent.tokens))]
                    feats_cache[key] = feats
                for i, tok in enumerate(sent.tokens):
                    tokens += 1
                    gold = tok.token_id
                    f = feats[i]
                    pred, _ = PerceptronNer.predict_type_and_position(model, f)
                    if pred == gold:
                        counts[pred] += 1
                        continue
                    counts[gold] += 1
                    counts[pred] += 1
                    errors += 1
                    model[gold].update(f, +learning_rate)
                    model[pred].update(f, -learning_rate)
                    averages[gold].update(f, counts[gold] * learning_rate)
                    averages[pred].update(f, -counts[pred] * learning_rate)
            if verbose:
                log.warning("Iteration %d, Token error = %.6f", it + 1, errors / max(tokens, 1))
            if dev_set is not None and verbose:
                copy = {k: v.copy() for k, v in model.items()}
                _average(averages, copy, counts)
                res = collect_evaluation_data(dev_set, PerceptronNer(copy, self.morphology).evaluate(dev_set))
                log.warning("\n%s", res.dump())
        _average(averages, model, counts)
        return PerceptronNer(model, self.morphology)


def _average(averages, model, counts):
    for t, cm in model.items():
        a = averages[t].weights
        c = counts.get(t, 0) or 1
        for k in list(cm.weights):
            cm.weights[k] = cm.weights[k] - a.get(k, 0.0) / c


def evaluation_report(reference: NerDataSet, prediction: NerDataSet, path: str):
    with open(path, "w", encoding="utf-8") as pw:
        pw.write("Evaluation Data Information:\n" + reference.info() + "\n")
        pw.write("Summary:\n" + collect_evaluation_data(reference, prediction).dump() + "\n\n")
        pw.write("Detailed Sentence Analysis:\n")
        for ts, ps in zip(reference.sentences, prediction.sentences):
            pw.write(ts.content + "\n")
            for tt, pt in zip(ts.tokens, ps.tokens):
                w = tt.word if tt.word == tt.normalized else f"{tt.word}:{tt.normalized}"
                pw.write(f"{w} {tt.token_id} -> {pt.token_id}\n")
            pw.write("\n")


# ---------------------------------------------------------------------------
# NEPostProcessor (sadeleştirilmiş): "Ankara'da" -> "Ankara", "İstanbullular" -> "İstanbul"
# ---------------------------------------------------------------------------

def strip_entity_suffixes(entity_text: str, morphology: Optional[TurkishMorphology] = None) -> str:
    words = entity_text.split(" ")
    last = TextUtil.normalize_apostrophes(words[-1])
    if "'" in last:
        words[-1] = last[: last.index("'")]
        return " ".join(words)
    if morphology is not None and last[:1].isupper():
        best = None
        low_last = _tr_lower(last)
        for a in morphology.analyze(last):
            lem = a.item.lemma
            if (a.item.secondary_pos in (SecondaryPos.ProperNoun, SecondaryPos.Abbreviation)
                    and len(lem) >= 2 and low_last.startswith(_tr_lower(lem)[:max(2, len(lem) - 1)])):
                if best is None or len(lem) > len(best):
                    best = lem
        if best:
            words[-1] = best if best[0].isupper() else _tr_upper_first(best)
    return " ".join(words)


# ---------------------------------------------------------------------------
# Model gerektirmeyen varlık çıkarıcı
# ---------------------------------------------------------------------------

@dataclass
class EntityMention:
    text: str              # metindeki hali: "Ankara'da"
    canonical: str         # ekleri atılmış: "Ankara"
    key: str               # karşılaştırma anahtarı: "ankara"
    kind: str              # PROPER | ABBRV | CAPITALIZED
    token_span: Tuple[int, int] = (0, 0)


_SENTENCE_BREAKS = {".", "!", "?", ":", "\"", "…", "...", "-", "(", "“", "”", "'"}


class ProperNounExtractor:
    """Morfoloji tabanlı varlık adayı çıkarıcı (NER modeli olmadan).

    Bir kelime şu durumlarda varlık sayılır:
      - belirsizlik giderme sonrası en iyi çözümlemesi Prop (özel isim) veya Abbrv (kısaltma) ise,
      - ya da cümle başında değilken büyük harfle başlıyor ve morfoloji tanımıyorsa.
    Ardışık varlık kelimeleri tek varlıkta birleştirilir ("Mustafa Kemal Atatürk").
    """

    def __init__(self, morphology: TurkishMorphology):
        self.morphology = morphology

    def extract(self, sentence: str) -> List[EntityMention]:
        sa = self.morphology.analyze_and_disambiguate(sentence)
        words = list(sa)
        flags: List[Optional[str]] = []
        for idx, swa in enumerate(words):
            w = swa.word_analysis.inp
            best = swa.best_analysis
            kind = None
            if not best.is_unknown() and best.item.secondary_pos == SecondaryPos.ProperNoun:
                kind = "PROPER"
            elif not best.is_unknown() and best.item.secondary_pos == SecondaryPos.Abbreviation:
                kind = "ABBRV"
            elif (idx > 0 and w[:1].isupper() and any(c.isalpha() for c in w)
                  and words[idx - 1].word_analysis.inp not in _SENTENCE_BREAKS):
                # Cümle ortasında büyük harf: "Ahmet Yılmaz" ('Yılmaz' sözlükte sıfat/fiil)
                kind = "CAPITALIZED"
            elif w[:1].isupper() and best.is_unknown() and any(c.isalpha() for c in w):
                kind = "CAPITALIZED"
            # Cümle başındaki büyük harf tek başına kanıt değil (Prop değilse eleyelim)
            if idx == 0 and kind == "CAPITALIZED":
                kind = None
            # Prop ama küçük harfle yazılmış (ör. 'kitabını' -> 'Kitab') -> yanlış pozitif
            if kind == "PROPER" and not w[:1].isupper():
                kind = None
            flags.append(kind)

        out: List[EntityMention] = []
        i = 0
        while i < len(words):
            if flags[i] is None:
                i += 1
                continue
            j = i
            while j + 1 < len(words) and flags[j + 1] is not None and "'" not in words[j].word_analysis.inp:
                j += 1
            text = " ".join(w.word_analysis.inp for w in words[i:j + 1])
            canonical = strip_entity_suffixes(text, self.morphology)
            kinds = {flags[k] for k in range(i, j + 1)}
            kind = "PROPER" if "PROPER" in kinds else ("ABBRV" if "ABBRV" in kinds else "CAPITALIZED")
            out.append(EntityMention(text, canonical, _tr_lower(canonical), kind, (i, j + 1)))
            i = j + 1
        return out


if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("zemberek").setLevel(logging.WARNING)
    m = TurkishMorphology.create_with_defaults()

    print("== Model gerektirmeyen varlık çıkarıcı ==")
    pe = ProperNounExtractor(m)
    for s in ["Mustafa Kemal Atatürk 1881'de Selanik'te doğdu.",
              "TBMM'de dün yapılan oturumda Ahmet Yılmaz konuştu.",
              "Kitabını okudum ve Google'da aradım."]:
        print(" ", s, "->", [(e.canonical, e.kind) for e in pe.extract(s)])

    print("\n== Perceptron NER: küçük örnek eğitim ==")
    train_lines = [
        "[PER Ahmet Yılmaz] dün [LOC Ankara'ya] gitti .",
        "[PER Ayşe] [ORG Türk Telekom'da] çalışıyor .",
        "[LOC İstanbul] çok kalabalık bir şehir .",
        "[PER Mehmet Öz] [LOC İzmir'den] geldi .",
        "[ORG Koç Holding] yeni yatırım açıkladı .",
        "[PER Zeynep Kaya] [ORG Boğaziçi Üniversitesi'nde] okuyor .",
        "Yarın [LOC Bursa'ya] gideceğiz .",
        "[PER Ali] ve [PER Veli] [LOC Antalya'da] tatil yaptı .",
    ] * 5
    ds = NerDataSet.from_lines(train_lines, AnnotationStyle.BRACKET)
    print(ds.info())
    t = time.time()
    ner = PerceptronNerTrainer(m).train(ds, None, iteration_count=5, learning_rate=0.1, verbose=False)
    print(f"eğitim: {time.time() - t:.1f} sn")
    for s in ["Ahmet Yılmaz yarın İzmir'e gidecek .", "Ayşe Koç Holding'de çalışıyor ."]:
        print(" ", s, "->", ner.find_named_entities(s).named_entities())
