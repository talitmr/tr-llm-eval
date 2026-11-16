"""
Türkçe LLM çıktıları için morfoloji tabanlı değerlendirme metrikleri.

Türkçe eklemeli bir dil: "kitap", "kitaplarımızdan", "kitabını" yüzeyde farklı ama aynı kelime.
İngilizce için tasarlanmış BLEU/ROUGE/anahtar kelime kontrolleri bu yüzden Türkçede yanıltıcı.
Bu modül morfolojik çözümleme kullanarak:

  1. Referans tabanlı   : yüzey + LEMMA + morfem düzeyinde BLEU / ROUGE-1/2/L / chrF / WER
  2. Kaynağa sadakat    : kaynakta olmayan sayılar, varlıklar (özel isimler) ve
                          OLUMSUZLUK ÇEVİRMELERİ ("geldi" -> "gelmedi", "var" -> "yok")
  3. Dil                : cümle bazında dil tespiti (yanıt gerçekten Türkçe mi, kod karışımı var mı)
  4. Akıcılık           : morfolojik geçerlilik, ASCII yazım (ç/ş/ğ eksik), argo/informal yazım,
                          yazım önerileri, (opsiyonel) text_quality skoru
  5. Üslup              : sen/siz tutarlılığı, zaman kipleri, okunabilirlik (Ateşman, Bezirci-Yılmaz),
                          sözcük çeşitliliği (lemma TTR, MTLD), tekrar oranı
  6. Talimata uyum      : eklerden bağımsız anahtar kelime / yasaklı kelime, uzunluk, dil, hitap kontrolü

Kullanım (kod):
    from tr_llm_eval import TurkishLLMEvaluator
    ev = TurkishLLMEvaluator()
    ev.evaluate(response="...", reference="...", source="...",
                constraints={"keywords": ["kitap"], "register": "siz", "max_words": 120})

Kullanım (komut satırı, JSONL):
    python tr_llm_eval.py girdi.jsonl -o sonuc.jsonl
    # her satır: {"id":..., "response":..., "reference":..., "source":..., "constraints": {...}}
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from zemberek import TurkishMorphology, TurkishSentenceExtractor
from zemberek.core.turkish import PrimaryPos, SecondaryPos, TurkishAlphabet
from zemberek.normalization.deasciifier.deasciifier import Deasciifier

from turkish_utils import (ORDINAL_TO_CARDINAL, STRING_TO_NUMBER, TurkishTextToNumberConverter,
                            get_lemmas, get_pos, syllable_count, word_distance)
from language_id import LanguageIdentifier
from named_entities import ProperNounExtractor, PerceptronNer, strip_entity_suffixes

logging.getLogger("zemberek").setLevel(logging.WARNING)

_ALPHABET = TurkishAlphabet.INSTANCE


def tr_lower(s: str) -> str:
    return s.translate(TurkishAlphabet.lower_map).lower()


# ===========================================================================
# Kelime ve metin çözümlemesi
# ===========================================================================

_FUNCTION_POS = {PrimaryPos.Punctuation, PrimaryPos.Conjunction, PrimaryPos.Determiner,
                 PrimaryPos.PostPositive, PrimaryPos.Question, PrimaryPos.Pronoun,
                 PrimaryPos.Interjection, PrimaryPos.Duplicator}
_CONTENT_POS = {PrimaryPos.Noun, PrimaryPos.Verb, PrimaryPos.Adjective, PrimaryPos.Adverb}
_SPECIAL_SEC = {SecondaryPos.Url, SecondaryPos.Email, SecondaryPos.Mention, SecondaryPos.HashTag,
                SecondaryPos.Emoticon, SecondaryPos.Date, SecondaryPos.Clock}
_TENSES = {"Past": "geçmiş (-di)", "Narr": "duyulan geçmiş (-miş)", "Fut": "gelecek (-ecek)",
           "Prog1": "şimdiki (-iyor)", "Prog2": "şimdiki (-mekte)", "Aor": "geniş (-ir)",
           "Cond": "şart (-se)", "Imp": "emir", "Opt": "istek (-e)", "Neces": "gereklilik (-meli)",
           "Desr": "dilek (-se)"}


@dataclass
class Word:
    surface: str
    lower: str
    lemma: str            # disambiguated lemma, küçük harf ("gelmek", "kitap", "ankara")
    root: str             # kök ("gel", "kitap")
    pos: str
    sec_pos: str
    morphemes: List[str]
    morph_segments: List[str]   # ["kitap", "+A3pl", "+P1pl", "+Abl"]
    known: bool                 # morfolojik çözümleme var mı
    is_punct: bool
    is_number: bool
    is_proper: bool
    is_special: bool            # URL, e-posta, tarih, saat, emoji ...
    negated: bool = False
    weak_2sg: bool = False      # eksiz emir kipi + isim okuması da var ("dönüş", "yap") -> zayıf "sen" kanıtı

    @property
    def is_word(self) -> bool:
        return not (self.is_punct or self.is_special)

    @property
    def is_content(self) -> bool:
        return self.is_word and not self.is_number and self.pos in {p.name for p in _CONTENT_POS}


@dataclass
class Sentence:
    text: str
    words: List[Word]


@dataclass
class TextAnalysis:
    text: str
    sentences: List[Sentence]

    @property
    def words(self) -> List[Word]:
        return [w for s in self.sentences for w in s.words]

    def tokens(self, view: str = "surface") -> List[str]:
        """Karşılaştırma için token dizisi. view: surface | lemma | root | morph"""
        out: List[str] = []
        for w in self.words:
            if not w.is_word:
                continue
            if view == "surface":
                out.append(w.lower)
            elif view == "lemma":
                out.append(w.lemma)
            elif view == "root":
                out.append(w.root)
            elif view == "morph":
                out.extend(w.morph_segments)
            else:
                raise ValueError(view)
        return out


# ===========================================================================
# n-gram metrikleri (bağımlılıksız)
# ===========================================================================

def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu(hyp: Sequence[str], ref: Sequence[str], max_n: int = 4) -> float:
    """Cümle BLEU, n>1 için +1 düzeltmesi (Lin & Och 2004). 0-100."""
    if not hyp or not ref:
        return 0.0
    log_p = 0.0
    for n in range(1, max_n + 1):
        h, r = _ngrams(hyp, n), _ngrams(ref, n)
        match = sum(min(c, r[g]) for g, c in h.items())
        total = max(sum(h.values()), 0)
        if n == 1:
            if match == 0:
                return 0.0
            p = match / total
        else:
            p = (match + 1) / (total + 1)
        log_p += math.log(p) / max_n
    bp = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref) / len(hyp))
    return 100 * bp * math.exp(log_p)


def _prf(match: float, n_hyp: float, n_ref: float, beta: float = 1.0) -> Tuple[float, float, float]:
    p = match / n_hyp if n_hyp else 0.0
    r = match / n_ref if n_ref else 0.0
    if p + r == 0:
        return p, r, 0.0
    b2 = beta * beta
    return p, r, (1 + b2) * p * r / (b2 * p + r)


def rouge_n(hyp: Sequence[str], ref: Sequence[str], n: int) -> float:
    h, r = _ngrams(hyp, n), _ngrams(ref, n)
    match = sum(min(c, r[g]) for g, c in h.items())
    return 100 * _prf(match, sum(h.values()), sum(r.values()))[2]


def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(hyp: Sequence[str], ref: Sequence[str]) -> float:
    return 100 * _prf(_lcs(hyp, ref), len(hyp), len(ref))[2]


def chrf(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    """chrF (Popović 2015), boşluklar atılarak, sacrebleu varsayılanlarıyla. 0-100."""
    h, r = hyp.replace(" ", ""), ref.replace(" ", "")
    scores = []
    for k in range(1, n + 1):
        hg, rg = _ngrams(h, k), _ngrams(r, k)
        if not hg or not rg:
            continue
        match = sum(min(c, rg[g]) for g, c in hg.items())
        p = match / sum(hg.values())
        rc = match / sum(rg.values())
        scores.append((p, rc))
    if not scores:
        return 0.0
    p = sum(s[0] for s in scores) / len(scores)
    rc = sum(s[1] for s in scores) / len(scores)
    if p + rc == 0:
        return 0.0
    b2 = beta * beta
    return 100 * (1 + b2) * p * rc / (b2 * p + rc)


def mtld(tokens: Sequence[str], threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010), ileri+geri ortalaması."""
    def one_way(seq):
        factors, types, count = 0.0, set(), 0
        for t in seq:
            count += 1
            types.add(t)
            if len(types) / count <= threshold:
                factors += 1
                types, count = set(), 0
        if count > 0:
            ttr = len(types) / count
            factors += (1 - ttr) / (1 - threshold) if ttr < 1 else 0
        return len(seq) / factors if factors > 0 else float(len(seq))
    if len(tokens) < 10:
        return float("nan")
    return (one_way(tokens) + one_way(list(reversed(tokens)))) / 2


# ===========================================================================
# Sayı çıkarımı
# ===========================================================================

_DATE_RE = re.compile(r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b")
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
_NUM_RE = re.compile(r"%?\d+(?:[.,]\d+)*%?")
_SCALES = {"bin": 1e3, "milyon": 1e6, "milyar": 1e9, "trilyon": 1e12, "katrilyon": 1e15}
# "85 milyon", "1,2 milyara", "372 bin'i"; "bina", "binmek" gibi kelimeleri dışarıda bırakır
_SCALE_RE = re.compile(r"\s+(katrilyon|trilyon|milyar|milyon|bin(?!a|di|er|iyor|me|mi[şs]|ece|ip))[\w']*",
                       re.IGNORECASE)


def _number_candidates(s: str) -> Set[float]:
    """'1.250,5' -> {1250.5}; '1,250' -> {1.25, 1250} (belirsiz, ikisi de); '3.5' -> {3.5}"""
    s = s.strip("%")
    out: Set[float] = set()
    if "." in s and "," in s:
        if s.rfind(",") > s.rfind("."):  # Türkçe: 1.250,5
            out.add(float(s.replace(".", "").replace(",", ".")))
        else:                            # İngilizce: 1,250.5
            out.add(float(s.replace(",", "")))
    elif "." in s:
        if re.fullmatch(r"\d{1,3}(\.\d{3})+", s):
            out.add(float(s.replace(".", "")))
            if s.count(".") == 1:
                out.add(float(s))
        else:
            try:
                out.add(float(s))
            except ValueError:
                pass
    elif "," in s:
        if re.fullmatch(r"\d{1,3}(,\d{3})+", s):
            out.add(float(s.replace(",", "")))
        if s.count(",") == 1:
            out.add(float(s.replace(",", ".")))
    else:
        out.add(float(s))
    return out


_FOLD = str.maketrans("çğıöşüâîûÇĞİÖŞÜÂÎÛ", "cgiosuaiuCGIOSUAIU")
_ABBR_SKIP = {"ve", "ile", "of", "and", "the", "de", "da"}


def _fold(s: str) -> str:
    """Türkçe karakterleri ASCII'ye indirip küçült (ASCII yazılmış metinleri eşleştirmek için)."""
    return tr_lower(s).translate(_FOLD)


def _is_abbreviation_of(abbr: str, words: Sequence[str]) -> bool:
    """abbr, kelimelerin sırayla alınmış (boş olmayan) önekleriyle oluşuyor mu?
    'tüik' ~ ['türkiye','istatistik','kurumu'], 'odtü' ~ ['orta','doğu','teknik','üniversitesi']"""
    a = re.sub(r"[^a-z0-9]", "", _fold(abbr))
    ws = [_fold(w) for w in words if _fold(w) not in _ABBR_SKIP]
    if len(a) < 2 or len(ws) < 2 or len(a) > 3 * len(ws):
        return False

    @lru_cache(maxsize=None)
    def match(ai: int, wi: int) -> bool:
        if wi == len(ws):
            return ai == len(a)
        w = ws[wi]
        for k in range(1, min(len(w), len(a) - ai) + 1):
            if w[:k] == a[ai:ai + k] and match(ai + k, wi + 1):
                return True
        return False

    return match(0, 0)


@dataclass
class NumberMention:
    text: str
    values: Set[float]
    kind: str  # digit | word | date | time

    def to_json(self):
        return {"text": self.text, "values": sorted(self.values), "kind": self.kind}


# ===========================================================================
# Değerlendirici
# ===========================================================================

class TurkishLLMEvaluator:
    def __init__(self, morphology: Optional[TurkishMorphology] = None,
                 ner_model_dir: Optional[str] = None, langid_group: str = "tr_group"):
        self.morphology = morphology or TurkishMorphology.create_with_defaults()
        self.extractor = TurkishSentenceExtractor()
        self.lid = LanguageIdentifier.from_internal_model_group(langid_group)
        self.entity_extractor = ProperNounExtractor(self.morphology)
        self.ner: Optional[PerceptronNer] = (
            PerceptronNer.load_model(ner_model_dir, self.morphology) if ner_model_dir else None)
        self._num_conv = TurkishTextToNumberConverter()
        self._informal_morph: Optional[TurkishMorphology] = None
        self._spell = None
        self._quality = None
        self._cache: Dict[str, TextAnalysis] = {}

    # -----------------------------------------------------------------------
    # Çözümleme
    # -----------------------------------------------------------------------

    def analyze(self, text: str) -> TextAnalysis:
        hit = self._cache.get(text)
        if hit is not None:
            return hit
        sentences: List[Sentence] = []
        for para in [p for p in re.split(r"\n\s*\n|\n", text) if p.strip()]:
            for s in self.extractor.from_paragraph(para.strip()):
                sentences.append(Sentence(s, self._analyze_sentence(s)))
        res = TextAnalysis(text, sentences)
        if len(self._cache) > 5000:
            self._cache.clear()
        self._cache[text] = res
        return res

    def _analyze_sentence(self, s: str) -> List[Word]:
        try:
            sa = list(self.morphology.analyze_and_disambiguate(s))
        except Exception:
            sa = []
        words: List[Word] = []
        for swa in sa:
            surf = swa.word_analysis.inp
            b = swa.best_analysis
            unknown = b.is_unknown()
            pos = b.item.primary_pos
            sec = b.item.secondary_pos
            low = tr_lower(surf)
            if unknown:
                base = low.split("'")[0] if "'" in low else low
                lemma = root = base
                morphs: List[str] = []
                segs = [base]
            else:
                lemma = tr_lower(b.item.lemma)
                root = tr_lower(b.item.root)
                morphs = [m.morpheme.id_ for m in b.morpheme_data_list]
                segs = [lemma] + ["+" + m.morpheme.id_ for m in b.morpheme_data_list[1:] if m.surface]
            is_punct = pos == PrimaryPos.Punctuation or (unknown and not any(c.isalnum() for c in surf))
            is_number = (pos == PrimaryPos.Numeral) or bool(re.fullmatch(r"[%\d.,:/'a-zığüşöç]*\d[\d.,:/%]*('.*)?", low))
            words.append(Word(
                surface=surf, lower=low, lemma=lemma, root=root, pos=pos.name, sec_pos=sec.name,
                morphemes=morphs, morph_segments=segs, known=not unknown, is_punct=is_punct,
                is_number=is_number,
                is_proper=sec in (SecondaryPos.ProperNoun, SecondaryPos.Abbreviation),
                is_special=sec in _SPECIAL_SEC,
                negated="Neg" in morphs,
                weak_2sg=("Imp" in morphs and "A2sg" in morphs
                          and any(get_pos(a) != PrimaryPos.Verb for a in swa.word_analysis)),
            ))
        return words

    # -----------------------------------------------------------------------
    # 1) Referans tabanlı metrikler
    # -----------------------------------------------------------------------

    def reference_metrics(self, response: str, reference: str) -> Dict[str, Any]:
        h, r = self.analyze(response), self.analyze(reference)
        out: Dict[str, Any] = {}
        for view in ("surface", "lemma", "morph"):
            ht, rt = h.tokens(view), r.tokens(view)
            out[view] = {
                "bleu": round(bleu(ht, rt), 2),
                "rouge1": round(rouge_n(ht, rt, 1), 2),
                "rouge2": round(rouge_n(ht, rt, 2), 2),
                "rougeL": round(rouge_l(ht, rt), 2),
            }
        hs, rs = h.tokens("surface"), r.tokens("surface")
        hl, rl = h.tokens("lemma"), r.tokens("lemma")
        out["chrf"] = round(chrf(" ".join(hs), " ".join(rs)), 2)
        out["wer_surface"] = round(word_distance(rs, hs) / max(len(rs), 1), 3)
        out["wer_lemma"] = round(word_distance(rl, hl) / max(len(rl), 1), 3)
        hc = {w.lemma for w in h.words if w.is_content}
        rc = {w.lemma for w in r.words if w.is_content}
        p, rec, f = _prf(len(hc & rc), len(hc), len(rc))
        out["content_lemma"] = {"precision": round(p, 3), "recall": round(rec, 3), "f1": round(f, 3),
                                "missing": sorted(rc - hc)[:25]}
        out["lemma_gain_rouge1"] = round(out["lemma"]["rouge1"] - out["surface"]["rouge1"], 2)
        return out

    # -----------------------------------------------------------------------
    # 2) Kaynağa sadakat
    # -----------------------------------------------------------------------

    def extract_numbers(self, text: str) -> List[NumberMention]:
        out: List[NumberMention] = []
        rest = text
        for rx, kind in ((_DATE_RE, "date"), (_TIME_RE, "time")):
            for m in rx.finditer(rest):
                out.append(NumberMention(m.group(0), set(), kind))
            rest = rx.sub(" ", rest)
        digit_matches = list(_NUM_RE.finditer(rest))
        for idx, m in enumerate(digit_matches):
            try:
                vals = _number_candidates(m.group(0))
            except ValueError:
                continue
            if not vals:
                continue
            # "1,2 milyar" / "85 milyon 372 bin": ölçek kelimesiyle çarpılmış (ve zincirlenmiş) değer de aday
            scaled = self._scaled_chain(rest, digit_matches, idx)
            vals = {round(v, 6) for v in vals | scaled}
            out.append(NumberMention(m.group(0), vals, "digit"))
        # yazıyla sayılar
        for s in self.analyze(text).sentences:
            run: List[Word] = []
            prev: Optional[Word] = None
            for w in s.words + [None]:  # type: ignore
                lemma = w.lemma if w else None
                # "1,2 milyar"daki "milyar" rakamın parçası; ayrı sayı sayma
                after_digit = (prev is not None and any(c.isdigit() for c in prev.surface)
                               and lemma in _SCALES)
                is_num_word = (w is not None and w.pos == "Numeral" and not (not run and after_digit)
                               and (lemma in STRING_TO_NUMBER or lemma in ORDINAL_TO_CARDINAL))
                if is_num_word:
                    run.append(w)
                elif run:
                    out.extend(self._spelled_numbers(run))
                    run = []
                prev = w
        return out

    @staticmethod
    def _scaled_chain(text: str, matches: List["re.Match"], start: int) -> Set[float]:
        """matches[start] rakamından başlayarak '<sayı> <ölçek>' zincirinin değerlerini döndürür.
        '85 milyon 372 bin' -> {85e6, 85372000}; ölçek yoksa boş küme."""
        out: Set[float] = set()
        total = 0.0
        prev_scale = float("inf")
        i = start
        while i < len(matches):
            m = matches[i]
            sm = _SCALE_RE.match(text, m.end())
            if not sm:
                break
            scale = _SCALES[sm.group(1).lower()]
            if scale >= prev_scale:
                break
            try:
                base = max(_number_candidates(m.group(0)))
            except ValueError:
                break
            total += base * scale
            out.add(total)
            prev_scale = scale
            # zincir devam ediyor mu: ölçek kelimesinden hemen sonra yeni bir rakam
            if i + 1 < len(matches) and text[sm.end():matches[i + 1].start()].strip() == "":
                i += 1
            else:
                break
        return out

    def _spelled_numbers(self, run: List[Word]) -> List[NumberMention]:
        words = [ORDINAL_TO_CARDINAL.get(w.lemma, w.lemma) for w in run]
        text = " ".join(w.surface for w in run)
        val = self._num_conv.convert(words)
        if val >= 0:
            if len(run) == 1 and val == 1:  # tek başına "bir" genelde belirsiz tanımlık
                return []
            return [NumberMention(text, {float(val)}, "word")]
        out = []
        for w, lw in zip(run, words):
            v = STRING_TO_NUMBER.get(lw)
            if v is not None and v != 1:
                out.append(NumberMention(w.surface, {float(v)}, "word"))
        return out

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        ents: Dict[str, Dict[str, str]] = {}
        for s in self.analyze(text).sentences:
            rule_ents = self.entity_extractor.extract(s.text)
            for e in rule_ents:
                ents.setdefault(e.key, {"text": e.text, "canonical": e.canonical, "key": e.key, "kind": e.kind})
            if self.ner is not None:
                # NER yalnızca morfolojik özel isim adaylarıyla örtüşüyorsa kabul (gürültülü metinde
                # cümle başı büyük harfleri varlık sanmasın); sınırları ve türü NER'den alınır.
                rule_words = {w for e in rule_ents for w in tr_lower(e.text).split()}
                for ne in self.ner.find_named_entities(s.text).named_entities():
                    if not any(tr_lower(t.word) in rule_words for t in ne.tokens):
                        continue
                    canon = strip_entity_suffixes(ne.content(), self.morphology)
                    k = tr_lower(canon)
                    ents.setdefault(k, {"text": ne.content(), "canonical": canon, "key": k, "kind": ne.type})
        # Birbirini kapsayanlardan en uzununu tut ("Mehmet Kaya" ⊂ "Genel Müdür Mehmet Kaya")
        kept: List[Dict[str, str]] = []
        for e in sorted(ents.values(), key=lambda x: -len(x["key"])):
            if not any(f" {e['key']} " in f" {k['key']} " for k in kept):
                kept.append(e)
        return kept

    def _capitalized_runs(self, text: str) -> List[List[str]]:
        """Ardışık büyük harfli kelime grupları (kısaltma açılımı adayları)."""
        runs: List[List[str]] = []
        for s in self.analyze(text).sentences:
            cur: List[str] = []
            for w in s.words + [None]:  # type: ignore
                if w is not None and w.is_word and w.surface[:1].isupper():
                    cur.append(tr_lower(w.surface.split("'")[0]))
                    continue
                if w is not None and cur and w.lower in _ABBR_SKIP:
                    cur.append(w.lower)
                    continue
                if len(cur) >= 2:
                    runs.append(cur)
                cur = []
        return runs

    def polarity_facts(self, text: str) -> Dict[str, Set[str]]:
        """Her içerik lemması için görülen kutuplar: {'gelmek': {'+'}, 'kitap/var': {'-'}}.
        Türkçede olumsuzluk ekle (-me), 'değil' ve 'yok' ile ifade edilir."""
        facts: Dict[str, Set[str]] = {}
        for s in self.analyze(text).sentences:
            ws = [w for w in s.words if not w.is_punct]
            for i, w in enumerate(ws):
                if w.lemma in ("değil", "var", "yok"):
                    continue
                if not w.is_content:
                    continue
                pol = "-" if w.negated else "+"
                nxt = ws[i + 1] if i + 1 < len(ws) else None
                key = w.lemma
                if nxt is not None and nxt.lemma == "değil":
                    pol = "+" if pol == "-" else "-"
                elif nxt is not None and nxt.lemma in ("var", "yok") and w.pos != "Verb":
                    key = f"{w.lemma}/var"
                    pol = "+" if nxt.lemma == "var" else "-"
                facts.setdefault(key, set()).add(pol)
        return facts

    def faithfulness(self, response: str, source: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        # sayılar
        rn, sn = self.extract_numbers(response), self.extract_numbers(source)
        s_vals = set().union(*[m.values for m in sn]) if sn else set()
        s_strs = {m.text for m in sn if m.kind in ("date", "time")}
        unsupported = [m.to_json() for m in rn
                       if (m.kind in ("date", "time") and m.text not in s_strs)
                       or (m.kind not in ("date", "time") and not (m.values & s_vals))]
        out["numbers"] = {
            "response_count": len(rn), "unsupported": unsupported,
            "support_ratio": round(1 - len(unsupported) / len(rn), 3) if rn else None,
        }

        # varlıklar
        re_, se = self.extract_entities(response), self.extract_entities(source)
        s_keys = {e["key"] for e in se}
        s_words = {w for e in se for w in e["key"].split()}
        src_low = tr_lower(source)
        src_fold = _fold(source)
        src_lemmas = {w.lemma for w in self.analyze(source).words}
        src_phrases = [e["key"].split() for e in se] + self._capitalized_runs(source)

        def supported(e):
            k = e["key"]
            if k in s_keys or k in src_low or k in src_lemmas or _fold(k) in src_fold:
                return True
            parts = k.split()
            if all(p in s_words or p in src_lemmas for p in parts):
                return True
            # kısaltma <-> açılım (TÜİK ~ Türkiye İstatistik Kurumu)
            if len(parts) == 1 and any(_is_abbreviation_of(k, ph) for ph in src_phrases):
                return True
            if len(parts) > 1 and any(_is_abbreviation_of(x["key"], parts) for x in se if " " not in x["key"]):
                return True
            return False

        bad = [e["canonical"] for e in re_ if not supported(e)]
        out["entities"] = {
            "response": [e["canonical"] for e in re_],
            "unsupported": bad,
            "support_ratio": round(1 - len(bad) / len(re_), 3) if re_ else None,
        }

        # olumsuzluk çevirmeleri
        rp, sp = self.polarity_facts(response), self.polarity_facts(source)
        flips = []
        for k, pols in rp.items():
            if k in sp and not (pols & sp[k]):
                flips.append({"lemma": k, "source": "".join(sorted(sp[k])), "response": "".join(sorted(pols))})
        out["negation_flips"] = flips

        # içerik örtüşmesi
        rc = {w.lemma for w in self.analyze(response).words if w.is_content}
        sc = {w.lemma for w in self.analyze(source).words if w.is_content}
        out["content_lemma_precision"] = round(len(rc & sc) / len(rc), 3) if rc else None
        out["source_coverage"] = round(len(rc & sc) / len(sc), 3) if sc else None
        out["novel_content_lemmas"] = sorted(rc - sc)[:30]
        return out

    # -----------------------------------------------------------------------
    # 3) Dil
    # -----------------------------------------------------------------------

    def language(self, text: str, min_chars: int = 12) -> Dict[str, Any]:
        sents = self.analyze(text).sentences
        total = tr_chars = 0
        foreign = []
        langs = Counter()
        for s in sents:
            n = len(s.text)
            if n < min_chars:
                continue
            lang, conf = self.lid.identify_with_confidence(s.text)
            langs[lang] += n
            total += n
            if lang == "tr":
                tr_chars += n
            else:
                foreign.append({"sentence": s.text[:120], "lang": lang, "confidence": round(conf, 2)})
        whole, conf = self.lid.identify_with_confidence(text) if text.strip() else ("unk", 0.0)
        non_latin = sum(1 for c in text if c.isalpha() and not ("a" <= c.lower() <= "z" or c in "çğıöşüâîûÇĞİÖŞÜÂÎÛ"))
        alpha = sum(1 for c in text if c.isalpha()) or 1
        return {
            "language": whole, "confidence": round(conf, 3),
            "turkish_ratio": round(tr_chars / total, 3) if total else None,
            "foreign_sentences": foreign[:10],
            "non_latin_char_ratio": round(non_latin / alpha, 4),
        }

    # -----------------------------------------------------------------------
    # 4) Akıcılık / yazım
    # -----------------------------------------------------------------------

    def _informal(self) -> TurkishMorphology:
        if self._informal_morph is None:
            self._informal_morph = (TurkishMorphology.builder(self.morphology.lexicon)
                                    .use_informal_analysis().build())
        return self._informal_morph

    def _spell_checker(self):
        if self._spell is None:
            from zemberek import TurkishSpellChecker
            self._spell = TurkishSpellChecker(self.morphology)
        return self._spell

    def fluency(self, text: str, suggestions: bool = True, full_quality: bool = False) -> Dict[str, Any]:
        a = self.analyze(text)
        cands = [w for w in a.words if w.is_word and not w.is_number and any(c.isalpha() for c in w.surface)]
        unknown = [w for w in cands if not w.known and not w.surface[:1].isupper()]
        informal = []
        for w in unknown:
            try:
                ia = list(self._informal().analyze(w.surface))
            except Exception:
                ia = []
            if ia:
                informal.append(w.surface)
        flat = " ".join(w.surface for w in cands)
        ascii_words = []
        if flat:
            for x, y in zip(flat.split(), Deasciifier(flat).convert_to_turkish().split()):
                if x != y:
                    ascii_words.append({"word": x, "fix": y})
        out: Dict[str, Any] = {
            "word_count": len(cands),
            "morph_valid_ratio": round(1 - len(unknown) / len(cands), 3) if cands else None,
            "unknown_words": sorted({w.surface for w in unknown})[:30],
            "informal_words": sorted(set(informal))[:30],
            "ascii_words": ascii_words[:20],
            "ascii_word_ratio": round(len(ascii_words) / len(cands), 3) if cands else 0.0,
        }
        if suggestions and unknown:
            sp = self._spell_checker()
            sugg = {}
            for w in list(dict.fromkeys(x.surface for x in unknown))[:10]:
                try:
                    s = sp.suggest_for_word(w)
                except Exception:
                    s = ()
                if s:
                    sugg[w] = list(s[:3])
            out["spelling_suggestions"] = sugg
        if full_quality:
            if self._quality is None:
                from text_quality import TextQualityAnalyzer
                self._quality = TextQualityAnalyzer(self.morphology)
            q = self._quality.analyze(text)
            out["quality_score"] = q.score
            out["lm_perplexity"] = round(q.lm_perplexity, 1)
            out["normalization_edit_ratio"] = round(q.normalization_edit_ratio, 3)
        return out

    # -----------------------------------------------------------------------
    # 5) Üslup / okunabilirlik / çeşitlilik
    # -----------------------------------------------------------------------

    def style(self, text: str) -> Dict[str, Any]:
        a = self.analyze(text)
        words = [w for w in a.words if w.is_word]
        alpha_words = [w for w in words if any(c.isalpha() for c in w.surface)]
        n_sent = max(len(a.sentences), 1)

        # hitap: sen / siz
        sen = siz = 0
        sen_ex, siz_ex = [], []
        for w in words:
            ms = set(w.morphemes)
            if w.lemma == "sen" or (("A2sg" in ms or "P2sg" in ms) and not w.weak_2sg):
                sen += 1
                sen_ex.append(w.surface)
            if w.lemma in ("siz",) or "A2pl" in ms or "P2pl" in ms:
                siz += 1
                siz_ex.append(w.surface)
        register = "yok" if sen == siz == 0 else ("siz" if sen == 0 else ("sen" if siz == 0 else "karışık"))

        # kipler
        tenses = Counter()
        for w in words:
            if w.pos == "Verb" or "Verb" in w.morphemes:
                for m in w.morphemes:
                    if m in _TENSES:
                        tenses[_TENSES[m]] += 1

        # okunabilirlik
        syl = [syllable_count(w.surface) for w in alpha_words]
        n_w = max(len(alpha_words), 1)
        avg_syl = sum(syl) / n_w
        avg_len = len(alpha_words) / n_sent
        atesman = 198.825 - 40.175 * avg_syl - 2.610 * avg_len
        h = Counter(min(s, 6) for s in syl)
        yod = math.sqrt(max(avg_len * ((h[3] / n_sent) * 0.84 + (h[4] / n_sent) * 1.5
                                       + (h[5] / n_sent) * 3.5 + (h[6] / n_sent) * 26.25), 0.0))

        def atesman_band(x):
            return ("çok kolay" if x >= 90 else "kolay" if x >= 70 else "orta" if x >= 50
                    else "zor" if x >= 30 else "çok zor")

        # çeşitlilik / tekrar
        lemmas = [w.lemma for w in alpha_words]
        tri = _ngrams(lemmas, 3)
        rep_tri = sum(c for c in tri.values() if c > 1)
        sent_texts = [tr_lower(s.text.strip()) for s in a.sentences if len(s.text.strip()) > 10]
        dup_sent = len(sent_texts) - len(set(sent_texts))

        morph_per_word = (sum(len(w.morphemes) for w in alpha_words if w.known)
                          / max(sum(1 for w in alpha_words if w.known), 1))
        return {
            "sentence_count": len(a.sentences),
            "word_count": len(alpha_words),
            "avg_words_per_sentence": round(avg_len, 2),
            "avg_syllables_per_word": round(avg_syl, 2),
            "morphemes_per_word": round(morph_per_word, 2),
            "register": register, "sen_count": sen, "siz_count": siz,
            "register_examples": {"sen": sen_ex[:5], "siz": siz_ex[:5]},
            "tenses": dict(tenses.most_common()),
            "question_sentences": sum(1 for s in a.sentences if s.text.rstrip().endswith("?")),
            "exclamations": text.count("!"),
            "readability": {"atesman": round(atesman, 1), "atesman_level": atesman_band(atesman),
                            "bezirci_yilmaz": round(yod, 2)},
            "diversity": {
                "lemma_ttr": round(len(set(lemmas)) / n_w, 3) if lemmas else None,
                "surface_ttr": round(len({w.lower for w in alpha_words}) / n_w, 3) if alpha_words else None,
                "lemma_mtld": None if len(lemmas) < 10 else round(mtld(lemmas), 1),
            },
            "repetition": {
                "repeated_lemma_trigram_ratio": round(rep_tri / max(sum(tri.values()), 1), 3),
                "duplicate_sentences": dup_sent,
            },
        }

    # -----------------------------------------------------------------------
    # 6) Talimata uyum
    # -----------------------------------------------------------------------

    @lru_cache(maxsize=4096)
    def _keyword_lemmas(self, kw: str) -> Tuple[Tuple[str, ...], ...]:
        """Bir anahtar kelimenin olası lemma dizileri (çok kelimeli olabilir)."""
        parts = kw.split()
        options: List[Set[str]] = []
        for p in parts:
            cand = {tr_lower(p)}
            for an in self.morphology.analyze(p):
                cand.add(tr_lower(an.item.lemma))
                cand.update(tr_lower(x) for x in get_lemmas(an))
            options.append(cand)
        return tuple(tuple(sorted(o)) for o in options)

    def _contains_keyword(self, a: TextAnalysis, kw: str) -> Optional[str]:
        opts = self._keyword_lemmas(kw)
        ws = [w for w in a.words if w.is_word]
        n = len(opts)
        for i in range(len(ws) - n + 1):
            ok = True
            for j in range(n):
                w = ws[i + j]
                lem_set = {w.lemma, w.root, w.lower}
                if not (lem_set & set(opts[j])):
                    ok = False
                    break
            if ok:
                return " ".join(w.surface for w in ws[i:i + n])
        low = tr_lower(a.text)
        return kw if tr_lower(kw) in low else None

    def instruction_checks(self, response: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        a = self.analyze(response)
        res: Dict[str, Any] = {}
        n_words = sum(1 for w in a.words if w.is_word)
        n_sent = len(a.sentences)
        if "keywords" in constraints:
            found = {k: self._contains_keyword(a, k) for k in constraints["keywords"]}
            res["keywords"] = {"pass": all(found.values()), "found_as": found}
        if "forbidden_words" in constraints:
            hits = {k: v for k in constraints["forbidden_words"] if (v := self._contains_keyword(a, k))}
            res["forbidden_words"] = {"pass": not hits, "violations": hits}
        for key, val, cmp in (("min_words", n_words, lambda v, c: v >= c), ("max_words", n_words, lambda v, c: v <= c),
                              ("min_sentences", n_sent, lambda v, c: v >= c),
                              ("max_sentences", n_sent, lambda v, c: v <= c)):
            if key in constraints:
                res[key] = {"pass": cmp(val, constraints[key]), "value": val, "limit": constraints[key]}
        if "language" in constraints:
            lang = self.language(response)
            want = constraints["language"]
            ok = lang["language"] == want and (want != "tr" or (lang["turkish_ratio"] or 0) >= 0.9)
            res["language"] = {"pass": ok, "detected": lang["language"], "turkish_ratio": lang["turkish_ratio"]}
        if "register" in constraints:
            st = self.style(response)
            want = constraints["register"]
            res["register"] = {"pass": st["register"] in (want, "yok"), "detected": st["register"]}
        if constraints.get("turkish_chars"):
            fl = self.fluency(response, suggestions=False)
            res["turkish_chars"] = {"pass": fl["ascii_word_ratio"] < 0.05, "ascii_word_ratio": fl["ascii_word_ratio"]}
        if constraints.get("ends_with_question"):
            res["ends_with_question"] = {"pass": response.rstrip().endswith("?")}
        res["all_pass"] = all(v["pass"] for v in res.values() if isinstance(v, dict) and "pass" in v)
        return res

    # -----------------------------------------------------------------------
    # Hepsi birden
    # -----------------------------------------------------------------------

    def evaluate(self, response: str, reference: Optional[str] = None, source: Optional[str] = None,
                 constraints: Optional[Dict[str, Any]] = None, full_quality: bool = False,
                 spelling_suggestions: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "language": self.language(response),
            "fluency": self.fluency(response, suggestions=spelling_suggestions, full_quality=full_quality),
            "style": self.style(response),
        }
        if reference:
            out["reference"] = self.reference_metrics(response, reference)
        if source:
            out["faithfulness"] = self.faithfulness(response, source)
        if constraints:
            out["instructions"] = self.instruction_checks(response, constraints)
        out["flags"] = self._flags(out)
        return out

    @staticmethod
    def _flags(r: Dict[str, Any]) -> List[str]:
        f = []
        lang = r["language"]
        n_words = r["style"]["word_count"]
        if n_words == 0:
            f.append("boş yanıt")
        elif lang["language"] == "unk" and n_words < 5:
            pass  # çok kısa yanıtta dil tespiti güvenilir değil
        elif lang["language"] != "tr":
            f.append(f"yanıt Türkçe değil ({lang['language']})")
        elif lang["turkish_ratio"] is not None and lang["turkish_ratio"] < 0.9:
            f.append(f"kod karışımı: cümlelerin %{100 - 100 * lang['turkish_ratio']:.0f}'i başka dilde")
        fl = r["fluency"]
        if fl["ascii_word_ratio"] >= 0.1:
            f.append(f"Türkçe karakter eksik (%{100 * fl['ascii_word_ratio']:.0f} kelime)")
        if fl["morph_valid_ratio"] is not None and fl["morph_valid_ratio"] < 0.85:
            f.append(f"tanınmayan kelime oranı yüksek (%{100 - 100 * fl['morph_valid_ratio']:.0f})")
        if fl["informal_words"]:
            f.append("informal yazım: " + ", ".join(fl["informal_words"][:5]))
        st = r["style"]
        if st["register"] == "karışık":
            f.append("sen/siz hitabı karışık")
        if st["repetition"]["duplicate_sentences"] > 0 or st["repetition"]["repeated_lemma_trigram_ratio"] > 0.2:
            f.append("tekrar var")
        fa = r.get("faithfulness")
        if fa:
            if fa["numbers"]["unsupported"]:
                f.append("kaynakta olmayan sayı: " + ", ".join(x["text"] for x in fa["numbers"]["unsupported"][:5]))
            if fa["entities"]["unsupported"]:
                f.append("kaynakta olmayan varlık: " + ", ".join(fa["entities"]["unsupported"][:5]))
            if fa["negation_flips"]:
                f.append("olumsuzluk çevrilmiş: " + ", ".join(x["lemma"] for x in fa["negation_flips"][:5]))
        ins = r.get("instructions")
        if ins and not ins["all_pass"]:
            f.append("talimat ihlali: " + ", ".join(k for k, v in ins.items() if isinstance(v, dict) and not v.get("pass", True)))
        return f


# ===========================================================================
# Komut satırı
# ===========================================================================

def _flatten(d: Any, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    elif isinstance(d, bool):
        out[prefix[:-1]] = float(d)
    elif isinstance(d, (int, float)) and not (isinstance(d, float) and math.isnan(d)):
        out[prefix[:-1]] = float(d)
    return out


def _demo(ev: TurkishLLMEvaluator):
    source = ("Türkiye İstatistik Kurumu'nun açıkladığı verilere göre 2023 yılında Türkiye'nin nüfusu "
              "85 milyon 372 bin kişiye ulaştı. Nüfus artış hızı bir önceki yıla göre yavaşladı. "
              "İstanbul 15 milyon 655 bin kişiyle en kalabalık il oldu.")
    reference = ("TÜİK verilerine göre Türkiye'nin nüfusu 2023'te 85 milyon 372 bine ulaştı ve artış hızı yavaşladı. "
                 "En kalabalık il 15,6 milyon nüfusla İstanbul oldu.")
    responses = {
        "iyi": "TÜİK'e göre Türkiye nüfusu 2023 yılında 85 milyon 372 bin kişiye çıktı ve nüfus artış hızı yavaşladı. "
               "En kalabalık şehir 15 milyon 655 bin kişiyle İstanbul'du.",
        "halüsinasyon": "TÜİK'e göre Türkiye nüfusu 2023 yılında 87 milyon kişiye çıktı ve artış hızı yavaşlamadı. "
                        "Ankara 6 milyon nüfusla ikinci sırada yer aldı.",
        "kötü dil": "Tuik'e gore turkiye nufusu 2023'te 85 milyon oldu. Population growth slowed down. "
                    "Sen de biliyosun, en kalabalik il istanbul. Sizce de öyle değil mi?",
    }
    constraints = {"keywords": ["nüfus", "İstanbul"], "max_words": 40, "language": "tr", "register": "siz",
                   "turkish_chars": True}
    for name, resp in responses.items():
        r = ev.evaluate(resp, reference=reference, source=source, constraints=constraints)
        ref, fa, st = r["reference"], r["faithfulness"], r["style"]
        print(f"\n=== {name} ===\n{resp}")
        print(f"  ROUGE-1 yüzey/lemma : {ref['surface']['rouge1']:.1f} / {ref['lemma']['rouge1']:.1f}   "
              f"BLEU yüzey/lemma: {ref['surface']['bleu']:.1f} / {ref['lemma']['bleu']:.1f}   chrF: {ref['chrf']:.1f}")
        print(f"  sayı desteği        : {fa['numbers']['support_ratio']}  desteksiz: {[x['text'] for x in fa['numbers']['unsupported']]}")
        print(f"  varlık desteği      : {fa['entities']['support_ratio']}  desteksiz: {fa['entities']['unsupported']}")
        print(f"  olumsuzluk çevirme  : {fa['negation_flips']}")
        print(f"  dil                 : {r['language']['language']} (TR oranı {r['language']['turkish_ratio']})")
        print(f"  hitap / okunabilirlik: {st['register']} / Ateşman {st['readability']['atesman']} ({st['readability']['atesman_level']})")
        print(f"  talimatlar          : {'GEÇTİ' if r['instructions']['all_pass'] else 'KALDI'} "
              f"{ {k: v['pass'] for k, v in r['instructions'].items() if isinstance(v, dict)} }")
        print(f"  UYARILAR            : {r['flags']}")


def main(argv: Optional[Sequence[str]] = None):
    ap = argparse.ArgumentParser(description="Türkçe LLM çıktısı değerlendirme")
    ap.add_argument("input", nargs="?", help="JSONL: response, reference?, source?, constraints?")
    ap.add_argument("-o", "--output", help="sonuç JSONL")
    ap.add_argument("--ner-model", help="perceptron NER model klasörü (ör. ner_model)")
    ap.add_argument("--full-quality", action="store_true", help="text_quality skorunu da hesapla (yavaş)")
    ap.add_argument("--no-suggestions", action="store_true", help="yazım önerilerini atla (hızlı)")
    args = ap.parse_args(argv)

    ev = TurkishLLMEvaluator(ner_model_dir=args.ner_model)
    if not args.input:
        _demo(ev)
        return

    results, sums, counts = [], Counter(), Counter()
    with open(args.input, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            resp = row.get("response") or row.get("output") or row.get("prediction") or ""
            r = ev.evaluate(resp, reference=row.get("reference"), source=row.get("source") or row.get("context"),
                            constraints=row.get("constraints"), full_quality=args.full_quality,
                            spelling_suggestions=not args.no_suggestions)
            r["id"] = row.get("id", line_no)
            if "model" in row:
                r["model"] = row["model"]
            results.append(r)
            for k, v in _flatten({k: v for k, v in r.items() if k not in ("id", "model")}).items():
                sums[k] += v
                counts[k] += 1
            print(f"[{r['id']}] uyarılar: {r['flags'] or '-'}", file=sys.stderr)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    keys = [k for k in sums if any(k.startswith(p) for p in (
        "reference.lemma.", "reference.surface.rouge1", "reference.chrf", "faithfulness.numbers.support_ratio",
        "faithfulness.entities.support_ratio", "language.turkish_ratio", "fluency.morph_valid_ratio",
        "fluency.ascii_word_ratio", "style.readability.atesman", "style.diversity.lemma_mtld", "instructions.all_pass"))]
    print(f"\n{len(results)} örnek — ortalamalar:")
    for k in sorted(keys):
        print(f"  {k:45} {sums[k] / counts[k]:8.3f}")


if __name__ == "__main__":
    main()
