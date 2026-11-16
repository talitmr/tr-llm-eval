# Lisans notu: Bu dosya zemberek-nlp (https://github.com/ahmetaa/zemberek-nlp;
# Copyright 2018 Ahmet A. Akın, Mehmet D. Akın; Apache License 2.0) kaynak kodundan
# Python'a uyarlanmış ve değiştirilmiştir. Ayrıntılar için NOTICE dosyasına bakın.
"""
Türkçe metin kalitesi ölçümü ve karşılaştırması.

Ölçülenler: morfolojik geçerlilik, normalizasyonun değiştirdiği kelime oranı, 2-gram dil modeli
perplexity'si, ASCII yazım (Türkçe karakter eksikliği), tekrarlanan semboller, sosyal medya öğeleri
(hashtag, mention, URL) ve dil tespiti. Skor 0-100 arasındadır (yüksek = daha kaliteli).

Kullanım:
    python text_quality.py                      # yerleşik örneklerle karşılaştırma
    python text_quality.py a.txt b.txt c.txt    # dosyaları karşılaştır
    python text_quality.py a.txt --json         # JSON çıktı
"""
from __future__ import annotations

import argparse
import difflib
import json
import logging
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import zemberek_compat  # noqa: F401  (NumPy 2 uyumluluğu, zemberek'ten önce import edilmeli)
from zemberek import TurkishMorphology, TurkishSentenceExtractor, TurkishSentenceNormalizer, TurkishTokenizer
from zemberek.core.turkish import TurkishAlphabet
from zemberek.normalization.deasciifier.deasciifier import Deasciifier
from zemberek.tokenization.token import Token

from language_id import LanguageIdentifier

logging.getLogger("zemberek").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# TextUtil.java karşılıkları
# ---------------------------------------------------------------------------

_SPACES = re.compile(r"[\s ​]+")
_SOFT_HYPHEN = re.compile(r"[­]")
_DQUOTES = re.compile(r"[“”»«″\u0093\u0094]|''")
_APOSTROPHES = re.compile(r"[\u0091\u0092′´`’‘]")
_HYPHENS = re.compile(r"[\u0096\u0097–]")


def normalize_spaces_and_soft_hyphens(text: str) -> str:
    """TextUtil.normalizeSpacesAndSoftHyphens"""
    return _SOFT_HYPHEN.sub("", _SPACES.sub(" ", text)).strip()


def normalize_apostrophes(text: str) -> str:
    """TextUtil.normalizeApostrophes"""
    return _APOSTROPHES.sub("'", text)


def normalize_quotes_hyphens(text: str) -> str:
    """TextUtil.normalizeQuotesHyphens"""
    text = _DQUOTES.sub('"', text)
    text = _APOSTROPHES.sub("'", text)
    return _HYPHENS.sub("-", text)


def contains_combining_diacritics(text: str) -> bool:
    """TextUtil.containsCombiningDiacritics: U+0300..U+036F aralığında karakter var mı."""
    return any(0x300 <= ord(c) <= 0x36F for c in text)


def digit_ratio(text: str) -> float:
    """TextUtil.digitRatio"""
    if not text.strip():
        return 0.0
    return sum(c.isdigit() for c in text) / len(text)


def uppercase_ratio(text: str) -> float:
    """TextUtil.uppercaseRatio"""
    if not text.strip():
        return 0.0
    return sum(c.isupper() for c in text) / len(text)


# ---------------------------------------------------------------------------
# NormalizationScripts.java karşılıkları
# ---------------------------------------------------------------------------

_REPEAT_SYMBOLS = frozenset("!.-><?(*)'\"_")


def remove_multiple_symbols(text: str) -> str:
    """NormalizationScripts.removeMultipleSymbols: '!!!' -> '!', '....' -> '.' vb."""
    if len(text) < 2:
        return text
    out = [text[0]]
    prev = text[0]
    for c in text[1:]:
        if c in _REPEAT_SYMBOLS and c == prev:
            continue
        prev = c
        out.append(c)
    return "".join(out)


_SOCIAL_TYPES = {Token.Type.HashTag, Token.Type.Mention, Token.Type.URL}


def is_social_noise(token: Token) -> bool:
    """NormalizationScripts içindeki tweet temizleme filtresinin tersi."""
    t = token.content
    return (
        token.type_ in _SOCIAL_TYPES
        or "_" in t
        or "#" in t
        or "@" in t
        or t == "RT"
    )


# Java'da Turkish olarak kabul eşiği (accepted * 1d / words.size() >= 0.3)
TURKISH_ACCEPT_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# TurkishSentenceNormalizer.probablyRequiresDeasciifier
# ---------------------------------------------------------------------------

def turkish_specific_ratio(text: str) -> float:
    """'ı' ve 'I' hariç Türkçeye özgü harflerin (çğöşüİ...) metne oranı."""
    if not text:
        return 0.0
    alphabet = TurkishAlphabet.INSTANCE
    n = sum(1 for c in text if c not in ("ı", "I") and alphabet.is_turkish_specific(c))
    return n / len(text)


def probably_requires_deasciifier(text: str) -> bool:
    """Java: oran < 0.1 ise metin muhtemelen ASCII yazılmış (ç->c, ş->s ...)."""
    return turkish_specific_ratio(text) < 0.1


# ---------------------------------------------------------------------------
# TextCleaner.java
# ---------------------------------------------------------------------------

_extractor = TurkishSentenceExtractor()


def clean_and_extract_sentences(lines: Sequence[str]) -> List[str]:
    """TextCleaner.cleanAndExtractSentences: '<' ile başlayan satırları at, boşlukları
    düzelt, cümlelere ayır, combining diacritic içeren cümleleri ele."""
    cleaned = [normalize_spaces_and_soft_hyphens(s) for s in lines if not s.startswith("<")]
    cleaned = [s for s in cleaned if s]
    sentences: List[str] = []
    for paragraph in cleaned:
        sentences.extend(_extractor.from_paragraph(paragraph))
    return [s for s in sentences if not contains_combining_diacritics(s)]


# ---------------------------------------------------------------------------
# Kalite raporu
# ---------------------------------------------------------------------------

_WORD_TYPES = {
    Token.Type.Word,
    Token.Type.WordWithSymbol,
    Token.Type.WordAlphanumerical,
    Token.Type.UnknownWord,
}


def turkish_lower(s: str) -> str:
    return s.translate(TurkishAlphabet.lower_map).lower()


@dataclass
class QualityReport:
    name: str
    char_count: int = 0
    sentence_count: int = 0
    word_count: int = 0
    dropped_sentences: int = 0          # TextCleaner'ın elediği cümleler

    # karakter düzeyi (TextUtil)
    digit_ratio: float = 0.0
    uppercase_ratio: float = 0.0
    repeated_symbol_ratio: float = 0.0   # removeMultipleSymbols ile silinen karakter oranı
    turkish_specific_ratio: float = 0.0
    ascii_word_ratio: float = 0.0        # deasciifier'ın düzelttiği kelime oranı
    needs_deasciifier: bool = False
    has_combining_diacritics: bool = False

    # token düzeyi
    social_token_ratio: float = 0.0      # hashtag / mention / URL / RT
    unknown_token_ratio: float = 0.0     # tokenizer'ın Unknown/UnknownWord dediği

    # dil
    language: str = ""                   # lang-id sonucu (tr, en, de, az ...)
    language_confidence: float = 0.0

    # morfoloji
    morph_accept_ratio: float = 0.0      # morfolojik çözümlemesi olan kelime oranı
    is_turkish: bool = False             # lang-id == tr ve morfolojik kabul >= 0.3
    unknown_words: List[str] = field(default_factory=list)

    # normalizasyon
    normalization_edit_ratio: float = 0.0  # normalizer'ın değiştirdiği kelime oranı
    normalized_text: str = ""

    # dil modeli
    lm_oov_ratio: float = 0.0
    lm_perplexity: float = 0.0

    score: float = 0.0                   # 0-100, yüksek = daha kaliteli


class TextQualityAnalyzer:
    # LM perplexity'sinin skora dönüşümünde kullanılan sınırlar (ln ölçeğinde).
    # 2-gram modelde düzgün metinler ~e^5-e^6, bozuk metinler e^9+ civarında çıkıyor.
    PPL_LN_GOOD = 5.0
    PPL_LN_BAD = 10.0

    WEIGHTS = {
        "morph": 0.35,
        "normalization": 0.25,
        "lm": 0.20,
        "surface": 0.20,
    }

    def __init__(self, morphology: Optional[TurkishMorphology] = None):
        self.morphology = morphology or TurkishMorphology.create_with_defaults()
        self.normalizer = TurkishSentenceNormalizer(self.morphology)
        self.lm = self.normalizer.lm  # aynı 2-gram modeli tekrar yüklemeyelim
        self.tokenizer = TurkishTokenizer.DEFAULT
        self.lid = LanguageIdentifier.from_internal_model_group("tr_group")

    # -- yardımcılar -----------------------------------------------------

    def _sentence_log_prob(self, words: List[str]):
        """<s> w1 ... wn </s> için toplam ln olasılık, token sayısı, OOV sayısı."""
        vocab = self.lm.vocabulary
        ids = [vocab.sentence_start_index] + [vocab.index_of(w) for w in words] + [vocab.sentence_end_index]
        oov = sum(1 for i in ids[1:-1] if i == vocab.unknown_word_index)
        total = 0.0
        for a, b in zip(ids, ids[1:]):
            total += float(self.lm.get_probability((a, b)))
        return total, len(ids) - 1, oov

    @staticmethod
    def _word_edit_ratio(a: List[str], b: List[str]) -> float:
        if not a and not b:
            return 0.0
        sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        return 1.0 - sm.ratio()

    # -- ana fonksiyon ---------------------------------------------------

    def analyze(self, text: str, name: str = "metin") -> QualityReport:
        r = QualityReport(name=name, char_count=len(text))
        lines = text.splitlines() or [text]

        raw_sentences: List[str] = []
        for ln in lines:
            ln = normalize_spaces_and_soft_hyphens(ln)
            if ln and not ln.startswith("<"):
                raw_sentences.extend(_extractor.from_paragraph(ln))
        sentences = clean_and_extract_sentences(lines)
        r.sentence_count = len(sentences)
        r.dropped_sentences = len(raw_sentences) - len(sentences)
        r.has_combining_diacritics = contains_combining_diacritics(text)

        flat = normalize_spaces_and_soft_hyphens(text)
        r.digit_ratio = digit_ratio(flat)
        r.uppercase_ratio = uppercase_ratio(flat)
        r.repeated_symbol_ratio = (len(flat) - len(remove_multiple_symbols(flat))) / max(len(flat), 1)
        r.turkish_specific_ratio = turkish_specific_ratio(flat)
        # Java'nın probablyRequiresDeasciifier eşiği (<0.1) düzgün metinleri de işaretliyor;
        # rapor için deasciifier'ın gerçekten kaç kelimeyi değiştirdiğine bakıyoruz.
        if flat:
            before_w = flat.split()
            after_w = Deasciifier(flat).convert_to_turkish().split()
            changed = sum(a != b for a, b in zip(before_w, after_w))
            r.ascii_word_ratio = changed / max(len(before_w), 1)
        r.needs_deasciifier = r.ascii_word_ratio >= 0.15
        r.language, r.language_confidence = self.lid.identify_with_confidence(flat)

        total_tokens = social = unknown_tok = 0
        words: List[str] = []
        accepted = 0
        unknown_words: Dict[str, None] = {}
        edit_ratios: List[float] = []
        normalized_sentences: List[str] = []
        lp_total = 0.0
        lp_tokens = 0
        oov_total = 0
        lm_words_total = 0

        for sent in sentences:
            # NormalizationScripts'teki ön işleme sırası
            s = remove_multiple_symbols(normalize_quotes_hyphens(sent))
            tokens = self.tokenizer.tokenize(s)
            total_tokens += len(tokens)
            for t in tokens:
                if is_social_noise(t):
                    social += 1
                if t.type_ in (Token.Type.Unknown, Token.Type.UnknownWord):
                    unknown_tok += 1
                if t.type_ in _WORD_TYPES and not is_social_noise(t):
                    words.append(t.content)
                    if self.morphology.analyze(t.content).is_correct():
                        accepted += 1
                    else:
                        unknown_words.setdefault(t.content, None)

            # normalizasyon: ne kadar değişiyor?
            try:
                norm = self.normalizer.normalize(s)
            except Exception:  # port bazı uç durumlarda hata verebiliyor
                norm = s
            normalized_sentences.append(norm)
            before = [turkish_lower(t.content) for t in tokens]
            after = [t.content for t in self.tokenizer.tokenize(norm)]
            edit_ratios.append(self._word_edit_ratio(before, after) * max(len(before), 1))

            # dil modeli (normalizer'ın modeli küçük harfli metinle eğitilmiş)
            lm_words = [turkish_lower(t.content) for t in tokens]
            if lm_words:
                lp, n, oov = self._sentence_log_prob(lm_words)
                lp_total += lp
                lp_tokens += n
                oov_total += oov
                lm_words_total += len(lm_words)

        r.word_count = len(words)
        r.social_token_ratio = social / max(total_tokens, 1)
        r.unknown_token_ratio = unknown_tok / max(total_tokens, 1)
        r.morph_accept_ratio = accepted / max(len(words), 1)
        r.is_turkish = r.language == "tr" and r.morph_accept_ratio >= TURKISH_ACCEPT_THRESHOLD
        r.unknown_words = list(unknown_words)[:20]
        r.normalization_edit_ratio = sum(edit_ratios) / max(total_tokens, 1)
        r.normalized_text = " ".join(normalized_sentences)
        r.lm_oov_ratio = oov_total / max(lm_words_total, 1)
        r.lm_perplexity = math.exp(-lp_total / lp_tokens) if lp_tokens else float("inf")

        r.score = self._score(r)
        return r

    def _score(self, r: QualityReport) -> float:
        if r.word_count == 0:
            return 0.0
        clamp = lambda x: max(0.0, min(1.0, x))

        morph = r.morph_accept_ratio
        norm = clamp(1.0 - 2.0 * r.normalization_edit_ratio)
        ln_ppl = math.log(r.lm_perplexity) if r.lm_perplexity > 0 else self.PPL_LN_BAD
        lm = clamp((self.PPL_LN_BAD - ln_ppl) / (self.PPL_LN_BAD - self.PPL_LN_GOOD))

        surface = 1.0
        surface -= min(0.35, r.ascii_word_ratio)
        if r.has_combining_diacritics:
            surface -= 0.15
        surface -= min(0.2, r.repeated_symbol_ratio * 5)
        surface -= min(0.2, r.social_token_ratio * 2)
        surface -= min(0.15, max(0.0, r.uppercase_ratio - 0.3))
        surface -= min(0.1, max(0.0, r.digit_ratio - 0.2))
        surface = clamp(surface)

        w = self.WEIGHTS
        total = w["morph"] * morph + w["normalization"] * norm + w["lm"] * lm + w["surface"] * surface
        if not r.is_turkish:
            total *= 0.3
        return round(100 * total, 1)

    def compare(self, texts: Dict[str, str]) -> List[QualityReport]:
        reports = [self.analyze(t, name) for name, t in texts.items()]
        return sorted(reports, key=lambda r: r.score, reverse=True)


# ---------------------------------------------------------------------------
# Çıktı
# ---------------------------------------------------------------------------

def print_table(reports: List[QualityReport]) -> None:
    cols = [
        ("Metin", lambda r: r.name[:22], 22),
        ("Skor", lambda r: f"{r.score:5.1f}", 5),
        ("Morf.✓", lambda r: f"{r.morph_accept_ratio:6.0%}", 6),
        ("Norm.Δ", lambda r: f"{r.normalization_edit_ratio:6.0%}", 6),
        ("PPL", lambda r: f"{r.lm_perplexity:7.0f}", 7),
        ("OOV", lambda r: f"{r.lm_oov_ratio:5.0%}", 5),
        ("ASCII", lambda r: f"{r.ascii_word_ratio:5.0%}", 5),
        ("Sembol", lambda r: f"{r.repeated_symbol_ratio:6.1%}", 6),
        ("Sosyal", lambda r: f"{r.social_token_ratio:6.0%}", 6),
        ("Dil", lambda r: f"{r.language} {r.language_confidence:.2f}", 7),
        ("TR?", lambda r: "evet" if r.is_turkish else "HAYIR", 5),
    ]
    print("  ".join(h.ljust(w) for h, _, w in cols))
    print("  ".join("-" * w for _, _, w in cols))
    for r in reports:
        print("  ".join(f(r).ljust(w) for _, f, w in cols))
    print()
    for r in reports:
        print(f"[{r.name}]")
        if r.unknown_words:
            print(f"   tanınmayan kelimeler : {', '.join(r.unknown_words[:10])}")
        print(f"   normalize edilmiş    : {r.normalized_text[:160]}")
    print(
        "\nMorf.✓: morfolojik olarak geçerli kelime oranı | Norm.Δ: normalizer'ın değiştirdiği oran"
        "\nPPL: 2-gram dil modeli perplexity (düşük = daha doğal) | ASCII: Türkçe karakteri eksik kelime oranı"
    )


SAMPLES = {
    "düzgün haber": (
        "Meteoroloji Genel Müdürlüğü, yarın İstanbul'da kuvvetli yağış beklendiğini açıkladı. "
        "Vatandaşların dikkatli olması ve gereksiz yolculuklardan kaçınması istendi."
    ),
    "ascii yazılmış": (
        "Meteoroloji Genel Mudurlugu, yarin Istanbul'da kuvvetli yagis beklendigini acikladi. "
        "Vatandaslarin dikkatli olmasi ve gereksiz yolculuklardan kacinmasi istendi."
    ),
    "sosyal medya": (
        "RT @haberci: yrn istanbulda yagmur varmıs!!!! dikkat edinnnn #hava "
        "bnce evde kalın yaa :) http://t.co/abc"
    ),
    "yazım hatalı": (
        "Meteorloji Genel Müdürlüğu, yarn İstanbulda kuvetli yağış beklendğini açıkldı. "
        "Vatandaşlarn dikatli olmasi ve gereksz yolculuklrdan kaçınması istendi."
    ),
    "ingilizce": (
        "The weather service announced heavy rain is expected in Istanbul tomorrow. "
        "Citizens were asked to be careful and avoid unnecessary travel."
    ),
}


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Türkçe metin kalitesi karşılaştırma")
    ap.add_argument("files", nargs="*", help="Karşılaştırılacak metin dosyaları (boşsa örnekler)")
    ap.add_argument("--json", action="store_true", help="JSON çıktı ver")
    args = ap.parse_args(argv)

    if args.files:
        texts = {}
        for p in args.files:
            with open(p, encoding="utf-8") as f:
                texts[p] = f.read()
    else:
        texts = SAMPLES

    analyzer = TextQualityAnalyzer()
    reports = analyzer.compare(texts)

    if args.json:
        json.dump([asdict(r) for r in reports], sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        print_table(reports)


if __name__ == "__main__":
    main()
