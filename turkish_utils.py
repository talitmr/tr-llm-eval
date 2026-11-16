# Lisans notu: Bu dosya zemberek-nlp (https://github.com/ahmetaa/zemberek-nlp;
# Copyright 2018 Ahmet A. Akın, Mehmet D. Akın; Apache License 2.0) kaynak kodundan
# Python'a uyarlanmış ve değiştirilmiştir. Ayrıntılar için NOTICE dosyasına bakın.
"""
Türkçe yardımcı araçlar:

  TurkishSyllableExtractor, syllable_count, TurkishHyphenator -> heceleme, satır sonu bölme
  char_distance, word_distance                               -> Levenshtein (karakter / kelime)
  TurkishTextToNumberConverter, TurkishNumbersExt            -> yazıyla sayı <-> rakam, sıra sayıları
  get_lemmas, get_stems, get_pos, morpheme_ids               -> morfolojik çözümleme yardımcıları
"""
from __future__ import annotations

import os
import re
from enum import Enum, auto
from typing import Dict, Iterable, List, Optional, Sequence

import zemberek_compat  # noqa: F401  (NumPy 2 uyumluluğu, zemberek'ten önce import edilmeli)
from zemberek.core.turkish import PrimaryPos, TurkishAlphabet

try:
    from pkg_resources import resource_filename

    _ORDINAL_FILE = resource_filename("zemberek", os.path.join("resources", "turkish-ordinal-numbers.txt"))
except Exception:  # pragma: no cover
    _ORDINAL_FILE = None

_ALPHABET = TurkishAlphabet.INSTANCE


# ---------------------------------------------------------------------------
# Heceleme  (TurkishSyllableExtractor.java)
# ---------------------------------------------------------------------------

class TurkishSyllableExtractor:
    """Türkçe kelimeleri hecelerine ayırır.

    Java'daki gibi 'tren', 'spor', 'strateji' gibi kelimeleri ayıramaz (boş liste döner).
    STRICT: 'kart', 'yoğurt' gibi iki ünsüzle biten kelimeleri de ayırmaz.
    """

    DEFAULT: "TurkishSyllableExtractor"
    STRICT: "TurkishSyllableExtractor"

    def __init__(self, strict: bool):
        self.strict = strict

    @staticmethod
    def _is_vowel(c: str) -> bool:
        return _ALPHABET.is_vowel(c)

    def get_syllables(self, word: str) -> List[str]:
        b = self.syllable_boundaries(word)
        if not b:
            return []
        return [word[b[i]: (b[i + 1] if i + 1 < len(b) else len(word))] for i in range(len(b))]

    def syllable_boundaries(self, word: str) -> List[int]:
        chars = list(word)
        out: List[int] = []
        last = len(chars)
        while last > 0:
            n = self._letter_count_for_last_syllable(chars, last)
            if n == -1:
                return []
            out.append(last - n)
            last -= n
        return out[::-1]

    def _letter_count_for_last_syllable(self, c: List[str], end: int) -> int:
        v = self._is_vowel
        if end == 0:
            return -1
        if v(c[end - 1]):
            if end == 1:
                return 1
            if v(c[end - 2]):
                return 1
            if end == 2:
                return 2
            if not v(c[end - 3]) and end == 3:
                return 3
            return 2
        if end == 1:
            return -1
        if v(c[end - 2]):
            if end == 2 or v(c[end - 3]):
                return 2
            if end == 3 or v(c[end - 4]):
                return 3
            if end == 4:
                return -1
            return 3
        if self.strict and not v(c[end - 2]):
            return -1
        if end == 2 or not v(c[end - 3]):
            return -1
        if end > 3 and not v(c[end - 4]):
            return 4
        return 3


TurkishSyllableExtractor.DEFAULT = TurkishSyllableExtractor(strict=False)
TurkishSyllableExtractor.STRICT = TurkishSyllableExtractor(strict=True)


def syllable_count(word: str) -> int:
    """Hece sayısı. Ayrılamayan (yabancı kökenli) kelimelerde ünlü sayısına düşer —
    Türkçede hece sayısı = ünlü sayısı kuralı neredeyse her zaman geçerlidir."""
    w = word.translate(TurkishAlphabet.lower_map).lower()
    s = TurkishSyllableExtractor.DEFAULT.get_syllables(w)
    if s:
        return len(s)
    return max(1, sum(1 for ch in w if _ALPHABET.is_vowel(ch))) if any(ch.isalpha() for ch in w) else 0


class TurkishHyphenator:
    """TurkishHyphenator.splitIndex: satır sonu bölme noktası."""

    def __init__(self, extractor: TurkishSyllableExtractor = TurkishSyllableExtractor.DEFAULT):
        self.extractor = extractor

    def split_index(self, word: str, space_available: int) -> int:
        if space_available >= len(word):
            return len(word)
        pieces = self.extractor.get_syllables(word)
        if not pieces:
            return -1
        remaining, index = space_available, 0
        for p in pieces:
            if len(p) < remaining:
                remaining -= len(p)
                index += 1
            else:
                break
        if index == 0:
            return -1
        return sum(len(p) for p in pieces[:index])


# ---------------------------------------------------------------------------
# Mesafeler  (CharDistance.java, WordDistance.java)
# ---------------------------------------------------------------------------

def char_distance(source: str, target: str) -> int:
    """Levenshtein mesafesi (CharDistance.distance)."""
    n, m = len(source), len(target)
    if n == 0 or m == 0:
        return abs(n - m)
    p = list(range(n + 1))
    for j in range(1, m + 1):
        tj = target[j - 1]
        d = [j] + [0] * n
        for i in range(1, n + 1):
            cost = 0 if source[i - 1] == tj else 1
            d[i] = min(d[i - 1] + 1, p[i] + 1, p[i - 1] + cost)
        p = d
    return p[n]


def word_distance(source: Sequence[str], target: Sequence[str]) -> int:
    """Kelime düzeyinde Levenshtein (WordDistance'ın dizi üzerinde kullanımı)."""
    n, m = len(source), len(target)
    if n == 0 or m == 0:
        return abs(n - m)
    p = list(range(n + 1))
    for j in range(1, m + 1):
        d = [j] + [0] * n
        for i in range(1, n + 1):
            cost = 0 if source[i - 1] == target[j - 1] else 1
            d[i] = min(d[i - 1] + 1, p[i] + 1, p[i - 1] + cost)
        p = d
    return p[n]


# ---------------------------------------------------------------------------
# Sayılar  (TurkishNumbers.java + TurkishTextToNumberConverter.java)
# ---------------------------------------------------------------------------

_NUMBER_TABLE = {
    0: "sıfır", 1: "bir", 2: "iki", 3: "üç", 4: "dört", 5: "beş", 6: "altı", 7: "yedi",
    8: "sekiz", 9: "dokuz", 10: "on", 20: "yirmi", 30: "otuz", 40: "kırk", 50: "elli",
    60: "altmış", 70: "yetmiş", 80: "seksen", 90: "doksan", 100: "yüz", 1000: "bin",
    1_000_000: "milyon", 1_000_000_000: "milyar", 1_000_000_000_000: "trilyon",
    1_000_000_000_000_000: "katrilyon",
}
STRING_TO_NUMBER: Dict[str, int] = {v: k for k, v in _NUMBER_TABLE.items()}
STRING_TO_NUMBER["atmış"] = 60  # Java'daki gibi


def _load_ordinals() -> Dict[str, str]:
    m: Dict[str, str] = {}
    if _ORDINAL_FILE and os.path.exists(_ORDINAL_FILE):
        with open(_ORDINAL_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and ":" in line:
                    k, v = line.split(":", 1)
                    m[k.strip()] = v.strip()
    return m


ORDINAL_MAP = _load_ordinals()
ORDINAL_TO_CARDINAL = {v: k for k, v in ORDINAL_MAP.items()}


class _T(Enum):
    ZERO = auto(); ONE = auto(); TWO_TO_NINE = auto(); TENS = auto(); HUNDRED = auto()
    THOUSAND = auto(); MILLION = auto(); BILLION = auto(); TRILLION = auto(); QUADRILLION = auto()


class _S(Enum):
    START = auto()
    ONES_1 = auto(); ONES_2 = auto(); ONES_3 = auto(); ONES_4 = auto(); ONES_5 = auto(); ONES_6 = auto()
    TENS_1 = auto(); TENS_2 = auto(); TENS_3 = auto()
    HUNDREDS_1 = auto(); HUNDREDS_2 = auto()
    THOUSAND = auto(); END = auto(); ERROR = auto()


def _type_of(n: int) -> _T:
    d = len(str(n))
    if d == 1:
        return _T.ONE if n == 1 else (_T.ZERO if n == 0 else _T.TWO_TO_NINE)
    return {2: _T.TENS, 3: _T.HUNDRED, 4: _T.THOUSAND, 7: _T.MILLION, 10: _T.BILLION,
            13: _T.TRILLION, 16: _T.QUADRILLION}[d]


_BIG = {_T.MILLION, _T.BILLION, _T.TRILLION, _T.QUADRILLION}


class TurkishTextToNumberConverter:
    """Yazıyla sayıyı rakama çevirir. ["seksen","iki","milyon","iki"] -> 82000002.
    Çevrilemezse -1 (Java ile aynı)."""

    def convert(self, words: Iterable[str]) -> int:
        total = 0
        to_add = 0
        prev_mil = 0
        state = _S.START

        def add_to_total(val: int):
            nonlocal total, to_add
            total += val if to_add == 0 else to_add * val
            to_add = 0

        for w in words:
            if w not in STRING_TO_NUMBER:
                return -1
            val = STRING_TO_NUMBER[w]
            t = _type_of(val)
            nxt = _S.ERROR
            if t == _T.ZERO:
                if state == _S.START:
                    nxt = _S.END if total == 0 else _S.ERROR
            elif t == _T.ONE:
                nxt = {_S.START: _S.ONES_1, _S.TENS_1: _S.ONES_3, _S.TENS_2: _S.ONES_4,
                       _S.HUNDREDS_1: _S.ONES_4, _S.TENS_3: _S.ONES_5, _S.HUNDREDS_2: _S.ONES_5,
                       _S.THOUSAND: _S.ONES_5}.get(state, _S.ERROR)
                if nxt != _S.ERROR:
                    to_add += 1
            elif t == _T.TWO_TO_NINE:
                nxt = {_S.START: _S.ONES_2, _S.TENS_1: _S.ONES_3, _S.TENS_2: _S.ONES_4,
                       _S.HUNDREDS_1: _S.ONES_4, _S.TENS_3: _S.ONES_5, _S.HUNDREDS_2: _S.ONES_5,
                       _S.THOUSAND: _S.ONES_6}.get(state, _S.ERROR)
                if nxt != _S.ERROR:
                    to_add += val
            elif t == _T.TENS:
                nxt = {_S.START: _S.TENS_1, _S.HUNDREDS_1: _S.TENS_2, _S.HUNDREDS_2: _S.TENS_3,
                       _S.THOUSAND: _S.TENS_3}.get(state, _S.ERROR)
                if nxt != _S.ERROR:
                    to_add += val
            elif t == _T.HUNDRED:
                nxt = {_S.START: _S.HUNDREDS_1, _S.ONES_2: _S.HUNDREDS_1, _S.THOUSAND: _S.HUNDREDS_2,
                       _S.ONES_6: _S.HUNDREDS_2}.get(state, _S.ERROR)
                if nxt != _S.ERROR:
                    to_add = 100 if to_add == 0 else to_add * 100
            elif t == _T.THOUSAND:
                if state in (_S.START, _S.ONES_2, _S.ONES_3, _S.ONES_4, _S.TENS_1, _S.TENS_2, _S.HUNDREDS_1):
                    add_to_total(val)
                    nxt = _S.THOUSAND
            elif t in _BIG:
                if state in (_S.ONES_1, _S.ONES_2, _S.ONES_3, _S.ONES_4, _S.TENS_1, _S.TENS_2, _S.HUNDREDS_1):
                    if prev_mil == 0 or prev_mil > val:
                        prev_mil = val
                        add_to_total(val)
                        nxt = _S.START
            state = nxt
            if state == _S.ERROR:
                return -1
        return to_add + total


class TurkishNumbersExt:
    """TurkishNumbers.java'nın portta olmayan metotları."""

    _conv = TurkishTextToNumberConverter()
    _SEP = re.compile(r"[0-9]+|[^0-9 ]+")

    @staticmethod
    def convert_to_number(text: str) -> int:
        """'üç yüz kırk iki' -> 342, çevrilemezse -1."""
        return TurkishNumbersExt._conv.convert(re.split(r"[ ]+", text.strip()))

    @staticmethod
    def convert_to_string(n: int) -> str:
        """342 -> 'üç yüz kırk iki' (port sürümü, baştaki sıfır hatası olmadan)."""
        from zemberek.morphology.analysis.tr.turkish_numbers import TurkishNumbers
        return TurkishNumbers.convert_to_string(n)

    @staticmethod
    def convert_number_to_string(s: str) -> str:
        s = s[1:] if s.startswith("+") else s
        zeros = len(s) - len(s.lstrip("0"))
        rest = s.lstrip("0")
        parts = ["sıfır"] * zeros
        if rest:
            parts.append(TurkishNumbersExt.convert_to_string(int(rest)))
        return " ".join(parts)

    @staticmethod
    def single_word_number_value(word: str) -> int:
        if word not in STRING_TO_NUMBER:
            raise ValueError(f"Sayı kelimesi değil: {word}")
        return STRING_TO_NUMBER[word]

    @staticmethod
    def replace_number_strings(words: Sequence[str]) -> List[str]:
        return [str(STRING_TO_NUMBER[w]) if w in STRING_TO_NUMBER else w for w in words]

    @staticmethod
    def separate_connected_numbers(word: str) -> List[str]:
        """'otuzbeş' -> ['otuz', 'beş']; sayı değilse [word]."""
        buf, out, found = "", [], False
        for ch in word:
            buf += ch
            if buf in STRING_TO_NUMBER:
                out.append(buf)
                buf = ""
                found = True
            else:
                found = False
        return out if found else [word]

    @staticmethod
    def convert_ordinal_number_string(s: str) -> str:
        """'21.' -> 'yirmi birinci'"""
        num = int(s.split(".")[0]) if s.endswith(".") else int(s)
        words = TurkishNumbersExt.convert_to_string(num).split()
        last = ORDINAL_MAP.get(words[-1])
        if last is None:
            raise ValueError(f"Sıra sayısı okunuşu bulunamadı: {words[-1]}")
        return " ".join(words[:-1] + [last])

    @staticmethod
    def separate_numbers(s: str) -> List[str]:
        return TurkishNumbersExt._SEP.findall(s)

    @staticmethod
    def has_number(s: str) -> bool:
        return any(c.isdigit() for c in s)

    @staticmethod
    def has_only_number(s: str) -> bool:
        return s.isdigit()


# ---------------------------------------------------------------------------
# SingleAnalysis yardımcıları (Java'da olup portta olmayan metotlar)
# ---------------------------------------------------------------------------

def _group_count(a) -> int:
    return int(a.group_boundaries.shape[0])


def _group_surface(group) -> str:
    return "".join(m.surface for m in group.morphemes)


def get_lemmas(a) -> List[str]:
    """SingleAnalysis.getLemmas: ['güzel', 'güzelleş', 'güzelleştir'] gibi."""
    lemmas = [a.item.root]
    prev = _group_surface(a.get_group(0))
    if prev != a.item.root and prev.endswith("ğ"):
        prev = prev[:-1] + "k"
    for i in range(1, _group_count(a)):
        g = a.get_group(i)
        stem = prev + g.morphemes[0].surface
        if stem.endswith("ğ"):
            stem = stem[:-1] + "k"
        if stem not in lemmas:
            lemmas.append(stem)
        prev = prev + _group_surface(g)
    return lemmas


def get_stems(a) -> List[str]:
    """SingleAnalysis.getStems"""
    stems = [a.get_stem()]
    prev = _group_surface(a.get_group(0))
    for i in range(1, _group_count(a)):
        g = a.get_group(i)
        stem = prev + g.morphemes[0].surface
        if stem not in stems:
            stems.append(stem)
        prev = prev + _group_surface(g)
    return stems


def get_pos(a) -> PrimaryPos:
    """SingleAnalysis.getPos: son çekim grubunun sözcük türü."""
    g = a.get_group(_group_count(a) - 1)
    for m in g.morphemes:
        if m.morpheme.pos is not None and m.morpheme.pos != PrimaryPos.Unknown:
            return m.morpheme.pos
    return PrimaryPos.Unknown


def morpheme_ids(a) -> List[str]:
    return [m.morpheme.id_ for m in a.morpheme_data_list]


if __name__ == "__main__":
    for w in ["merhaba", "kitaplarımızdan", "kart", "tren", "saat", "anlaşılamayacak"]:
        print(f"{w:18} DEFAULT={TurkishSyllableExtractor.DEFAULT.get_syllables(w)} "
              f"STRICT={TurkishSyllableExtractor.STRICT.get_syllables(w)} count={syllable_count(w)}")
    print()
    for t in ["üç yüz kırk iki", "seksen iki milyon iki", "bin dokuz yüz seksen dört", "bir bin", "on bir", "iki yüz bin"]:
        print(f"{t!r:30} -> {TurkishNumbersExt.convert_to_number(t)}")
    print(TurkishNumbersExt.separate_connected_numbers("otuzbeş"), TurkishNumbersExt.convert_ordinal_number_string("21."),
          TurkishNumbersExt.convert_number_to_string("007"))
    print("char_distance(kitap, kitaplar) =", char_distance("kitap", "kitaplar"))
    print("hyphen split 'anlaşılamayacak' @8 ->", TurkishHyphenator().split_index("anlaşılamayacak", 8))
