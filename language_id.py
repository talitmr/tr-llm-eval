# Lisans notu: Bu dosya zemberek-nlp (https://github.com/ahmetaa/zemberek-nlp;
# Copyright 2018 Ahmet A. Akın, Mehmet D. Akın; Apache License 2.0) kaynak kodundan
# Python'a uyarlanmış ve değiştirilmiştir. Ayrıntılar için NOTICE dosyasına bakın.
"""
Karakter n-gram dil modelleriyle dil tespiti (saf Python).

Model dosyaları ./langid_models altındadır:
  tr_group   -> az, de, en, ku, ky, tr, uz, unk  (Türkçe odaklı, hızlı)
  compressed -> 60 dil

Örnek:
    from language_id import LanguageIdentifier
    lid = LanguageIdentifier.from_internal_model_group("tr_group")
    lid.identify("Bugün hava çok güzel")        # 'tr'
    lid.identify_with_confidence("Hello world") # ('en', 0.99)
"""
from __future__ import annotations

import math
import os
import re
import struct
from array import array
from typing import BinaryIO, Dict, List, Optional, Sequence, Tuple

MODELS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "langid_models")

# ---------------------------------------------------------------------------
# MultiLevelMphf (Java) - yalnızca deserialize + get(String)
# ---------------------------------------------------------------------------

_HASH_MULTIPLIER = 16777619
_INITIAL_HASH_SEED = 0x811C9DC5
_MASK32 = 0xFFFFFFFF


def _utf16_units(s: str) -> List[int]:
    """Java String.charAt() UTF-16 kod birimleri üzerinden çalışır; BMP dışı karakterler
    için de aynı sonucu almak adına aynısını yapıyoruz."""
    if all(ord(c) < 0x10000 for c in s):
        return [ord(c) for c in s]
    b = s.encode("utf-16-be")
    return [(b[i] << 8) | b[i + 1] for i in range(0, len(b), 2)]


def mphf_hash(units: Sequence[int], seed: int) -> int:
    """MultiLevelMphf.hash(String, seed). Java int taşmasını 32-bit maske ile taklit eder."""
    d = seed if seed > 0 else _INITIAL_HASH_SEED
    for a in units:
        d = ((d ^ a) * _HASH_MULTIPLIER) & _MASK32
    return d & 0x7FFFFFFF


def _read_int(f: BinaryIO) -> int:
    return struct.unpack(">i", f.read(4))[0]


def _read_int_array(f: BinaryIO, n: int) -> array:
    a = array("i")
    a.frombytes(f.read(4 * n))
    if struct.pack("=i", 1) != struct.pack(">i", 1):  # little-endian makinede çevir
        a.byteswap()
    return a


class MultiLevelMphf:
    __slots__ = ("levels",)

    def __init__(self, levels: List[Tuple[int, int, bytes, array]]):
        # her seviye: (key_amount, bucket_amount, seed_bytes, failed_indexes)
        self.levels = levels

    @staticmethod
    def deserialize(f: BinaryIO) -> "MultiLevelMphf":
        level_count = _read_int(f)
        levels = []
        for _ in range(level_count):
            key_count = _read_int(f)
            bucket_amount = _read_int(f)
            seeds = f.read(bucket_amount)
            failed_count = _read_int(f)
            failed = _read_int_array(f, failed_count)
            levels.append((key_count, bucket_amount, seeds, failed))
        return MultiLevelMphf(levels)

    def get(self, units: Sequence[int], initial_hash: int) -> int:
        levels = self.levels
        for i, (key_amount, bucket_amount, seeds, _) in enumerate(levels):
            seed = seeds[initial_hash % bucket_amount]  # bytes -> 0..255 (Java'daki & 0xff)
            if seed != 0:
                h = mphf_hash(units, seed) % key_amount
                return h if i == 0 else levels[i - 1][3][h]
        raise RuntimeError("MPHF: buraya gelinmemeli")


# ---------------------------------------------------------------------------
# DoubleLookup (Java) - yalnızca okuma
# ---------------------------------------------------------------------------

def _read_double_lookup(f: BinaryIO) -> Tuple[float, ...]:
    n = _read_int(f)
    return struct.unpack(f">{n}d", f.read(8 * n))


def _read_java_utf(f: BinaryIO) -> str:
    n = struct.unpack(">H", f.read(2))[0]
    return f.read(n).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# CompressedCharNgramModel
# ---------------------------------------------------------------------------

class CompressedCharNgramModel:
    UNK_CHAR_PENALTY = -10.0
    BACK_OFF = -2.0
    FINGER_PRINT_MASK = (1 << 16) - 1

    def __init__(self, order: int, model_id: str, mphfs, prob_data, lookups):
        self.order = order
        self.id = model_id
        self.mphfs = mphfs          # 1 tabanlı listeler (index 0 kullanılmıyor)
        self.prob_data = prob_data  # her eleman: bytes (count*3) -> [fp_hi, fp_lo, prob_idx]
        self.lookups = lookups
        self._cache: Dict[str, float] = {}

    @staticmethod
    def load(path: str) -> "CompressedCharNgramModel":
        with open(path, "rb") as f:
            order = _read_int(f)
            model_id = _read_java_utf(f)
            mphfs: List[Optional[MultiLevelMphf]] = [None] * (order + 1)
            prob_data: List[Optional[bytes]] = [None] * (order + 1)
            lookups: List[Optional[Tuple[float, ...]]] = [None] * (order + 1)
            for i in range(1, order + 1):
                lookups[i] = _read_double_lookup(f)
                count = _read_int(f)
                prob_data[i] = f.read(count * 3)
                mphfs[i] = MultiLevelMphf.deserialize(f)
        return CompressedCharNgramModel(order, model_id, mphfs, prob_data, lookups)

    def gram_probability(self, gram: str) -> float:
        """Basit 'stupid back-off' ile doğal log olasılık."""
        cached = self._cache.get(gram)
        if cached is not None:
            return cached
        result = self._gram_probability(_utf16_units(gram))
        if len(self._cache) < 500_000:
            self._cache[gram] = result
        return result

    def _gram_probability(self, units: List[int]) -> float:
        o = len(units)
        if o == 0:
            return self.UNK_CHAR_PENALTY
        if o > self.order:
            raise ValueError(f"Gram uzunluğu ({o}) model derecesinden ({self.order}) büyük")
        fp = mphf_hash(units, -1)
        idx = self.mphfs[o].get(units, fp)
        data = self.prob_data[o]
        j = idx * 3
        if (fp & self.FINGER_PRINT_MASK) == ((data[j] << 8) | data[j + 1]):
            return self.lookups[o][data[j + 2]]
        return self.BACK_OFF + self._gram_probability(units[:-1])


# ---------------------------------------------------------------------------
# LanguageIdentifier
# ---------------------------------------------------------------------------

_REMOVE_CHARS = re.compile(r"[0-9\"#$%^&*()_+\-=/|\\<>{}\[\];:,]")
_WHITESPACE = re.compile(r"\s+")


class LanguageIdentifier:
    UNKNOWN = "unk"

    def __init__(self, models: Dict[str, CompressedCharNgramModel]):
        if not models:
            raise ValueError("En az bir model gerekli")
        self.models = models
        self.order = next(iter(models.values())).order

    # -- yükleme ---------------------------------------------------------

    @classmethod
    def from_internal_model_group(cls, group_id: str = "tr_group",
                                  root: str = MODELS_ROOT) -> "LanguageIdentifier":
        group_dir = os.path.join(root, group_id)
        with open(os.path.join(group_dir, "langs.txt"), encoding="utf-8") as f:
            langs = [s.strip() for s in f.read().strip().split(",") if s.strip()]
        return cls({l: CompressedCharNgramModel.load(os.path.join(group_dir, l + ".clm")) for l in langs})

    @classmethod
    def from_internal_models(cls, root: str = MODELS_ROOT) -> "LanguageIdentifier":
        """60 dilin tamamı (models/compressed)."""
        d = os.path.join(root, "compressed")
        files = sorted(p for p in os.listdir(d) if p.endswith(".clm"))
        return cls({p[:-4]: CompressedCharNgramModel.load(os.path.join(d, p)) for p in files})

    def languages(self) -> List[str]:
        return [l for l in self.models if l != self.UNKNOWN]

    # -- yardımcılar -----------------------------------------------------

    @staticmethod
    def preprocess(s: str) -> str:
        s = _REMOVE_CHARS.sub("", s)
        s = _WHITESPACE.sub(" ", s)
        return s.lower()

    def _stepping(self, content: str, gram_amount: int) -> List[int]:
        n = len(content)
        if n <= self.order:
            return []
        limit = n - self.order + 1
        if gram_amount <= 0:
            return list(range(limit))
        count = min(gram_amount, limit)
        step = max(3, limit // count)  # Java: varsayılan adım en az 3
        vals = []
        sampling = start = 0
        for _ in range(count):
            vals.append(sampling)
            sampling += step
            if sampling >= limit:
                start += 1
                sampling = start
        return vals

    def _grams(self, content: str, starts: List[int]) -> List[str]:
        o = self.order
        return [content[i:i + o] for i in starts]

    def _model_scores(self, grams: List[str]) -> Dict[str, float]:
        return {mid: sum(m.gram_probability(g) for g in grams) for mid, m in self.models.items()}

    # -- genel API -------------------------------------------------------

    def identify(self, text: str, max_sample_count: int = -1) -> str:
        """LanguageIdentifier.identify(input) / identify(input, maxSampleCount)"""
        clean = self.preprocess(text)
        if len(clean) < self.order:
            return self.UNKNOWN
        amount = len(clean) - 1 if max_sample_count <= 0 else max_sample_count
        scores = self._model_scores(self._grams(clean, self._stepping(clean, amount)))
        return max(scores, key=scores.get)

    def get_scores(self, text: str, max_sample_count: int = -1) -> List[Tuple[str, float]]:
        """Her dil için olasılık (softmax), büyükten küçüğe. Java'daki identifyConf skoru."""
        clean = self.preprocess(text)
        if len(clean) < self.order:
            return []
        amount = len(clean) if max_sample_count <= 0 else max_sample_count
        scores = self._model_scores(self._grams(clean, self._stepping(clean, amount)))
        mx = max(scores.values())
        z = sum(math.exp(v - mx) for v in scores.values())
        probs = {k: math.exp(v - mx) / z for k, v in scores.items()}
        return sorted(probs.items(), key=lambda kv: kv[1], reverse=True)

    def identify_with_confidence(self, text: str, max_sample_count: int = -1) -> Tuple[str, float]:
        scores = self.get_scores(text, max_sample_count)
        return scores[0] if scores else (self.UNKNOWN, 0.0)

    def contains_language(self, content: str, language: str, slice_size: int = 50) -> bool:
        """Metnin herhangi bir diliminde `language` baskın mı (karışık dilli metinler için)."""
        if slice_size < 10:
            raise ValueError("slice_size 10'dan küçük olamaz")
        content = self.preprocess(content)
        if len(content) < self.order:
            return False
        slice_size = min(slice_size, len(content))
        begin = 0
        while begin < len(content):
            end = begin + slice_size
            if len(content) - end < slice_size:
                end = len(content)
            piece = content[begin:end]
            scores = self._model_scores(self._grams(piece, self._stepping(piece, len(piece))))
            if max(scores, key=scores.get) == language:
                return True
            begin = end
        return False


if __name__ == "__main__":
    import sys
    import time

    t = time.time()
    lid = LanguageIdentifier.from_internal_model_group("tr_group")
    print(f"tr_group yüklendi ({time.time() - t:.2f} sn): {sorted(lid.models)}")
    samples = sys.argv[1:] or [
        "Merhaba, bugün hava çok güzel ve parka gitmeyi düşünüyorum.",
        "merhaba bugun hava cok guzel ve parka gitmeyi dusunuyorum",
        "Hello, the weather is really nice today.",
        "Heute ist das Wetter sehr schön.",
        "Salam, bu gün hava çox gözəldir.",
    ]
    for s in samples:
        lang, conf = lid.identify_with_confidence(s)
        print(f"  {lid.identify(s):4} ({lang} {conf:.2f})  {s}")
