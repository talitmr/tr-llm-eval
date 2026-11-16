"""zemberek-python 0.2.3 için NumPy 2 uyumluluk ve determinizm yamaları.

Kullanım: zemberek'ten herhangi bir şey import etmeden önce bu modülü import edin. Yamalar
import anında uygulanır. Sözlük yamasının etkili olması için TurkishMorphology bu modülden
sonra oluşturulmalıdır.

    import zemberek_compat  # noqa: F401

1) NumPy 2 uyumluluğu
---------------------
zemberek-python, Java'daki 32 bit tamsayı ve işaretli bayt davranışını taklit etmek için NumPy
skalerleri (np.int8, np.int32) kullanır. NumPy 2 ile gelen yeni tip yükseltme kuralları (NEP 50)
nedeniyle bu skalerlerle sınır dışı bir Python tamsayısı aynı işlemde kullanıldığında
OverflowError oluşur:

  * GramDataArray.get_probability_rank:  int8 dizisi elemanı & 255   -> 255 int8'e sığmıyor
  * Mphf.rshift / GramDataArray.rshift:  np.int32 % 0x100000000      -> sabit int32'ye sığmıyor

Bu iki nokta dil modeli (lm.2gram.slm) kullanan her şeyi bozar: TurkishSentenceNormalizer,
TurkishSpellChecker'ın bağlamlı önerileri vb. Yamalar aynı sonucu saf Python tamsayılarıyla
hesaplar. NumPy 1.x altında da sonuç değişmez.

2) Determinizm
--------------
Python string hash'i her süreçte rastgele tohumlandığı için (PYTHONHASHSEED), zemberek'in set
üzerinden dolaştığı yerlerde sonuç sırası çalıştırmadan çalıştırmaya değişir:

  * RootLexicon.__iter__ öğeleri bir set'ten verir. Kök geçişleri bu sırayla eklendiği için
    TurkishMorphology.analyze() aynı kelimenin çözümlemelerini her seferinde farklı sırada
    döndürür ("[kitap:Noun]" / "[Kitap:Noun, Prop]"). Çözümleme sırasına bağlı kodlar
    (NER öznitelikleri, eşit skorda ilk adayı seçen her şey) bu yüzden değişken sonuç verir.
    Yama: sözlük dosyasındaki sıra (Java'daki LinkedHashSet davranışı).
  * StemTransitionsMapBased.get_transitions_ascii_tolerant sonucu set olarak döndürür
    (ASCII toleranslı morfoloji, normalizer'da kullanılır). Yama: ekleme sırasını koruyan
    tekilleştirme.
  * TurkishSpellChecker.get_unranked_suggestions adayları set'te toplar. Sonraki sıralama
    kararlı (stable) olduğu için eşit skorlu adaylar rastgele sırada kalır. Yama: adaylar
    alfabetik sıralanır, eşit skorda alfabetik sıra korunur.
"""
from typing import Tuple

from zemberek.core.hash.mphf import Mphf
from zemberek.core.turkish import TurkishAlphabet
from zemberek.lm.compression.gram_data_array import GramDataArray
from zemberek.morphology.lexicon.root_lexicon import DictionaryItemIterator, RootLexicon
from zemberek.morphology.morphotactics.turkish_morphotactics import StemTransitionsMapBased
from zemberek.normalization.turkish_spell_checker import TurkishSpellChecker

__all__ = ["apply"]

_applied = False


def _rshift(val: int, n: int) -> int:
    """İşaretsiz sağa kaydırma (Java'daki >>>). int() ile NumPy skaleri Python tamsayısına çevrilir."""
    return (int(val) % 0x100000000) >> n


def _patch_numpy2() -> None:
    Mphf.rshift = staticmethod(_rshift)
    GramDataArray.rshift = staticmethod(_rshift)

    # Sayfaları int8 dizisi yerine bytes olarak tut: bytes indekslemesi 0-255 arası Python
    # tamsayısı döndürdüğü için "& 255" / "& 0xFF" ifadeleri NumPy 1'deki sonucu verir.
    original_init = GramDataArray.__init__

    def __init__(self, file):
        original_init(self, file)
        self.data = [page.tobytes() for page in self.data]

    GramDataArray.__init__ = __init__


def _patch_determinism() -> None:
    # id_map, add_() içinde item_set ile aynı öğeleri sözlük dosyası sırasıyla tutar.
    def lexicon_iter(self):
        return DictionaryItemIterator(tuple(self.id_map.values()))

    RootLexicon.__iter__ = lexicon_iter

    def get_transitions_ascii_tolerant(self, stem: str):
        self.lock.acquire_read()
        try:
            result = {}  # dict: ekleme sırasını koruyan küme
            if stem in self.single_stems:
                result[self.single_stems[stem]] = None
            elif stem in self.multi_stems:
                result.update(dict.fromkeys(self.multi_stems[stem]))

            ascii_stems = dict.fromkeys(self.ascii_keys.get(TurkishAlphabet.INSTANCE.to_ascii(stem), []))
            for st in ascii_stems:
                if st in self.single_stems:
                    result[self.single_stems[st]] = None
                elif st in self.multi_stems:
                    result.update(dict.fromkeys(self.multi_stems[st]))
            return tuple(result)
        finally:
            self.lock.release_read()

    StemTransitionsMapBased.get_transitions_ascii_tolerant = get_transitions_ascii_tolerant

    original_unranked = TurkishSpellChecker.get_unranked_suggestions

    def get_unranked_suggestions(self, word: str) -> Tuple[str, ...]:
        return tuple(sorted(original_unranked(self, word)))

    TurkishSpellChecker.get_unranked_suggestions = get_unranked_suggestions


def apply() -> None:
    global _applied
    if _applied:
        return
    _patch_numpy2()
    _patch_determinism()
    _applied = True


apply()
