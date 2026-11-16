"""Cümle ayırma, tokenizasyon, morfolojik analiz, normalizasyon ve yazım denetimi örnekleri."""
import logging
import time

from zemberek import (
    TurkishMorphology,
    TurkishSentenceExtractor,
    TurkishSentenceNormalizer,
    TurkishSpellChecker,
    TurkishTokenizer,
)

logging.basicConfig(level=logging.WARNING)

t0 = time.time()
morphology = TurkishMorphology.create_with_defaults()
print(f"Morfoloji yüklendi ({time.time() - t0:.1f} sn)\n")

# 1) Cümle ayırma
extractor = TurkishSentenceExtractor()
metin = "Dr. Ahmet Bey bugün İstanbul'a geldi. Yarın Ankara'ya gidecek mi? Bilmiyorum."
print("== Cümle ayırma ==")
for s in extractor.from_paragraph(metin):
    print("  ", s)

# 2) Tokenizasyon
print("\n== Tokenizasyon ==")
tokenizer = TurkishTokenizer.DEFAULT
for tok in tokenizer.tokenize("Saat 12:30'da info@ornek.com'a 3,5 TL gönderdim :)"):
    print(f"   {tok.content:<20} {tok.type_.name}")

# 3) Morfolojik analiz (tek kelime, tüm olası çözümlemeler)
print("\n== Kelime analizi: 'kitaplarımızdan' ==")
for a in morphology.analyze("kitaplarımızdan"):
    print("  ", a.format_string())

# 4) Cümle analizi + belirsizlik giderme (disambiguation)
print("\n== Cümle analizi + disambiguation ==")
cumle = "Yarın kar yağacak ama yollar açık kalacakmış."
for wa in morphology.analyze_and_disambiguate(cumle).best_analysis():
    print(f"   {wa.get_stem():<12} lemma={wa.item.lemma:<10} {wa.format_string()}")

# 5) Normalizasyon (gürültülü metni düzeltme)
print("\n== Normalizasyon ==")
normalizer = TurkishSentenceNormalizer(morphology)
for g in ["Yrn okua gidicem", "mrb nasılsın bugun ne yapcaksin"]:
    print(f"   {g!r:40} -> {normalizer.normalize(g)!r}")

# 6) Yazım denetimi
print("\n== Yazım denetimi ==")
spell = TurkishSpellChecker(morphology)
for w in ["okuyorum", "okuyrum", "kitapalr"]:
    ok = morphology.analyze(w).is_correct()
    print(f"   {w:<10} doğru={ok}" + ("" if ok else f"  öneriler={spell.suggest_for_word(w)[:5]}"))
