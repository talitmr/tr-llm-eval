# tr-llm-eval

Türkçe LLM çıktıları için morfoloji tabanlı değerlendirme metrikleri. Saf Python, Java gerektirmez.

Türkçe sondan eklemeli bir dil: "kitap", "kitaplarımızdan" ve "kitabını" yüzeyde farklı ama aynı kelime.
İngilizce için tasarlanmış BLEU/ROUGE ve anahtar kelime kontrolleri bu yüzden Türkçede yanıltıcı sonuç
veriyor. Bu proje morfolojik çözümleme kullanarak Türkçeye uygun metrikler hesaplar. Dil tespiti,
adlandırılmış varlık tanıma (NER), heceleme ve yazıyla sayıları rakama çevirme gibi yardımcı araçları da içerir.

## Kurulum

```bash
git clone https://github.com/<kullanıcı>/tr-llm-eval.git
cd tr-llm-eval
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`numpy<2` zorunludur, çünkü morfoloji bağımlılığı NumPy 2 ile normalizasyon adımında çöküyor. Python 3.9 ile test edildi.

## Hızlı başlangıç

```python
from tr_llm_eval import TurkishLLMEvaluator

ev = TurkishLLMEvaluator(ner_model_dir="ner_model")   # NER modeli opsiyonel
r = ev.evaluate(
    response="TÜİK'e göre nüfus 2023'te 87 milyona çıktı ve artış hızı yavaşlamadı.",
    source="Türkiye İstatistik Kurumu'na göre nüfus 2023'te 85 milyon 372 bine ulaştı. Artış hızı yavaşladı.",
    reference="TÜİK verilerine göre nüfus 2023'te 85 milyon 372 bine ulaştı ve artış hızı yavaşladı.",
    constraints={"keywords": ["nüfus"], "register": "siz", "language": "tr", "max_words": 50},
)
print(r["flags"])
# ['kaynakta olmayan sayı: 87', 'olumsuzluk çevrilmiş: yavaşlamak']
```

Komut satırından toplu değerlendirme yapmak için JSONL kullanılır. Her satırda `response` alanı zorunludur; `reference`, `source` ve `constraints` isteğe bağlıdır:

```bash
.venv/bin/python tr_llm_eval.py samples/batch_input.jsonl -o sonuc.jsonl --ner-model ner_model
.venv/bin/python tr_llm_eval.py      # örnek yanıtlarla demo
```

İlk yüklemede morfoloji yaklaşık 5-8 saniye sürer. Sonraki analizler hızlıdır. Senaryo bazlı örnekler için
[Kullanım senaryoları](#kullanım-senaryoları) bölümüne bakın.

## Metrikler

| Bölüm | Ne ölçer | Neden Türkçeye özel |
|---|---|---|
| `reference` | BLEU, ROUGE-1/2/L **yüzey, lemma ve morfem** düzeyinde; chrF; WER; içerik lemma F1 | "kitabını" ile "kitapları" yüzeyde eşleşmez, lemmada eşleşir |
| `faithfulness` | Kaynakta olmayan **sayılar** (rakam, yazıyla, sıra sayısı, tarih/saat), **varlıklar** (TÜİK ↔ Türkiye İstatistik Kurumu gibi kısaltma eşleşmeleri dahil), **olumsuzluk çevirmeleri**, kaynak kapsamı | Olumsuzluk ekle ifade edilir: "geldi"→"gelmedi", "var"→"yok", "… değil" |
| `language` | Belge ve cümle düzeyinde dil, Türkçe cümle oranı, Latin dışı karakter oranı | Kod karışımı tespiti |
| `fluency` | Morfolojik geçerlilik, tanınmayan kelimeler, **ASCII yazım** (ç/ş/ğ eksik), **informal yazım** ("gelcem"), yazım önerileri, isteğe bağlı kalite skoru ve perplexity | |
| `style` | **sen/siz tutarlılığı**, zaman kipleri, **Ateşman** ve **Bezirci-Yılmaz** okunabilirlik, lemma TTR / MTLD, tekrar oranı | Hitap karışıklığı tipik bir LLM hatası |
| `instructions` | Eklerden bağımsız anahtar/yasaklı kelime ("not almak" ↔ "not alınız"), uzunluk, dil, hitap, Türkçe karakter | |

## Kullanım senaryoları

Aşağıdaki örnekler repo klasöründe çalıştırılır. Yorum satırlarındaki çıktılar gerçek çıktılardır.
Değerlendiriciyi bir kez oluşturup tüm örneklerde kullanabilirsiniz:

```python
from tr_llm_eval import TurkishLLMEvaluator

ev = TurkishLLMEvaluator(ner_model_dir="ner_model")
```

### 1. RAG / özetleme: kaynakta olmayan bilgiyi yakalama

Model bir kaynak metne dayanarak cevap veriyorsa, `source` verin. Kaynakta geçmeyen sayılar ve kişi/kurum
adları ile anlamı tersine çeviren olumsuzluklar işaretlenir.

```python
kaynak = ("Şirketin 2024 yılı net kârı 1,2 milyar TL oldu. Genel Müdür Ayşe Demir, "
          "yeni yatırımların İzmir'de yapılacağını açıkladı. Temettü dağıtılmayacak.")
yanit = ("Şirket 2024'te 1,5 milyar TL net kâr elde etti. Genel Müdür Mehmet Kaya, "
         "yatırımların İzmir'de yapılacağını ve temettü dağıtılacağını söyledi.")

r = ev.evaluate(yanit, source=kaynak)
print(r["flags"])
# ['kaynakta olmayan sayı: 1,5', 'kaynakta olmayan varlık: Genel Müdür Mehmet Kaya', 'olumsuzluk çevrilmiş: dağıtmak']

fa = r["faithfulness"]
print(fa["numbers"]["unsupported"])   # [{'text': '1,5', 'values': [1.5, 1500000000.0], 'kind': 'digit'}]
print(fa["entities"]["unsupported"])  # ['Genel Müdür Mehmet Kaya']
print(fa["negation_flips"])           # [{'lemma': 'dağıtmak', 'source': '-', 'response': '+'}]
```

"1,2 milyar" ↔ "1.200.000.000", "85 milyon 372 bin" ↔ "85.372.000" ve "iki bin yirmi dört" ↔ "2024" gibi
farklı yazımlar aynı sayı olarak eşleşir. "TÜİK" de kaynaktaki "Türkiye İstatistik Kurumu"nun kısaltması olarak
tanınır. `values` listesi bir sayının olası okunuşlarını içerir ("1,5 milyar" için 1,5 ve 1.500.000.000).

### 2. Referans cevapla karşılaştırma (QA, çeviri, özet)

Altın cevap varsa `reference` verin. Yüzey metrikleri Türkçede ekler yüzünden düşük çıkar; lemma ve morfem
düzeyindeki metrikler anlamsal örtüşmeyi daha iyi yansıtır.

```python
referans = "Toplantı yarın saat onda başlayacak ve tüm ekiplerin katılması gerekiyor."
yanit = "Yarınki toplantı saat 10'da başlıyor; bütün ekipler katılmalı."

m = ev.reference_metrics(yanit, referans)
print({v: m[v]["rouge1"] for v in ("surface", "lemma", "morph")})
# {'surface': 22.22, 'lemma': 66.67, 'morph': 60.0}
print(m["chrf"])                        # 41.83
print(m["content_lemma"]["missing"])    # ['gerekmek']  -> referansta olup cevapta olmayan içerik
```

### 3. Talimata uyum (IFEval tarzı kontroller)

Anahtar kelimeler eklerden bağımsız aranır: "fatura" için "Faturanızı", "ödeme yapmak" için
"Ödemenizi yaptıktan" kabul edilir.

```python
talimat = {
    "keywords": ["fatura", "ödeme yapmak"],   # geçmesi gerekenler (lemma düzeyinde)
    "forbidden_words": ["kredi kartı"],       # geçmemesi gerekenler
    "register": "siz",                        # "siz" ya da "sen"
    "max_words": 40,                          # ayrıca min_words, min_sentences, max_sentences
    "language": "tr",
    "turkish_chars": True,                    # ç/ş/ğ/ı/ö/ü eksik yazımı ihlal sayar
}
yanit = ("Faturanızı uygulama üzerinden ödeyebilirsiniz. Ödemenizi yaptıktan sonra dekontunuzu saklayın, "
         "sen de bilirsin geç kalırsan faiz işler.")

c = ev.instruction_checks(yanit, talimat)
print(c["all_pass"])               # False
print(c["keywords"]["found_as"])   # {'fatura': 'Faturanızı', 'ödeme yapmak': 'Ödemenizi yaptıktan'}
print(c["register"])               # {'pass': False, 'detected': 'karışık'}
```

### 4. Dil ve yazım kalitesi (chatbot çıktıları)

Referans ya da kaynak olmadan da kullanılabilir. Araya giren yabancı dilde cümleleri ve Türkçe karakter
eksikliklerini bulur.

```python
yanit = ("Merhaba! Siparişiniz kargoya verildi. Your tracking number will be sent by email. "
         "Iyi gunler dileriz, tesekkurler.")

r = ev.evaluate(yanit)
print(r["flags"])
# ["kod karışımı: cümlelerin %42'i başka dilde", 'Türkçe karakter eksik (%19 kelime)', 'tanınmayan kelime oranı yüksek (%25)']
print(r["language"]["foreign_sentences"])
# [{'sentence': 'Your tracking number will be sent by email.', 'lang': 'en', 'confidence': 1.0}]
print(r["fluency"]["ascii_words"])
# [{'word': 'Iyi', 'fix': 'İyi'}, {'word': 'gunler', 'fix': 'günler'}, {'word': 'tesekkurler', 'fix': 'teşekkürler'}]
```

`ev.evaluate(..., full_quality=True)` ile 0-100 arası kalite skoru ve dil modeli perplexity'si de hesaplanır
(daha yavaştır).

### 5. Okunabilirlik, çeşitlilik ve tekrar

```python
yanit = ("Yapay zekâ, bilgisayarların insan benzeri görevleri yerine getirmesini sağlayan bir teknolojidir. "
         "Günümüzde sağlık, eğitim ve ulaşım gibi birçok alanda kullanılmaktadır. "
         "Yapay zekâ, bilgisayarların insan benzeri görevleri yerine getirmesini sağlayan bir teknolojidir.")

st = ev.style(yanit)
print(st["readability"])   # {'atesman': 44.9, 'atesman_level': 'zor', 'bezirci_yilmaz': 19.0}
print(st["repetition"])    # {'repeated_lemma_trigram_ratio': 0.621, 'duplicate_sentences': 1}
print(st["diversity"])     # {'lemma_ttr': 0.645, 'surface_ttr': 0.645, 'lemma_mtld': 31.0}
```

Ateşman skoru 0-100 arasıdır (yüksek = daha kolay). Bezirci-Yılmaz değeri yaklaşık olarak okumak için
gereken eğitim yılını gösterir.

### 6. Toplu değerlendirme ve model karşılaştırma

Her satır bir örnektir. `model` alanı verilirse sonuçlara aynen taşınır, böylece modelleri karşılaştırabilirsiniz.
Örnek dosya: [`samples/batch_input.jsonl`](samples/batch_input.jsonl).

```json
{"id": "q2", "model": "model_a", "response": "Başvurunuzu web sitemiz üzerinden yapabilirsiniz. ...", "constraints": {"keywords": ["başvuru", "belge"], "register": "siz", "max_words": 40}}
```

```bash
.venv/bin/python tr_llm_eval.py samples/batch_input.jsonl -o sonuc.jsonl --ner-model ner_model --no-suggestions
# [q1] uyarılar: -
# [q1] uyarılar: ['Türkçe karakter eksik (%43 kelime)', 'tanınmayan kelime oranı yüksek (%29)', 'kaynakta olmayan sayı: 1,5']
# [q2] uyarılar: -
# [q2] uyarılar: ["kod karışımı: cümlelerin %35'i başka dilde", 'tanınmayan kelime oranı yüksek (%23)', 'sen/siz hitabı karışık', 'talimat ihlali: register']
#
# 4 örnek — ortalamalar:
#   faithfulness.numbers.support_ratio               0.750
#   instructions.all_pass                            0.750
#   reference.lemma.rouge1                          68.570
#   ...
```

Sonuçları model bazında özetlemek için:

```python
import json
from collections import defaultdict

by_model = defaultdict(list)
for line in open("sonuc.jsonl", encoding="utf-8"):
    r = json.loads(line)
    by_model[r["model"]].append(r)

for model, rs in by_model.items():
    n_flags = sum(len(r["flags"]) for r in rs)
    pass_rate = sum(r["instructions"]["all_pass"] for r in rs) / len(rs)
    print(f"{model}: uyarı sayısı={n_flags}, talimat geçme oranı={pass_rate:.0%}")
# model_a: uyarı sayısı=0, talimat geçme oranı=100%
# model_b: uyarı sayısı=7, talimat geçme oranı=50%
```

Seçenekler: `--ner-model` NER modelini kullanır, `--full-quality` kalite skorunu ekler (yavaş),
`--no-suggestions` yazım önerilerini atlar (hızlı). Girdi alanları: `response` (ya da `output` / `prediction`),
`reference`, `source` (ya da `context`), `constraints`, `id`, `model`.

### 7. Araçları tek başına kullanmak

**Dil tespiti**

```python
from language_id import LanguageIdentifier

lid = LanguageIdentifier.from_internal_model_group("tr_group")   # Türkçe odaklı, 8 dil
print(lid.identify("Bugün hava çok güzel."))                      # tr
print(lid.identify_with_confidence("Salam, bu gün hava çox gözəldir."))   # ('az', 0.99...)

lid60 = LanguageIdentifier.from_internal_models()                 # 60 dil
print(lid60.identify("Bonjour, comment allez-vous ?"))            # fr
```

**Sayılar ve heceleme**

```python
from turkish_utils import TurkishNumbersExt, TurkishSyllableExtractor, syllable_count

print(TurkishNumbersExt.convert_to_number("iki bin yirmi dört"))        # 2024
print(TurkishNumbersExt.convert_number_to_string("1453"))               # bin dört yüz elli üç
print(TurkishNumbersExt.convert_ordinal_number_string("21."))           # yirmi birinci
print(TurkishSyllableExtractor.DEFAULT.get_syllables("kitaplarımızdan"))  # ['ki', 'tap', 'la', 'rı', 'mız', 'dan']
print(syllable_count("anlaşılamayacak"))                                # 7
```

**Varlık tanıma (NER)**

```python
from zemberek import TurkishMorphology
from named_entities import PerceptronNer, ProperNounExtractor

m = TurkishMorphology.create_with_defaults()
ner = PerceptronNer.load_model("ner_model", m)
print(ner.find_named_entities("Mustafa Kemal Atatürk 1919'da Samsun'a çıktı.").named_entities())
# [[PER Mustafa Kemal Atatürk], [LOC Samsun'a]]

# Model gerektirmeyen, morfoloji tabanlı özel isim çıkarıcı
print([(e.canonical, e.kind) for e in ProperNounExtractor(m).extract("Ayşe Demir dün TBMM'de konuştu.")])
# [('Ayşe Demir', 'PROPER'), ('TBMM', 'ABBRV')]
```

**Metin kalitesi skoru**

```python
from text_quality import TextQualityAnalyzer

qa = TextQualityAnalyzer(m)
for r in qa.compare({"model_a": "Toplantı yarın saat onda başlayacak.",
                     "model_b": "toplanti yarin 10da baslicak!!!"}):
    print(r.name, r.score, r.language)
# model_a 96.9 tr
# model_b 26.5 tr
```

Komut satırından: `.venv/bin/python text_quality.py a.txt b.txt` ya da `--json`.

### 8. Kendi verinizle NER eğitmek

Eğitim dosyasında her satır bir cümledir. Varlıklar `[TÜR kelimeler]` biçiminde işaretlenir (BRACKET formatı;
ENAMEX ve OpenNLP formatları da desteklenir):

```text
[PER Ahmet Yılmaz] dün [LOC Ankara'ya] gitti .
[ORG Koç Holding] yeni yatırım açıkladı .
```

```python
from zemberek import TurkishMorphology
from named_entities import NerDataSet, AnnotationStyle, PerceptronNerTrainer, collect_evaluation_data

m = TurkishMorphology.create_with_defaults()
train = NerDataSet.load("train.txt", AnnotationStyle.BRACKET)
test = NerDataSet.load("test.txt", AnnotationStyle.BRACKET)

ner = PerceptronNerTrainer(m).train(train, test, iteration_count=7, learning_rate=0.1)
print(collect_evaluation_data(test, ner.evaluate(test)).dump())
ner.save_model("benim_ner_modelim")

ev = TurkishLLMEvaluator(ner_model_dir="benim_ner_modelim")   # değerlendiricide kullanmak için
```

## Dosyalar

| Dosya | Ne yapar |
|---|---|
| `tr_llm_eval.py` | LLM değerlendirme |
| `text_quality.py` | Metin kalitesi skoru (0-100) ve karşılaştırma |
| `language_id.py` | Dil tespiti (60 dil ya da Türkçe odaklı 8 dil) |
| `named_entities.py` | Perceptron NER (eğitim, tahmin, değerlendirme) ve modelsiz özel isim çıkarıcı |
| `turkish_utils.py` | Heceleme, yazıyla sayıyı rakama çevirme, sıra sayıları, Levenshtein, `get_lemmas` / `get_pos` |
| `train_ner_wikiann.py` | WikiANN-tr ile NER modelini yeniden eğitir |
| `examples.py` | Cümle ayırma, tokenizasyon, morfolojik analiz, normalizasyon ve yazım denetimi örnekleri |
| `langid_models/` | Dil tespiti model dosyaları |
| `ner_model/` | WikiANN-tr ile eğitilmiş NER modeli (PER / LOC / ORG) |

## NER modeli

`ner_model/` klasöründeki model, WikiANN Türkçe eğitim setinin 19.803 cümlesiyle 7 iterasyon eğitildi.
WikiANN test setinin ilk 1000 cümlesindeki sonuçlar:

| Metrik | Değer |
|---|---|
| NE token kesinliği | 0.974 |
| NE token duyarlılığı | 0.955 |
| Tam varlık eşleşmesi | 0.849 |

WikiANN, Wikipedia'dan otomatik etiketlenmiştir. Haber ve diyalog metinlerinde doğruluk daha düşük olabilir.
Kendi verinizle eğitmek için `named_entities.py` modülünün başındaki örneğe bakın. BRACKET, ENAMEX ve
OpenNLP anotasyon formatları desteklenir. Orijinal Java uygulamasıyla eğitilmiş modeller de yüklenebilir.

```bash
.venv/bin/pip install pyarrow
.venv/bin/python train_ner_wikiann.py          # yaklaşık 4 dakika
```

## Sınırlamalar

- Skor eşikleri ve ağırlıkları kalibre edilmedi. Kendi verinizle ayarlamanız önerilir.
- Olumsuzluk kontrolü lemma düzeyinde çalışır. Aynı fiil kaynakta hem olumlu hem olumsuz geçiyorsa çevirme işaretlenmez.
- Morfoloji sözlükte olmayan yeni kelimeleri tanımaz. Yeni marka ve terimler "tanınmayan kelime" olarak görünebilir.
- Dil tespiti, orijinal Java uygulamasının çıktısıyla birebir karşılaştırılmadı. Sonuçlar tutarlı görünüyor.

## Lisans ve atıf

Apache License 2.0. Bu proje [zemberek-nlp](https://github.com/ahmetaa/zemberek-nlp)'nin (Ahmet A. Akın,
Mehmet D. Akın; Apache 2.0) Python'a çevrilmiş parçalarını ve dil modeli dosyalarını içerir.
[zemberek-python](https://github.com/Loodos/zemberek-python) bağımlılık olarak kullanılır. NER modeli
WikiANN (Pan et al., 2017) ile eğitilmiştir. Ayrıntılar için [NOTICE](NOTICE) dosyasına bakın.
