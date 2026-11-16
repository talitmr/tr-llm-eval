"""
WikiANN Türkçe verisiyle perceptron NER modeli eğitir.

Veri HuggingFace'ten parquet olarak indirilir (pyarrow gerekir) (unimelb-nlp/wikiann, config=tr),
BRACKET anotasyon formatına çevrilir, eğitilir ve ./ner_model altına kaydedilir.

    python train_ner_wikiann.py                 # 20000 eğitim cümlesi, 7 iterasyon
    python train_ner_wikiann.py --train 5000    # daha hızlı deneme
"""
import argparse
import logging
import os
import time
import urllib.request

from zemberek import TurkishMorphology

from named_entities import (AnnotationStyle, NerDataSet, PerceptronNerTrainer, collect_evaluation_data,
                          evaluation_report)

TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
PARQUET_URL = "https://huggingface.co/api/datasets/unimelb-nlp/wikiann/parquet/tr/{split}/0.parquet"


def to_bracket(tokens, tags):
    parts, i = [], 0
    while i < len(tokens):
        t = TAGS[tags[i]]
        if t.startswith("B-"):
            typ, j = t[2:], i + 1
            while j < len(tokens) and TAGS[tags[j]] == "I-" + typ:
                j += 1
            parts.append(f"[{typ} " + " ".join(tokens[i:j]) + "]")
            i = j
        else:
            parts.append(tokens[i])
            i += 1
    return " ".join(parts)


def fetch(split: str, n: int, cache_dir: str) -> str:
    """Parquet dosyasını tek istekte indirir (datasets-server satır API'si istek sınırına takılıyor)."""
    import pyarrow.parquet as pq

    os.makedirs(cache_dir, exist_ok=True)
    pq_path = os.path.join(cache_dir, f"wikiann-tr-{split}.parquet")
    if not os.path.exists(pq_path):
        for attempt in range(4):
            try:
                urllib.request.urlretrieve(PARQUET_URL.format(split=split), pq_path)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
    table = pq.read_table(pq_path, columns=["tokens", "ner_tags"]).to_pylist()
    lines = []
    for row in table[:n]:
        toks = row["tokens"]
        if any("[" in t or "]" in t for t in toks):
            continue
        lines.append(to_bracket(toks, row["ner_tags"]))
    path = os.path.join(cache_dir, f"wikiann-tr-{split}-{n}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--iterations", type=int, default=7)
    ap.add_argument("--out", default="ner_model")
    ap.add_argument("--data-dir", default="ner_data")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    print("Veri indiriliyor...")
    train_path = fetch("train", args.train, args.data_dir)
    test_path = fetch("test", args.test, args.data_dir)

    m = TurkishMorphology.create_with_defaults()
    train = NerDataSet.load(train_path, AnnotationStyle.BRACKET)
    test = NerDataSet.load(test_path, AnnotationStyle.BRACKET)
    print(train.info())

    t = time.time()
    ner = PerceptronNerTrainer(m).train(train, None, args.iterations, 0.1)
    print(f"Eğitim: {time.time() - t:.0f} sn")

    pred = ner.evaluate(test)
    print(collect_evaluation_data(test, pred).dump())
    ner.save_model(args.out)
    evaluation_report(test, pred, os.path.join(args.out, "test-report.txt"))
    print(f"Model kaydedildi: {args.out}/  (rapor: {args.out}/test-report.txt)")


if __name__ == "__main__":
    main()
