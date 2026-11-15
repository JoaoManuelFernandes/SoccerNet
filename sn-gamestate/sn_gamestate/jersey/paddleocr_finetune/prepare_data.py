"""
prepare_data.py
===============
Prepara o dataset para fine-tuning do PaddleOCR recognizer (PP-OCRv4).

A partir dos crops em all_crops/{jersey}/*.jpg gera:
  - train_list.txt   → path\tlabel  (80%)
  - val_list.txt     → path\tlabel  (20%)
  - jersey_dict.txt  → dicionário de caracteres (0-9 + espaço)

Uso:
    python prepare_data.py \
        --crops-dir /home/joao/soccernet/data/dataset_jersey/all_crops \
        --out-dir   /home/joao/soccernet/data/dataset_jersey/paddleocr_data
"""

import argparse
import os
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops-dir", default="/home/joao/soccernet/data/dataset_jersey/all_crops")
    ap.add_argument("--out-dir",   default="/home/joao/soccernet/data/dataset_jersey/paddleocr_data")
    ap.add_argument("--val-ratio", type=float, default=0.20)
    ap.add_argument("--min-samples", type=int, default=10,
                    help="Ignorar classes com menos amostras do que este valor")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    crops_dir = Path(args.crops_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []  # (abs_path, label_str)

    class_dirs = sorted([d for d in crops_dir.iterdir() if d.is_dir()])
    print(f"Classes encontradas: {len(class_dirs)}")

    for cls_dir in class_dirs:
        label = cls_dir.name  # "7", "23", etc.
        # Validar que é número entre 0-99
        try:
            n = int(label)
            assert 0 <= n <= 99
        except (ValueError, AssertionError):
            print(f"  Ignorar pasta inválida: {cls_dir.name}")
            continue

        imgs = sorted(cls_dir.glob("*.jpg"))
        if len(imgs) < args.min_samples:
            print(f"  Ignorar #{label}: só {len(imgs)} amostras (< {args.min_samples})")
            continue

        for img in imgs:
            all_samples.append((str(img), label))

    print(f"Total amostras: {len(all_samples)}")

    # Shuffle e split por classe (stratified)
    from collections import defaultdict
    by_class = defaultdict(list)
    for path, label in all_samples:
        by_class[label].append(path)

    train_lines = []
    val_lines   = []

    for label, paths in sorted(by_class.items(), key=lambda x: int(x[0])):
        random.shuffle(paths)
        n_val = max(1, int(len(paths) * args.val_ratio))
        val_paths   = paths[:n_val]
        train_paths = paths[n_val:]
        for p in train_paths:
            train_lines.append(f"{p}\t{label}")
        for p in val_paths:
            val_lines.append(f"{p}\t{label}")
        print(f"  #{label:>2}: {len(train_paths):>5} train | {len(val_paths):>4} val")

    random.shuffle(train_lines)
    random.shuffle(val_lines)

    train_file = out_dir / "train_list.txt"
    val_file   = out_dir / "val_list.txt"

    train_file.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    val_file.write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    print(f"\nTrain: {len(train_lines)} amostras → {train_file}")
    print(f"Val:   {len(val_lines)} amostras → {val_file}")

    # Dicionário de caracteres — só dígitos 0-9
    dict_file = out_dir / "jersey_dict.txt"
    dict_file.write_text("\n".join(list("0123456789")) + "\n", encoding="utf-8")
    print(f"Dict:  {dict_file}  (10 caracteres: 0-9)")


if __name__ == "__main__":
    main()
