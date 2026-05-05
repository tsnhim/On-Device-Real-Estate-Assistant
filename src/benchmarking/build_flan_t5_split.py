from __future__ import annotations

import argparse
from pathlib import Path

from src.benchmarking.flan_t5_data import (
    CANONICAL_SPLIT_SEED,
    NOTEBOOK_DATASET_NAME,
    NOTEBOOK_TEST_SIZE,
    build_qa_pairs_from_hf_train_split,
    save_pairs_jsonl,
    save_split_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a canonical FLAN-T5 benchmark split from the notebook-compatible dataset flow."
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/data/flan_t5_baseline",
        help="Directory for the canonical pair cache and split manifest.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CANONICAL_SPLIT_SEED,
        help="Canonical split seed. The original notebook omitted an explicit split seed, so this saved split becomes the reproducible benchmark reference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    pairs_path = output_dir / "qa_pairs.jsonl"
    split_path = output_dir / "split_manifest.json"

    from datasets import Dataset
    from datasets import load_dataset

    train_split = load_dataset(NOTEBOOK_DATASET_NAME)["train"]
    pairs = build_qa_pairs_from_hf_train_split(train_split)
    hf_pairs = Dataset.from_list([pair.as_dict() for pair in pairs])
    split = hf_pairs.train_test_split(test_size=NOTEBOOK_TEST_SIZE, seed=args.seed)
    train_pairs = split["train"]
    eval_pairs = split["test"]
    source_name = NOTEBOOK_DATASET_NAME
    split_notes = (
        "Original notebook used tokenized.train_test_split(test_size=0.1) without an explicit split seed. "
        f"This canonical benchmark split rebuilds the same QA extraction flow and saves a deterministic split with seed={args.seed}."
    )

    all_pairs = []
    all_pairs.extend(train_pairs)
    all_pairs.extend(eval_pairs)
    deduped = {}
    for item in all_pairs:
        deduped[item["example_id"]] = item

    save_pairs_jsonl(
        (
            type("Obj", (), {"as_dict": lambda self, item=item: item})()
            for item in deduped.values()
        ),
        pairs_path,
    )
    save_split_manifest(
        output_path=split_path,
        source_name=source_name,
        source_pairs_path=pairs_path,
        total_pairs=len(deduped),
        train_ids=[item["example_id"] for item in train_pairs],
        eval_ids=[item["example_id"] for item in eval_pairs],
        split_seed=args.seed,
        split_test_size=NOTEBOOK_TEST_SIZE,
        split_notes=split_notes,
    )

    print(f"Saved pair cache to {pairs_path}")
    print(f"Saved split manifest to {split_path}")
    print(f"Train examples: {len(train_pairs)}")
    print(f"Eval examples: {len(eval_pairs)}")


if __name__ == "__main__":
    main()
