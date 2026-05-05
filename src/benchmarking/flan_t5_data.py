from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


NOTEBOOK_DATASET_NAME = "zillow/real_estate_v1"
NOTEBOOK_TEST_SIZE = 0.1
CANONICAL_SPLIT_SEED = 42
TRAINING_PREFIX = "answer question: "


@dataclass(frozen=True)
class QAPair:
    example_id: str
    input_text: str
    target_text: str
    topic: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "input_text": self.input_text,
            "target_text": self.target_text,
            "topic": self.topic,
        }


def clean_text(text: str) -> str:
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"#+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_example_id(input_text: str, target_text: str, topic: str | None = None) -> str:
    joined = "\n".join([input_text, target_text, topic or ""])
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def build_qa_pairs_from_hf_train_split(train_split: Iterable[dict[str, Any]]) -> list[QAPair]:
    pairs: list[QAPair] = []

    for example in train_split:
        messages = example["messages"]
        for i in range(len(messages) - 1):
            if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
                input_text = clean_text(messages[i]["content"])
                target_text = clean_text(messages[i + 1]["content"])
                pairs.append(
                    QAPair(
                        example_id=build_example_id(input_text, target_text, None),
                        input_text=input_text,
                        target_text=target_text,
                        topic=None,
                    )
                )

    return pairs


def load_local_retrieval_pairs(records_path: Path) -> list[QAPair]:
    raw = json.loads(records_path.read_text(encoding="utf-8"))
    pairs: list[QAPair] = []
    for item in raw:
        question = clean_text(item["question"])
        answer = clean_text(item["answer"])
        topic = item.get("topic")
        pairs.append(
            QAPair(
                example_id=build_example_id(question, answer, topic),
                input_text=question,
                target_text=answer,
                topic=topic,
            )
        )
    return pairs


def save_pairs_jsonl(pairs: Iterable[QAPair], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair.as_dict(), ensure_ascii=True) + "\n")


def load_pairs_jsonl(path: Path) -> list[QAPair]:
    pairs: list[QAPair] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            pairs.append(
                QAPair(
                    example_id=item["example_id"],
                    input_text=item["input_text"],
                    target_text=item["target_text"],
                    topic=item.get("topic"),
                )
            )
    return pairs


def save_split_manifest(
    *,
    output_path: Path,
    source_name: str,
    source_pairs_path: Path,
    total_pairs: int,
    train_ids: list[str],
    eval_ids: list[str],
    split_seed: int | None,
    split_test_size: float,
    split_notes: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_name": source_name,
        "source_pairs_path": str(source_pairs_path).replace("\\", "/"),
        "total_pairs": total_pairs,
        "split_test_size": split_test_size,
        "split_seed": split_seed,
        "split_notes": split_notes,
        "train_count": len(train_ids),
        "eval_count": len(eval_ids),
        "train_ids": train_ids,
        "eval_ids": eval_ids,
    }
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_split_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def split_examples_from_manifest(path: Path) -> dict[str, list[QAPair]]:
    manifest = load_split_manifest(path)
    pairs = load_pairs_jsonl(Path(manifest["source_pairs_path"]))
    pair_by_id = {pair.example_id: pair for pair in pairs}
    return {
        "train": [pair_by_id[item_id] for item_id in manifest["train_ids"]],
        "eval": [pair_by_id[item_id] for item_id in manifest["eval_ids"]],
    }
