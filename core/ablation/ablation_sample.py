import argparse
import json
import os
import random
from collections import defaultdict
from typing import Any


def sample_equal_by_task(items: list[Any], count: int, rng: random.Random) -> list[Any]:
    if count <= 0 or not items:
        return []

    by_task = defaultdict(list)
    for item in items:
        by_task[item["task"]].append(item)

    tasks = list(by_task)
    rng.shuffle(tasks)
    base, extra = divmod(count, len(tasks))

    sampled = []
    for idx, task in enumerate(tasks):
        quota = base + int(idx < extra)
        bucket = by_task[task]
        if quota <= len(bucket):
            sampled.extend(rng.sample(bucket, quota))
        else:
            sampled.extend(rng.choices(bucket, k=quota))

    rng.shuffle(sampled)
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample a task-balanced ablation training shard without model generation."
    )
    parser.add_argument("--train_data", type=str, default="data/train.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/precomputed")
    parser.add_argument("--shard_size", type=int, default=8)
    parser.add_argument("--shard_idx", type=int, required=True)
    parser.add_argument("--seed", type=int, default=910)
    args = parser.parse_args()
    if args.shard_size < 1:
        parser.error("--shard_size must be at least 1")

    rows = []
    with open(args.train_data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {args.train_data}")

    rng = random.Random(args.seed + args.shard_idx)
    sampled_rows = sample_equal_by_task(rows, args.shard_size, rng)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"shard_{args.shard_idx:04d}.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for row in sampled_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sampled_rows)} sampled ablation rows to {output_path}")


if __name__ == "__main__":
    main()
