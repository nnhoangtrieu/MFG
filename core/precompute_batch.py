import argparse
import json
import os
import random
from collections import defaultdict
from typing import Any

import torch
from tqdm import tqdm

from output import Output
from properties import get_properties
from prompt import CURRENT_BEST_FEEDBACK, FOLLOW_UP_FEEDBACK
from reward import check_all_properties_improved, score
from utils import load


def check_is_strictly_successful(
    input_properties: dict[str, float],
    gen_properties: dict[str, float],
    properties: list[str],
    target_change: dict[str, float] | None,
    similarity: float,
    good_similarity_threshold: float = 0.6,
) -> bool:
    if target_change is None:
        return False

    if any(
        prop not in input_properties
        or prop not in gen_properties
        or input_properties[prop] is None
        or gen_properties[prop] is None
        for prop in properties
    ):
        return False

    all_properties_improved, _ = check_all_properties_improved(input_properties, gen_properties, properties)
    if not all_properties_improved:
        return False

    if similarity < good_similarity_threshold:
        return False

    for prop in properties:
        target = target_change[prop]
        gen_change = gen_properties[prop] - input_properties[prop]
        if prop == "mutagenicity":
            if gen_change > target:
                return False
        elif gen_change < target:
            return False

    return True


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
        sampled.extend(rng.sample(bucket, quota) if quota <= len(bucket) else rng.choices(bucket, k=quota))

    rng.shuffle(sampled)
    return sampled


def choose_current_best(
    current_best: Output | None,
    candidate: Output,
    acceptable_similarity_threshold: float = 0.4,
    good_similarity_threshold: float = 0.6,
) -> Output | None:
    if current_best is None:
        if (
            candidate.is_valid
            and candidate.similarity >= acceptable_similarity_threshold
            and candidate.all_properties_improved
        ):
            return candidate
        return None

    return max(
        [current_best, candidate],
        key=lambda result: (
            result.is_valid,
            result.similarity >= acceptable_similarity_threshold,
            result.all_properties_improved,
            result.num_properties_improved,
            result.similarity >= good_similarity_threshold,
            result.relative_improvement or 0.0,
            result.similarity,
        ),
    )


def generate_batch(
    model,
    tokenizer,
    generation_mode: str,
    messages_batch: list[list[dict[str, str]]],
    max_new_tokens: int = 96,
    num_beams: int = 20,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> list[str]:
    device = next(model.parameters()).device

    try:
        encoded = tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
        )
    except (TypeError, ValueError):
        prompts = [
            tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            for messages in messages_batch
        ]
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            return_tensors="pt",
            return_dict=True,
            padding=True,
        )

    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_len = encoded["input_ids"].shape[1]

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if generation_mode == "sampling":
        generate_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
    elif generation_mode == "beam":
        generate_kwargs.update({"num_beams": num_beams, "early_stopping": True})
    else:
        raise ValueError(f"Unsupported generation mode: {generation_mode}")

    with torch.no_grad():
        outputs = model.generate(**encoded, **generate_kwargs)

    return [tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip() for output in outputs]


def to_precomputed_row(state: dict[str, Any], num_context_turns: int) -> dict[str, Any]:
    return {
        "messages": state["messages"],
        "input_smiles": state["input_smiles"],
        "current_best": state["current_best"].gen_smiles if state["current_best"] is not None else None,
        "input_properties": state["input_properties"],
        "all_outputs": state["all_outputs"],
        "properties": state["properties"],
        "num_context_turns": num_context_turns,
        "target_props": state["target_props"],
        "target_similarity": state["target_similarity"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--train_data", type=str, default="data/train.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/precomputed")
    parser.add_argument("--path_base_model", type=str, default="/GeLLMO-P6-Llama")
    parser.add_argument("--path_adapter", type=str, default="")
    parser.add_argument("--shard_size", type=int, default=8)
    parser.add_argument("--shard_idx", type=int, required=True)
    parser.add_argument("--min_context_turns", type=int, default=0)
    parser.add_argument("--max_context_turns", type=int, default=1)
    parser.add_argument("--seed", type=int, default=910)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--generation_mode", choices=["sampling", "beam"], default="sampling")
    parser.add_argument("--acceptable_similarity_threshold", type=float, default=0.4)
    parser.add_argument("--good_similarity_threshold", type=float, default=0.6)
    parser.add_argument("--batch_size", type=int, default=24)
    args = parser.parse_args()

    if args.min_context_turns < 0 or args.min_context_turns > args.max_context_turns:
        raise ValueError("--min_context_turns must be between 0 and --max_context_turns")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1")

    rows = []
    with open(args.train_data, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line.strip()))

    rng = random.Random(args.seed + args.shard_idx)
    sample = sample_equal_by_task(rows, args.shard_size, rng)
    precomputed_rows: list[dict[str, Any] | None] = [None] * len(sample)
    active_states = []

    for idx, row in enumerate(sample):
        properties = row["properties"]
        input_smiles = row["input_smiles"]
        state = {
            "idx": idx,
            "messages": row["messages"],
            "input_smiles": input_smiles,
            "input_properties": get_properties(input_smiles, properties),
            "all_outputs": [],
            "current_best": None,
            "properties": properties,
            "target_change": row["target_change"],
            "target_props": row["target_props"],
            "target_similarity": row["target_similarity"],
            "planned_context_turns": rng.randint(args.min_context_turns, args.max_context_turns),
        }

        if state["planned_context_turns"] == 0:
            precomputed_rows[idx] = to_precomputed_row(state, 0)
        else:
            active_states.append(state)

    model, tokenizer = load(args.path_base_model, args.path_adapter) if active_states else (None, None)

    with tqdm(total=len(sample), desc=f"Precomputing shard {args.shard_idx}") as progress:
        progress.update(len(sample) - len(active_states))

        for turn in range(args.max_context_turns):
            current_states = [
                state
                for state in active_states
                if precomputed_rows[state["idx"]] is None and turn < state["planned_context_turns"]
            ]
            if not current_states:
                continue

            for start in range(0, len(current_states), args.batch_size):
                batch = current_states[start:start + args.batch_size]
                raw_outputs = generate_batch(
                    model,
                    tokenizer,
                    args.generation_mode,
                    [state["messages"] for state in batch],
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

                for state, raw_output in zip(batch, raw_outputs):
                    current_best = state["current_best"]
                    result = score(
                        input_smiles=state["input_smiles"],
                        input_properties=state["input_properties"],
                        raw_output=raw_output,
                        all_outputs=state["all_outputs"],
                        current_best=current_best.gen_smiles if current_best is not None else None,
                        properties=state["properties"],
                        target_props=state["target_props"],
                        target_similarity=state["target_similarity"],
                        acceptable_similarity_threshold=args.acceptable_similarity_threshold,
                        good_similarity_threshold=args.good_similarity_threshold,
                    )

                    if result.is_valid and check_is_strictly_successful(
                        input_properties=state["input_properties"],
                        gen_properties=result.gen_properties,
                        properties=state["properties"],
                        target_change=state["target_change"],
                        similarity=result.similarity,
                    ):
                        precomputed_rows[state["idx"]] = to_precomputed_row(state, turn)
                        progress.update(1)
                        continue

                    if result.is_valid and result.gen_canonical_smiles is not None:
                        state["all_outputs"].append(result.gen_canonical_smiles)

                    if result.is_valid:
                        state["current_best"] = choose_current_best(
                            state["current_best"],
                            candidate=result,
                            acceptable_similarity_threshold=args.acceptable_similarity_threshold,
                            good_similarity_threshold=args.good_similarity_threshold,
                        )

                    state["messages"] = state["messages"] + [{"role": "assistant", "content": raw_output}]
                    follow_up = FOLLOW_UP_FEEDBACK
                    if state["current_best"] is not None:
                        follow_up = CURRENT_BEST_FEEDBACK.format(current_best=state["current_best"].gen_smiles)
                    state["messages"] += [{"role": "user", "content": f"{result.feedback}\n{follow_up}"}]

                    if turn + 1 == state["planned_context_turns"]:
                        precomputed_rows[state["idx"]] = to_precomputed_row(state, state["planned_context_turns"])
                        progress.update(1)

    if any(row is None for row in precomputed_rows):
        raise RuntimeError("Internal error: not all rows were precomputed")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/shard_{args.shard_idx:04d}.jsonl", "w", encoding="utf-8") as f:
        for row in precomputed_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
