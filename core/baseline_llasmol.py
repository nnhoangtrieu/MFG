"""Evaluate LlaSMol-Mistral-7B with the prompt and feedback loop from main/.

Run from the main/ directory, for example:

    python core/llasmol_inference.py --tasks BDP BDQ --num_turns 1
    python core/llasmol_inference.py --tasks BDP BDQ --num_turns 2

LlaSMol-Mistral-7B is a LoRA adapter, so this script loads it on top of
mistralai/Mistral-7B-v0.1 and merges the adapter before inference.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference import (
    BUDGET,
    FINAL_SELECTION_CRITERIAS,
    PROGRESS_SELECTION,
    best_optimized_result,
    choose_current_best,
    get_best_result,
    load_benchmark,
    result_to_dict,
    summarize_by_task,
    summarize_metrics,
)
from prompt import (
    CURRENT_BEST_FEEDBACK,
    FOLLOW_UP_FEEDBACK,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from properties import get_properties
from reward import score
from utils import task_to_properties


ADAPTER_ID = "osunlp/LlaSMol-Mistral-7B"
BASE_MODEL_ID = "mistralai/Mistral-7B-v0.1"
NUM_BEAMS = 20
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def format_llasmol_prompt(messages: list[dict[str, str]]) -> str:
    """Serialize main/'s messages using LlaSMol's native training template."""
    system_prompt = messages[0]["content"]
    conversation = messages[1:]
    prompt = ""

    for index, message in enumerate(conversation):
        if message["role"] == "user":
            content = message["content"]
            if index == 0:
                content = f"{system_prompt}\n\n{content}"
            prompt += f"<s>[INST] {content} [/INST]"
        else:
            prompt += f" {message['content']}</s>"

    return prompt


def load_llasmol(
    adapter: str,
    base_model: str,
    dtype: torch.dtype,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = "<pad>"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        adapter,
        torch_dtype=dtype,
    )
    model = model.merge_and_unload()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.eval()
    return model, tokenizer


def generate(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    num_return_sequences: int,
    max_new_tokens: int,
) -> list[str]:
    prompt = format_llasmol_prompt(messages)
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_length = encoded["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=NUM_BEAMS,
            num_return_sequences=num_return_sequences,
            do_sample=False,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stop_strings=["</SMILES>"]
        )

    return [
        tokenizer.decode(
            sequence[prompt_length:],
            skip_special_tokens=True,
        ).strip()
        for sequence in output_ids
    ]


def save_results(
    output_path: Path,
    config: dict,
    records: list[dict],
) -> dict:
    metrics = {
        selection: summarize_by_task(records, selection)
        for selection in FINAL_SELECTION_CRITERIAS
    }
    payload = {
        "config": config,
        "metrics": metrics,
        "results": records,
    }
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    temporary_path.replace(output_path)
    return metrics


def main():
    resume_parser = argparse.ArgumentParser(add_help=False)
    resume_parser.add_argument("--resume", type=Path)
    resume_args, _ = resume_parser.parse_known_args()
    resume_payload = None
    resume_config = {}
    if resume_args.resume:
        if not resume_args.resume.is_file():
            resume_parser.error(f"--resume path does not exist: {resume_args.resume}")
        with resume_args.resume.open(encoding="utf-8") as file:
            resume_payload = json.load(file)
        resume_config = {
            key: value
            for key, value in resume_payload.get("config", {}).items()
            if key not in {"budget", "num_beams"}
        }

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate osunlp/LlaSMol-Mistral-7B with main/'s prompt, feedback, "
            "scoring, and fixed 20-candidate budget."
        )
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        required=resume_args.resume is None,
    )
    parser.add_argument("--num_turns", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument(
        "--adapter",
        default=ADAPTER_ID,
        help="LlaSMol adapter ID or local adapter path.",
    )
    parser.add_argument(
        "--base_model",
        default=BASE_MODEL_ID,
        help="Mistral base model ID or local model path.",
    )
    parser.add_argument(
        "--dtype",
        choices=DTYPES,
        default="bfloat16",
    )
    parser.add_argument("--acceptable_similarity_threshold", type=float, default=0.4)
    parser.add_argument("--good_similarity_threshold", type=float, default=0.6)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/LlaSMol-Mistral-7B"),
    )
    parser.add_argument(
        "--limit_per_task",
        type=int,
        help="Evaluate only the first N examples per task, useful for smoke tests.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Continue an existing results JSON using its saved configuration.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    parser.add_argument("--seed", type=int, default=910)
    parser.set_defaults(**resume_config)
    args = parser.parse_args()

    if not 1 <= args.num_turns <= BUDGET:
        parser.error(f"--num_turns must be between 1 and {BUDGET}")
    if not args.tasks:
        parser.error("--tasks is required because the resume file does not contain it")
    if args.limit_per_task is not None and args.limit_per_task < 1:
        parser.error("--limit_per_task must be at least 1")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"resume", "overwrite"}
    }
    config["budget"] = BUDGET
    config["num_beams"] = NUM_BEAMS

    if args.resume:
        saved_config = resume_payload.get("config", {})
        for key in ("adapter", "base_model"):
            if saved_config.get(key) and saved_config[key] != getattr(args, key):
                parser.error(
                    f"resume file uses {key}={saved_config[key]}, but this run "
                    f"uses {getattr(args, key)}"
                )
        records = resume_payload["results"]
        output_path = args.resume
    else:
        records = []
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / (
            f"LlaSMol-Mistral-7B_{'_'.join(args.tasks)}_{args.num_turns}turn.json"
        )
        if output_path.exists() and not args.overwrite:
            parser.error(
                f"output already exists: {output_path}; use --resume or --overwrite"
            )

    benchmark = load_benchmark(args.tasks)
    completed_cases = Counter(
        (record["task"], record["input_smiles"]) for record in records
    )
    success_count = sum(
        record["best_optimized"][PROGRESS_SELECTION] is not None for record in records
    )
    model = None
    tokenizer = None

    for task in args.tasks:
        task_rows = benchmark[task]
        if args.limit_per_task is not None:
            task_rows = task_rows[: args.limit_per_task]

        progress = tqdm(task_rows, desc=f"Inference {task}")
        for row in progress:
            input_smiles = row["input_smiles"]
            case = (task, input_smiles)
            if completed_cases[case]:
                completed_cases[case] -= 1
                continue

            if model is None:
                print(f"Loading {args.adapter} on {args.base_model}")
                model, tokenizer = load_llasmol(
                    args.adapter,
                    args.base_model,
                    DTYPES[args.dtype],
                )

            properties = task_to_properties(task)
            input_properties = get_properties(input_smiles, properties)
            record = {
                "input_smiles": input_smiles,
                "task": task,
                "original_input_properties": row["properties"],
                "recalculated_properties": input_properties,
                "output": {},
            }
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_prompt(input_smiles, properties),
                },
            ]
            all_outputs = []
            all_results = []
            current_best = None
            used_budget = 0

            for turn in range(1, args.num_turns + 1):
                turn_budget = (
                    BUDGET // args.num_turns
                    if turn < args.num_turns
                    else BUDGET - used_budget
                )
                raw_outputs = generate(
                    model,
                    tokenizer,
                    messages,
                    num_return_sequences=turn_budget,
                    max_new_tokens=args.max_new_tokens,
                )
                print(raw_outputs)
                used_budget += turn_budget

                turn_results = [
                    score(
                        input_smiles=input_smiles,
                        input_properties=input_properties,
                        raw_output=raw_output,
                        all_outputs=all_outputs,
                        current_best=(
                            current_best.gen_smiles
                            if current_best is not None
                            else None
                        ),
                        properties=properties,
                        target_props=None,
                        target_similarity=None,
                        acceptable_similarity_threshold=(
                            args.acceptable_similarity_threshold
                        ),
                        good_similarity_threshold=args.good_similarity_threshold,
                    )
                    for raw_output in raw_outputs
                ]
                selected = get_best_result(
                    turn_results,
                    args.acceptable_similarity_threshold,
                    args.good_similarity_threshold,
                )

                if selected.is_valid and selected.gen_canonical_smiles is not None:
                    all_outputs.append(selected.gen_canonical_smiles)
                if selected.is_valid:
                    current_best = choose_current_best(
                        current_best,
                        selected,
                        args.acceptable_similarity_threshold,
                        args.good_similarity_threshold,
                    )

                messages.append({"role": "assistant", "content": selected.raw_output})
                if current_best is not None:
                    feedback = (
                        f"{selected.feedback}\n"
                        f"{CURRENT_BEST_FEEDBACK.format(current_best=current_best.gen_smiles)}"
                    )
                else:
                    feedback = f"{selected.feedback}\n{FOLLOW_UP_FEEDBACK}"
                messages.append({"role": "user", "content": feedback})

                record["output"][turn] = {
                    "selected": result_to_dict(selected),
                    "candidates": [result_to_dict(result) for result in turn_results],
                }
                all_results.extend(turn_results)

            best_results = {
                selection: best_optimized_result(
                    all_results,
                    selection,
                    input_properties,
                )
                for selection in FINAL_SELECTION_CRITERIAS
            }
            record["best_optimized"] = {
                selection: result_to_dict(result) if result else None
                for selection, result in best_results.items()
            }
            records.append(record)
            success_count += int(best_results[PROGRESS_SELECTION] is not None)
            current_metrics = summarize_metrics(records, PROGRESS_SELECTION)
            progress.set_postfix(
                success=f"{success_count}/{len(records)}",
                SR=f"{current_metrics['SR']:.2f}%",
                Sim=f"{current_metrics['Sim']:.4f}",
                RI=f"{current_metrics['RI']:.4f}",
            )
            save_results(output_path, config, records)

    metrics = save_results(output_path, config, records)
    print(json.dumps({"metrics": metrics}, indent=2))
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
