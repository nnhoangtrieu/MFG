import argparse
import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm
from transformers import pipeline, set_seed

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


def generate(
    generator,
    messages: list[dict[str, str]],
    num_beams: int,
    num_return_sequences: int,
    max_new_tokens: int,
    decoding: str,
    enable_thinking: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> list[str]:
    generation_args = {
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": num_return_sequences,
        "return_full_text": False,
    }

    if decoding == "sample":
        generation_args.update(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,

        )
    else:
        generation_args.update(
            do_sample=False,
            num_beams=num_beams,
            early_stopping=True,
        )

    outputs = generator(
        messages,
        tokenizer_encode_kwargs={"enable_thinking": enable_thinking},
        **generation_args,
    )
    return [output["generated_text"].strip() for output in outputs]


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
            resume_parser.error(
                f"--resume path does not exist: {resume_args.resume}"
            )
        with resume_args.resume.open(encoding="utf-8") as file:
            resume_payload = json.load(file)
        resume_config = resume_payload.get("config", {})

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a Hugging Face chat model with Transformers pipeline, "
            "the main/ prompts and scoring, and a 20-candidate budget."
        )
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--tasks",
        nargs="+",
        required=resume_args.resume is None,
    )
    parser.add_argument("--num_turns", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument(
        "--decoding",
        choices=["beam", "sample"],
        default="beam",
        help="Use beam search or multinomial sampling.",
    )
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument(
        "--enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Thinking is disabled because the task expects one tagged SMILES.",
    )
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--acceptable_similarity_threshold", type=float, default=0.4)
    parser.add_argument("--good_similarity_threshold", type=float, default=0.6)
    parser.add_argument(
        "--output_dir",
        default="results/general_llm_transformers",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help=(
            "Path to an existing results JSON file. Saved arguments are reused "
            "as defaults."
        ),
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=20,
        help="Save a checkpoint every N completed examples (default: 20).",
    )
    parser.add_argument("--seed", type=int, default=910)
    parser.set_defaults(**resume_config)
    args = parser.parse_args()

    if not 1 <= args.num_turns <= args.budget:
        parser.error(f"--num_turns must be between 1 and {args.budget}")
    if not args.tasks:
        parser.error(
            "--tasks is required because the resume file does not contain it"
        )
    if args.save_every < 1:
        parser.error("--save_every must be at least 1")

    set_seed(args.seed)
    config = {
        key: value
        for key, value in vars(args).items()
        if key != "resume"
    }
    config["budget"] = args.budget

    if args.resume:
        records = resume_payload["results"]
        output_path = args.resume
    else:
        records = []
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_name = Path(args.model.rstrip("/")).name
        output_path = output_dir / (
            f"{model_name}_{'_'.join(args.tasks)}_{args.num_turns}turn.json"
        )

    benchmark = load_benchmark(args.tasks)
    completed_cases = Counter(
        (record["task"], record["input_smiles"]) for record in records
    )
    success_count = sum(
        record["best_optimized"][PROGRESS_SELECTION] is not None
        for record in records
    )
    generator = None

    for task in args.tasks:
        task_rows = benchmark[task]

        progress = tqdm(task_rows, desc=f"Inference {task}")
        for row in progress:
            input_smiles = row["input_smiles"]
            case = (task, input_smiles)
            if completed_cases[case]:
                completed_cases[case] -= 1
                continue

            if generator is None:
                generator = pipeline(
                    task="text-generation",
                    model=args.model,
                    device_map=args.device_map,
                    dtype=args.dtype,
                    trust_remote_code=args.trust_remote_code,
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
                    args.budget // args.num_turns
                    if turn < args.num_turns
                    else args.budget - used_budget
                )
                raw_outputs = generate(
                    generator,
                    messages,
                    num_beams=args.num_beams,
                    num_return_sequences=turn_budget,
                    max_new_tokens=args.max_new_tokens,
                    decoding=args.decoding,
                    enable_thinking=args.enable_thinking,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
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

                messages.append(
                    {"role": "assistant", "content": selected.raw_output}
                )
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
                    "candidates": [
                        result_to_dict(result) for result in turn_results
                    ],
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
            if len(records) % args.save_every == 0:
                save_results(output_path, config, records)

    metrics = save_results(output_path, config, records)
    print(json.dumps({"metrics": metrics}, indent=2))
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
