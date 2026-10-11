import os 
import copy
import torch
import argparse
import json
import random
from tqdm import tqdm
from prompt import build_user_prompt, FOLLOW_UP_FEEDBACK, SYSTEM_PROMPT, CURRENT_BEST_FEEDBACK
from utils import task_to_properties, load
from reward import score
from datasets import load_dataset
from properties import get_properties
from output import Output

NUM_BEAMS = 20 # Follow GeLLMO experiment
BUDGET = 20 # Follow GeLLMO experiment
FINAL_SELECTION_CRITERIAS = ["RI_strict04", "RI_strict05", "RI_strict06"]
PROGRESS_SELECTION = "RI_strict05"
STRICT_RI_MIN_SIMILARITY = {
    "RI_strict04": 0.4,
    "RI_strict05": 0.5,
    "RI_strict06": 0.6,
}


def choose_current_best(current_best: Output | None, candidate: Output, acceptable_similarity_threshold: float = 0.4, good_similarity_threshold: float = 0.6):
    if current_best is None: 
        if candidate.is_valid and candidate.similarity >= acceptable_similarity_threshold and candidate.all_properties_improved: 
            return candidate
        else:
            return None 
    else:
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

def get_best_result(results, 
                    acceptable_similarity_threshold: float = 0.4,
                    good_similarity_threshold: float = 0.6) -> Output:
    return max(
        results,
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

def strict_ri_candidate(
    result: Output,
    input_properties: dict[str, float],
    min_similarity: float,
) -> Output | None:
    if not result.is_valid or not result.gen_canonical_smiles or result.similarity < min_similarity:
        return None

    try:
        rounded_input = {
            prop: round(float(value), 2)
            for prop, value in input_properties.items()
        }
        rounded_output = {
            prop: round(float(result.gen_properties[prop]), 2)
            for prop in rounded_input
        }
    except (KeyError, TypeError, ValueError):
        return None

    if not rounded_input:
        return None

    relative_improvements = []
    for prop, input_value in rounded_input.items():
        output_value = rounded_output[prop]
        is_improved = (
            output_value < input_value
            if prop == "mutagenicity"
            else output_value > input_value
        )
        if not is_improved:
            return None
        relative_improvements.append(
            (output_value - 0.1) / 0.1
            if input_value == 0.0 or abs(input_value) < 0.01
            else abs(output_value - input_value) / abs(input_value)
        )

    strict_result = copy.copy(result)
    strict_result.relative_improvement = (
        sum(relative_improvements) / len(relative_improvements)
    )
    strict_result.all_properties_improved = True
    return strict_result


def best_optimized_result(
    results: list[Output],
    selection: str,
    input_properties: dict[str, float] | None = None,
) -> Output | None:
    if selection in STRICT_RI_MIN_SIMILARITY:
        if not input_properties:
            return None
        strict_results = [
            strict_result
            for result in results
            if (
                strict_result := strict_ri_candidate(
                    result,
                    input_properties,
                    STRICT_RI_MIN_SIMILARITY[selection],
                )
            ) is not None
        ]
        if not strict_results:
            return None
        return max(
            strict_results,
            key=lambda result: (result.relative_improvement, result.similarity),
        )

    improved = [result for result in results if result.all_properties_improved]
    if not improved:
        return None

    return max(improved, key=lambda result: (result.relative_improvement, result.similarity))


def summarize_metrics(records: list[dict], selection: str) -> dict[str, float]:
    successes = [
        record["best_optimized"][selection]
        for record in records
        if record["best_optimized"][selection]
    ]
    return {
        "SR": 100.0 * len(successes) / len(records) if records else 0.0,
        "Sim": (
            sum(result["similarity"] for result in successes) / len(successes)
            if successes
            else 0.0
        ),
        "RI": (
            sum(result["relative_improvement"] for result in successes) / len(successes)
            if successes
            else 0.0
        ),
        "n": len(records),
        "n_success": len(successes),
    }


def summarize_by_task(records: list[dict], selection: str) -> dict[str, dict[str, float]]:
    tasks = sorted({record["task"] for record in records})
    return {
        task: summarize_metrics(
            [record for record in records if record["task"] == task],
            selection,
        )
        for task in tasks
    }


def result_to_dict(result):
    return {
        "raw_output": result.raw_output,
        "gen_smiles": result.gen_canonical_smiles,
        "properties": result.gen_properties,
        "similarity": result.similarity,
        "relative_improvement": result.relative_improvement,
        "reward": result.reward,
        "all_properties_improved": result.all_properties_improved,
        "is_successful": getattr(result, "is_successful", result.all_properties_improved),
        "is_valid": result.is_valid,
    }


def load_benchmark(tasks: list[str]): 
    dataset = load_dataset("NingLab/MuMOInstruct", split="test")
    # rows = [] 

    data = {c: [] for c in tasks}

    for row in tqdm(dataset, desc="Loading benchmark data"): 
        if row["instr_setting"] != "unseen": 
            continue 

        properties = [t for t in str(row['task']).split("+")]
        for task in tasks: 
            if task_to_properties(task) == sorted(properties): 
                data[task].append(
                    {
                        "input_smiles": str(row["source_smiles"]).strip(),
                        "task": task,
                        "properties": row["properties"]
                    }
                )

    return data 


def generate(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int = 192,
    num_beams: int = 20,
    num_return_sequences: int = 20,
) -> list[str]:
    device = next(model.parameters()).device

    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_len = encoded["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            do_sample=False,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return [
        tokenizer.decode(output_id[prompt_len:], skip_special_tokens=True).strip()
        for output_id in output_ids
    ]


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_base_model", type=str, default="/users/PAS1440/nnhoangtrieu/rdkit_llm/GeLLMO-P6-Llama")
    parser.add_argument("--path_checkpoint", type=str, required=True)
    parser.add_argument("--tasks", type=str, nargs="+", default=None)
    parser.add_argument("--num_turns", type=int, default=2)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--acceptable_similarity_threshold", type=float, default=0.4)
    parser.add_argument("--good_similarity_threshold", type=float, default=0.6)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=910)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load model, tokenizer, and benchmark data
    model, tokenizer = load(args.path_base_model, args.path_checkpoint)
    data = load_benchmark(args.tasks)

    # Process each row in the benchmark data
    final_results = []
    success_count = 0

    for task in args.tasks: 
        progress = tqdm(data[task], desc=f"Inference {task}")
        
        for row in progress: 
            input_smiles, task, original_input_properties = row["input_smiles"], row["task"], row["properties"]
            properties = task_to_properties(task)
            input_properties = get_properties(input_smiles, properties)

            record = {
                "input_smiles": input_smiles, 
                "task": task,
                "original_input_properties": original_input_properties,
                "recalculated_properties": input_properties,
                "output": {}
            }
            
            # Build initial messages for the conversation
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(input_smiles, properties)}, # Be careful of orders of tasks
            ]
            
            # Tracking
            all_outputs = []
            all_results = []
            current_best = None
            used_budget = 0 

            for turn in range(1, args.num_turns + 1): 
                num_return_sequences = int(args.budget // args.num_turns)
                raw_outputs = generate(
                    model, 
                    tokenizer,
                    messages=messages,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=NUM_BEAMS,
                    num_return_sequences=num_return_sequences if turn < args.num_turns else args.budget - used_budget,
                )
                used_budget += num_return_sequences

                turn_results = []
                for raw_output in raw_outputs:
                    result = score(input_smiles=input_smiles, 
                                   input_properties=input_properties,
                                   raw_output=raw_output,
                                   all_outputs=all_outputs,
                                   current_best=current_best.gen_smiles if current_best else None,
                                   properties=properties,
                                   target_props=None, 
                                   target_similarity=None,
                                   acceptable_similarity_threshold=args.acceptable_similarity_threshold,
                                   good_similarity_threshold=args.good_similarity_threshold)
                    turn_results.append(result)

                # Extract the best result to be the input for the next turn
                result = get_best_result(turn_results, args.acceptable_similarity_threshold, args.good_similarity_threshold)
                
                # Update all_outputs if the selected result is valid
                if result.is_valid and result.gen_canonical_smiles is not None:
                    all_outputs.append(result.gen_canonical_smiles)

                if result.is_valid: 
                    current_best = choose_current_best(current_best, candidate=result,
                                                       acceptable_similarity_threshold=args.acceptable_similarity_threshold,
                                                       good_similarity_threshold=args.good_similarity_threshold)

                # Append feedback and follow-up prompt for the next turn
                messages = messages + [{"role": "assistant", "content": result.raw_output}]

                if current_best is not None: 
                    messages += [{"role": "user", "content": f"{result.feedback}\n{CURRENT_BEST_FEEDBACK.format(current_best=current_best.gen_smiles)}"}]
                else:
                    messages += [{"role": "user", "content": f"{result.feedback}\n{FOLLOW_UP_FEEDBACK}"}]


                record["output"][turn] = {
                    "selected": result_to_dict(result),
                    "candidates": [result_to_dict(turn_result) for turn_result in turn_results],
                }
                all_results.extend(turn_results)

            # Extract the best result from the whole run
            best_results = {
                selection: best_optimized_result(
                    all_results,
                    selection,
                    record["recalculated_properties"],
                )
                for selection in FINAL_SELECTION_CRITERIAS
            }
            record["best_optimized"] = {
                selection: result_to_dict(result) if result else None
                for selection, result in best_results.items()
            }

            final_results.append(record)
            success_count += int(best_results[PROGRESS_SELECTION] is not None)
            current_metrics = summarize_metrics(final_results, PROGRESS_SELECTION)
            progress.set_postfix(
                success=f"{success_count}/{len(final_results)}",
                SR=f"{current_metrics['SR']:.2f}%",
                Sim=f"{current_metrics['Sim']:.4f}",
                RI=f"{current_metrics['RI']:.4f}",
            )

        # Summarize metrics with different criteria and save the results to a JSON file
        metrics = {
            selection: summarize_by_task(final_results, selection)
            for selection in FINAL_SELECTION_CRITERIAS
        }
        print(json.dumps({"metrics": metrics}, indent=2))

        os.makedirs(args.output_dir, exist_ok=True)
        with open(f"{args.output_dir}/{'_'.join(args.tasks)}_{args.num_turns}turn.json", "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "results": final_results}, f, indent=2)




if __name__ == "__main__":
    main()
