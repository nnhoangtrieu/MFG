import os 
import json 
import random 
import argparse
from tqdm import tqdm 
from datasets import load_dataset 
from prompt import SYSTEM_PROMPT, build_user_prompt
from properties import tanimoto_similarity
from utils import properties_to_task


def sample_rows_per_task(
    rows: list[dict],
    samples_per_task: int,
    rng: random.Random,
) -> list[dict]:
    rows_by_task: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_task.setdefault(row["task"], []).append(row)

    sampled_rows = []
    for task in sorted(rows_by_task):
        task_rows = rows_by_task[task]
        sample_size = min(samples_per_task, len(task_rows))
        sampled_rows.extend(rng.sample(task_rows, k=sample_size))

    return sampled_rows


def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--seed", type=int, default=910, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="data/train.jsonl", help="Output file path for the dataset")
    parser.add_argument("--samples_per_task", type=int, default=100, help="Maximum number of samples to keep for each task")
    combination_group = parser.add_mutually_exclusive_group(required=True)
    combination_group.add_argument("--include", type=str, nargs="+", help="List of task combinations to sample from")
    combination_group.add_argument("--exclude", type=str, nargs="+", help="List of task combinations to skip")
    args = parser.parse_args() 

    # Create random number generator with the specified seed for reproducibility
    rng = random.Random(args.seed)

    # Determine which tasks to include or exclude based on the provided arguments
    include_tasks = args.include if args.include else None
    exclude_tasks = args.exclude if args.exclude else None

    # Load the dataset and process each row to build the final dataset
    dataset = load_dataset("NingLab/MuMOInstruct", split="train")
    rows = [] 
    for row in tqdm(dataset, desc="Building dataset"): 
        # original task: bbbp+drd2+plogp -> properties: ['bbbp', 'drd2', 'plogp'] -> task: 'BDP'
        properties = sorted([task.strip() for task in str(row["task"]).split("+")])
        task = properties_to_task(properties)

        # Determine whether to include or exclude the current task based on the provided arguments
        if include_tasks is not None:
            should_add = any(task == t for t in include_tasks)
        else:
            should_add = not any(task == t for t in exclude_tasks)

        # If the task should be added, extract input SMILES, input properties, target change, and target similarity, and append to the rows list
        if should_add:
            input_smiles = str(row["source_smiles"]).strip() 
            target_smiles = str(row["target_smiles"]).strip()
            target_similarity = float(tanimoto_similarity(input_smiles, target_smiles))
            rows.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(input_smiles, properties)},
                ],
                "input_smiles": input_smiles,
                "task": task,
                "properties": properties,
                "target_change": {k: v["change"] for k, v in row["properties"].items()},
                "target_props": {k: v["target"] for k, v in row["properties"].items()},
                "target_similarity": target_similarity
            })

    # Sample a specified number of rows per task and shuffle the final dataset
    rows = sample_rows_per_task(rows, args.samples_per_task, rng)
    rng.shuffle(rows)

    # Log the all tasks used and final dataset size 
    task_counts = {}
    for row in rows:
        task_counts[row["task"]] = task_counts.get(row["task"], 0) + 1
    print(f"Task counts: {task_counts}")
    print(f"Final dataset size: {len(rows)} samples")
    
    # Save the final dataset and used tasks 
    os.makedirs(args.output, exist_ok=True)
    with open(f"{args.output}/train.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(f"{args.output}/tasks_count.json", "w", encoding="utf-8") as f:
        json.dump(task_counts, f, indent=4)


if __name__ == "__main__":
    main()
