import os 
import sys
import json 
import random 
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


user_prompt = [
    "Modify the given molecule to adjust the specified molecular properties by substituting functional groups while keeping changes to the core structure minimal. Output only the SMILES of the modified molecule, wrapped in <SMILES> </SMILES> tags.",
    "Your goal is to fine-tune the specified molecular properties of the given compound with minimal structural changes. Make the necessary adjustments and return the modified molecule in a SMILES format enclosed in <SMILES> </SMILES> tags.",
    "Adjust the structure of the given molecule to target the specified adjustments in molecular properties. Retain the core structure as much as possible. Respond with only the SMILES of the modified molecule enclosed in <SMILES> </SMILES> tags.",
    "Modify the given molecular structure to target specific property changes, aiming to keep structural adjustments minimal. Respond solely with the SMILES notation for the adjusted molecule, enclosed within <SMILES> </SMILES> tags.",
    "Alter the given molecule to meet the desired property changes with the least structural alteration possible. Output only the adjusted molecule in SMILES format, using <SMILES> </SMILES> tags."
]
user_prompt.append("Your task is to modify the given molecule to adjust specific molecular properties while keeping structural changes as minimal as possible. Your response should only contain a valid SMILES representation of the modified molecule enclosed with <SMILES> </SMILES> tag.")



def build_user_prompt(
    original_smiles: str,
    properties: list[str],
    instr_idx: int,
    property_instructions: dict[str, str],
) -> str:
    instructions = [property_instructions[prop] for prop in properties if prop in property_instructions]
    if len(instructions) > 1:
        goals = ", ".join(instructions[:-1]) + f" and {instructions[-1]}"
    elif instructions:
        goals = instructions[0]
    else:
        goals = ""

    return f"""{user_prompt[instr_idx]}
%%% Input: <SMILES>{original_smiles}</SMILES>
%%% Adjust: {goals}"""


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
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--include", type=str, nargs="+", help="List of task combinations to sample from")
    data_group.add_argument("--exclude", type=str, nargs="+", help="List of task combinations to skip")
    data_group.add_argument("--existing_data", type=str, help="Existing JSONL dataset to use without rebuilding or sampling")
    args = parser.parse_args() 

    if args.existing_data:
        rows = []
        with open(args.existing_data, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        print(f"Loaded {len(rows)} samples from {args.existing_data}")
    else:
        from datasets import load_dataset
        from tqdm import tqdm
        from tasks import PROPERTY_INSTRUCTIONS
        from properties import tanimoto_similarity, get_properties
        from utils import properties_to_task

        # Create random number generator with the specified seed for reproducibility
        rng = random.Random(args.seed)

        # Determine which tasks to include or exclude based on the provided arguments
        include_tasks = args.include if args.include else None
        exclude_tasks = args.exclude if args.exclude else None

        # Load the dataset and process each row to build the final dataset
        dataset = load_dataset("NingLab/MuMOInstruct", split="train")
        rows = []
        for row in tqdm(dataset, desc="Building dataset"):
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
                        {
                            "role": "user",
                            "content": build_user_prompt(
                                input_smiles,
                                properties,
                                row["instr_idx"],
                                PROPERTY_INSTRUCTIONS,
                            ),
                        },
                    ],
                    "input_smiles": input_smiles,
                    "input_properties": get_properties(input_smiles, properties),
                    "properties": properties,
                    "task": task,
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
