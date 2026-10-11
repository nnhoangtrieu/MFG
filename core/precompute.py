import os 
import json 
import torch 
import random 
import argparse
from collections import defaultdict
from typing import Any
from tqdm import tqdm 
from reward import check_all_properties_improved, score
from properties import get_properties
from prompt import FOLLOW_UP_FEEDBACK, CURRENT_BEST_FEEDBACK
from utils import load
from output import Output

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
        else:
            if gen_change < target:
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
        if quota <= len(bucket):
            sampled.extend(rng.sample(bucket, quota))
        else:
            sampled.extend(rng.choices(bucket, k=quota))

    rng.shuffle(sampled)
    return sampled

def generate(
    model,
    tokenizer,
    generation_mode,
    messages: list[dict[str, str]], 
    max_new_tokens: int = 96,
    num_beams: int = 20,
    temperature: float = 1.0,
    top_p: float = 0.9
) -> str: 
    device = next(model.parameters()).device 

    encoded = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_len = encoded["input_ids"].shape[1]


    if generation_mode == "sampling":
        with torch.no_grad(): 
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )[0] 
    elif generation_mode == "beam":
        with torch.no_grad(): 
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )[0]
    else:
        raise ValueError(f"Unsupported generation mode: {generation_mode}")

    return tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip()

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


def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--train_data", type=str, default="data/train.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/precomputed")
    parser.add_argument("--path_base_model", type=str, default="/users/PAS1440/nnhoangtrieu/rdkit_llm/GeLLMO-P6-Llama")
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
    args = parser.parse_args()

    # Load raw dataset
    rows = []
    with open(args.train_data, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line.strip()))

    # Sample a shard
    rng = random.Random(args.seed + args.shard_idx)
    sample = sample_equal_by_task(rows, args.shard_size, rng)

    # Load model and tokenizer only if max_context_turns > 0
    if args.max_context_turns > 0: 
        model, tokenizer = load(args.path_base_model, args.path_adapter)   
    else: 
        model, tokenizer = None, None 
    
    # Precompute the dataset with context turns
    precomputed_rows = []
    for row in tqdm(sample, desc=f"Precomputing shard {args.shard_idx}"): 
        # Sample number of context turns for this input
        num_context_turns = rng.randint(args.min_context_turns, args.max_context_turns) 

        # Extract input SMILES, tasks, and calculate input properties
        messages, properties, input_smiles= row["messages"], row["properties"], row["input_smiles"]
        target_change, target_props, target_similarity = row["target_change"], row["target_props"], row["target_similarity"]
        input_properties = get_properties(input_smiles, properties)

        # Track seen SMILES and the previous output.
        all_outputs = []
        current_best = None 

        # Flag for successful generation before exhausting context turns
        is_added = False

        # Generate, score, and provide feedback for the specified number of context turns
        for t in range(num_context_turns): 
            # Generate output from the model based on current messages
            raw_output = generate(
                model,
                tokenizer,
                generation_mode=args.generation_mode,
                messages=messages,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                temperature=args.temperature,
                top_p=args.top_p
            )

            # Score generated output
            result = score(input_smiles=input_smiles, 
                           input_properties=input_properties,
                           raw_output=raw_output,
                           all_outputs=all_outputs, 
                           current_best=current_best.gen_smiles if current_best is not None else None,
                           properties=properties,
                           target_props=target_props,
                           target_similarity=target_similarity,
                           acceptable_similarity_threshold=args.acceptable_similarity_threshold,
                           good_similarity_threshold=args.good_similarity_threshold
                           )

            # Backtrack (still stay at current state) if successful and add to precomputed rows
            is_strictly_successful = (
                result.is_valid
                and check_is_strictly_successful(
                    input_properties=input_properties,
                    gen_properties=result.gen_properties,
                    properties=properties,
                    target_change=target_change,
                    similarity=result.similarity
                )
            )
            if is_strictly_successful: 
                precomputed_rows.append({
                    "messages": messages, 
                    "input_smiles": input_smiles,
                    "current_best": current_best.gen_smiles if current_best is not None else None,
                    "input_properties": input_properties,
                    "all_outputs": all_outputs,
                    "properties": properties,
                    "num_context_turns": t,
                    "target_props": target_props,
                    "target_similarity": target_similarity
                })
                is_added = True
                break
            
            # Update seen SMILES if valid and canonical SMILES is available
            if result.is_valid and result.gen_canonical_smiles is not None:
                all_outputs.append(result.gen_canonical_smiles)

            if result.is_valid:
                current_best = choose_current_best(current_best, candidate=result, 
                                                   acceptable_similarity_threshold=args.acceptable_similarity_threshold, 
                                                   good_similarity_threshold=args.good_similarity_threshold)

            
            # Append feedback and follow-up prompt for the next turn
            messages = messages + [{"role": "assistant", "content": raw_output}]

            if current_best is not None: 
                messages += [{"role": "user", "content": f"{result.feedback}\n{CURRENT_BEST_FEEDBACK.format(current_best=current_best.gen_smiles)}"}]
            else:
                messages += [{"role": "user", "content": f"{result.feedback}\n{FOLLOW_UP_FEEDBACK}"}]



        # Add row if not successful within context turns
        if not is_added:
            precomputed_rows.append({
                "messages": messages,
                "input_smiles": input_smiles,
                "current_best": current_best.gen_smiles if current_best is not None else None,
                "input_properties": input_properties,
                "properties": properties,
                "all_outputs": all_outputs,
                "num_context_turns": num_context_turns,
                "target_props": target_props,
                "target_similarity": target_similarity
            })

    # Save the precomputed rows to a new file for this shard
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/shard_{args.shard_idx:04d}.jsonl", "w", encoding="utf-8") as f:
        for row in precomputed_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    
if __name__ == "__main__":
    main()
