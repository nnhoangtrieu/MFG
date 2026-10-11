# api stands for all properties improved 
# npi stands for number of properties improved
import numpy as np
from properties import tanimoto_similarity, get_properties, composite_ri
from utils import extract_smiles, canonicalize, get_mol, is_kekulizable
from output import Output, output_wrong_format, output_invalid_gen_smiles, output_identical_gen_smiles, output_duplicate_gen_smiles, output_very_low_similarity
from prompt import get_property_feedback, get_similarity_feedback

PROPERTY_REWARD_THRESHOLD = 1.25
FINAL_REWARD_LOWER_BOUND = -1.0

def check_all_properties_improved(input_properties: dict[str, float], gen_properties: dict[str, float], properties: list[str]) -> tuple[bool, float]:
    num_properties_improved = 0
    for prop in properties:
        input_val, gen_val = input_properties[prop], gen_properties[prop]
        if prop == "mutagenicity":
            is_improved = gen_val < input_val
        else:
            is_improved = gen_val > input_val

        if is_improved:
            num_properties_improved += 1

    fraction_properties_improved = num_properties_improved / len(properties) if properties else 0.0
    return num_properties_improved == len(properties), fraction_properties_improved

def property_reward(
    x: dict[str, float | None],
    baseline_properties: dict[str, float | None],
    target_properties: dict[str, float | None],
    properties: list[str]
): 
    all_props_improved, num_props_improved, reward, property_feedbacks = True, 0, 0, []

    for prop in sorted(properties): 
        x_val, baseline_val = x[prop], baseline_properties[prop]

        is_improved, prop_feedback = get_property_feedback(x, baseline_properties, prop)
        property_feedbacks.append(prop_feedback)

        if is_improved: 
            num_props_improved += 1
        else: 
            all_props_improved = False 

        # During inference, there is no known target property, but we only need the feedback
        if target_properties is None: 
            continue

        # Calculate the actual change and target change for the property, taking into account whether higher or lower values are better (e.g., for mutagenicity, lower is better).
        actual_change = x_val - baseline_val if prop != "mutagenicity" else baseline_val - x_val
        target_change = target_properties[prop] - baseline_val if prop != "mutagenicity" else baseline_val - target_properties[prop]
        reward += np.clip(actual_change / target_change, -PROPERTY_REWARD_THRESHOLD, PROPERTY_REWARD_THRESHOLD)  # Ensure reward is within [-1, 1]

    normalized_reward = reward / len(properties)
    return normalized_reward, num_props_improved, all_props_improved, property_feedbacks

def similarity_reward(similarity: float, target_similarity: float | None, good_similarity_threshold: float) -> tuple[float, str]:
    # During inference, there is no known target similarity, but we only need the feedback
    if target_similarity is None:
        similarity_reward = 0.0 
    else:
        similarity_reward = 1.0 if similarity >= good_similarity_threshold else similarity / target_similarity
        # similarity_reward = max(1.0, similarity / target_similarity)
    return similarity_reward, get_similarity_feedback(similarity)



def score(
    input_smiles: str,
    input_properties: dict[str, float],
    raw_output: str, 
    all_outputs: list[str],
    current_best: str | None,
    properties: list[str],
    target_props: dict[str, float] | None = None,
    target_similarity: float | None = None,
    acceptable_similarity_threshold: float = 0.4,
    good_similarity_threshold: float = 0.6
) -> Output: 
    feedback = ["- Feedback:"]

    # Extract SMILES from <SMILES></SMILES> tags in the raw output
    gen_smiles, is_clean_output = extract_smiles(raw_output)

    # Reject if output format is incorrect 
    if gen_smiles is None or not is_clean_output:
        return output_wrong_format(raw_output)
    

    # Check if the generated SMILES is valid and kekulizable
    gen_mol = get_mol(gen_smiles) 
    if gen_mol is None or not is_kekulizable(gen_mol):
        return output_invalid_gen_smiles(raw_output, gen_smiles)
    
    # Canonicalize SMILES
    gen_canonical_smiles = canonicalize(gen_smiles) 
    input_canonical_smiles = canonicalize(input_smiles) 

    # Precheck for identical or duplicate generated SMILES
    if gen_canonical_smiles == input_canonical_smiles: 
        return output_identical_gen_smiles(raw_output, gen_smiles)  
    if bool(gen_canonical_smiles and gen_canonical_smiles in all_outputs): 
        return output_duplicate_gen_smiles(raw_output, gen_smiles)

    # Compute similarity to input
    try: 
        similarity = float(tanimoto_similarity(input_smiles, gen_smiles))
        current_best_similarity = float(tanimoto_similarity(input_smiles, current_best)) if current_best else None
    except Exception: 
        return output_invalid_gen_smiles(raw_output, gen_smiles)

    if similarity < acceptable_similarity_threshold:
        return output_very_low_similarity(raw_output, gen_smiles, gen_canonical_smiles, similarity, reward=-(2.0-similarity))

    # Compute properties for input and generated molecules
    gen_properties = get_properties(gen_smiles, properties)
    current_best_is_input = current_best is None or canonicalize(current_best) == input_canonical_smiles

    if current_best_is_input: 
        current_best_properties = input_properties 
    else: 
        current_best_properties = get_properties(current_best, properties)

    # Compute rewards and feedback for similarity and properties
    sim_reward, similarity_feedback = similarity_reward(similarity, target_similarity, good_similarity_threshold)
    gen_vs_input_prop_reward, gen_vs_input_npi, gen_vs_input_api, property_feedback = property_reward(
        x=gen_properties,
        baseline_properties=input_properties,
        target_properties=target_props if target_props else None, # None for inference
        properties=properties
    )
    input_comparison_reward = gen_vs_input_prop_reward + sim_reward

    
    # Bonus if all properties improved over input and similarity is good
    if gen_vs_input_api and similarity >= acceptable_similarity_threshold: 
        print("API bonus applied")
        input_comparison_reward *= (1 + similarity) 

    # Calculate the reward of current best compared to input 
    current_best_comparison_reward = 0.0
    if not current_best_is_input: 
        cb_sim_reward, _ = similarity_reward(current_best_similarity, target_similarity, good_similarity_threshold)
        cb_vs_input_prop_reward, _, cb_vs_input_api, _ = property_reward(
            x=current_best_properties,
            baseline_properties=input_properties,
            target_properties=target_props if target_props else None, # None for inference
            properties=properties
        )
        current_best_comparison_reward = cb_vs_input_prop_reward + cb_sim_reward

        if cb_vs_input_api and current_best_similarity >= acceptable_similarity_threshold:
            current_best_comparison_reward *= (1 + current_best_similarity)


    # Final reward calculation
    if current_best_is_input:
        reward = input_comparison_reward
    else:
        print(f"IR: {input_comparison_reward:.4f}, CBR: {current_best_comparison_reward:.4f}")
        delta = input_comparison_reward - current_best_comparison_reward
        reward = input_comparison_reward + 2 * max(0.0, delta)  
    
    output = Output(
        gen_smiles=gen_smiles,
        gen_canonical_smiles=gen_canonical_smiles,
        is_valid=True,
        is_identical=False,
        is_duplicate=False,
        similarity=similarity,
        raw_output=raw_output,
        input_properties=input_properties,
        gen_properties=gen_properties,
        reward=max(FINAL_REWARD_LOWER_BOUND, reward),
        feedback="\n".join(feedback + [similarity_feedback] + property_feedback),
        relative_improvement=composite_ri(input_properties, gen_properties, properties),
        all_properties_improved=gen_vs_input_api,
        num_properties_improved=gen_vs_input_npi
    )

    return output