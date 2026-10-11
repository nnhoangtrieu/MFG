from tasks import PROPERTY_INSTRUCTIONS, PROPERTY_LABELS

def build_user_prompt(original_smiles: str, properties: list[str]) -> str: 
    instructions = [PROPERTY_INSTRUCTIONS[prop] for prop in sorted(properties) if prop in PROPERTY_INSTRUCTIONS]
    if len(instructions) > 1:
        goals = ", ".join(instructions[:-1]) + f" and {instructions[-1]}"
    elif instructions:
        goals = instructions[0]
    else:
        goals = ""

    return f"""Your task is to modify the given molecule to adjust specific molecular properties while keeping structural changes as minimal as possible. Your response should only contain a valid SMILES representation of the modified molecule enclosed with <SMILES> </SMILES> tag.
%%% Input: <SMILES>{original_smiles}</SMILES>
%%% Adjust: {goals}"""

def get_property_feedback(
    x: dict[str, float | None],
    baseline_properties: dict[str, float | None],
    prop: str
): 
    is_improved = (x[prop] > baseline_properties[prop]) if prop != "mutagenicity" else (x[prop] < baseline_properties[prop])
    return is_improved, PROPERTY_FEEDBACK.format(
        property_label=PROPERTY_LABELS[prop],
        baseline_val=baseline_properties[prop],
        x_val=x[prop],
        change=x[prop] - baseline_properties[prop],
        improvement_status='improved' if (x[prop] > baseline_properties[prop] if prop != "mutagenicity" else x[prop] < baseline_properties[prop]) else 'not improved',
        direction="higher is better" if prop != "mutagenicity" else "lower is better"
    )

def get_similarity_feedback(similarity: float) -> str:
    if similarity >= 0.6: 
        return SIMILARITY_FEEDBACK.format(similarity=similarity, similarity_status="Very High; very good.")
    elif similarity >= 0.5: 
        return SIMILARITY_FEEDBACK.format(similarity=similarity, similarity_status="High; good.")
    elif similarity >= 0.4: 
        return SIMILARITY_FEEDBACK.format(similarity=similarity, similarity_status="Moderate; acceptable.")
    else: 
        return SIMILARITY_FEEDBACK.format(similarity=similarity, similarity_status="Low; not acceptable.")


SYSTEM_PROMPT = """You are an expert chemist in molecular optimization.
Your task is to modify the given molecule to improve the requested molecular properties while keeping structural changes as minimal as possible.
Each optimization turn must contain exactly ONE valid modified SMILES enclosed in <SMILES> </SMILES> tags.
You will receive feedback after each candidate. Use that feedback to improve the next candidate while keeping the molecule structurally similar to the original.
Rules:
- Output exactly one <SMILES>...</SMILES> block.
- Do not repeat previous candidates or the input."""


FOLLOW_UP_FEEDBACK = (
    "Revise the molecule using this feedback. Prioritize fixing properties marked worse, "
    "preserve features marked better, and keep the molecule structurally similar to the original. "
    "Return exactly ONE new SMILES enclosed in <SMILES> </SMILES> tags."
)

FEEDBACK_FORMAT_ERROR = (
    "- Feedback:\n Format error - output exactly one candidate enclosed as "
    "<SMILES>...</SMILES> with no extra text."
)

FEEDBACK_IDENTICAL_GEN_SMILES = (
    "- Feedback:\n Generated SMILES is identical to the input. You need to change "
    "the molecule structure to improve the properties."
)

FEEDBACK_DUPLICATE_GEN_SMILES = (
    "- Feedback:\n Generated SMILES is a duplicate of a previously generated molecule."
)

FEEDBACK_INVALID_GEN_SMILES = (
    "- Feedback:\n Generated SMILES is invalid. Output exactly one valid SMILES inside <SMILES>...</SMILES>."
)

FEEDBACK_UNACCEPTABLE_SIMILARITY = (
    "- Feedback:\n The generated molecule is too structurally different from the input. Make smaller modifications to improve the properties while keeping the molecule similar to the original."
)

FEEDBACK_ACCEPTABLE_SIMILARITY_AND_API = (
    "- Feedback:\n Similarity to input molecule is acceptable and all properties improved. "
    "Now focus more on similarity to the input while maintaining these improvements."
)

FEEDBACK_ACCEPTABLE_SIMILARITY_AND_NOT_API = (
    "- Feedback:\n Similarity to input molecule is acceptable but not all properties improved. "
    "Focus on improving all properties over this molecule while maintaining similarity to the original input."
)

FEEDBACK_GOOD_SIMILARITY_AND_API = (
    "- Feedback:\n Similarity to input molecule is good and all properties improved. "
    "Focus solely on further improving the properties."
)

FEEDBACK_GOOD_SIMILARITY_AND_NOT_API = (
    "- Feedback:\n Similarity to input molecule is good but not all properties improved. "
    "Focus on improving all properties over this molecule while maintaining similarity to the original input."
)

PROPERTY_FEEDBACK = (
    "- {property_label}: {baseline_val:.2f} -> {x_val:.2f} ({change:+.2f}, {improvement_status}; {direction})"
)

SIMILARITY_FEEDBACK = (
    "- Similarity to input molecule: {similarity:.2f} ({similarity_status})"
)

CURRENT_BEST_FEEDBACK = (
    "<SMILES>{current_best}</SMILES> has improved all the properties and met the similarity threshold. Follow up on this molecule to further improve the properties while maintaining similarity to the original input."
)