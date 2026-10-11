from dataclasses import dataclass 
from prompt import (FEEDBACK_FORMAT_ERROR, 
                    FEEDBACK_IDENTICAL_GEN_SMILES,
                    FEEDBACK_DUPLICATE_GEN_SMILES, 
                    FEEDBACK_INVALID_GEN_SMILES,
                    FEEDBACK_UNACCEPTABLE_SIMILARITY)

FAILED_REWARD = -2.0
FAILED_RELATIVE_IMPROVEMENT = -5.0

@dataclass 
class Output: 
    gen_smiles: str 
    gen_canonical_smiles: str 
    raw_output: str | None
    similarity: float 
    input_properties: dict[str, float | None]
    gen_properties: dict[str, float | None] 
    relative_improvement: float | None
    reward: float 
    feedback: str
    is_valid: bool
    is_identical: bool
    is_duplicate: bool 
    all_properties_improved: bool
    num_properties_improved: int

def output_wrong_format(raw_output: str) -> Output:
    return Output(
            gen_smiles=None, 
            gen_canonical_smiles=None,
            is_valid=False,
            is_identical=False,
            is_duplicate=False, 
            similarity=0.0,
            raw_output=raw_output,
            relative_improvement=FAILED_RELATIVE_IMPROVEMENT,
            input_properties={},
            gen_properties={},
            reward=FAILED_REWARD,
            feedback=FEEDBACK_FORMAT_ERROR,
            all_properties_improved=False,
            num_properties_improved=0
        )

def output_invalid_gen_smiles(raw_output: str, gen_smiles: str) -> Output:
    return Output(
            gen_smiles=gen_smiles, 
            gen_canonical_smiles=None,
            is_valid=False,
            is_identical=False,
            is_duplicate=False, 
            similarity=0.0,
            raw_output=raw_output,
            relative_improvement=FAILED_RELATIVE_IMPROVEMENT,
            input_properties={},
            gen_properties={},
            reward=FAILED_REWARD,
            feedback=FEEDBACK_INVALID_GEN_SMILES,
            all_properties_improved=False,
            num_properties_improved=0
        )

def output_identical_gen_smiles(raw_output: str, gen_smiles: str) -> Output:
    return Output(
            gen_smiles=gen_smiles, 
            gen_canonical_smiles=None,
            is_valid=False,
            is_identical=True,
            is_duplicate=False, 
            similarity=0.0,
            raw_output=raw_output,
            relative_improvement=FAILED_RELATIVE_IMPROVEMENT,
            input_properties={},
            gen_properties={},
            reward=FAILED_REWARD,
            feedback=FEEDBACK_IDENTICAL_GEN_SMILES,
            all_properties_improved=False,
            num_properties_improved=0
        )

def output_duplicate_gen_smiles(raw_output: str, gen_smiles: str) -> Output:
    return Output(
            gen_smiles=gen_smiles, 
            gen_canonical_smiles=None,
            is_valid=False,
            is_identical=False,
            is_duplicate=True, 
            similarity=0.0,
            raw_output=raw_output,
            relative_improvement=FAILED_RELATIVE_IMPROVEMENT,
            input_properties={},
            gen_properties={},
            reward=FAILED_REWARD,
            feedback=FEEDBACK_DUPLICATE_GEN_SMILES,
            all_properties_improved=False,
            num_properties_improved=0
        )

def output_very_low_similarity(
    raw_output: str,
    gen_smiles: str,
    gen_canonical_smiles: str,
    similarity: float,
    reward: float
) -> Output:
    return Output(
            gen_smiles=gen_smiles,
            gen_canonical_smiles=gen_canonical_smiles,
            is_valid=False,
            is_identical=False,
            is_duplicate=False,
            similarity=similarity,
            raw_output=raw_output,
            relative_improvement=FAILED_RELATIVE_IMPROVEMENT,
            input_properties={},
            gen_properties={},
            reward=reward,
            feedback=FEEDBACK_UNACCEPTABLE_SIMILARITY,
            all_properties_improved=False,
            num_properties_improved=0
        )
