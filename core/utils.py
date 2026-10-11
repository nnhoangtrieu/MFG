import re
from peft import PeftModel
from rdkit import Chem 
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tasks import TASK_DIRECTIONAL_CHANGE_MEDIANS 



def load(path_base_model: str, path_adapter: str | None = None) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(path_base_model, 
                                                 torch_dtype="bfloat16",
                                                 device_map="auto",
                                                 trust_remote_code=True)
    
    adapter = (path_adapter or "").strip()
    if adapter:
        model = PeftModel.from_pretrained(model, 
                                        adapter,
                                        is_trainable=False) 

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        path_base_model,
        padding_side="left", 
        trust_remote_code=True
    )

    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer

def extract_smiles(raw_output: str) -> tuple[Optional[str], bool]: 
    clean_match = re.fullmatch(r"<SMILES>(.*?)</SMILES>", raw_output.strip(), re.DOTALL)
    
    if clean_match: 
        return clean_match.group(1).strip(), True
    
    match = re.search(r"<SMILES>(.*?)</SMILES>", raw_output, re.DOTALL)
    if match:
        return match.group(1).strip(), False
    
    return None, False


def canonicalize(smiles: str | None) -> str | None: 
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None
    

def get_mol(smiles: str): 
    return Chem.MolFromSmiles(smiles)


def is_kekulizable(mol: Chem.Mol) -> bool: 
    try:
        Chem.Kekulize(mol)
        return True
    except:
        return False


def task_to_properties(task: str) -> list[str]: 
    task = task.lower() 
    tasks = set()

    if "q" in task: 
        tasks.add("qed") 
    if "p" in task: 
        tasks.add("plogp") 
    if "d" in task: 
        tasks.add("drd2") 
    if "h" in task: 
        tasks.add("hia") 
    if "b" in task: 
        tasks.add("bbbp") 
    if "m" in task: 
        tasks.add("mutagenicity") 
    return sorted(list(tasks))

def properties_to_task(properties) -> str: 
    prop_to_char = {
        "qed": "Q",
        "plogp": "P",
        "drd2": "D",
        "hia": "H",
        "bbbp": "B",
        "mutagenicity": "M",
    }
    return "".join(sorted(prop_to_char[prop] for prop in properties if prop in prop_to_char))

def approximate_median_directional_change(task: str, prop: str) -> float:
    median_change = TASK_DIRECTIONAL_CHANGE_MEDIANS[task][prop]
    return abs(float(median_change))

if __name__ == "__main__":
    # Test the functions
    test_smiles = "CCO"
    print(f"Canonical SMILES of {test_smiles}: {canonicalize(test_smiles)}")
    print(f"Extracted SMILES from raw output: {extract_smiles('<SMILES>CCO</SMILES>')}")
    print(f"Is the molecule kekulizable? {is_kekulizable(get_mol(test_smiles))}")
    print(f"Properties for task 'BPQ': {task_to_properties('BPQ')}")
    print(f"task for properties {{'qed', 'plogp', 'drd2', 'hia', 'bbbp', 'mutagenicity'}}: {properties_to_task({'qed', 'plogp', 'drd2', 'hia', 'bbbp', 'mutagenicity'})}")
