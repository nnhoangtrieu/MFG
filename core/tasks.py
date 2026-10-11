import json
from pathlib import Path

PROPERTY_LABELS = {
    "qed": "QED (drug-likeness)",
    "plogp": "Penalized lipophilicity",
    "drd2": "Binding affinity to DRD2",
    "hia": "Human intestinal absorption",
    "bbbp": "Blood-brain barrier permeability",
    "mutagenicity": "Mutagenicity (safety)",
}

PROPERTY_INSTRUCTIONS = {
    "qed": "increase QED",
    "plogp": "increase Penalized octanol-water partition coefficient (penalized logP)",
    "drd2": "increase DRD2 inhibition",
    "bbbp": "increase BBB permeability",
    "hia": "increase Intestinal absorption",
    "mutagenicity": "decrease Mutagenicity",
}

TASK_DIRECTIONAL_CHANGE_MEDIANS_PATH = Path("core/task_directional_change_medians.json")
if TASK_DIRECTIONAL_CHANGE_MEDIANS_PATH.exists():
    with TASK_DIRECTIONAL_CHANGE_MEDIANS_PATH.open("r") as f:
        TASK_DIRECTIONAL_CHANGE_MEDIANS = json.load(f)
else:
    TASK_DIRECTIONAL_CHANGE_MEDIANS = {}


if __name__ == "__main__":
    import json
    import numpy as np
    from tqdm import tqdm
    from datasets import load_dataset
    from utils import task_to_properties, properties_to_task

    train_dataset = load_dataset("NingLab/MuMOInstruct", split="train")

    task_dic: dict[str, dict[str, list]] = {}

    for data in tqdm(train_dataset):
        task = properties_to_task(data["task"].split('+'))
        properties = task_to_properties(task)

        if task not in task_dic:
            task_dic[task] = {prop: [] for prop in properties}

        for prop in properties:
            task_dic[task][prop].append(
                data["properties"][prop]["change"]
            )

    distribution_dic: dict[str, dict[str, list[float]]] = {}
    median_dic: dict[str, dict[str, float]] = {}

    for task, change_dic in task_dic.items():

        distribution_dic[task] = {}
        median_dic[task] = {}

        for prop_key, change_list in change_dic.items():
            distribution_dic[task][prop_key] = [
                float(np.mean(change_list)),
                float(np.std(change_list)),
            ]
            directional_change_list = [
                -change if prop_key == "mutagenicity" else change
                for change in change_list
            ]
            median_dic[task][prop_key] = float(np.median(directional_change_list))

    
    # with open("core/task_change_distributions.json", "w") as f: 
    #     json.dump(distribution_dic, f, indent=4)

    with open("core/task_directional_change_medians.json", "w") as f:
        json.dump(median_dic, f, indent=4)


