import os 
from swift.rewards import ORM, orms
from reward import score


class MoleculeReward(ORM):
    def __call__(
        self,
        completions: list[str],
        input_smiles: list[str],
        input_properties: list[dict[str, float]],
        properties: list[list[str]],
        all_outputs: list[list[str]],
        current_best: list[str],
        target_props: list[dict[str, float]],
        target_similarity: list[float],
        **kwargs,
    ) -> list[float]:
        rewards = []

        for idx in range(len(completions)): 
            result = score(
                input_smiles=input_smiles[idx],
                input_properties=input_properties[idx],
                raw_output=completions[idx],
                all_outputs=all_outputs[idx],
                current_best=current_best[idx],
                properties=properties[idx],
                target_props=target_props[idx],
                target_similarity=target_similarity[idx],
                acceptable_similarity_threshold=float(os.getenv("ACCEPTABLE_SIMILARITY_THRESHOLD")),
                good_similarity_threshold=float(os.getenv("GOOD_SIMILARITY_THRESHOLD")),
            )
            rewards.append(float(result.reward))

            smiles = result.gen_canonical_smiles if result.gen_canonical_smiles else result.raw_output

            print(f"{smiles} - RI: {result.relative_improvement:.4f} - sim: {result.similarity:.4f} - r: {result.reward:.4f}")
            if (idx + 1) % int(os.getenv("NUM_GENERATIONS")) == 0:
                print("-"*20)
        return rewards 


orms["molecule"] = MoleculeReward
