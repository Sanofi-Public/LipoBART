"""We use this script to generate embeddings for all of the SMILES in the ding dataset using our
own methods."""
import json

import hydra
import pandas as pd
from deepchem.feat import CircularFingerprint, RDKitDescriptors
from omegaconf import DictConfig


def generate_cfp(smiles):
    """Generate circular fingerprints for all SMILES in the dataset."""
    f = CircularFingerprint(1024, is_counts_based=True, chiral=True)
    features = f.featurize(list(smiles))
    return {k: list(v) for k, v in zip(smiles, features)}


def generate_rdkit(smiles):
    """Generate RDKit features for all SMILES in the dataset."""
    featurizer = RDKitDescriptors()
    features = featurizer.featurize(smiles)
    return {smiles: list(feature) for smiles, feature in zip(smiles, features)}


@hydra.main(config_path="..", config_name="config")
def main(cfg: DictConfig):
    if cfg.embeddings.input.endswith("csv"):
        df = pd.read_csv(cfg.embeddings.input)
        # collect all smiles from columns m1, m2, m3, m4 and remove duplicates
        smiles = list(
            set(
                df["m1"].values.tolist()
                + df["m2"].values.tolist()
                + df["m3"].values.tolist()
                + df["m4"].values.tolist()
            )
        )
        mol2fp = generate_rdkit(smiles)
    else:
        with open(cfg.embeddings.input) as f:
            mol2fp = json.load(f)
            smiles = list(mol2fp.keys())

    if cfg.embeddings.method == "cfp":
        mol2fp = generate_cfp(smiles)
    elif cfg.embeddings.method == "rdkit":
        mol2fp = generate_rdkit(smiles)

    with open(cfg.embeddings.output, "w") as f:
        json.dump(mol2fp, f)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
