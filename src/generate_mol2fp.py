"""A simple script to convert a list of smiles and feature file (.npz) generated from grover into a
mol2fp dictionary."""
import json
import os

import hydra
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


def generate_gcn(smiles, fingerprint_file):
    """Generate GCN embeddings for a list of smiles."""
    with open(smiles) as f:
        mols = [line.strip() for line in f][1:]
    features = np.load(fingerprint_file)[:572, :]
    print(len(molsmiles), len(features))
    assert len(mols) == len(features)
    return {k: v.tolist() for k, v in zip(mols, features)}


def generate_cfp(smiles):
    """Generate circular fingerprints for a list of smiles."""
    import deepchem as dc

    with open(smiles) as f:
        mols = [line.strip() for line in f][1:]
    featurizer = dc.feat.CircularFingerprint(is_counts_based=True, chiral=True)
    features = featurizer.featurize(mols)
    return {k: v.tolist() for k, v in zip(mols, features)}


def generate_expert(smiles, fingerprint_file):
    # load data
    with open(smiles) as f:
        mols = [line.strip() for line in f][1:]
    features = np.load(fingerprint_file)["features"]
    print(len(mols), len(features))
    assert len(mols) == len(features)
    return {mol: feature.tolist() for mol, feature in zip(mols, features)}


def generate_grover(smiles, fingerprint_file):
    # load data
    with open(smiles) as f:
        mols = [line.strip() for line in f][1:]
    features = np.load(fingerprint_file)["fps"]
    print(len(mols), len(features))
    assert len(mols) == len(features)
    return {mol: feature.tolist() for mol, feature in zip(mols, features)}


@hydra.main(config_path="..", config_name="config")
def main(cfg: DictConfig):
    smiles_path = os.path.join(get_original_cwd(), cfg.fingerprints.smiles_path)
    fingerprint_path = os.path.join(get_original_cwd(), cfg.fingerprints.fingerprint_path)
    if cfg.fingerprints.type == "cfp":
        mol2fp = generate_cfp(smiles_path)
    if cfg.fingerprints.type == "expert":
        mol2fp = generate_expert(smiles_path, fingerprint_path)
    if cfg.fingerprints.type == "grover":
        mol2fp = generate_grover(smiles_path, fingerprint_path)
    if cfg.fingerprints.type == "gcn":
        mol2fp = generate_gcn(smiles_path, fingerprint_path)
    with open(os.path.join(get_original_cwd(), cfg.fingerprints.output_path), "w") as f:
        json.dump(mol2fp, f)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
