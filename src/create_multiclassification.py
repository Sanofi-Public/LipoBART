"""Create a multiclass classification dataset from figure 2 in Liu, S., Cheng, Q., Wei, T.
et al. Membrane-destabilizing ionizable phospholipids for organ-selective mRNA delivery and
CRISPR–Cas gene editing. Nat. Mater. 20, 701–710 (2021).
https://doi.org/10.1038/s41563-020-00886-0
"""

import os
import re

import hydra
import pandas as pd
from omegaconf import DictConfig
from rdkit import Chem


def get_lipid_family(lipid_name):
    """We divide the lipids into 7 families based on the Amine group as shown in figure 2 of the
    paper.

    Parameters:
    - lipid_name (str): Name of the lipid in the form nAxPm
        here nA is the amine ID
        Pm is the alkylated dioxaphospholane oxide molecule ID
        x is the number of Pm molecules attached to the amine group
    """
    # make some regex rules for assigning lipid class
    # the regex rules are based on the amine group
    class_dict = {
        "^[1-6]A1": 0,
        "^([7-9]|1[0-3])A1": 1,
        "^1[4-8]A1": 2,
        "^(19|2[0-8])A2": 3,
        "2[0-8]A3": 4,
        "2[3-8]A4": 5,
        "28A5": 6,
    }

    # check which class the amine group falls into
    for key in class_dict.keys():
        pattern = re.compile(key)
        if re.search(pattern, lipid_name):
            return class_dict[key]


def canonicalize_smiles(smiles):
    """Canonicalize a SMILES string using RDKit.

    Parameters:
    - smiles (str): Input SMILES string.

    Returns:
    - str: Canonicalized SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is not None:
        canon_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return canon_smiles
    else:
        raise ValueError(f"Invalid SMILES string: {smiles}")


@hydra.main(config_path="..", config_name="config")
def main(cfg: DictConfig):
    # Read in the smiles of m1 (cationic) and the associated target
    base_path = hydra.utils.get_original_cwd()
    df_smiles = pd.read_csv(
        os.path.join(base_path, cfg.data.multiclass.smiles), index_col=0
    )
    df_target = pd.read_csv(
        os.path.join(base_path, cfg.data.multiclass.target), index_col=0
    )

    # df_target is a matrix where the combination of row name and column name is a unique lipid
    # concatenate row and column names to get a set of keys for a dictionary
    # populate the values with the respective target
    # this dictionary will be used to map the target to the smiles
    target_dict = {}
    for i in df_target.index:
        for j in df_target.columns:
            target_dict[i + j] = df_target.loc[i, j]

    # df_smiles.combined is the smiles structure and the index is the lipid name
    # convert to a dictionary of lipid name: smiles structure
    smiles_dict = df_smiles.to_dict()["combined"]
    # canonicalize the smiles
    for key in smiles_dict.keys():
        smiles_dict[key] = canonicalize_smiles(smiles_dict[key])

    # Smiles_dict and target_dict have the same keys
    # create a new dictionary mapping the values of smiles_dict to the values of target_dict
    smiles_target_dict = {}
    for key in smiles_dict.keys():
        smiles_target_dict[smiles_dict[key]] = target_dict[key]

    # reverse the smiles to name dictionary
    smiles_name_dict = {v: k for k, v in smiles_dict.items()}

    # Read in the helper lipids
    df_all = pd.read_csv(os.path.join(base_path, cfg.data.multiclass.helper_lipids))

    # LNPs from ding et al. are extracted from a combination of papers
    # However, looking at Zhang et al. 2020, the proxy measure for transfection efficiency is completely different
    # additionally the y2 binary targets are arbitrarily assigned and I do not agree with them
    # moving forward for our multi-classification task, we will only use the LNPs from Liu et al. 2021
    # drop all other LNPs
    df_iphos = df_all.iloc[:572, :].copy()

    # create a new column for the target and populate with the target from the dictionary
    # canonicalize the m1 smiles
    df_iphos["m1"] = list(smiles_target_dict.keys())
    df_iphos["name"] = df_iphos["m1"].apply(lambda x: smiles_name_dict[x])
    df_iphos["family"] = df_iphos["name"].apply(get_lipid_family)
    df_iphos["y1"] = 0
    df_iphos["y1"] = list(smiles_target_dict.values())
    df_iphos["y2"] = df_iphos["y1"].apply(lambda x: 1 if x >= 2 else 0)
    # move y1 and y2 to the front
    cols = df_iphos.columns.tolist()
    cols = cols[-3:] + cols[:-3]
    df_iphos = df_iphos[
        ["name", "family", "y1", "y2", "p1", "p2", "p3", "p4", "m1", "m2", "m3", "m4"]
    ]
    # write to csv
    df_iphos.to_csv(os.path.join(base_path, cfg.data.multiclass.output), index=False)

    # get a set of all smiles in m1 m2 m3 m4 and write to iphos_smiles.txt
    smiles_set = set(
        df_iphos["m1"].values.tolist()
        + df_iphos["m2"].values.tolist()
        + df_iphos["m3"].values.tolist()
        + df_iphos["m4"].values.tolist()
    )
    with open("iphos_smiles.txt", "w") as f:
        f.write("smiles\n")
        for item in smiles_set:
            f.write("%s\n" % item)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
