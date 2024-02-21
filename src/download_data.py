"""Script to download data from Ding et al repository."""

import json
import os
from io import StringIO

import hydra
import pandas as pd
import requests
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


@hydra.main(config_path="..", config_name="config")
def main(cfg: DictConfig):
    """Download data from Ding et al repository."""
    base_path = os.path.join(get_original_cwd(), cfg.data.download_dir)

    # Download all_data.csv
    r = requests.get(
        "https://anonymous.4open.science/api/repo/Lipid_Nanoparticle_Design/file/data/all_data.csv",
        timeout=10,
    )
    df = pd.read_csv(StringIO(r.text))
    df.to_csv(os.path.join(base_path, "all_data.csv"), index=False)

    # Download split.json for train / val / test splits
    r = requests.get(
        "https://anonymous.4open.science/api/repo/Lipid_Nanoparticle_Design/file/data/split.json",
        timeout=10,
    )
    with open(os.path.join(base_path, "split.json"), "w") as f:
        json.dump(r.json(), f)

    # Download json fingerprint files
    fp_files = ["mol2fp.json", "mol2fp_grover.json", "mol2fp_grover_large.json"]
    for file_ in fp_files:
        r = requests.get(
            f"https://anonymous.4open.science/api/repo/Lipid_Nanoparticle_Design/file/model/features/{file_}",
            timeout=20,
        )
        fp_dict = r.json()
        with open(os.path.join(base_path, file_), "w") as f:
            json.dump(fp_dict, f)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
