# Lipid Nanoparticle Design

Repository containing data, code and walkthrough for methods in the paper [*Representations of lipid nanoparticles using large language models for transfection efficiency prediction*](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btae342/7684951).

We aim to design LNPs for delivery of mRNA therapeutics. With this goal in mind we therefore define a successful LNP as one that has following attributes: biodegradeble, non-toxic, synthesizable, stable (pH/temperature) and (most importantly) transfection efficiency.
We design predictive models that estimate these qualities with the intention of using them in screening/selecting the best candidates for experimental testing and development.

We reproduce the results of [*Ding et al., 2023*](https://arxiv.org/abs/2308.01402) in this repository. We then also compare how are methods compare on this benchmark dataset.

## Environment Setup

Dependency management is done via [poetry](https://python-poetry.org/).

```
pip install poetry
pip install tensorflow    # tensorflow is installed separately
poetry install
```

## Package Structure

We organize our code into the following structure:
- `data`: contains all data used in our experiments. This includes CSVs with lipids SMILES and properties, and corresponding fingerprint JSON files from different embedding methods. Those ending in `_alldata` correspond to SMILES from *Ding et al*.
- `src`: contains the source code for downloading data from *Ding et al.*, splitting data for multiclass classification, running the tournament script for classification, and some helpers.
- `notebooks`: contains notebooks for data exploration, as well as gcn training and embedding extraction, and downstream classification with different methods.

## Fine-Tuning MegaMolBART 
We found that the best performing model for predicting LNP transfection efficiency relies on embedding LNPs with a large language model. Here we outline our method for finetuning NVIDIA's MegaMolBART model on the swisslipid dataset.

1. Download, Install and Setup MegaMolBART pre-trained model. We recommend installing using the container.
  - [repository](https://github.com/NVIDIA/MegaMolBART)
  - [tutorial](https://docs.nvidia.com/bionemo-framework/0.4.0/notebooks/MMB_GenerativeAI_Inference_with_examples.html) 
2. Download the [swisslipid dataset](https://www.swisslipids.org/)
3. Edit the model configuration file such that (i.e. `megamolbart_pretrain_base.yaml`):
  - `restore_from_path` points to the path of the pre-trained MegaMolBART `.nemo` file
  - adjust `trainer` and `batch_size` params optimal to your system settings (number and memory of GPUs)
  - `dataset_path` is set to the location of the downloaded swisslipid dataset.
    -   *edit the swisslipid dataset such that it conforms to the datamodel of the MegaMolBART example dataset. Split the data into 99% training 0.5% test, 0.5% validation*
4. Run the pre-training script with this configuration ([instructions here] (https://github.com/NVIDIA/MegaMolBART/blob/dev/QUICKSTART.md#training-megamolbart))
5. Stop training once validation_molecular_accuracy converges
  
