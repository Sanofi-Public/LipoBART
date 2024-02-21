# Lipid Nanoparticle Design

Repository containing data, code and walkthrough for methods in the paper [*PAPER NAME*](www.google.com).

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

[LNP Design Pipeline](lnp-design-diagram.png)
