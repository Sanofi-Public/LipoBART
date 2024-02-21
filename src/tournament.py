"""A tournament experiment to compare the performance of different fingerprints validation metric
is weighted F1 score."""
import json
import os
import pickle
import random

import deepchem as dc
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.metrics import (
    classification_report,
    log_loss,
    mean_squared_error,
    roc_auc_score,
)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def train_gcn(model, train_data, val_data, epochs, patience, multiclass: bool = True):
    val_scores = []
    train_scores = []
    best_val_score = float("inf")
    stop_epoch = epochs
    print("Training GCN...")
    for i in range(epochs):
        model.fit(train_data, nb_epoch=1, checkpoint_interval=0)
        train_pred = model.predict(train_data).squeeze()
        val_pred = model.predict(val_data).squeeze()
        if multiclass:
            train_scores.append(log_loss(train_data.y, train_pred, labels=[0, 1, 2, 3]))
            val_scores.append(log_loss(val_data.y, val_pred, labels=[0, 1, 2, 3]))
        else:
            train_scores.append(log_loss(train_data.y, train_pred))
            val_scores.append(log_loss(val_data.y, val_pred))

        # ----- Early Stopping -----
        if i > patience and min(val_scores[-patience:]) > best_val_score:
            print(f"Early stopping after {i - patience} epochs")
            stop_epoch = i - patience
            model.restore()
            break

        if val_scores[-1] < best_val_score and i > 100:
            best_val_score = val_scores[-1]
            model.save_checkpoint()

        if i % 100 == 0:
            print(f"epoch: {i}, train_loss {train_scores[-1]}, val_loss {val_scores[-1]}")

    return model


def catboost(X, y, folds, seed, multiclass: bool = True):
    """Folds is a list of integers, each integer represents a fold."""
    score_list = []
    for i in range(len(np.unique(folds))):
        print(f"Fold {i+1}")
        # Get the indices of the training and test sets
        train_index = np.where(folds != i)[0]
        test_index = np.where(folds == i)[0]
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        if len(train_index) <= 1 or len(test_index) <= 1 or np.unique(y_test).shape[0] < 2:
            print("Not enough data for this fold")
            score_list.append(np.nan)
            continue

        # Create a CatBoostRegressor model
        if multiclass:
            model = CatBoostClassifier(
                iterations=5000,
                learning_rate=0.005,
                depth=4,
                random_seed=seed,
                classes_count=4,
                verbose=False,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test, prediction_type="Class")
            report = classification_report(y_test, preds, output_dict=True, zero_division=0.0)
            score = report["accuracy"]
        else:
            model = CatBoostClassifier(
                iterations=5000,
                learning_rate=0.005,
                depth=4,
                random_seed=seed,
                verbose=False,
            )
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, probs, labels=[0, 1])
        print(f"score: {score:.3f}")
        score_list.append(score)

    # compute average score excluding NaNs
    print(f"Average Score:  {np.nanmean(score_list):.3f}")

    return score_list


def gcn_cv(dataset, folds, hyper_params, seed, epochs, patience, multiclass: bool = True):
    """Do a similar thing to the catboost function, but with a GCN model."""
    score_list = []
    for i in range(len(np.unique(folds))):
        print(f"Fold {i+1}")
        # --- split the data into train, test, and validation sets --- #
        train_index = np.where(folds != i)[0]
        val_index = np.random.choice(train_index, size=int(len(train_index) * 0.1), replace=False)
        train_index = np.setdiff1d(train_index, val_index)
        test_index = np.where(folds == i)[0]
        train_dataset = dataset.select(train_index)
        test_dataset = dataset.select(test_index)
        val_dataset = dataset.select(val_index)
        if (
            len(train_index) <= 1
            or len(val_index) <= 1
            or len(test_index) <= 1
            or np.unique(test_dataset.y).shape[0] < 2
        ):
            print("Not enough data for this fold")
            score_list.append(np.nan)
            continue

        # --- Train the GCN --- #
        if multiclass:
            model = dc.models.GraphConvModel(
                n_tasks=1, mode="classification", n_classes=4, **hyper_params
            )
        else:
            model = dc.models.GraphConvModel(n_tasks=1, mode="classification", **hyper_params)
        model = train_gcn(model, train_dataset, val_dataset, epochs, patience, multiclass)

        # --- Get Embeddings from GCN --- #
        train_embedding = model.predict_embedding(train_dataset)[: len(train_index)]
        val_embedding = model.predict_embedding(val_dataset)[: len(val_index)]
        test_embedding = model.predict_embedding(test_dataset)[: len(test_index)]

        # -- Train a CatBoost model on the embeddings --- #
        if multiclass:
            model = CatBoostClassifier(
                iterations=5000,
                learning_rate=0.005,
                depth=4,
                random_seed=seed,
                classes_count=4,
                verbose=False,
            )
            train_embedding = np.concatenate([train_embedding, val_embedding])
            train_y = np.concatenate([train_dataset.y, val_dataset.y])
            model.fit(train_embedding, train_y, verbose=False)
            preds = model.predict(test_embedding, prediction_type="Class")
            report = classification_report(
                test_dataset.y, preds, output_dict=True, zero_division=0.0
            )
            test_score = report["accuracy"]
        else:
            model = CatBoostClassifier(
                iterations=5000,
                learning_rate=0.005,
                depth=4,
                random_seed=seed,
                verbose=False,
            )
            model.fit(
                train_embedding,
                train_dataset.y,
                eval_set=(val_embedding, val_dataset.y),
                verbose=False,
                use_best_model=True,
            )
            test_probs = model.predict_proba(test_embedding)[:, 1]
            test_score = roc_auc_score(test_dataset.y, test_probs, labels=[0, 1])

        print(f"score: {test_score:.3f}")
        score_list.append(test_score)

    print(f"Average Score:  {np.nanmean(score_list):.3f}")
    return score_list


@hydra.main(config_path="../", config_name="config")
def main(cfg: DictConfig):
    # --- Initialize Logging and Random Seeds --- #
    base_path = get_original_cwd()
    np.random.seed(cfg.tournament.train.seed)
    random.seed(cfg.tournament.train.seed)
    metrics = {}

    # Read in the data containing transfection efficiency targets
    df = pd.read_csv(os.path.join(base_path, cfg.tournament.data_path))
    smiles = np.array(df["m1"])
    if cfg.tournament.train.multiclass:
        y = np.array(df["y1"])
    else:
        y = np.array(df["y2"])
    folds = np.array(df["family"])

    # --- Initialize GCN --- #
    hyperparams = {
        "batch_size": cfg.tournament.train.hyperparameters.batch_size,
        "dense_layer_size": cfg.tournament.train.hyperparameters.dense_width,
        "graph_conv_layers": [cfg.tournament.train.hyperparameters.conv_depth]
        * cfg.tournament.train.hyperparameters.conv_depth,
        "dropout": [cfg.tournament.train.hyperparameters.conv_dropout]
        * cfg.tournament.train.hyperparameters.conv_depth
        + [cfg.tournament.train.hyperparameters.dense_dropout],
        "learning_rate": cfg.tournament.train.hyperparameters.learning_rate,
    }
    print("GCN")
    featurizer = dc.feat.ConvMolFeaturizer()
    fp = featurizer.featurize(smiles)
    dataset = dc.data.NumpyDataset(fp, y)
    metrics["GCN"] = gcn_cv(
        dataset,
        folds,
        hyperparams,
        cfg.tournament.train.seed,
        cfg.tournament.train.hyperparameters.epochs,
        cfg.tournament.train.hyperparameters.patience,
        cfg.tournament.train.multiclass,
    )

    # --- CatBoost with other embeddings --- #
    for fp_file in cfg.tournament.fingerprints:
        fp_name = fp_file.split("_")[-1].split(".")[0]
        print(f"CatBoost with {fp_name}")
        file_path = os.path.join(base_path, fp_file)
        with open(file_path) as f:
            fp_dict = json.load(f)

        fp = np.array([fp_dict[mol] for mol in smiles])
        metrics[fp_name] = catboost(
            fp, y, folds, cfg.tournament.train.seed, cfg.tournament.train.multiclass
        )

    # Make a line plot of the scores for each method, each method is a different colored line
    plt.figure(figsize=(10, 10))
    for name, metric in metrics.items():
        plt.plot(metric, label=name)

    plt.legend()
    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.title("CatBoost AUC for TE Classification")
    # display x axis as integers
    n_folds = len(np.unique(folds))
    plt.xticks(np.arange(0, n_folds, 1))

    output_dir = os.path.join(get_original_cwd(), cfg.tournament.output_dir)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    plt.savefig(os.path.join(output_dir, "auc.png"), dpi=300)

    for name, metric in metrics.items():
        print(f"{name}: {np.nanmean(metric):.3f} +- {np.nanstd(metric):.3f}")

    # save the metrics in a pickle file
    with open(os.path.join(output_dir, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
