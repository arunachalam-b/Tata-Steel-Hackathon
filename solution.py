from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


DATA_DIR = Path("datasets")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"
SUBMISSION_PATH = Path("expected_submission.csv")
HIGH_PRECISION_SUBMISSION_PATH = Path("expected_submission_high_precision.csv")

TARGET_COL = "Y"
ID_COL = "CoilID"

N_SPLITS = 10
N_SEEDS = 5
BASE_SEED = 42

RECALL_SAFETY_MARGIN = 0.90


@dataclass
class TrainedEnsemble:
    feature_columns: list[str]
    oof_probabilities: np.ndarray
    oof_targets: np.ndarray
    test_probabilities: np.ndarray


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train = train.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    train[TARGET_COL] = train[TARGET_COL].astype(int)
    return train, test


STAGE_GROUPS = {
    "stage_geom": ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9"],
    "stage_temp": ["X22", "X23", "X24", "X25", "X26", "X27", "X28", "X29", "X30", "X31", "X32", "X33"],
    "stage_count": ["X34", "X35", "X36", "X37", "X38"],
    "stage_ratio": ["X41", "X42", "X43", "X44", "X45", "X46", "X47", "X48", "X49"],
}

LOG_SCALE_COLS = ["X34", "X35", "X36", "X37", "X38", "X42", "X46", "X48"]
SIGN_COLS = ["X45"]
KEY_DISCRIMINATORS = ["X45", "X36", "X38", "X37", "X42", "X34", "X35", "X13", "X10", "X15"]


def _aggregate_block(name: str, block: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"{name}_mean": block.mean(axis=1),
            f"{name}_std": block.std(axis=1),
            f"{name}_max": block.max(axis=1),
            f"{name}_min": block.min(axis=1),
            f"{name}_range": block.max(axis=1) - block.min(axis=1),
            f"{name}_zero_frac": (block == 0).mean(axis=1),
            f"{name}_nan_frac": block.isna().mean(axis=1),
        }
    )


def _pairwise_deltas(block: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = list(block.columns)
    deltas = {}
    for left, right in zip(cols[:-1], cols[1:]):
        deltas[f"{prefix}_{left}_minus_{right}"] = block[left] - block[right]
    return pd.DataFrame(deltas)


def _log_transforms(block: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({f"log1p_{c}": np.log1p(block[c].clip(lower=0)) for c in block.columns})


def _stage_activity(combined: pd.DataFrame) -> pd.DataFrame:
    count_cols = [c for c in STAGE_GROUPS["stage_count"] if c in combined.columns]
    activity = (combined[count_cols] > 0).astype(np.int8).add_suffix("_active")
    activity["active_stage_count"] = activity.sum(axis=1).astype(np.int16)
    return activity


def _sign_indicators(combined: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in SIGN_COLS if c in combined.columns]
    return pd.DataFrame({f"{c}_is_negative": (combined[c] < 0).astype(np.int8) for c in available})


def _key_interactions(combined: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in KEY_DISCRIMINATORS if c in combined.columns]
    interactions: dict[str, pd.Series] = {}
    for i, left in enumerate(available):
        for right in available[i + 1 :]:
            denom = combined[right].replace(0, np.nan).abs() + 1e-6
            interactions[f"{left}_over_{right}"] = combined[left] / denom
            interactions[f"{left}_times_{right}"] = combined[left] * combined[right]
    return pd.DataFrame(interactions)


def build_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    raw_features = [c for c in train.columns if c not in (ID_COL, TARGET_COL)]
    combined = pd.concat([train[raw_features], test[raw_features]], axis=0, ignore_index=True)

    new_blocks: list[pd.DataFrame] = [combined]

    missing_indicator_cols = [c for c in raw_features if combined[c].isna().any()]
    if missing_indicator_cols:
        new_blocks.append(combined[missing_indicator_cols].isna().astype(np.int8).add_suffix("_isna"))

    log_cols_available = [c for c in LOG_SCALE_COLS if c in combined.columns]
    if log_cols_available:
        new_blocks.append(_log_transforms(combined[log_cols_available]))

    for name, cols in STAGE_GROUPS.items():
        available = [c for c in cols if c in combined.columns]
        block = combined[available]
        new_blocks.append(_aggregate_block(name, block))
        if name == "stage_temp":
            new_blocks.append(_pairwise_deltas(block, "temp_delta"))

    new_blocks.append(_stage_activity(combined))
    new_blocks.append(_sign_indicators(combined))
    new_blocks.append(_key_interactions(combined))

    combined = pd.concat(new_blocks, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    enriched_features = [c for c in combined.columns]

    train_features = combined.iloc[: len(train)].reset_index(drop=True)
    test_features = combined.iloc[len(train) :].reset_index(drop=True)
    return train_features, test_features, enriched_features


def make_lgbm(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=15,
        max_depth=-1,
        min_child_samples=5,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.2,
        objective="binary",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def make_xgb(seed: int, scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=4,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.2,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=200,
        verbosity=0,
    )


def make_catboost(seed: int, scale_pos_weight: float) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=3000,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=3.0,
        random_seed=seed,
        loss_function="Logloss",
        eval_metric="PRAUC",
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=200,
        verbose=False,
        allow_writing_files=False,
    )


def fit_lgbm_fold(model: LGBMClassifier, x_train, y_train, x_valid, y_valid) -> LGBMClassifier:
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="average_precision",
        callbacks=[early_stopping(stopping_rounds=200, verbose=False), log_evaluation(period=0)],
    )
    return model


def fit_xgb_fold(model: XGBClassifier, x_train, y_train, x_valid, y_valid) -> XGBClassifier:
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    return model


def fit_catboost_fold(model: CatBoostClassifier, x_train, y_train, x_valid, y_valid) -> CatBoostClassifier:
    model.fit(x_train, y_train, eval_set=(x_valid, y_valid), use_best_model=True, verbose=False)
    return model


def train_with_cv(
    train_features: pd.DataFrame,
    targets: np.ndarray,
    test_features: pd.DataFrame,
    feature_columns: list[str],
) -> TrainedEnsemble:
    n_train = len(train_features)
    n_test = len(test_features)
    oof_probabilities = np.zeros(n_train, dtype=np.float64)
    test_probabilities = np.zeros(n_test, dtype=np.float64)

    pos_weight = float((targets == 0).sum()) / max(float((targets == 1).sum()), 1.0)

    train_matrix = train_features[feature_columns]
    test_matrix = test_features[feature_columns]

    total_runs = 0
    for seed_offset in range(N_SEEDS):
        seed = BASE_SEED + seed_offset
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for train_idx, valid_idx in splitter.split(train_matrix, targets):
            x_train = train_matrix.iloc[train_idx]
            x_valid = train_matrix.iloc[valid_idx]
            y_train = targets[train_idx]
            y_valid = targets[valid_idx]

            x_train_filled = x_train.fillna(-9999.0)
            x_valid_filled = x_valid.fillna(-9999.0)
            test_filled = test_matrix.fillna(-9999.0)

            lgbm = fit_lgbm_fold(make_lgbm(seed), x_train, y_train, x_valid, y_valid)
            xgb = fit_xgb_fold(make_xgb(seed, pos_weight), x_train, y_train, x_valid, y_valid)
            cat = fit_catboost_fold(
                make_catboost(seed, pos_weight), x_train_filled, y_train, x_valid_filled, y_valid
            )

            valid_prob = (
                lgbm.predict_proba(x_valid)[:, 1]
                + xgb.predict_proba(x_valid)[:, 1]
                + cat.predict_proba(x_valid_filled)[:, 1]
            ) / 3.0
            test_prob = (
                lgbm.predict_proba(test_matrix)[:, 1]
                + xgb.predict_proba(test_matrix)[:, 1]
                + cat.predict_proba(test_filled)[:, 1]
            ) / 3.0

            oof_probabilities[valid_idx] += valid_prob
            test_probabilities += test_prob
            total_runs += 1

    oof_probabilities /= N_SEEDS
    test_probabilities /= total_runs

    return TrainedEnsemble(
        feature_columns=feature_columns,
        oof_probabilities=oof_probabilities,
        oof_targets=targets,
        test_probabilities=test_probabilities,
    )


def recall_one_threshold(oof_probabilities: np.ndarray, oof_targets: np.ndarray) -> float:
    positive_probs = oof_probabilities[oof_targets == 1]
    if len(positive_probs) == 0:
        return 0.5
    return float(positive_probs.min()) * RECALL_SAFETY_MARGIN


def best_f1_threshold(oof_probabilities: np.ndarray, oof_targets: np.ndarray) -> float:
    candidates = np.unique(np.round(oof_probabilities, 4))
    best_threshold = 0.5
    best_score = -1.0
    for candidate in candidates:
        predictions = (oof_probabilities >= candidate).astype(int)
        score = f1_score(oof_targets, predictions, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(candidate)
    return best_threshold


def evaluate_threshold(oof_probabilities: np.ndarray, oof_targets: np.ndarray, threshold: float) -> dict:
    predictions = (oof_probabilities >= threshold).astype(int)
    return {
        "threshold": threshold,
        "recall": recall_score(oof_targets, predictions, zero_division=0),
        "precision": precision_score(oof_targets, predictions, zero_division=0),
        "f1": f1_score(oof_targets, predictions, zero_division=0),
        "predicted_positives": int(predictions.sum()),
        "actual_positives": int(oof_targets.sum()),
    }


def describe_oof(oof_probabilities: np.ndarray, oof_targets: np.ndarray) -> None:
    pos_probs = np.sort(oof_probabilities[oof_targets == 1])
    neg_probs = np.sort(oof_probabilities[oof_targets == 0])
    print(f"OOF ROC-AUC: {roc_auc_score(oof_targets, oof_probabilities):.4f}")
    print(f"OOF PR-AUC : {average_precision_score(oof_targets, oof_probabilities):.4f}")
    print(f"positive probs lowest 10: {pos_probs[:10]}")
    print(f"positive probs highest 5: {pos_probs[-5:]}")
    print(f"negative probs highest 10: {neg_probs[-10:]}")
    for k in (5, 10, 15, 20, 30):
        threshold = pos_probs[min(k - 1, len(pos_probs) - 1)]
        metrics = evaluate_threshold(oof_probabilities, oof_targets, threshold)
        print(
            f"  drop {k - 1} hardest positives -> threshold={threshold:.5f} "
            f"recall={metrics['recall']:.3f} precision={metrics['precision']:.3f} "
            f"predicted={metrics['predicted_positives']}"
        )


def write_submission(test_ids: pd.Series, test_probabilities: np.ndarray, threshold: float, path: Path) -> pd.DataFrame:
    predictions = (test_probabilities >= threshold).astype(int)
    submission = pd.DataFrame({ID_COL: test_ids.values, TARGET_COL: predictions})
    submission.to_csv(path, index=False)
    return submission


def print_metrics(label: str, metrics: dict) -> None:
    print(f"\n=== {label} ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")


def main() -> None:
    train, test = load_datasets()
    train_features, test_features, feature_columns = build_features(train, test)
    targets = train[TARGET_COL].to_numpy()

    ensemble = train_with_cv(train_features, targets, test_features, feature_columns)

    print("=== Out-of-fold diagnostics ===")
    describe_oof(ensemble.oof_probabilities, ensemble.oof_targets)

    recall_threshold = recall_one_threshold(ensemble.oof_probabilities, ensemble.oof_targets)
    f1_threshold = best_f1_threshold(ensemble.oof_probabilities, ensemble.oof_targets)

    recall_metrics = evaluate_threshold(ensemble.oof_probabilities, ensemble.oof_targets, recall_threshold)
    f1_metrics = evaluate_threshold(ensemble.oof_probabilities, ensemble.oof_targets, f1_threshold)

    print_metrics("Recall=1.0 threshold (primary submission)", recall_metrics)
    print_metrics("Best-F1 threshold (alternative submission)", f1_metrics)

    primary = write_submission(test[ID_COL], ensemble.test_probabilities, recall_threshold, SUBMISSION_PATH)
    alternative = write_submission(
        test[ID_COL], ensemble.test_probabilities, f1_threshold, HIGH_PRECISION_SUBMISSION_PATH
    )

    print(f"\nPrimary submission   -> {SUBMISSION_PATH.resolve()}  shape={primary.shape}  defects={int(primary[TARGET_COL].sum())}")
    print(f"Alternative submission -> {HIGH_PRECISION_SUBMISSION_PATH.resolve()}  shape={alternative.shape}  defects={int(alternative[TARGET_COL].sum())}")

    sample = pd.read_csv(SAMPLE_PATH)
    expected_columns = list(sample.columns)
    assert list(primary.columns) == expected_columns, f"column mismatch: {primary.columns} vs {expected_columns}"
    assert set(primary[ID_COL]) == set(test[ID_COL]), "CoilID mismatch with test set"
    print(f"\nFormat check passed: columns={expected_columns}, row_count={len(primary)}")


if __name__ == "__main__":
    main()
