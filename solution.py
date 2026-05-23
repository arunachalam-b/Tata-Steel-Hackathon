from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


DATA_DIR = Path("datasets")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"

PRIMARY_SUBMISSION = Path("expected_submission.csv")
TOPK_SUBMISSIONS = {
    10: Path("expected_submission_top10.csv"),
    15: Path("expected_submission_top15.csv"),
    17: Path("expected_submission_top17.csv"),
    20: Path("expected_submission_top20.csv"),
    25: Path("expected_submission_top25.csv"),
    35: Path("expected_submission_top35.csv"),
    50: Path("expected_submission_top50.csv"),
    75: Path("expected_submission_top75.csv"),
    100: Path("expected_submission_top100.csv"),
}
F1_SUBMISSION = Path("expected_submission_best_f1.csv")
GAP_SUBMISSION = Path("expected_submission_natural_gap.csv")
RANKED_PREDICTIONS = Path("test_predictions_ranked.csv")

TARGET_COL = "Y"
ID_COL = "CoilID"

N_SPLITS = 5
N_SEEDS = 3
BASE_SEED = 42
RECALL_SAFETY_MARGIN = 0.90

ROLLING_WINDOWS = [3, 5, 10, 20, 50]
LAG_OFFSETS = [1, 2, 3, 5]
TARGET_WINDOWS = [5, 10, 20, 50]

STAGE_GROUPS = {
    "stage_geom": ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9"],
    "stage_temp": ["X22", "X23", "X24", "X25", "X26", "X27", "X28", "X29", "X30", "X31", "X32", "X33"],
    "stage_count": ["X34", "X35", "X36", "X37", "X38"],
    "stage_ratio": ["X41", "X42", "X43", "X44", "X45", "X46", "X47", "X48", "X49"],
}

LOG_SCALE_COLS = ["X34", "X35", "X36", "X37", "X38", "X42", "X46", "X48"]


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


def _log_transforms(block: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({f"log1p_{c}": np.log1p(block[c].clip(lower=0)) for c in block.columns})


def _temperature_deltas(block: pd.DataFrame) -> pd.DataFrame:
    cols = list(block.columns)
    return pd.DataFrame(
        {f"tempdelta_{a}_{b}": block[a] - block[b] for a, b in zip(cols[:-1], cols[1:])}
    )


def _stage_activity(combined: pd.DataFrame) -> pd.DataFrame:
    count_cols = [c for c in STAGE_GROUPS["stage_count"] if c in combined.columns]
    activity = (combined[count_cols] > 0).astype(np.int8).add_suffix("_active")
    activity["active_stage_count"] = activity.sum(axis=1).astype(np.int16)
    return activity


def _missing_indicators(combined: pd.DataFrame, raw_cols: list[str]) -> pd.DataFrame:
    interesting = [c for c in raw_cols if combined[c].isna().any() and combined[c].isna().mean() > 0.01]
    return combined[interesting].isna().astype(np.int8).add_suffix("_isna") if interesting else pd.DataFrame()


def _rolling_neighborhood_features(sorted_block: pd.DataFrame, raw_cols: list[str]) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    for window in ROLLING_WINDOWS:
        roll = sorted_block[raw_cols].rolling(window=window, center=True, min_periods=1)
        blocks.append(roll.mean().add_suffix(f"_rmean{window}"))
        blocks.append(roll.std().add_suffix(f"_rstd{window}"))
    for lag in LAG_OFFSETS:
        blocks.append(sorted_block[raw_cols].shift(lag).add_suffix(f"_lag{lag}"))
        blocks.append(sorted_block[raw_cols].shift(-lag).add_suffix(f"_fwd{lag}"))
    deviation = sorted_block[raw_cols] - sorted_block[raw_cols].rolling(window=10, center=True, min_periods=1).mean()
    blocks.append(deviation.add_suffix("_dev10"))
    return pd.concat(blocks, axis=1)


def build_static_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    raw_cols = [c for c in train.columns if c not in (ID_COL, TARGET_COL)]
    combined = pd.concat(
        [train.drop(columns=[TARGET_COL]), test], axis=0, ignore_index=True
    ).sort_values(ID_COL).reset_index(drop=True)

    blocks: list[pd.DataFrame] = [combined]
    blocks.append(combined[[ID_COL]].assign(coil_rank=np.arange(len(combined))).drop(columns=[ID_COL]))

    missing = _missing_indicators(combined, raw_cols)
    if not missing.empty:
        blocks.append(missing)

    log_cols = [c for c in LOG_SCALE_COLS if c in combined.columns]
    if log_cols:
        blocks.append(_log_transforms(combined[log_cols]))

    for name, cols in STAGE_GROUPS.items():
        available = [c for c in cols if c in combined.columns]
        block = combined[available]
        blocks.append(_aggregate_block(name, block))
        if name == "stage_temp":
            blocks.append(_temperature_deltas(block))

    blocks.append(_stage_activity(combined))
    blocks.append(_rolling_neighborhood_features(combined, raw_cols))

    features = pd.concat(blocks, axis=1)
    features = features.loc[:, ~features.columns.duplicated()]
    feature_columns = [c for c in features.columns if c != ID_COL]
    return features, feature_columns


def compute_target_encoding(
    all_coil_ids_sorted: np.ndarray,
    fold_train_ids: np.ndarray,
    fold_train_targets: np.ndarray,
) -> pd.DataFrame:
    coil_to_target = dict(zip(fold_train_ids, fold_train_targets))
    target_series = pd.Series(
        [coil_to_target.get(coil, np.nan) for coil in all_coil_ids_sorted],
        dtype=float,
    )
    is_train_mask = target_series.notna().astype(int)
    filled_targets = target_series.fillna(0.0)

    encodings = {ID_COL: all_coil_ids_sorted}
    for window in TARGET_WINDOWS:
        rolling_sum = filled_targets.rolling(window=window, center=True, min_periods=1).sum()
        train_counts = is_train_mask.rolling(window=window, center=True, min_periods=1).sum()
        loo_sum = rolling_sum - filled_targets
        loo_count = (train_counts - is_train_mask).replace(0, np.nan)
        encodings[f"te_neighbor_{window}"] = (loo_sum / loo_count).to_numpy()
    return pd.DataFrame(encodings)


def make_lgbm(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=600,
        learning_rate=0.04,
        num_leaves=15,
        max_depth=-1,
        min_child_samples=5,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.6,
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
        n_estimators=600,
        learning_rate=0.04,
        max_depth=4,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda=0.2,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )


def make_catboost(seed: int, scale_pos_weight: float) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=600,
        learning_rate=0.05,
        depth=5,
        l2_leaf_reg=3.0,
        random_seed=seed,
        loss_function="Logloss",
        scale_pos_weight=scale_pos_weight,
        verbose=False,
        allow_writing_files=False,
    )


def predict_fold_ensemble(
    seed: int,
    pos_weight: float,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train_filled = x_train.fillna(-9999.0)
    x_valid_filled = x_valid.fillna(-9999.0)
    x_test_filled = x_test.fillna(-9999.0)

    lgbm = make_lgbm(seed).fit(x_train, y_train)
    xgb = make_xgb(seed, pos_weight).fit(x_train, y_train)
    cat = make_catboost(seed, pos_weight).fit(x_train_filled, y_train)

    valid_per_model = np.column_stack(
        [
            lgbm.predict_proba(x_valid)[:, 1],
            xgb.predict_proba(x_valid)[:, 1],
            cat.predict_proba(x_valid_filled)[:, 1],
        ]
    )
    test_per_model = np.column_stack(
        [
            lgbm.predict_proba(x_test)[:, 1],
            xgb.predict_proba(x_test)[:, 1],
            cat.predict_proba(x_test_filled)[:, 1],
        ]
    )
    valid_mean = valid_per_model.mean(axis=1)
    test_mean = test_per_model.mean(axis=1)
    return valid_mean, test_mean, test_per_model


def _select_by_coil(static_features: pd.DataFrame, coil_ids: np.ndarray) -> pd.DataFrame:
    indexed = static_features.set_index(ID_COL)
    return indexed.loc[coil_ids].reset_index()


def _attach_target_encoding(rows: pd.DataFrame, te_table: pd.DataFrame) -> pd.DataFrame:
    return rows.merge(te_table, on=ID_COL, how="left")


def train_with_cv(
    static_features: pd.DataFrame,
    static_feature_columns: list[str],
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    targets: np.ndarray,
) -> TrainedEnsemble:
    n_train = len(train_ids)
    n_test = len(test_ids)
    n_base_models = 3

    oof_per_model = np.zeros((n_train, n_base_models), dtype=np.float64)
    test_per_model = np.zeros((n_test, n_base_models), dtype=np.float64)

    all_coil_ids_sorted = static_features.sort_values(ID_COL)[ID_COL].to_numpy()
    te_table = compute_target_encoding(all_coil_ids_sorted, train_ids, targets)
    enriched = static_features.merge(te_table, on=ID_COL, how="left")
    train_table = _select_by_coil(enriched, train_ids)
    test_table = _select_by_coil(enriched, test_ids)

    te_columns = [f"te_neighbor_{w}" for w in TARGET_WINDOWS]
    feature_columns = static_feature_columns + te_columns
    pos_weight = float((targets == 0).sum()) / max(float((targets == 1).sum()), 1.0)

    x_test = test_table[feature_columns]
    total_test_runs = 0

    for seed_offset in range(N_SEEDS):
        seed = BASE_SEED + seed_offset
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        seed_oof_per_model = np.zeros_like(oof_per_model)
        for train_idx, valid_idx in splitter.split(train_table, targets):
            x_train = train_table.iloc[train_idx][feature_columns]
            x_valid = train_table.iloc[valid_idx][feature_columns]
            x_train_filled = x_train.fillna(-9999.0)
            x_valid_filled = x_valid.fillna(-9999.0)
            x_test_filled = x_test.fillna(-9999.0)

            lgbm = make_lgbm(seed).fit(x_train, targets[train_idx])
            xgb = make_xgb(seed, pos_weight).fit(x_train, targets[train_idx])
            cat = make_catboost(seed, pos_weight).fit(x_train_filled, targets[train_idx])

            seed_oof_per_model[valid_idx, 0] = lgbm.predict_proba(x_valid)[:, 1]
            seed_oof_per_model[valid_idx, 1] = xgb.predict_proba(x_valid)[:, 1]
            seed_oof_per_model[valid_idx, 2] = cat.predict_proba(x_valid_filled)[:, 1]

            test_per_model[:, 0] += lgbm.predict_proba(x_test)[:, 1]
            test_per_model[:, 1] += xgb.predict_proba(x_test)[:, 1]
            test_per_model[:, 2] += cat.predict_proba(x_test_filled)[:, 1]
            total_test_runs += 1
        oof_per_model += seed_oof_per_model

    oof_per_model /= N_SEEDS
    test_per_model /= total_test_runs

    oof_probabilities = oof_per_model.mean(axis=1)
    test_probabilities = test_per_model.mean(axis=1)

    meta = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    meta.fit(oof_per_model, targets)
    oof_meta = meta.predict_proba(oof_per_model)[:, 1]
    test_meta = meta.predict_proba(test_per_model)[:, 1]

    if roc_auc_score(targets, oof_meta) > roc_auc_score(targets, oof_probabilities):
        print(f"[stacker] meta-AUC={roc_auc_score(targets, oof_meta):.4f} beats mean-AUC={roc_auc_score(targets, oof_probabilities):.4f}")
        oof_probabilities = oof_meta
        test_probabilities = test_meta
    else:
        print(f"[stacker] keeping mean (mean-AUC={roc_auc_score(targets, oof_probabilities):.4f} >= meta-AUC={roc_auc_score(targets, oof_meta):.4f})")

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


def write_submission_from_predictions(test_ids: np.ndarray, predictions: np.ndarray, path: Path) -> pd.DataFrame:
    submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: predictions.astype(int)})
    submission.to_csv(path, index=False)
    return submission


def write_topk_submission(test_ids: np.ndarray, test_probabilities: np.ndarray, k: int, path: Path) -> pd.DataFrame:
    k = min(k, len(test_probabilities))
    threshold = np.sort(test_probabilities)[-k]
    predictions = (test_probabilities >= threshold).astype(int)
    if predictions.sum() != k:
        top_idx = np.argsort(test_probabilities)[-k:]
        predictions = np.zeros_like(predictions)
        predictions[top_idx] = 1
    return write_submission_from_predictions(test_ids, predictions, path)


def write_threshold_submission(test_ids: np.ndarray, test_probabilities: np.ndarray, threshold: float, path: Path) -> pd.DataFrame:
    return write_submission_from_predictions(test_ids, (test_probabilities >= threshold).astype(int), path)


def write_ranked_predictions(test_ids: np.ndarray, test_probabilities: np.ndarray, path: Path) -> pd.DataFrame:
    ranked = pd.DataFrame({ID_COL: test_ids, "probability": test_probabilities})
    ranked = ranked.sort_values("probability", ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked.to_csv(path, index=False)
    return ranked


def find_largest_gap_cutoff(ranked: pd.DataFrame, search_top_n: int = 80) -> tuple[int, float]:
    top_probs = ranked["probability"].to_numpy()[:search_top_n]
    gaps = top_probs[:-1] - top_probs[1:]
    largest_gap_index = int(np.argmax(gaps))
    cut_rank = largest_gap_index + 1
    cut_prob = float(top_probs[largest_gap_index])
    return cut_rank, cut_prob


def write_gap_submission(ranked: pd.DataFrame, test_ids: np.ndarray, test_probabilities: np.ndarray, path: Path) -> tuple[pd.DataFrame, int, float]:
    cut_rank, cut_prob = find_largest_gap_cutoff(ranked)
    submission = write_topk_submission(test_ids, test_probabilities, cut_rank, path)
    return submission, cut_rank, cut_prob


def print_metrics(label: str, metrics: dict) -> None:
    print(f"\n=== {label} ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")


def main() -> None:
    train, test = load_datasets()
    static_features, static_feature_columns = build_static_features(train, test)

    train_ids = train[ID_COL].to_numpy()
    test_ids = test[ID_COL].to_numpy()
    targets = train[TARGET_COL].to_numpy()

    ensemble = train_with_cv(
        static_features=static_features,
        static_feature_columns=static_feature_columns,
        train_ids=train_ids,
        test_ids=test_ids,
        targets=targets,
    )

    print("\n=== Out-of-fold diagnostics ===")
    describe_oof(ensemble.oof_probabilities, ensemble.oof_targets)

    recall_threshold = recall_one_threshold(ensemble.oof_probabilities, ensemble.oof_targets)
    f1_threshold = best_f1_threshold(ensemble.oof_probabilities, ensemble.oof_targets)

    print_metrics(
        "OOF metrics @ recall=1.0 threshold",
        evaluate_threshold(ensemble.oof_probabilities, ensemble.oof_targets, recall_threshold),
    )
    print_metrics(
        "OOF metrics @ best-F1 threshold",
        evaluate_threshold(ensemble.oof_probabilities, ensemble.oof_targets, f1_threshold),
    )

    test_train_ratio = (targets == 1).mean()
    expected_positives_in_test = int(round(len(test_ids) * test_train_ratio))
    print(f"\nExpected positives in test (base rate {test_train_ratio:.4f} * {len(test_ids)}): ~{expected_positives_in_test}")

    primary = write_threshold_submission(test_ids, ensemble.test_probabilities, recall_threshold, PRIMARY_SUBMISSION)
    best_f1 = write_threshold_submission(test_ids, ensemble.test_probabilities, f1_threshold, F1_SUBMISSION)
    ranked = write_ranked_predictions(test_ids, ensemble.test_probabilities, RANKED_PREDICTIONS)
    gap_sub, gap_rank, gap_prob = write_gap_submission(ranked, test_ids, ensemble.test_probabilities, GAP_SUBMISSION)

    print(f"\nPrimary (recall=1.0)  -> {PRIMARY_SUBMISSION.name:38s} defects={int(primary[TARGET_COL].sum())}")
    print(f"Best F1 threshold     -> {F1_SUBMISSION.name:38s} defects={int(best_f1[TARGET_COL].sum())}")
    print(f"Natural gap (rank {gap_rank})  -> {GAP_SUBMISSION.name:38s} defects={int(gap_sub[TARGET_COL].sum())}  prob_cutoff={gap_prob:.4f}")

    for k, path in TOPK_SUBMISSIONS.items():
        sub = write_topk_submission(test_ids, ensemble.test_probabilities, k, path)
        cutoff = ranked.iloc[k - 1]["probability"]
        print(f"Top-{k:<3d}              -> {path.name:38s} defects={int(sub[TARGET_COL].sum())}  prob_cutoff={cutoff:.4f}")

    print(f"\nRanked test predictions -> {RANKED_PREDICTIONS.name}")
    print("Top 30 most confident test predictions:")
    print(ranked.head(30).to_string(index=False))

    sample = pd.read_csv(SAMPLE_PATH)
    expected_columns = list(sample.columns)
    assert list(primary.columns) == expected_columns, f"column mismatch: {primary.columns} vs {expected_columns}"
    assert set(primary[ID_COL]) == set(test_ids), "CoilID mismatch with test set"
    print(f"\nFormat check passed: columns={expected_columns}, row_count={len(primary)}")


if __name__ == "__main__":
    main()
