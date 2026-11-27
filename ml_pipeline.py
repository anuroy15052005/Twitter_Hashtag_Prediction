"""Train, evaluate, and run a multi-output tweet weather classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

SENTIMENT_COLUMNS = [f"s{i}" for i in range(1, 6)]
TIMING_COLUMNS = [f"w{i}" for i in range(1, 5)]
WEATHER_COLUMNS = [f"k{i}" for i in range(1, 16)]
TARGET_COLUMNS = SENTIMENT_COLUMNS + TIMING_COLUMNS + WEATHER_COLUMNS
MODEL_FORMAT_VERSION = 1


def build_vectorizer() -> FeatureUnion:
    """Combine word and character features for short, noisy tweet text."""
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=200_000,
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=2,
                    max_features=200_000,
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def load_training_data(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    frame = pd.read_csv(path)
    required = {"tweet", *TARGET_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"{path} contains no training rows")

    texts = frame["tweet"].fillna("").astype(str).tolist()
    targets: dict[str, np.ndarray] = {}

    for name, columns in (
        ("sentiment", SENTIMENT_COLUMNS),
        ("timing", TIMING_COLUMNS),
        ("weather", WEATHER_COLUMNS),
    ):
        try:
            values = frame[columns].apply(pd.to_numeric, errors="raise").to_numpy(
                dtype=np.float32
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"Target columns for {name} must be numeric") from error

        if not np.isfinite(values).all():
            raise ValueError(f"Target columns for {name} contain missing or infinite values")
        if np.any((values < 0) | (values > 1)):
            raise ValueError(f"Target columns for {name} must contain values from 0 to 1")

        if name in ("sentiment", "timing"):
            if np.any(values.sum(axis=1) == 0):
                raise ValueError(f"Every row must have a {name} label")
            targets[name] = values.argmax(axis=1).astype(np.int64)
        else:
            targets[name] = (values >= 0.5).astype(np.uint8)

    return texts, targets


def _new_estimators(random_state: int) -> dict[str, Any]:
    classifier_options = {
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 5000,
        "random_state": random_state,
    }
    return {
        "sentiment": LinearSVC(**classifier_options),
        "timing": LinearSVC(**classifier_options),
        "weather": OneVsRestClassifier(LinearSVC(**classifier_options), n_jobs=1),
    }


def _fit_estimators(
    features: Any, targets: dict[str, np.ndarray], random_state: int
) -> dict[str, Any]:
    estimators = _new_estimators(random_state)
    for name, estimator in estimators.items():
        estimator.fit(features, targets[name])
    return estimators


def _predict_targets(
    vectorizer: FeatureUnion, estimators: dict[str, Any], texts: list[str]
) -> dict[str, np.ndarray]:
    features = vectorizer.transform(texts)
    return {
        name: np.asarray(estimator.predict(features))
        for name, estimator in estimators.items()
    }


def _calculate_metrics(
    targets: dict[str, np.ndarray], predictions: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    metrics = {}
    for name in ("sentiment", "timing"):
        metrics[name] = {
            "accuracy": float(accuracy_score(targets[name], predictions[name])),
            "macro_f1": float(
                f1_score(targets[name], predictions[name], average="macro", zero_division=0)
            ),
            "weighted_f1": float(
                f1_score(
                    targets[name],
                    predictions[name],
                    average="weighted",
                    zero_division=0,
                )
            ),
        }

    metrics["weather"] = {
        "micro_f1": float(
            f1_score(targets["weather"], predictions["weather"], average="micro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(targets["weather"], predictions["weather"], average="macro", zero_division=0)
        ),
        "exact_match": float(
            accuracy_score(targets["weather"], predictions["weather"])
        ),
    }
    return metrics


def train_pipeline(
    texts: list[str],
    targets: dict[str, np.ndarray],
    model_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    for name in ("sentiment", "timing", "weather"):
        if len(texts) != len(targets[name]):
            raise ValueError(f"Text and {name} target row counts do not match")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1")

    indices = np.arange(len(texts))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=targets["sentiment"],
    )
    train_texts = [texts[index] for index in train_indices]
    validation_texts = [texts[index] for index in validation_indices]
    train_targets = {name: values[train_indices] for name, values in targets.items()}
    validation_targets = {
        name: values[validation_indices] for name, values in targets.items()
    }

    validation_vectorizer = build_vectorizer()
    train_features = validation_vectorizer.fit_transform(train_texts)
    validation_estimators = _fit_estimators(
        train_features, train_targets, random_state
    )
    validation_predictions = _predict_targets(
        validation_vectorizer, validation_estimators, validation_texts
    )
    metrics = _calculate_metrics(validation_targets, validation_predictions)

    final_vectorizer = build_vectorizer()
    all_features = final_vectorizer.fit_transform(texts)
    final_estimators = _fit_estimators(all_features, targets, random_state)

    bundle = {
        "format_version": MODEL_FORMAT_VERSION,
        "target_columns": {
            "sentiment": SENTIMENT_COLUMNS,
            "timing": TIMING_COLUMNS,
            "weather": WEATHER_COLUMNS,
        },
        "vectorizer": final_vectorizer,
        "estimators": final_estimators,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)
    metrics_path = model_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def load_model_bundle(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if bundle.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError(f"Unsupported model format in {path}")
    return bundle


def predict_submission(
    bundle: dict[str, Any], test_path: Path, output_path: Path, sample_path: Path | None
) -> None:
    frame = pd.read_csv(test_path)
    missing = sorted({"id", "tweet"}.difference(frame.columns))
    if missing:
        raise ValueError(f"{test_path} is missing required columns: {', '.join(missing)}")

    texts = frame["tweet"].fillna("").astype(str).tolist()
    predictions = _predict_targets(bundle["vectorizer"], bundle["estimators"], texts)
    submission = pd.DataFrame({"id": frame["id"]})

    for name, columns in (
        ("sentiment", SENTIMENT_COLUMNS),
        ("timing", TIMING_COLUMNS),
    ):
        class_predictions = predictions[name].astype(int)
        if np.any((class_predictions < 0) | (class_predictions >= len(columns))):
            raise ValueError(f"Model produced an invalid {name} class")
        for class_index, column in enumerate(columns):
            submission[column] = (class_predictions == class_index).astype(np.uint8)

    weather_predictions = predictions["weather"].astype(np.uint8)
    if weather_predictions.shape != (len(frame), len(WEATHER_COLUMNS)):
        raise ValueError("Model produced an invalid weather prediction shape")
    for class_index, column in enumerate(WEATHER_COLUMNS):
        submission[column] = weather_predictions[:, class_index]

    if sample_path is not None:
        expected_columns = pd.read_csv(sample_path, nrows=0).columns.tolist()
        if set(expected_columns) != set(submission.columns):
            raise ValueError(
                f"Submission columns do not match the sample file {sample_path}"
            )
        submission = submission[expected_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Wrote {len(submission)} predictions to {output_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and use a multi-output Twitter weather classifier."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="evaluate and train a model")
    train_parser.add_argument("--data", type=Path, default=Path("train.csv"))
    train_parser.add_argument(
        "--model", type=Path, default=Path("artifacts/weather_tfidf.joblib")
    )
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--random-state", type=int, default=42)

    predict_parser = commands.add_parser("predict", help="write a test submission")
    predict_parser.add_argument("--data", type=Path, default=Path("test.csv"))
    predict_parser.add_argument(
        "--model", type=Path, default=Path("artifacts/weather_tfidf.joblib")
    )
    predict_parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    predict_parser.add_argument(
        "--sample-submission", type=Path, default=Path("sampleSubmission.csv")
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "train":
        texts, targets = load_training_data(args.data)
        metrics = train_pipeline(
            texts,
            targets,
            args.model,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        print(json.dumps(metrics, indent=2))
        print(f"Saved final model trained on all rows to {args.model}")
        print(f"Saved validation metrics to {args.model.with_suffix('.metrics.json')}")
    else:
        bundle = load_model_bundle(args.model)
        sample_path = args.sample_submission if args.sample_submission.exists() else None
        predict_submission(bundle, args.data, args.output, sample_path)


if __name__ == "__main__":
    main()
