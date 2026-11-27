import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ml_pipeline import (
    SENTIMENT_COLUMNS,
    TIMING_COLUMNS,
    WEATHER_COLUMNS,
    load_model_bundle,
    load_training_data,
    predict_submission,
    train_pipeline,
)


class MlPipelineTests(unittest.TestCase):
    def test_load_training_data_maps_category_and_multilabel_targets(self):
        row = {"tweet": "Sunny and cold"}
        row.update({column: 0.0 for column in SENTIMENT_COLUMNS})
        row.update({column: 0.0 for column in TIMING_COLUMNS})
        row.update({column: 0.0 for column in WEATHER_COLUMNS})
        row["s3"] = 0.8
        row["w2"] = 1.0
        row["k2"] = 0.5
        frame = pd.DataFrame([row])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            frame.to_csv(path, index=False)
            texts, targets = load_training_data(path)

        self.assertEqual(texts, ["Sunny and cold"])
        self.assertEqual(targets["sentiment"].tolist(), [2])
        self.assertEqual(targets["timing"].tolist(), [1])
        self.assertEqual(targets["weather"][0, 1], 1)
        self.assertEqual(int(targets["weather"].sum()), 1)

    def test_train_and_predict_round_trip(self):
        texts = [f"weather tweet token {index % 5} {'rain' if index % 2 else 'sun'}" for index in range(50)]
        targets = {
            "sentiment": np.asarray([index % 5 for index in range(50)]),
            "timing": np.asarray([index % 4 for index in range(50)]),
            "weather": np.asarray(
                [
                    [int((index + label) % 4 == 0) for label in range(len(WEATHER_COLUMNS))]
                    for index in range(50)
                ],
                dtype=np.uint8,
            ),
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            model_path = directory_path / "model.joblib"
            train_pipeline(texts, targets, model_path, test_size=0.2, random_state=42)

            test_path = directory_path / "test.csv"
            pd.DataFrame(
                {"id": [101, 102], "tweet": ["sunny weather", "heavy rain"]}
            ).to_csv(test_path, index=False)
            sample_path = directory_path / "sample.csv"
            pd.DataFrame(columns=["id", *SENTIMENT_COLUMNS, *TIMING_COLUMNS, *WEATHER_COLUMNS]).to_csv(
                sample_path, index=False
            )
            output_path = directory_path / "submission.csv"
            bundle = load_model_bundle(model_path)
            predict_submission(bundle, test_path, output_path, sample_path)
            result = pd.read_csv(output_path)

        self.assertEqual(result.shape, (2, 25))
        self.assertEqual(result["id"].tolist(), [101, 102])
        self.assertEqual(result[SENTIMENT_COLUMNS].sum(axis=1).tolist(), [1, 1])
        self.assertEqual(result[TIMING_COLUMNS].sum(axis=1).tolist(), [1, 1])


if __name__ == "__main__":
    unittest.main()
