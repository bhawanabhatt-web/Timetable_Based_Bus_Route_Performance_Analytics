"""
spark_backend.py -- loads your real best_model/ PipelineModel and serves
predictions. Shared by route_popularity_gui_spark.py (standalone) and
bus_route_dashboard.py (merged dashboard).
"""

import json
import os
from pathlib import Path

from pyspark.sql import Row, SparkSession
from pyspark.ml import PipelineModel

PROJECT_ROOT = Path(
    os.environ.get(
        "PROJECT_ROOT",
        r"D:\Fourth Semester\Big Data\Project\Timetable-Based Bus Route Performance Analytics",
    )
)
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "output"))
FEATURE_PARQUET_PATH = str(OUTPUT_DIR / "final_feature_dataset.parquet")
FEATURE_METADATA_PATH = str(OUTPUT_DIR / "feature_metadata.json")
BEST_MODEL_PATH = str(OUTPUT_DIR / "best_model")

NUMERIC_FEATURES = [
    "fare_publication_count", "hour_of_day", "is_first_stop", "is_peak_hour",
    "operator_routes", "operator_trip_count", "stop_activity",
    "stop_progress_pct", "stop_sequence", "total_stops", "unique_stops",
]
CATEGORICAL_FEATURES = [
    "fare_publication_level", "fare_status", "journey_type", "operator_size",
    "operator_workload", "route_complexity", "scheduled_time",
    "stop_busyness", "stop_position",
]


class SparkBackend:
    """Owns the SparkSession, the real PipelineModel, and lookup helpers."""

    def __init__(self, status_callback=print):
        self.status = status_callback
        self.status("Starting Spark session...")
        self.spark = (
            SparkSession.builder
            .appName("RoutePopularityGUI")
            .master("local[4]")
            .config("spark.driver.memory", "4g")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate()
        )
        self.spark.sparkContext.setLogLevel("WARN")

        self.status("Loading model (best_model/)...")
        self.model = PipelineModel.load(BEST_MODEL_PATH)

        with open(FEATURE_METADATA_PATH) as f:
            self.feature_metadata = json.load(f)
        self.target_column = self.feature_metadata["target_column"]

        # The label indexer is always stage 0 of the saved pipeline -- its
        # .labels gives the index -> class-name mapping the classifier's
        # "prediction" column refers to. Only these are REAL classes.
        label_indexer_model = self.model.stages[0]
        self.real_classes = list(label_indexer_model.labels)
        self.label_names = {i: name for i, name in enumerate(self.real_classes)}
        # Any valid class works here -- this column is required by the
        # pipeline's schema (the label indexer stage needs it to exist) but
        # is NOT used to compute the prediction itself.
        self._dummy_target_value = self.real_classes[0]

        self.status("Reading feature dataset...")
        self.schedule = self.spark.read.parquet(FEATURE_PARQUET_PATH)

        candidate_features = self.feature_metadata["candidate_feature_columns"]
        self.numeric_features = [c for c in NUMERIC_FEATURES if c in candidate_features]
        self.categorical_features = [c for c in CATEGORICAL_FEATURES if c in candidate_features]
        self.all_features = self.numeric_features + self.categorical_features

        self.status("Ready.")

    def get_route_list(self):
        id_cols = [c for c in ["line_ref", "line_name"] if c in self.schedule.columns]
        return self.schedule.select(*id_cols).distinct().orderBy("line_ref").toPandas()

    def get_categorical_options(self, column, limit=20):
        rows = (
            self.schedule.groupBy(column).count()
            .orderBy("count", ascending=False)
            .limit(limit)
            .toPandas()
        )
        return rows[column].tolist()

    def get_numeric_median(self, column):
        return self.schedule.approxQuantile(column, [0.5], 0.01)[0]

    def predict_for_route(self, line_ref):
        row_df = (
            self.schedule.filter(self.schedule.line_ref == line_ref)
            .select(*self.all_features)
            .limit(1)
        )
        row = row_df.collect()
        if not row:
            return None, None, None
        feature_values = row[0].asDict()
        pred_class, probs = self._predict(feature_values)
        return pred_class, probs, feature_values

    def predict_whatif(self, feature_values: dict):
        return self._predict(feature_values)

    def _predict(self, feature_values: dict):
        payload = dict(feature_values)
        payload[self.target_column] = self._dummy_target_value  # schema placeholder only

        for c in self.numeric_features:
            payload[c] = float(payload[c])

        row = Row(**payload)
        input_df = self.spark.createDataFrame([row])
        result = self.model.transform(input_df)
        pred_row = result.select("prediction", "probability").collect()[0]
        pred_idx = int(pred_row["prediction"])
        pred_class = self.label_names.get(pred_idx, f"Unrecognised (index {pred_idx})")
        probs_array = pred_row["probability"].toArray()

        # handleInvalid="keep" on the label StringIndexer reserves one extra
        # index beyond the real classes for "unseen" values, so the
        # probability vector can be one slot longer than self.real_classes.
        # That reserved slot has ~0 probability in practice (no training row
        # has that label) and only clutters a bar chart, so we report the
        # real classes' probabilities and fold anything else into "Other".
        probs = {name: float(probs_array[i]) for i, name in enumerate(self.real_classes)}
        leftover = sum(float(p) for i, p in enumerate(probs_array) if i >= len(self.real_classes))
        if leftover > 1e-6:
            probs["Other"] = leftover

        return pred_class, probs

    def stop(self):
        self.spark.stop()
