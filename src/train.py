"""
Training Pipeline for PM2.5 Prediction
- Uses time-based split (no data leakage)
- Trains Linear Regression, Random Forest, GBT
- Uses lag_1 to lag_24 features
"""
import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# -----------------------
# Spark Session
# -----------------------
spark = (SparkSession.builder
    .appName("PM25 Training Model")
    .master("local[*]")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# -----------------------
# Paths
# -----------------------
INPUT_PATH = "data/processed/features"
LR_MODEL_PATH = "models/pm25_lr_model"
RF_MODEL_PATH = "models/pm25_rf_model"
GBT_MODEL_PATH = "models/pm25_gbt_model"
METRICS_PATH = "data/processed/model_metrics"
PRED_LR_PATH = "data/processed/predictions"
PRED_RF_PATH = "data/processed/predictions_rf"
PRED_GBT_PATH = "data/processed/predictions_gbt"

RUN_RF = os.getenv("RUN_RF", "1") != "0"
RUN_GBT = os.getenv("RUN_GBT", "1") != "0"


TREE_SAMPLE_FRAC = float(os.getenv("TREE_SAMPLE_FRAC", "0.3"))
RF_NUM_TREES = int(os.getenv("RF_NUM_TREES", "30"))
RF_MAX_DEPTH = int(os.getenv("RF_MAX_DEPTH", "6"))
RF_MAX_BINS = int(os.getenv("RF_MAX_BINS", "32"))
GBT_MAX_ITER = int(os.getenv("GBT_MAX_ITER", "50"))
GBT_MAX_DEPTH = int(os.getenv("GBT_MAX_DEPTH", "5"))

# -----------------------
# Load feature data
# -----------------------
print("📂 Loading feature data...")
df = spark.read.parquet(INPUT_PATH)
print(f"   Total records: {df.count()}")

# -----------------------
# Feature vector
# -----------------------
feature_cols = ["hour", "day_of_week", "month"] + [f"lag_{i}" for i in range(1, 25)]

print(f"\n🔧 Creating feature vector with {len(feature_cols)} features:")
print(f"   {feature_cols}")

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

df = assembler.transform(df)

# -----------------------
# Train / Test split (Time-Series based)
# -----------------------
print("\n⏰ Time-based split (no data leakage)...")

df_sorted = df.orderBy("datetime").dropna()

total_count = df_sorted.count()
print(f"   Total valid records: {total_count}")

split_index = int(total_count * 0.8)

all_datetimes = df_sorted.select("datetime").orderBy("datetime").collect()
split_datetime = all_datetimes[split_index]["datetime"]

train_df = df_sorted.filter(col("datetime") < split_datetime)
test_df = df_sorted.filter(col("datetime") >= split_datetime)

train_count = train_df.count()
test_count = test_df.count()

print("\n📊 Train/Test Split:")
print(f"   Train: {train_count} records ({train_count/total_count*100:.1f}%)")
print(f"   Test: {test_count} records ({test_count/total_count*100:.1f}%)")
print(f"   Split datetime: {split_datetime}")

print(f"\n   Train date range: {train_df.select(col('datetime')).first()[0]}")
print(f"   Test date range: {test_df.select(col('datetime')).first()[0]}")


def evaluate_metrics(predictions, label_col="pm25"):
    metrics = {}
    for metric in ["rmse", "mae", "r2"]:
        evaluator = RegressionEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName=metric
        )
        metrics[metric] = evaluator.evaluate(predictions)
    return metrics


def save_predictions(predictions, output_path):
    result = predictions.select(
        "datetime",
        col("pm25").alias("actual_pm25"),
        col("prediction").alias("predicted_pm25")
    ).withColumnRenamed("datetime", "date")
    result.write.mode("overwrite").parquet(output_path)


metrics_rows = []

# -----------------------
# Train Linear Regression
# -----------------------
print("\n🚀 Training Linear Regression...")
lr = LinearRegression(
    featuresCol="features",
    labelCol="pm25",
    regParam=0.1
)

lr_model = lr.fit(train_df)
lr_predictions = lr_model.transform(test_df)
lr_metrics = evaluate_metrics(lr_predictions)

print(f"   RMSE: {lr_metrics['rmse']:.2f}")
print(f"   MAE: {lr_metrics['mae']:.2f}")
print(f"   R²: {lr_metrics['r2']:.4f}")

print("\n💾 Saving Linear Regression model...")
lr_model.write().overwrite().save(LR_MODEL_PATH)
print(f"   ✅ Linear Regression: {LR_MODEL_PATH}")

save_predictions(lr_predictions, PRED_LR_PATH)
print(f"   ✅ Predictions: {PRED_LR_PATH}")

metrics_rows.append(("linear_regression", lr_metrics["rmse"], lr_metrics["mae"], lr_metrics["r2"]))


# -----------------------
# Prepare sampled data for tree models (to avoid OOM)
# -----------------------
tree_train_df = train_df
tree_test_df = test_df
if TREE_SAMPLE_FRAC < 1.0:
    tree_train_df = train_df.sample(withReplacement=False, fraction=TREE_SAMPLE_FRAC, seed=42)
    tree_test_df = test_df.sample(withReplacement=False, fraction=min(1.0, TREE_SAMPLE_FRAC), seed=42)
    print(f"\n🌲 Tree models using sampled data: {TREE_SAMPLE_FRAC:.2f}")

# -----------------------
# Train Random Forest Regressor
# -----------------------
if RUN_RF:
    print("\n🌲 Training Random Forest...")
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="pm25",
        numTrees=RF_NUM_TREES,
        maxDepth=RF_MAX_DEPTH,
        maxBins=RF_MAX_BINS,
        subsamplingRate=0.8,
        featureSubsetStrategy="auto",
        seed=42
    )
    rf_model = rf.fit(tree_train_df)
    rf_predictions = rf_model.transform(tree_test_df)
    rf_metrics = evaluate_metrics(rf_predictions)
    print(f"   RMSE: {rf_metrics['rmse']:.2f}")
    print(f"   MAE: {rf_metrics['mae']:.2f}")
    print(f"   R²: {rf_metrics['r2']:.4f}")

    print("\n💾 Saving Random Forest model...")
    rf_model.write().overwrite().save(RF_MODEL_PATH)
    print(f"   ✅ Random Forest: {RF_MODEL_PATH}")

    save_predictions(rf_predictions, PRED_RF_PATH)
    print(f"   ✅ Predictions: {PRED_RF_PATH}")

    metrics_rows.append(("random_forest", rf_metrics["rmse"], rf_metrics["mae"], rf_metrics["r2"]))
else:
    print("\n🌲 Random Forest skipped (RUN_RF=0)")

# -----------------------
# Train GBT Regressor
# -----------------------
if RUN_GBT:
    print("\n🌟 Training GBT Regressor...")
    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="pm25",
        maxIter=GBT_MAX_ITER,
        maxDepth=GBT_MAX_DEPTH,
        stepSize=0.05,
        subsamplingRate=0.8,
        seed=42
    )
    gbt_model = gbt.fit(tree_train_df)
    gbt_predictions = gbt_model.transform(tree_test_df)
    gbt_metrics = evaluate_metrics(gbt_predictions)
    print(f"   RMSE: {gbt_metrics['rmse']:.2f}")
    print(f"   MAE: {gbt_metrics['mae']:.2f}")
    print(f"   R²: {gbt_metrics['r2']:.4f}")

    print("\n💾 Saving GBT model...")
    gbt_model.write().overwrite().save(GBT_MODEL_PATH)
    print(f"   ✅ GBT Regressor: {GBT_MODEL_PATH}")

    save_predictions(gbt_predictions, PRED_GBT_PATH)
    print(f"   ✅ Predictions: {PRED_GBT_PATH}")

    metrics_rows.append(("gbt_regression", gbt_metrics["rmse"], gbt_metrics["mae"], gbt_metrics["r2"]))
else:
    print("\n🌟 GBT Regressor skipped (RUN_GBT=0)")

# -----------------------
# Results Summary
# -----------------------
print("\n" + "=" * 60)
print("📊 MODEL RESULTS")
print("=" * 60)
print(f"{'Model':<25} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
print("-" * 60)
for name, rmse, mae, r2 in metrics_rows:
    print(f"{name:<25} {rmse:>10.2f} {mae:>10.2f} {r2:>10.4f}")
print("=" * 60)

metrics_df = spark.createDataFrame(metrics_rows, ["model", "rmse", "mae", "r2"])
if os.path.exists(METRICS_PATH):
    try:
        spark.catalog.clearCache()
        old_df = spark.read.parquet(METRICS_PATH)
        new_models = [row[0] for row in metrics_rows]
        old_df = old_df.filter(~col("model").isin(new_models))
        metrics_df = old_df.unionByName(metrics_df)
    except Exception:
        pass

# Write to temp path then replace to avoid read/write conflict
metrics_df = metrics_df.coalesce(1)
_tmp_path = METRICS_PATH + "_tmp"
if os.path.exists(_tmp_path):
    shutil.rmtree(_tmp_path)
metrics_df.write.mode("overwrite").parquet(_tmp_path)
if os.path.exists(METRICS_PATH):
    shutil.rmtree(METRICS_PATH)
shutil.move(_tmp_path, METRICS_PATH)
print(f"   Metrics: {METRICS_PATH}")

print("\n✅ TRAINING COMPLETE!")
spark.stop()
