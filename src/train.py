"""
Training Pipeline for PM2.5 Prediction
- Uses time-based split (no data leakage)
- Trains Linear Regression and Random Forest
- Uses lag_1 to lag_24 features
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# -----------------------
# Spark Session
# -----------------------
spark = SparkSession.builder \
    .appName("PM25 Training Model") \
    .master("local[*]") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# -----------------------
# Paths
# -----------------------
INPUT_PATH = "data/processed/features"
LR_MODEL_PATH = "models/pm25_lr_model"
RF_MODEL_PATH = "models/pm25_rf_model"
METRICS_PATH = "data/processed/model_metrics"

# -----------------------
# Load feature data
# -----------------------
print("📂 Loading feature data...")
df = spark.read.parquet(INPUT_PATH)
print(f"   Total records: {df.count()}")

# -----------------------
# Feature vector
# -----------------------
# Features: hour, day_of_week, month, lag_1..lag_24
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

# Sort by datetime and drop nulls
df_sorted = df.orderBy("datetime").dropna()

# Get total count
total_count = df_sorted.count()
print(f"   Total valid records: {total_count}")

# Calculate split point (80% for train)
split_index = int(total_count * 0.8)

# Get split datetime
all_datetimes = df_sorted.select("datetime").orderBy("datetime").collect()
split_datetime = all_datetimes[split_index]["datetime"]

# Create train/test sets
train_df = df_sorted.filter(col("datetime") < split_datetime)
test_df = df_sorted.filter(col("datetime") >= split_datetime)

train_count = train_df.count()
test_count = test_df.count()

print(f"\n📊 Train/Test Split:")
print(f"   Train: {train_count} records ({train_count/total_count*100:.1f}%)")
print(f"   Test: {test_count} records ({test_count/total_count*100:.1f}%)")
print(f"   Split datetime: {split_datetime}")

# Check for class imbalance
print(f"\n   Train date range: {train_df.select(col('datetime')).first()[0]}")
print(f"   Test date range: {test_df.select(col('datetime')).first()[0]}")


def evaluate_metrics(predictions, label_col="pm25"):
    """Calculate RMSE, MAE, R2"""
    metrics = {}
    for metric in ["rmse", "mae", "r2"]:
        evaluator = RegressionEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName=metric
        )
        metrics[metric] = evaluator.evaluate(predictions)
    return metrics


# -----------------------
# Train Linear Regression
# -----------------------
print("\n🚀 Training Linear Regression...")
lr = LinearRegression(
    featuresCol="features",
    labelCol="pm25",
    regParam=0.1  # Add regularization to prevent overfitting
)

lr_model = lr.fit(train_df)
lr_predictions = lr_model.transform(test_df)
lr_metrics = evaluate_metrics(lr_predictions)

print(f"   RMSE: {lr_metrics['rmse']:.2f}")
print(f"   MAE: {lr_metrics['mae']:.2f}")
print(f"   R²: {lr_metrics['r2']:.4f}")

# Save Linear Regression model
print("\n💾 Saving Linear Regression model...")
lr_model.write().overwrite().save(LR_MODEL_PATH)
print(f"   ✅ Linear Regression: {LR_MODEL_PATH}")

# -----------------------
# Train Random Forest Regressor (SKIPPED - Out of Memory)
# -----------------------
# print("\n🌲 Training Random Forest...")
# rf = RandomForestRegressor(
#     featuresCol="features",
#     labelCol="pm25",
#     numTrees=100,
#     maxDepth=10,
#     seed=42
# )
# rf_model = rf.fit(train_df)
# rf_predictions = rf_model.transform(test_df)
# rf_metrics = evaluate_metrics(rf_predictions)
# print(f"   RMSE: {rf_metrics['rmse']:.2f}")
# print(f"   MAE: {rf_metrics['mae']:.2f}")
# print(f"   R²: {rf_metrics['r2']:.4f}")

# -----------------------
# Results Summary
# -----------------------
print("\n" + "=" * 60)
print("📊 MODEL RESULTS")
print("=" * 60)
print(f"{'Model':<25} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
print("-" * 60)
print(f"{'Linear Regression':<25} {lr_metrics['rmse']:>10.2f} {lr_metrics['mae']:>10.2f} {lr_metrics['r2']:>10.4f}")
print("=" * 60)
print(f"🏆 Best Model: Linear Regression")
print("\n💾 Saving models...")
lr_model.write().overwrite().save(LR_MODEL_PATH)
print(f"   Linear Regression: {LR_MODEL_PATH}")

# Save metrics
metrics_rows = [
    ("linear_regression", lr_metrics["rmse"], lr_metrics["mae"], lr_metrics["r2"])
]

metrics_df = spark.createDataFrame(
    metrics_rows,
    ["model", "rmse", "mae", "r2"]
)

metrics_df.write.mode("overwrite").parquet(METRICS_PATH)
print(f"   Metrics: {METRICS_PATH}")

print("\n✅ TRAINING COMPLETE! (Random Forest skipped due to memory constraints)")
spark.stop()
