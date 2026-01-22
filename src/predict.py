import os
from pyspark.sql import SparkSession
from pyspark.ml.regression import (
    LinearRegressionModel,
    RandomForestRegressionModel
)
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col

spark = (
    SparkSession.builder
    .appName("pm25-predict")
    .master("local[1]")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

MODEL_TYPE = os.getenv("MODEL_TYPE", "lr").lower()

df = spark.read.parquet("data/processed/features")
print("Feature data loaded")
df.show(5)

assembler = VectorAssembler(
    inputCols=["hour", "day_of_week", "month"] + [f"lag_{i}" for i in range(1, 25)],
    outputCol="features"
)

df_features = assembler.transform(df)

if MODEL_TYPE in {"lr", "linear", "linear_regression"}:
    model_path = "models/pm25_lr_model"
    output_path = "data/processed/predictions"
    model = LinearRegressionModel.load(model_path)
    model_name = "Linear Regression"
elif MODEL_TYPE in {"rf", "random_forest", "randomforest"}:
    model_path = "models/pm25_rf_model"
    output_path = "data/processed/predictions_rf"
    model = RandomForestRegressionModel.load(model_path)
    model_name = "Random Forest"
else:
    raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")

print(f"Model loaded successfully: {model_name} ({model_path})")

predictions = model.transform(df_features)

result = predictions.select(
    "datetime",
    col("pm25").alias("actual_pm25"),
    col("prediction").alias("predicted_pm25")
).withColumnRenamed("datetime", "date")

result.show(10)

result.write.mode("overwrite").parquet(output_path)

spark.stop()
print("STEP 6 PREDICT DONE")
