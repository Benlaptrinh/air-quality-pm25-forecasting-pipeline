"""
ETL Pipeline for PM2.5 Data
- Read raw JSON files from multiple sensors
- Normalize schema
- Aggregate to hourly
- Save to Parquet
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, to_timestamp, avg, hour, dayofweek, month,
    when, isnan, isnull, date_add
)
import os

# -----------------------
# Spark Session
# -----------------------
spark = SparkSession.builder \
    .appName("PM25_ETL") \
    .master("local[*]") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# -----------------------
# Paths
# -----------------------
RAW_DIR = "data/raw"
OUTPUT_CLEAN = "data/processed/pm25_clean"
OUTPUT_HOURLY = "data/processed/pm25_hourly"

# -----------------------
# Read all raw JSON files
# -----------------------
print("📂 Reading raw JSON files...")

# Read all sensor_*.json files
raw_files = [f for f in os.listdir(RAW_DIR) if f.startswith("sensor_") and f.endswith(".json")]

if not raw_files:
    raise ValueError(f"No sensor files found in {RAW_DIR}")

print(f"   Found {len(raw_files)} raw files")

# Read and union all files
dfs = []
for f in raw_files:
    filepath = os.path.join(RAW_DIR, f)
    # Read with multiLine option for proper JSON parsing
    df_temp = spark.read \
        .option("multiLine", "true") \
        .option("mode", "PERMISSIVE") \
        .json(filepath)
    dfs.append(df_temp)
    
df_raw = dfs[0]
for df_other in dfs[1:]:
    df_raw = df_raw.unionByName(df_other)

print(f"   Total raw records: {df_raw.count()}")

# -----------------------
# Extract and normalize columns
# -----------------------
print("\n🔧 Normalizing schema...")

# OpenAQ response structure varies, try different paths
# Handle nested structure from our ingest script
if "results" in df_raw.columns:
    df_expanded = df_raw.selectExpr("explode(results) as r", "sensor_id")
    
    df_normalized = df_expanded.select(
        col("r.value").alias("pm25"),
        to_timestamp(col("r.period.datetimeTo.utc")).alias("datetime"),
        col("r.parameter.name").alias("parameter"),
        col("r.period.label").alias("period_label"),
        col("sensor_id")
    )
else:
    # Direct structure
    df_normalized = df_raw.select(
        col("value").alias("pm25"),
        to_timestamp(col("period.datetimeTo.utc")).alias("datetime"),
        col("parameter.name").alias("parameter"),
        col("period.label").alias("period_label")
    )

# -----------------------
# Filter valid PM2.5 data
# -----------------------
print("\n🧹 Filtering valid PM2.5 data...")

df_clean = df_normalized.filter(
    (col("parameter") == "pm25") &
    (col("pm25").isNotNull()) &
    (~isnan(col("pm25"))) &
    (col("pm25") > 0) &
    (col("pm25") != -999) &
    (col("datetime").isNotNull())
).select(
    "sensor_id",
    "datetime",
    "pm25"
)

print(f"   Clean records: {df_clean.count()}")

# Save clean data
df_clean.write.mode("overwrite").parquet(OUTPUT_CLEAN)
print(f"   ✅ Saved to: {OUTPUT_CLEAN}")

# -----------------------
# Hourly Aggregation (IMPORTANT: Increases row count!)
# -----------------------
print("\n⏰ Aggregating to hourly (this increases data size)...")

# Vì data đã là hourly từ API, nên KHÔNG cần aggregate lại!
# Chỉ cần select các columns cần thiết
from pyspark.sql.functions import date_format

df_hourly = df_clean.select(
    "sensor_id",
    "datetime",
    hour("datetime").alias("hour"),
    dayofweek("datetime").alias("day_of_week"),
    month("datetime").alias("month"),
    "pm25"
)

df_hourly = df_hourly.orderBy("sensor_id", "datetime")

# Reorder columns
df_hourly = df_hourly.select(
    "sensor_id",
    "datetime",
    "hour",
    "day_of_week",
    "month",
    "pm25"
)

df_hourly = df_hourly.orderBy("sensor_id", "datetime")

print(f"   Hourly records: {df_hourly.count()}")

# Save hourly data
df_hourly.write.mode("overwrite").parquet(OUTPUT_HOURLY)
print(f"   ✅ Saved to: {OUTPUT_HOURLY}")

# -----------------------
# Summary
# -----------------------
print("\n" + "=" * 60)
print("📊 ETL SUMMARY")
print("=" * 60)
print(f"Raw files processed: {len(raw_files)}")
print(f"Clean records: {df_clean.count()}")
print(f"Hourly records: {df_hourly.count()}")
print("=" * 60)

spark.stop()
