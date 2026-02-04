from pyspark.sql import SparkSession
from pyspark.sql.functions import avg as avg_fn, col, to_date
from pyspark.sql.window import Window

# -----------------------
# Spark Session
# -----------------------
spark = (
    SparkSession.builder
    .appName("PM25 EDA")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# -----------------------
# Paths
# -----------------------
INPUT_PATH = "data/processed/pm25_clean"
OUTPUT_DIR = "data/processed/eda"

# -----------------------
# Load clean data
# -----------------------
df = spark.read.parquet(INPUT_PATH)
df = df.withColumn("date", to_date("datetime"))
df.createOrReplaceTempView("pm25_clean")

# -----------------------
# Spark SQL: Daily average (trend over time)
# -----------------------
daily_avg_sql = """
SELECT
  date,
  AVG(pm25) AS pm25_avg,
  COUNT(*) AS n
FROM pm25_clean
GROUP BY date
ORDER BY date
"""

df_daily = spark.sql(daily_avg_sql)

# 7-day moving average for smoother trend
window_7d = Window.orderBy("date").rowsBetween(-6, 0)
df_daily_trend = df_daily.withColumn(
    "pm25_avg_7d",
    avg_fn(col("pm25_avg")).over(window_7d)
)

df_daily_trend.write.mode("overwrite").parquet(f"{OUTPUT_DIR}/daily_trend")

# -----------------------
# Spark SQL: Distribution by day of week
# -----------------------
weekday_sql = """
SELECT
  dayofweek(date) AS day_of_week,
  COUNT(*) AS n,
  AVG(pm25) AS pm25_avg,
  PERCENTILE_APPROX(pm25, 0.5) AS pm25_median,
  MIN(pm25) AS pm25_min,
  MAX(pm25) AS pm25_max
FROM pm25_clean
GROUP BY day_of_week
ORDER BY day_of_week
"""

df_weekday = spark.sql(weekday_sql)
df_weekday.write.mode("overwrite").parquet(f"{OUTPUT_DIR}/weekday_stats")

# -----------------------
# Spark SQL: Weekly trend
# -----------------------
weekly_sql = """
SELECT
  year(date) AS year,
  weekofyear(date) AS week_of_year,
  AVG(pm25) AS pm25_avg,
  COUNT(*) AS n
FROM pm25_clean
GROUP BY year, week_of_year
ORDER BY year, week_of_year
"""

df_weekly = spark.sql(weekly_sql)
df_weekly.write.mode("overwrite").parquet(f"{OUTPUT_DIR}/weekly_trend")

print("EDA DONE.")
print("Daily trend:", f"{OUTPUT_DIR}/daily_trend")
print("Weekday stats:", f"{OUTPUT_DIR}/weekday_stats")
print("Weekly trend:", f"{OUTPUT_DIR}/weekly_trend")

spark.stop()
