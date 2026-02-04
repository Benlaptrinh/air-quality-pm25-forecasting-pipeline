from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

# -----------------------
# App Config
# -----------------------
st.set_page_config(
    page_title="AQI Big Data Dashboard",
    page_icon="AQI",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

DAILY_TREND_DIR = DATA_DIR / "eda" / "daily_trend"
WEEKLY_TREND_DIR = DATA_DIR / "eda" / "weekly_trend"
WEEKDAY_STATS_DIR = DATA_DIR / "eda" / "weekday_stats"
METRICS_DIR = DATA_DIR / "model_metrics"
PRED_DIR = DATA_DIR / "predictions"
FEATURES_DIR = DATA_DIR / "features"
MODEL_DIR = BASE_DIR / "models" / "pm25_lr_model"

# -----------------------
# Helpers
# -----------------------
@st.cache_data(show_spinner=False)
def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def day_of_week_label(d: int) -> str:
    # Spark dayofweek: 1=Sunday, 2=Monday, ... 7=Saturday
    mapping = {
        1: "Sun",
        2: "Mon",
        3: "Tue",
        4: "Wed",
        5: "Thu",
        6: "Fri",
        7: "Sat",
    }
    return mapping.get(int(d), str(d))


WEEKDAY_ORDER = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

# -----------------------
# Sidebar
# -----------------------
st.sidebar.header("AQI PM2.5 Demo")
show_raw_tables = st.sidebar.checkbox("Show raw tables", value=False)
max_pred_rows = st.sidebar.slider("Prediction rows", min_value=50, max_value=1000, value=200, step=50)
enable_forecast = st.sidebar.checkbox("Enable next-hour forecast", value=True)

# -----------------------
# Main
# -----------------------
st.title("AQI Big Data (PM2.5) Dashboard")
st.caption("Spark pipeline outputs: EDA trends, model metrics, predictions")

# -----------------------
# Model Metrics
# -----------------------
st.subheader("Model Metrics")
if METRICS_DIR.exists():
    df_metrics = read_parquet(METRICS_DIR)
    st.dataframe(df_metrics, use_container_width=True)

    if "r2" in df_metrics.columns:
        best = df_metrics.sort_values("r2", ascending=False).head(1)
        if not best.empty:
            row = best.iloc[0]
            col1, col2, col3 = st.columns(3)
            col1.metric("Best Model", str(row.get("model", "n/a")))
            col2.metric("R2", f"{row.get('r2', 0):.4f}")
            col3.metric("RMSE", f"{row.get('rmse', 0):.2f}")
else:
    st.warning("Metrics not found. Run training first.")

st.divider()

# -----------------------
# EDA
# -----------------------
st.subheader("EDA Trends")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**Daily Trend (with 7-day moving average)**")
    if DAILY_TREND_DIR.exists():
        df_daily = read_parquet(DAILY_TREND_DIR)
        if "date" in df_daily.columns:
            df_daily["date"] = pd.to_datetime(df_daily["date"])
            df_daily = df_daily.sort_values("date")
        chart_cols = [c for c in ["pm25_avg", "pm25_avg_7d"] if c in df_daily.columns]
        if chart_cols:
            st.line_chart(df_daily.set_index("date")[chart_cols])
        if show_raw_tables:
            st.dataframe(df_daily.head(50), use_container_width=True)
    else:
        st.info("Daily trend not found. Run EDA.")

with col_right:
    st.markdown("**Weekly Trend**")
    if WEEKLY_TREND_DIR.exists():
        df_weekly = read_parquet(WEEKLY_TREND_DIR)
        if {"year", "week_of_year"}.issubset(df_weekly.columns):
            df_weekly["week_of_year"] = df_weekly["week_of_year"].astype(int)
            df_weekly["year_week"] = (
                df_weekly["year"].astype(int).astype(str)
                + "-"
                + df_weekly["week_of_year"].astype(int).map(lambda x: f"{x:02d}")
            )
            df_weekly = df_weekly.sort_values(["year", "week_of_year"])
            if "pm25_avg" in df_weekly.columns:
                st.line_chart(df_weekly.set_index("year_week")["pm25_avg"])
        if show_raw_tables:
            st.dataframe(df_weekly.head(50), use_container_width=True)
    else:
        st.info("Weekly trend not found. Run EDA.")

st.subheader("Weekday Statistics")
if WEEKDAY_STATS_DIR.exists():
    df_weekday = read_parquet(WEEKDAY_STATS_DIR)
    if "day_of_week" in df_weekday.columns:
        df_weekday["day_label"] = df_weekday["day_of_week"].apply(day_of_week_label)
        df_weekday["day_label"] = pd.Categorical(
            df_weekday["day_label"], categories=WEEKDAY_ORDER, ordered=True
        )
        df_weekday = df_weekday.sort_values("day_label")
    if "pm25_avg" in df_weekday.columns:
        st.bar_chart(df_weekday.set_index("day_label")["pm25_avg"])
    if show_raw_tables:
        st.dataframe(df_weekday, use_container_width=True)
else:
    st.info("Weekday stats not found. Run EDA.")

st.divider()

# -----------------------
# Predictions
# -----------------------
st.subheader("Predictions (Actual vs Predicted)")
if PRED_DIR.exists():
    df_pred = read_parquet(PRED_DIR)

    # Normalize column names if needed
    if "date" in df_pred.columns:
        df_pred["date"] = pd.to_datetime(df_pred["date"])
        df_pred = df_pred.dropna(subset=["date"])

        # Aggregate duplicates (multiple sensors may share the same timestamp)
        agg_cols = [c for c in ["actual_pm25", "predicted_pm25"] if c in df_pred.columns]
        if agg_cols:
            df_pred = (
                df_pred.groupby("date", as_index=False)[agg_cols]
                .mean()
            )

        df_pred = df_pred.sort_values("date")

    if len(df_pred) > max_pred_rows:
        df_pred = df_pred.tail(max_pred_rows)

    cols = [c for c in ["actual_pm25", "predicted_pm25"] if c in df_pred.columns]
    if cols:
        st.line_chart(df_pred.set_index("date")[cols])
    st.dataframe(df_pred.head(50), use_container_width=True)
else:
    st.info("Predictions not found. Run prediction step.")

st.divider()

# -----------------------
# Next-hour forecast (t+1)
# -----------------------
st.subheader("Next-hour Forecast (t+1)")
if not enable_forecast:
    st.info("Forecast is disabled in the sidebar.")
elif not FEATURES_DIR.exists() or not MODEL_DIR.exists():
    st.warning("Forecast requires features and a trained model.")
else:
    try:
        with st.spinner("Computing forecast..."):
            from datetime import timedelta
            from pyspark.sql import SparkSession
            from pyspark.sql.functions import col
            from pyspark.ml.feature import VectorAssembler
            from pyspark.ml.regression import LinearRegressionModel

            @st.cache_resource(show_spinner=False)
            def get_spark() -> SparkSession:
                spark = (
                    SparkSession.builder
                    .appName("pm25-forecast")
                    .master("local[1]")
                    .config("spark.ui.enabled", "false")
                    .config("spark.driver.host", "127.0.0.1")
                    .config("spark.driver.bindAddress", "127.0.0.1")
                    .getOrCreate()
                )
                spark.sparkContext.setLogLevel("WARN")
                return spark

            @st.cache_resource(show_spinner=False)
            def load_lr_model() -> LinearRegressionModel:
                return LinearRegressionModel.load(str(MODEL_DIR))

            spark = get_spark()
            model = load_lr_model()

            df_feat = spark.read.parquet(str(FEATURES_DIR))
            latest = df_feat.orderBy(col("datetime").desc()).limit(1).collect()[0]

            last_dt = latest["datetime"]
            last_pm25 = float(latest["pm25"])

            # Build t+1 feature row by shifting lags and advancing time
            next_dt = last_dt + timedelta(hours=1)
            next_hour = int(next_dt.hour)
            next_dow = int(next_dt.isoweekday() % 7 + 1)  # Spark dayofweek (Sun=1)
            next_month = int(next_dt.month)

            row = {
                "hour": next_hour,
                "day_of_week": next_dow,
                "month": next_month,
            }

            # Shift lags: lag_1 = last pm25, lag_2 = last lag_1, ...
            row["lag_1"] = last_pm25
            for i in range(2, 25):
                row[f"lag_{i}"] = float(latest[f"lag_{i-1}"])

            spark_row = spark.createDataFrame([row])
            feature_cols = ["hour", "day_of_week", "month"] + [f"lag_{i}" for i in range(1, 25)]
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            spark_row = assembler.transform(spark_row)

            pred = model.transform(spark_row).collect()[0]["prediction"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Forecast (t+1)", f"{pred:.3f}")
        c2.metric("Last observed pm25", f"{last_pm25:.3f}")
        c3.metric("Forecast time", str(next_dt))

        st.caption(
            "Forecast uses the latest available timestamp and shifts lag features for one-step-ahead prediction."
        )
    except Exception as exc:
        st.warning(f"Forecast unavailable: {exc}")

st.caption("Tip: run the pipeline to refresh outputs before demo.")
