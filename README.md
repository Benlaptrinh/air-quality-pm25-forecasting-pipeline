# Air Quality PM2.5 Forecasting Pipeline (Apache Spark)

End-to-end Spark pipeline to ingest PM2.5 data, clean it, engineer time-series features, train a simple regression model, and run predictions.

## Data source
- OpenAQ API (daily measurements)
- Script: `src/ingest_openaq.py`
- Optional micro-batch ingest: `src/ingest_openaq_stream.py` (hourly)

## Project structure
- `src/ingest_openaq.py` - fetch raw data from OpenAQ
- `src/ingest_openaq_stream.py` - micro-batch ingest (hourly loop)
- `src/etl.py` - clean raw JSON and write parquet
- `src/eda.py` - Spark SQL EDA (daily/weekly stats, trends)
- `src/features.py` - feature engineering (day_of_week, month, lag_1)
- `src/train.py` - Spark ML training (Linear Regression + Random Forest) + metrics
- `src/predict.py` - load model and write predictions
- `data/` - local data (ignored by git)
- `models/` - trained model (ignored by git)
- `notebooks/` - optional notebooks (placeholders)

## Pipeline
```
Ingest (OpenAQ API) -> ETL (Spark) -> EDA (Spark SQL) -> Features -> Train (LR/RF) -> Predict
```

## Setup
1) Create a virtualenv (optional but recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

3) Configure OpenAQ API key
Create a `.env` file:
```
OPENAQ_API_KEY=your_key_here
```

## Run the pipeline
1) Ingest raw data (single run)
```bash
python src/ingest_openaq.py
```

2) ETL (raw JSON -> clean parquet)
```bash
python src/etl.py
```
Note: `src/etl.py` reads from `data/raw/hourly/*.json` by default.
If you only ran the single ingest script, either move the file into `data/raw/hourly/` or change `RAW_PATH`.

3) EDA (Spark SQL: daily/weekly stats + trend)
```bash
python src/eda.py
```

4) Feature engineering
```bash
python src/features.py
```

5) Train models (Linear Regression + Random Forest)
```bash
python src/train.py
```

6) Predict
```bash
python src/predict.py
```
To run Random Forest predictions:
```bash
MODEL_TYPE=rf python src/predict.py
```

## Model evaluation
After running `src/train.py`, metrics are saved to `data/processed/model_metrics/` and can be summarized here:

| Model | RMSE | MAE | R2 |
| --- | --- | --- | --- |
| Linear Regression | (fill after run) | (fill after run) | (fill after run) |
| Random Forest | (fill after run) | (fill after run) | (fill after run) |

## Optional: near real-time micro-batch
Start the ingest loop and let it write new raw files:
```bash
python src/ingest_openaq_stream.py
```

For a quick demo, set a shorter interval:
```bash
OPENAQ_POLL_SECONDS=600 python src/ingest_openaq_stream.py
```

Then run ETL, features, train, and predict again to refresh outputs.

## Outputs
- Raw JSON (single run): `data/raw/pm25_sensor_5049_page_1.json`
- Raw JSON (hourly loop): `data/raw/hourly/*.json`
- Clean parquet: `data/processed/pm25_clean/`
- EDA outputs: `data/processed/eda/`
- Feature parquet: `data/processed/features/`
- Metrics: `data/processed/model_metrics/`
- Model (LR): `models/pm25_lr_model/`
- Model (RF): `models/pm25_rf_model/`
- Predictions (LR): `data/processed/predictions/`
- Predictions (RF): `data/processed/predictions_rf/`

## Notes
- Output folders under `data/` and `models/` are ignored by git (see `.gitignore`).
- Spark runs in local mode. If you see Spark URL/loopback errors on ARM VMs, set:
```bash
export SPARK_LOCAL_IP=127.0.0.1
```
