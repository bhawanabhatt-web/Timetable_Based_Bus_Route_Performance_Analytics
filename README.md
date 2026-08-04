# Timetable-Based Bus Route Performance Analytics

## Project Overview
This project builds a big data analytics pipeline using **PySpark** to process UK Bus Open
Data Service (BODS) timetable and fare data, engineer features, and train machine learning
models to classify bus routes into **High / Medium / Low** popularity tiers. Results are
stored in **MySQL** and surfaced through an interactive prediction dashboard.

- **Records processed:** 926,481 (after cleaning)
- **Data source:** UK Bus Open Data Service (BODS) — Stagecoach, Oxfordshire
- **Best model:** Random Forest (Accuracy 69.98%, F1-score 69.91%)

## Repository Structure
```
.
├── data/
│   ├── raw_data_csv/           # XML→CSV converted raw files
│   └── cleaned/                # Cleaned Parquet/CSV per dataset
├── output/
│   ├── final_schedule.parquet  # Final merged + feature-engineered dataset
│   ├── final_feature_dataset.csv
│   ├── feature_metadata.json
│   ├── cv_results.json         # Cross-validation results per model
│   ├── feature_importance.json
│   └── eval_predictions/       # Per-model test-set predictions
├── database/
│   ├── schema.sql              # Full MySQL schema (DDL)
│   ├── bus_performance_analytics_dump.sql  # Full data dump
│   └── sample_queries.sql      # Example analytical queries
├── notebooks/                  # Jupyter notebooks (01_ingestion → 06_dashboard)
├── src/
│   ├── process_raw_data.py     # XML → CSV parser
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── model_training.py       # Logistic Regression / Decision Tree / Random Forest
│   ├── mysql_writer.py         # Parameterised MySQL persistence
│   └── dashboard.py            # Prediction dashboard
├── docs/
│   ├── architecture_diagram.png
│   ├── er_diagram.png
│   └── report.pdf              # Final coursework report
├── requirements.txt
├── .env.example                # Environment variable template (no real credentials)
└── README.md
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/bhawanabhatt-web/Timetable_Based_Bus_Route_Performance_Analytics.git
cd Timetable_Based_Bus_Route_Performance_Analytics
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Copy `.env.example` to `.env` and fill in your own values — **never commit real credentials**:
```
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password_here
MYSQL_DATABASE=bus_performance_analytics
SPARK_CORES=8
SPARK_DRIVER_MEMORY=4g
RAW_CSV_DIR=data/raw_data_csv
CLEANED_DIR=data/cleaned
OUTPUT_DIR=output
MYSQL_JDBC_JAR=jdbc/mysql-connector-j-9.7.0.jar
```

### 5. Set up MySQL
```bash
mysql -u root -p < database/schema.sql
mysql -u root -p bus_performance_analytics < database/bus_performance_analytics_dump.sql
```

### 6. Run the pipeline (in order)
```bash
python src/process_raw_data.py        # XML -> CSV
python src/data_cleaning.py           # Clean + validate (PySpark)
python src/feature_engineering.py     # Feature engineering + MySQL write
python src/model_training.py          # Train & compare 3 ML models
python src/dashboard.py               # Launch prediction dashboard
```

## Tools & Technologies
| Tool | Purpose |
|---|---|
| PySpark 3.5.8 | Distributed data processing & MLlib model training |
| Pandas | Data inspection & final-stage visualisation |
| MySQL 8.0 | Persistent structured storage |
| SQLAlchemy | Parameterised, injection-safe database access |
| Matplotlib / Plotly | Visualisation |
| Git / GitHub | Version control |

## Big Data Evidence
- Dataset: 926,481 rows after cleaning (exceeds 100,000-record requirement)
- Spark configured with `local[8]`, 16 shuffle partitions
- Caching and repartitioning applied on join keys (`source_file`, `vehicle_journey_code`)
- Spark UI screenshots included in `docs/report.pdf` (Section 5.4)

## Machine Learning
Three PySpark MLlib classifiers were trained and compared using 3-fold cross-validation
and an 80:20 stratified train/test split:

| Model | Accuracy | F1-score | Training Time (s) |
|---|---|---|---|
| Logistic Regression (baseline) | 0.5665 | 0.5707 | 1,131.6 |
| Decision Tree | 0.6056 | 0.5932 | 2,550.4 |
| **Random Forest (selected)** | **0.6998** | **0.6991** | 17,298.8 |

## Security Considerations
- MySQL credentials loaded from environment variables (`.env`), never hard-coded
- All database operations use parameterised SQLAlchemy queries
- Only publicly available BODS data used — no personal or sensitive data processed

## Ethical & Legal Considerations
This project uses only publicly available BODS data, consistent with GDPR principles.
Predictions are intended to support, not replace, human decision-making in transport planning.

## Video Demonstration
https://www.youtube.com/watch?v=zexjuApUtMY

## Report
Full technical report available at `docs/report.pdf`.

## Licence
Academic coursework submission for Softwarica College of IT & E-Commerce (Coventry University).

