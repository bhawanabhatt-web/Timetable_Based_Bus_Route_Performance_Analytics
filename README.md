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
Timetable_Based_Bus_Route_Performance_Analytics/
├── data/
│   ├── raw/                    
│   ├── raw_data_csv/           
│   └── cleaned_parquet/         
├── docs/
│   ├── Bhawana_Kumari_Bhatta_240620.pdf   
│   ├── architecture diagram.png
│   ├── data source.png
│   ├── er.png
│   ├── spark1.png
│   ├── spark2.png
│   ├── userinterface1.png
│   ├── userinterface2.png
│   └── user interface3.png
|
│   
│   
│   
├── jdbc/                        
├── notebooks/                    
│                                 
├── output/                       
│                                  
├── src/
│   ├── ingestion/
│   │   ├── poll_locations.py      
│   │   └── process_raw_data.py    
│   ├── spark_backend.py          
│   ├── bus_route_dashboard.py    
├── .env                    
├── .gitignore
├── requirements.txt
└── README.md



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


### 4. Run the pipeline (in order)
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

