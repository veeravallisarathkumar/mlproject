Student Performance Prediction – End-to-End ML Pipeline
📌 Project Overview

This project is an end-to-end machine learning pipeline that predicts student performance based on demographic and academic features. It covers the entire ML workflow — from data ingestion to deployment — and is packaged as a Flask web application for real-time predictions.

The goal is to demonstrate skills in Data Science, MLOps, and Cloud Deployment using Python, Scikit-learn, CatBoost, Flask, and AWS EC2.

🚀 Features

Data Ingestion: Reads raw data and splits into train/test sets.

Data Transformation: Cleans data, handles missing values, encodes categorical variables, and scales numerical features.

Model Training: Trains multiple ML models (Random Forest, CatBoost), evaluates performance, and selects the best model.

Model Serialization: Saves the trained model and preprocessing pipeline as artifacts.

Prediction Pipeline: Loads the saved model and preprocessor for real-time inference.

Flask Web App: Interactive UI (templates/) for entering student details and getting predictions.

Exception Handling & Logging: Custom error tracking and detailed logs for debugging.

Deployment: Flask app hosted on AWS EC2, version controlled with Git & GitHub.

🛠️ Tech Stack

Programming: Python

ML & Data Processing: Pandas, NumPy, Scikit-learn, CatBoost

Web Framework: Flask

Database/Storage: CSV (local) | AWS Athena (for SQL queries)

Version Control: Git, GitHub

Cloud: AWS EC2 (hosting)

Others: Logging, Custom Exception Handling, Pipelines

📂 Project Structure
mlproject/
│
├── artifacts/               # Saved models, preprocessor, train/test data
├── catboost_info/           # CatBoost training logs
├── notebook/                # Jupyter notebooks (EDA, experiments)
├── src/                     # Source code
│   ├── components/          # ML pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/            # End-to-end pipelines
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   ├── utils.py             # Utility functions (save/load objects, etc.)
│   ├── logger.py            # Logging configuration
│   └── exception.py         # Custom exception handling
├── templates/               # HTML files for Flask app
├── app.py                   # Flask application
├── requirements.txt         # Project dependencies
├── setup.py                 # Packaging setup
└── README.md                # Project documentation

⚡ How to Run Locally
1️⃣ Clone the Repository
git clone https://github.com/veeravallisarathkumar/mlproject.git
cd mlproject

2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run Training Pipeline
python src/pipeline/training_pipeline.py


This will:

Ingest data

Transform data

Train models

Save artifacts (preprocessor + model)

5️⃣ Run Flask App
python app.py


App will run on http://127.0.0.1:5000/

Enter student details in the form and get predictions instantly.

☁️ Deployment (AWS EC2)

Flask app deployed on AWS EC2 instance.

Artifacts and logs managed via GitHub version control.

Project can be extended to use AWS S3 + Athena for cloud-based storage and querying.

📈 Results

Achieved high accuracy/F1-score using CatBoost Classifier.

Pipeline is modular, reusable, and production-ready.

Real-time predictions available via web interface.

🔮 Future Enhancements

Integrate Power BI dashboard for student insights.

Automate retraining pipeline with CI/CD.

Store artifacts in AWS S3 and use SageMaker for training.

👨‍💻 Author

Sarath Kumar Veeravalli

LinkedIn

GitHub