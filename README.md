# Student Performance Predictor (Machine Learning Website)

## What it does
A Streamlit web app that estimates a student's final score using five input features:
- Study hours per day
- Attendance percentage
- Previous exam score
- Sleep hours per night
- Assignments completed out of 10

The model is a real, trained scikit-learn Pipeline using median imputation, standard scaling, and Linear Regression. The saved model is `student_model.joblib`.

## Dataset note
`student_performance.csv` contains **synthetic sample data generated for this demonstration**. It is not real student data and should not be used to make consequential academic decisions. For a serious project, replace it with an appropriate, consented, real dataset and retrain the model.

## Setup (Windows)
1. Install Python 3.10 or newer from https://www.python.org/downloads/ and select "Add Python to PATH".
2. Extract this ZIP file.
3. Open Command Prompt in the extracted `student_performance_ml` folder.
4. Create and activate a virtual environment:
   ```bat
   py -m venv .venv
   .venv\Scripts\activate
   ```
5. Install dependencies:
   ```bat
   pip install -r requirements.txt
   ```
6. Start the website:
   ```bat
   streamlit run app.py
   ```
7. Open the local URL Streamlit prints, usually http://localhost:8501.

## Setup (macOS/Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` — interactive website
- `student_performance.csv` — example dataset
- `student_model.joblib` — trained model bundle and evaluation metrics
- `train_model.py` — script to retrain the model from the CSV
- `requirements.txt` — Python dependencies

## Model evaluation
The app shows MAE, RMSE, and R² from a held-out 20% test split. The current model's scores are recorded in the bundle; metrics can vary if you replace the dataset.
