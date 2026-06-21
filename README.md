# Asthma Prediction Dashboard

An interactive web-based dashboard for predicting asthma risk probability using machine learning. The application provides two input methods:
1. **Single Patient**: Manual data entry for individual assessments with instant results.
2. **Import Data**: Bulk processing via CSV file upload for batch predictions.

Built with Python and [Streamlit](https://streamlit.io/).

---

## 🛠 Prerequisites

Ensure you have **Python 3.8+** installed on your system. You can verify this by running:

```bash
python --version
```

---

## 🚀 Installation & Setup

1. **Clone or Download the Project**
   Download the project files to your local machine.

2. **Navigate to the Project Directory**
   Open your terminal or command prompt (cmd/PowerShell) and navigate to the project folder:
   ```bash
   cd "path/to/project/folder"
   ```
   *(Replace `path/to/project/folder` with the actual path where you saved the files, e.g., `cd "C:\Users\User\Documents\CDCS230\SEMESTER 6\CSP650 - FYP\CODING"`)*

3. **Install Dependencies**
   It is recommended (but optional) to create a virtual environment first.
   
   Install the required Python libraries using `pip`:
   ```bash
   pip install -r requirements.txt
   ```
   
   If `requirements.txt` is not available, you can install the packages manually:
   ```bash
   pip install streamlit pandas numpy matplotlib plotly scikit-learn xgboost joblib reportlab
   ```

---

## ▶️ Running the Application

To start the dashboard, run the following command in your terminal:

```bash
streamlit run main.py
```

Wait for a few seconds. Streamlit will automatically open the application in your default web browser (usually at `http://localhost:8501`).

---

## 📖 Usage Guide

### 1. Single Patient Mode (Default)
- Use the sidebar to select "Single Patient".
- Click **"Add New"** to start a new assessment.
- Fill in the patient's vitals (Name, Age, BMI) and clinical symptoms.
- Click **"Predict"** to see the risk analysis.
- View detailed charts (Radar Chart for symptoms, BMI gauge, etc.) and risk probability.

### 2. Import Data Mode (Batch Processing)
- Use the sidebar to select **"Import Data"**.
- Click **"Browse files"** to upload a CSV file containing multiple patient records.
- Once uploaded, click **"Submit"** to process the data.
- You can select individual patients from the dropdown to detailed analysis.
- Click **"Generate All Reports (ZIP)"** to create and download PDF reports for all patients in the batch.

---

## 📂 File Structure

- `main.py`: The main application code.
- `requirements.txt`: List of dependencies (if created).
- `xgboost_best_final-new.pkl`: The trained XGBoost model file used for predictions.
- `asthma-test-dataset.csv`: Sample dataset (optional, for reference).
---

## ⚠️ Troubleshooting

- **"Model file not found"**: Ensure `xgboost_best_final-new.pkl` is in the same directory as `main.py`.
- **"ModuleNotFoundError"**: Run `pip install -r requirements.txt` again to ensure all libraries are installed.
- **Port already in use**: If port 8501 is busy, Streamlit will use the next available port (e.g., 8502). Check the terminal output for the correct URL.

---

## 📝 Credits
Developed for CSP650 - Final Year Project.
