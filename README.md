# 🌦️ WeatherSense-ML
### End-to-End Humidity Prediction | ISRO Weather Station, Red Fort Delhi

An end-to-end machine learning pipeline that predicts humidity levels (Dry / Moderate / Humid) from real ISRO weather sensor data using the EM Algorithm with both Bayesian Network and Random Forest classifiers.

---

## 📌 About

WeatherSense-ML processes 39,992 hourly weather readings from the ISRO station at Red Fort, New Delhi (2009–2020). It handles 26,008 missing sensor values using the EM Algorithm, then trains two models — a **Bayesian Network** for probabilistic humidity classification and a **Random Forest** for higher-accuracy predictions. Both models are available interactively in the app.

---

## 📊 Results

| Algorithm | Accuracy | Notes |
|---|---|---|
| Bayesian Network (EM + BN) | ~67% | Probabilistic output — full probability distribution |
| Random Forest (EM + RF) | **83.37%** | Higher accuracy — ensemble of 200 decision trees |

- **26,008 missing values** filled with zero data loss using EM Algorithm
- All **6 seasonal Delhi scenarios** predicted correctly by both models

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/KaranKumar282828/WeatherSense-ML.git
cd WeatherSense-ML
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Place `2015-16_ISRO.xlsx` in the project root folder.

> ⚠️ Dataset not included due to file size. Download from ISRO Atmospheric Science Division.

### 5. Run
```bash
streamlit run app.py
```

---

## 🗺️ Pipeline

```
Raw ISRO Data
     ↓
Data Cleaning       ← Remove sentinel values (9999.9), impossible readings
     ↓
EM Algorithm        ← Fill 26,008 missing values using IterativeImputer
     ↓
Discretization      ← Convert continuous features to categorical bins (BN only)
     ↓
Model Training      ← Bayesian Network (pgmpy) + Random Forest (scikit-learn)
     ↓
Prediction          ← Low / Moderate / High humidity with probability scores
```

---

## 🛠️ Tech Stack

- **Scikit-learn** — EM Algorithm (IterativeImputer), Random Forest Classifier
- **pgmpy** — Bayesian Network (DiscreteBayesianNetwork), Variable Elimination inference
- **Streamlit** — Interactive 6-step web application
- **Plotly** — Probability distribution charts, heatmaps, feature importance
- **NumPy / Pandas** — Data processing and feature engineering

---

## 📂 Project Structure

```
WeatherSense-ML/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── .gitignore             # Git ignore rules
└── 2015-16_ISRO.xlsx      # Dataset (not included — add manually)
```

---

## 👨‍💻 Author

**Karan Kumar** — B.Tech ECE, IIIT Kalyani  
[GitHub](https://github.com/KaranKumar282828) · [LinkedIn](https://linkedin.com/in/karankumar75055)
