# 🌦️ WeatherSense-ML
### End-to-End Humidity Prediction | ISRO Weather Station, Red Fort Delhi

An end-to-end machine learning pipeline that predicts humidity levels (Dry / Moderate / Humid) from real ISRO weather sensor data using the EM Algorithm and Bayesian Network.

---

## 📌 About

WeatherSense-ML processes 39,992 hourly weather readings from the ISRO station at Red Fort, New Delhi (2009–2020). It handles 26,008 missing sensor values using the EM Algorithm, then trains a Bayesian Network for probabilistic humidity classification. A Random Forest model is included for accuracy benchmarking.

---

## 📊 Results

| Algorithm | Accuracy |
|---|---|
| Bayesian Network (EM + BN) | ~67% |
| Random Forest (EM + RF) | **83.37%** |

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
Raw ISRO Data → Data Cleaning → EM Algorithm → Discretization → Bayesian Network → Prediction

---

## 🛠️ Tech Stack

- **Scikit-learn** — EM Algorithm (IterativeImputer), Random Forest
- **pgmpy** — Bayesian Network, Variable Elimination inference
- **Streamlit** — Interactive 6-step web application
- **Plotly** — Visualizations and probability charts
- **NumPy / Pandas** — Data processing

---

## 👨‍💻 Author

**Karan Kumar** — B.Tech ECE, IIIT Kalyani  
[GitHub](https://github.com/KaranKumar282828) · [LinkedIn](https://linkedin.com/in/karankumar75055)
