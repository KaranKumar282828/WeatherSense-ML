# ==============================================================
#  WeatherSense-ML — Humidity Prediction System
#  ISRO Weather Station, Red Fort, New Delhi (2009–2020)
#
#  HOW TO RUN:
#    pip install streamlit pandas numpy scikit-learn pgmpy plotly openpyxl
#    streamlit run app.py
#
#  Place '2015-16_ISRO.xlsx' in the same folder as app.py
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings, time, os
warnings.filterwarnings('ignore')

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

st.set_page_config(
    page_title="WeatherSense-ML | Humidity Prediction",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background: linear-gradient(180deg,#1a3a5c 0%,#2e86c1 100%); }
[data-testid="stSidebar"] * { color: white !important; }
.metric-card {
    background: rgba(46,134,193,0.12); border-radius: 12px; padding: 20px;
    border-left: 5px solid #2e86c1; margin-bottom: 10px;
}
.metric-card h3 { margin:0; color:inherit; font-size:28px; }
.metric-card p  { margin:0; opacity:0.7; font-size:13px; }
.step-box {
    background: rgba(46,134,193,0.15); border-radius:10px; padding:16px 20px;
    border-left:4px solid #2e86c1; margin:8px 0; color:inherit;
}
.success-box {
    background: rgba(39,174,96,0.15); border-radius:10px; padding:16px 20px;
    border-left:4px solid #27ae60; margin:8px 0; color:inherit;
}
.warn-box {
    background: rgba(243,156,18,0.15); border-radius:10px; padding:16px 20px;
    border-left:4px solid #f39c12; margin:8px 0; color:inherit;
}
.algo-card-bn {
    background: rgba(46,134,193,0.12); border-radius:12px; padding:18px;
    border: 2px solid #2e86c1; margin-bottom:10px;
}
.algo-card-rf {
    background: rgba(39,174,96,0.12); border-radius:12px; padding:18px;
    border: 2px solid #27ae60; margin-bottom:10px;
}
.acc-badge {
    display:inline-block; padding:6px 16px; border-radius:20px;
    font-size:22px; font-weight:bold; margin:8px 0;
}
.predict-result {
    text-align:center; padding:30px; border-radius:16px;
    font-size:22px; font-weight:bold; margin:20px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────
for k in ['df_raw','df_clean','df_imputed','df_bn',
          'bn_model','bn_inference','bn_accuracy',
          'rf_model','rf_accuracy','rf_feat_cols',
          'selected_algo']:
    if k not in st.session_state:
        st.session_state[k] = None

DATASET_PATH = "2015-16_ISRO.xlsx"

LABEL_MAP = {
    'Low':    {'display': 'Dry',      'emoji': '☀️',  'color': '#f39c12'},
    'Medium': {'display': 'Moderate', 'emoji': '🌤️', 'color': '#2980b9'},
    'High':   {'display': 'Humid',    'emoji': '💧',  'color': '#27ae60'},
}

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌦️ WeatherSense-ML")
    st.markdown("**Humidity Prediction System**")
    st.markdown("*ISRO Dataset | Red Fort, Delhi*")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠  Home",
         "📂  1. Load Dataset",
         "🧹  2. Data Cleaning",
         "🔧  3. EM Algorithm",
         "🏗️  4. Train Model",
         "📊  5. Visualisations",
         "🔮  6. Predict"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Pipeline Status**")
    for label, done in [
        ("Data Loaded",  st.session_state['df_raw']      is not None),
        ("Data Cleaned", st.session_state['df_clean']    is not None),
        ("EM Done",      st.session_state['df_imputed']  is not None),
        ("BN Trained",   st.session_state['bn_model']    is not None),
        ("RF Trained",   st.session_state['rf_model']    is not None),
    ]:
        st.markdown(f"{'✅' if done else '⬜'} {label}")

    if st.session_state['bn_accuracy'] or st.session_state['rf_accuracy']:
        st.markdown("---")
        st.markdown("**Accuracy Comparison**")
        if st.session_state['bn_accuracy']:
            bn_acc = st.session_state['bn_accuracy']
            st.markdown(f"🔵 BN:  **{bn_acc*100:.1f}%**")
        if st.session_state['rf_accuracy']:
            rf_acc = st.session_state['rf_accuracy']
            st.markdown(f"🟢 RF:  **{rf_acc*100:.1f}%**")


# ══════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.title("🌦️ WeatherSense-ML")
    st.markdown("### End-to-End Humidity Prediction | ISRO Weather Station — Red Fort, Delhi")
    st.markdown("---")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class='step-box'>
        <b>📌 About this Project</b><br><br>
        WeatherSense-ML is an end-to-end machine learning pipeline that predicts
        humidity levels (Dry / Moderate / Humid) from real ISRO sensor data.<br><br>
        It uses the <b>EM Algorithm</b> to intelligently handle missing sensor readings,
        then trains a <b>Bayesian Network</b> for probabilistic humidity classification.
        A <b>Random Forest</b> model is included for accuracy benchmarking.<br><br>
        <b>Dataset:</b> ISRO Weather Station — Red Fort, New Delhi (2009–2020)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🗺️ How to use this app")
        for icon, title, desc in [
            ("📂","Step 1 — Load Dataset",   "Load ISRO Excel file"),
            ("🧹","Step 2 — Data Cleaning",  "Remove invalid sensor codes"),
            ("🔧","Step 3 — EM Algorithm",   "Fill missing values intelligently"),
            ("🏗️","Step 4 — Train Model",    "Train Bayesian Network and/or Random Forest"),
            ("📊","Step 5 — Visualisations", "Explore seasonal trends and patterns"),
            ("🔮","Step 6 — Predict",        "Select algorithm → predict humidity in real-time"),
        ]:
            st.markdown(f"**{icon} {title}:** {desc}")

    with col2:
        st.markdown("### 📡 Dataset Info")
        for k, v in {
            "Station":  "ISRO0516",
            "Location": "Red Fort, Delhi",
            "Records":  "~39,992 hourly",
            "Period":   "2009 – 2020",
            "Target":   "Humidity (Low / Medium / High)",
        }.items():
            st.markdown(f"**{k}:** {v}")

        st.markdown("### 🔬 Two Algorithms")
        st.markdown("""
        <div class='algo-card-bn'>
        <b>🔵 Bayesian Network</b><br>
        Probabilistic graphical model.<br>
        Outputs full probability distribution.
        </div>
        <div class='algo-card-rf'>
        <b>🟢 Random Forest</b><br>
        Ensemble of 200 decision trees.<br>
        Higher accuracy benchmark.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 1: LOAD DATASET
# ══════════════════════════════════════════════════════════════
elif page == "📂  1. Load Dataset":
    st.title("📂 Step 1: Load Dataset")
    st.markdown("---")

    if not os.path.exists(DATASET_PATH):
        st.error(f"❌ `{DATASET_PATH}` not found in the same folder as app.py!")
        uploaded = st.file_uploader("Upload the dataset here:", type=['xlsx'])
        if uploaded:
            with open(DATASET_PATH, 'wb') as f:
                f.write(uploaded.read())
            st.success("✅ Uploaded!")

    if os.path.exists(DATASET_PATH):
        if st.button("🚀 Load Dataset", type="primary", use_container_width=True):
            with st.spinner("Loading..."):
                df = pd.read_excel(DATASET_PATH, skiprows=7)
                st.session_state['df_raw'] = df
            st.success("✅ Dataset loaded!")

    if st.session_state['df_raw'] is not None:
        df = st.session_state['df_raw']
        st.markdown("### 📊 Overview")
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(f"<div class='metric-card'><h3>{df.shape[0]:,}</h3><p>Total Records</p></div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='metric-card'><h3>{df.shape[1]}</h3><p>Columns</p></div>", unsafe_allow_html=True)
        with c3: st.markdown(f"<div class='metric-card'><h3>{df.isnull().sum().sum():,}</h3><p>Raw NaN Values</p></div>", unsafe_allow_html=True)
        with c4: st.markdown(f"<div class='metric-card'><h3>{df['Year'].nunique()}</h3><p>Years Covered</p></div>", unsafe_allow_html=True)

        st.markdown("### 👁️ Raw Data Preview")
        st.dataframe(df.head(st.slider("Rows to show:", 5, 50, 10)), use_container_width=True)
        st.markdown("### 📈 Statistical Summary")
        st.dataframe(df.describe().round(3), use_container_width=True)
        st.markdown("---")
        st.markdown("<div class='success-box'>✅ Loaded! Go to <b>Step 2: Data Cleaning</b></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2: DATA CLEANING
# ══════════════════════════════════════════════════════════════
elif page == "🧹  2. Data Cleaning":
    st.title("🧹 Step 2: Data Cleaning")
    st.markdown("---")

    if st.session_state['df_raw'] is None:
        st.warning("⚠️ Please load the dataset first (Step 1).")
        st.stop()

    df = st.session_state['df_raw'].copy()

    st.markdown("""
    <div class='step-box'>
    <b>🔍 Sentinel Values in ISRO Dataset:</b><br><br>
    • <b>9999.9</b> → Sensor failure / no data recorded<br>
    • <b>&gt; 100%</b> → Physically impossible humidity<br>
    • <b>&gt; 100 mm/hr</b> → Invalid rainfall codes (e.g. 469, 502, 1016)
    </div>
    """, unsafe_allow_html=True)

    rows = []
    for col in ['air_temp','windspeed','pressure','humidity','rainfall']:
        n9  = int((df[col] >= 9999).sum())
        nov = int((df[col] > 100).sum()) if col in ['humidity','rainfall'] else 0
        rows.append({"Column": col, "9999.9 codes": n9,
                     "Impossible values": nov, "Total to fix": n9+nov})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if st.button("🧹 Run Data Cleaning", type="primary", use_container_width=True):
        with st.spinner("Cleaning..."):
            for col in ['air_temp','windspeed','pressure','humidity']:
                df.loc[df[col] >= 9999, col] = np.nan
            df.loc[df['humidity'] > 100, 'humidity'] = np.nan
            df.loc[df['rainfall'] > 100, 'rainfall'] = np.nan
            df.loc[df['air_temp'] < -40, 'air_temp'] = np.nan
            df_clean = df[['Year','Month','Day','Hour',
                           'air_temp','windspeed','pressure','humidity','rainfall']].copy()
            st.session_state['df_clean'] = df_clean
        st.success("✅ Cleaning complete!")

    if st.session_state['df_clean'] is not None:
        df_clean = st.session_state['df_clean']
        after, total = [], 0
        for col in ['air_temp','windspeed','pressure','humidity','rainfall']:
            c = int(df_clean[col].isnull().sum())
            total += c
            after.append({"Column": col, "Missing": c,
                          "Missing %": f"{c/len(df_clean)*100:.1f}%"})
        st.markdown("### ✅ After Cleaning")
        st.dataframe(pd.DataFrame(after), use_container_width=True, hide_index=True)
        c1,c2 = st.columns(2)
        c1.metric("Total Missing", f"{total:,}")
        c2.metric("Records", f"{len(df_clean):,}")

        fig = px.bar(pd.DataFrame(after), x="Column", y="Missing",
                     title="Missing Values After Cleaning",
                     color="Missing", color_continuous_scale="Blues", text="Missing %")
        fig.update_traces(textposition='outside')
        fig.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        st.markdown("<div class='success-box'>✅ Cleaned! Go to <b>Step 3: EM Algorithm</b></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3: EM ALGORITHM
# ══════════════════════════════════════════════════════════════
elif page == "🔧  3. EM Algorithm":
    st.title("🔧 Step 3: EM Algorithm — Missing Data Imputation")
    st.markdown("---")

    if st.session_state['df_clean'] is None:
        st.warning("⚠️ Please complete Step 2 first.")
        st.stop()

    df_clean = st.session_state['df_clean']

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='step-box'>
        <b>E-Step (Expectation)</b><br><br>
        Look at complete rows, learn relationships between features.
        Use those relationships to <b>ESTIMATE</b> the missing values.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='step-box'>
        <b>M-Step (Maximization)</b><br><br>
        Use complete + estimated data to <b>IMPROVE</b> the model.
        Better model → more accurate estimates next round.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='warn-box'>
    ⟳ Repeat E → M → E → M for <b>10 iterations</b> until values converge (stop changing).
    Better than mean imputation — uses actual relationships between weather features!
    </div>
    """, unsafe_allow_html=True)

    features = ['air_temp','windspeed','pressure','humidity','rainfall']
    missing_before = {c: int(df_clean[c].isnull().sum()) for c in features}

    c1,c2 = st.columns(2)
    with c1: n_iters   = st.slider("EM Iterations:", 5, 20, 10)
    with c2: rand_seed = st.number_input("Random Seed:", value=42, step=1)

    if st.button("🚀 Run EM Algorithm", type="primary", use_container_width=True):
        bar = st.progress(0, text="Starting EM...")
        with st.spinner("Running EM (~30 seconds)..."):
            for i in range(n_iters):
                bar.progress((i+1)/n_iters, text=f"EM Iteration {i+1}/{n_iters}...")
                time.sleep(0.05)
            em  = IterativeImputer(max_iter=n_iters, random_state=int(rand_seed))
            arr = em.fit_transform(df_clean[features])
            df_imp = pd.DataFrame(arr, columns=features)
            df_imp['humidity']  = df_imp['humidity'].clip(0, 100)
            df_imp['air_temp']  = df_imp['air_temp'].clip(-10, 50)
            df_imp['windspeed'] = df_imp['windspeed'].clip(0, None)
            df_imp['rainfall']  = df_imp['rainfall'].clip(0, 100)
            df_imp['Year']  = df_clean['Year'].values
            df_imp['Month'] = df_clean['Month'].values
            df_imp['Day']   = df_clean['Day'].values
            df_imp['Hour']  = df_clean['Hour'].values
            st.session_state['df_imputed'] = df_imp
        bar.progress(1.0, text="Done!")
        st.success("✅ EM Complete! All missing values filled.")

    if st.session_state['df_imputed'] is not None:
        df_imp = st.session_state['df_imputed']
        missing_after = {c: int(df_imp[c].isnull().sum()) for c in features}

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Before EM', x=features,
                             y=[missing_before[c] for c in features], marker_color='#e74c3c'))
        fig.add_trace(go.Bar(name='After EM',  x=features,
                             y=[missing_after[c]  for c in features], marker_color='#27ae60'))
        fig.update_layout(barmode='group', height=320, title="Missing Values: Before vs After EM")
        st.plotly_chart(fig, use_container_width=True)

        c1,c2,c3 = st.columns(3)
        c1.metric("Missing Before", f"{sum(missing_before.values()):,}")
        c2.metric("Missing After",  "0 ✅")
        c3.metric("Humidity Mean",  f"{df_imp['humidity'].mean():.1f}%")

        st.markdown("---")
        st.markdown("<div class='success-box'>✅ EM done! Go to <b>Step 4: Train Model</b></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 4: TRAIN MODEL
# ══════════════════════════════════════════════════════════════
elif page == "🏗️  4. Train Model":
    st.title("🏗️ Step 4: Train Model")
    st.markdown("---")

    if st.session_state['df_imputed'] is None:
        st.warning("⚠️ Please complete Step 3 first.")
        st.stop()

    df_imp = st.session_state['df_imputed']

    if st.session_state['bn_accuracy'] or st.session_state['rf_accuracy']:
        st.markdown("### 📊 Accuracy Comparison")
        c1, c2 = st.columns(2)
        with c1:
            bn_acc = st.session_state['bn_accuracy']
            val    = f"{bn_acc*100:.2f}%" if bn_acc else "Not trained yet"
            st.markdown(f"""
            <div class='algo-card-bn'>
            <b>🔵 Bayesian Network</b><br>
            <span style='font-size:32px; font-weight:900; color:#2e86c1'>{val}</span><br>
            <small>EM + Bayesian Network</small>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            rf_acc = st.session_state['rf_accuracy']
            val2   = f"{rf_acc*100:.2f}%" if rf_acc else "Not trained yet"
            st.markdown(f"""
            <div class='algo-card-rf'>
            <b>🟢 Random Forest</b><br>
            <span style='font-size:32px; font-weight:900; color:#27ae60'>{val2}</span><br>
            <small>EM + Random Forest</small>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("---")

    st.markdown("### ⚙️ Humidity Bins (used by both algorithms)")
    c1, c2 = st.columns(2)
    with c1: hum_low  = st.slider("Low  if humidity <",  10, 60, 40)
    with c2: hum_high = st.slider("High if humidity >",  40, 90, 75)

    st.markdown("---")

    st.markdown("""
    <div class='algo-card-bn'>
    <b>🔵 Algorithm 1 — Bayesian Network</b><br>
    Probabilistic graphical model. Learns conditional probability tables from data.
    Structure: air_temp, windspeed, pressure, rainfall → humidity
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        temp_cold = st.slider("Temp: Cold if < (°C)", -10, 25, 15, key="tc")
        temp_hot  = st.slider("Temp: Hot  if > (°C)",  20, 45, 30, key="th")
    with col2:
        n_train_bn = st.slider("BN Training samples:", 1000, 15000, 10000, 500, key="nbn")

    if st.button("🔵 Train Bayesian Network", type="primary", use_container_width=True):
        with st.spinner("Discretising + Training Bayesian Network..."):
            df_bn = df_imp[['air_temp','windspeed','pressure','humidity','rainfall']].copy()
            df_bn['air_temp']  = pd.cut(df_bn['air_temp'],
                bins=[-np.inf, temp_cold, temp_hot, np.inf], labels=['Cold','Warm','Hot'])
            df_bn['windspeed'] = pd.cut(df_bn['windspeed'],
                bins=[-np.inf, 0.5, 2.0, np.inf], labels=['Low','Medium','High'])
            df_bn['pressure']  = pd.cut(df_bn['pressure'],
                bins=[-np.inf, 975, 985, np.inf], labels=['Low','Medium','High'])
            df_bn['rainfall']  = pd.cut(df_bn['rainfall'],
                bins=[-np.inf, 0.001, 5.0, np.inf], labels=['None','Light','Heavy'])
            df_bn['humidity']  = pd.cut(df_bn['humidity'],
                bins=[0, hum_low, hum_high, 100.1], labels=['Low','Medium','High'])
            for col in df_bn.columns:
                df_bn[col] = df_bn[col].astype(str).replace('nan', np.nan)
            df_bn = df_bn.dropna()

            bn_model = DiscreteBayesianNetwork([
                ('air_temp','humidity'), ('windspeed','humidity'),
                ('pressure','humidity'), ('rainfall','humidity'),
            ])

            train_bn, test_bn = train_test_split(df_bn, test_size=0.2,
                                                  random_state=42, stratify=df_bn['humidity'])
            bn_model.fit(train_bn, estimator=BayesianEstimator,
                         prior_type='BDeu', equivalent_sample_size=5)
            bn_inf = VariableElimination(bn_model)

            test_samp = test_bn.sample(n=min(200, len(test_bn)), random_state=42)
            y_true_bn, y_pred_bn = [], []
            for _, row in test_samp.iterrows():
                res = bn_inf.query(variables=['humidity'],
                                   evidence={'air_temp':  row['air_temp'],
                                             'windspeed': row['windspeed'],
                                             'pressure':  row['pressure'],
                                             'rainfall':  row['rainfall']},
                                   show_progress=False)
                probs = dict(zip(res.state_names['humidity'], res.values))
                y_pred_bn.append(max(probs, key=probs.get))
                y_true_bn.append(row['humidity'])

            bn_acc = accuracy_score(y_true_bn, y_pred_bn)
            st.session_state['bn_model']    = bn_model
            st.session_state['bn_inference']= bn_inf
            st.session_state['bn_accuracy'] = bn_acc
            st.session_state['df_bn']       = df_bn

        st.success(f"✅ Bayesian Network trained! Accuracy: **{bn_acc*100:.2f}%**")
        st.rerun()

    st.markdown("---")

    st.markdown("""
    <div class='algo-card-rf'>
    <b>🟢 Algorithm 2 — Random Forest</b><br>
    Ensemble of 200 decision trees. Uses richer features including Month, Hour, Season.
    Higher accuracy — benchmark against Bayesian Network.
    </div>
    """, unsafe_allow_html=True)

    if st.button("🟢 Train Random Forest", type="primary", use_container_width=True):
        with st.spinner("Training Random Forest..."):
            df_rf = df_imp.copy()

            def get_season(m):
                if m in [12,1,2]:    return 0
                elif m in [3,4,5]:   return 1
                elif m in [6,7,8,9]: return 2
                else:                return 3

            df_rf['Season']    = df_rf['Month'].apply(get_season)
            df_rf['Is_day']    = ((df_rf['Hour'] >= 6) & (df_rf['Hour'] <= 18)).astype(int)
            df_rf['Is_rain']   = (df_rf['rainfall'] > 0.001).astype(int)
            df_rf['temp_pres'] = df_rf['air_temp'] * df_rf['pressure'] / 1000
            df_rf['label']     = pd.cut(df_rf['humidity'],
                bins=[0, hum_low, hum_high, 100.1], labels=['Low','Medium','High'])
            df_rf = df_rf.dropna()

            feat_cols = ['air_temp','windspeed','pressure','rainfall',
                         'Month','Hour','Season','Is_day','Is_rain','temp_pres']
            X = df_rf[feat_cols]
            y = df_rf['label']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)

            rf = RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                class_weight='balanced', random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            rf_acc = accuracy_score(y_test, y_pred)

            st.session_state['rf_model']     = rf
            st.session_state['rf_accuracy']  = rf_acc
            st.session_state['rf_feat_cols'] = feat_cols
            st.session_state['rf_X_test']    = X_test
            st.session_state['rf_y_test']    = y_test
            st.session_state['rf_y_pred']    = y_pred

        st.success(f"✅ Random Forest trained! Accuracy: **{rf_acc*100:.2f}%**")
        st.rerun()

    if st.session_state['bn_accuracy'] and st.session_state['rf_accuracy']:
        st.markdown("---")
        st.markdown("### 📊 Detailed Comparison")

        bn_acc = st.session_state['bn_accuracy']
        rf_acc = st.session_state['rf_accuracy']

        fig = go.Figure(go.Bar(
            x=['Bayesian Network', 'Random Forest'],
            y=[bn_acc*100, rf_acc*100],
            marker_color=['#2e86c1','#27ae60'],
            text=[f"{bn_acc*100:.2f}%", f"{rf_acc*100:.2f}%"],
            textposition='outside'
        ))
        fig.add_hline(y=80, line_dash="dash", line_color="red",
                      annotation_text="80% Target", annotation_position="right")
        fig.update_layout(title="Algorithm Accuracy Comparison",
                          yaxis_title="Accuracy (%)", yaxis_range=[0,100],
                          height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        if st.session_state['rf_model']:
            st.markdown("**🟢 Random Forest — Feature Importance**")
            feat_cols = st.session_state['rf_feat_cols']
            fi = sorted(zip(feat_cols, st.session_state['rf_model'].feature_importances_),
                        key=lambda x: -x[1])
            fig2 = px.bar(x=[f[1]*100 for f in fi], y=[f[0] for f in fi],
                          orientation='h', title="Which features matter most?",
                          color=[f[1]*100 for f in fi], color_continuous_scale='Greens')
            fig2.update_layout(height=320, yaxis={'categoryorder':'total ascending'},
                               showlegend=False, xaxis_title="Importance (%)")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.markdown("<div class='success-box'>✅ Both models trained! Now go to <b>Predict</b></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 5: VISUALISATIONS
# ══════════════════════════════════════════════════════════════
elif page == "📊  5. Visualisations":
    st.title("📊 Step 5: Visualisations")
    st.markdown("---")

    if st.session_state['df_imputed'] is None:
        st.warning("⚠️ Please complete Steps 1–3 first.")
        st.stop()

    df = st.session_state['df_imputed']
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Distributions", "🗓️ Seasonal Trends",
        "🔗 Relationships",  "🗺️ Heatmap"
    ])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x='humidity', nbins=50, title="Humidity Distribution",
                               color_discrete_sequence=['steelblue'])
            fig.add_vline(x=df['humidity'].mean(),   line_color='red',    line_dash='dash',
                          annotation_text=f"Mean={df['humidity'].mean():.1f}%")
            fig.add_vline(x=df['humidity'].median(), line_color='orange', line_dash='dash',
                          annotation_text=f"Median={df['humidity'].median():.1f}%")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.histogram(df, x='air_temp', nbins=50, title="Temperature Distribution",
                                color_discrete_sequence=['#e74c3c'])
            fig2.update_layout(height=380)
            st.plotly_chart(fig2, use_container_width=True)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Avg Humidity",  f"{df['humidity'].mean():.1f}%")
        c2.metric("Avg Temp",      f"{df['air_temp'].mean():.1f}°C")
        c3.metric("Avg Wind",      f"{df['windspeed'].mean():.2f} m/s")
        c4.metric("Avg Pressure",  f"{df['pressure'].mean():.1f} hPa")

    with tab2:
        col1, col2 = st.columns(2)
        monthly = df.groupby('Month')[['humidity','air_temp']].mean().reset_index()
        monthly['Month_Name'] = monthly['Month'].apply(lambda x: months[int(x)-1])
        with col1:
            fig = px.bar(monthly, x='Month_Name', y='humidity',
                         title="Average Humidity by Month",
                         color='humidity', color_continuous_scale='Blues',
                         text=monthly['humidity'].round(1))
            fig.update_traces(textposition='outside')
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.line(monthly, x='Month_Name', y='air_temp',
                           title="Average Temperature by Month",
                           markers=True, color_discrete_sequence=['#e74c3c'])
            fig2.update_layout(height=380)
            st.plotly_chart(fig2, use_container_width=True)

        hourly = df.groupby('Hour')['humidity'].mean().reset_index()
        fig3 = px.area(hourly, x='Hour', y='humidity',
                       title="Average Humidity by Hour of Day",
                       color_discrete_sequence=['steelblue'])
        fig3.update_layout(height=300, xaxis_title="Hour (0–23)")
        fig3.update_xaxes(tickmode='linear', dtick=2)
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        feat = st.selectbox("Feature to compare with Humidity:",
                            ['air_temp','windspeed','pressure','rainfall'])
        samp = df.sample(n=min(5000, len(df)), random_state=42)
        fig  = px.scatter(samp, x=feat, y='humidity',
                          title=f"{feat} vs Humidity",
                          opacity=0.3, trendline='ols',
                          color='Month', color_continuous_scale='RdYlBu')
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

        corr = df[['air_temp','windspeed','pressure','humidity','rainfall']].corr()
        fig2 = px.imshow(corr, text_auto='.2f', title="Feature Correlation Heatmap",
                         color_continuous_scale='RdBu_r', aspect='auto')
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        pivot = df.groupby(['Month','Hour'])['humidity'].mean().unstack()
        pivot.index = months
        fig = px.imshow(pivot, title="Humidity Heatmap: Month × Hour of Day",
                        color_continuous_scale='Blues',
                        labels=dict(x="Hour of Day", y="Month", color="Humidity %"),
                        aspect='auto')
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
        st.info("🔵 Darker = Higher Humidity. July–September (monsoon) shows highest humidity.")


# ══════════════════════════════════════════════════════════════
# PAGE 6: PREDICT
# ══════════════════════════════════════════════════════════════
elif page == "🔮  6. Predict":
    st.title("🔮 Step 6: Real-World Humidity Prediction")
    st.markdown("---")

    bn_ready = st.session_state['bn_model'] is not None
    rf_ready = st.session_state['rf_model'] is not None

    if not bn_ready and not rf_ready:
        st.warning("⚠️ Please train at least one model in Step 4 first.")
        st.stop()

    st.markdown("### 🔀 Select Algorithm for Prediction")
    options = []
    if bn_ready: options.append("🔵 Bayesian Network")
    if rf_ready: options.append("🟢 Random Forest (Higher Accuracy)")

    algo_choice = st.radio("Choose algorithm:", options,
                           label_visibility="collapsed",
                           horizontal=True)
    use_bn = "Bayesian" in algo_choice

    if use_bn:
        acc = st.session_state['bn_accuracy']
        st.markdown(f"""
        <div class='algo-card-bn'>
        <b>🔵 Using: Bayesian Network</b><br>
        <b>Accuracy: {acc*100:.2f}%</b>
        </div>
        """, unsafe_allow_html=True)
    else:
        acc = st.session_state['rf_accuracy']
        st.markdown(f"""
        <div class='algo-card-rf'>
        <b>🟢 Using: Random Forest</b><br>
        <b>Accuracy: {acc*100:.2f}%</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    def disc_temp(v): return 'Cold' if v < 15 else ('Hot' if v > 30 else 'Warm')
    def disc_wind(v): return 'Low'  if v < 0.5 else ('High' if v > 2.0 else 'Medium')
    def disc_pres(v): return 'Low'  if v < 975 else ('High' if v > 985 else 'Medium')
    def disc_rain(v): return 'None' if v <= 0.001 else ('Heavy' if v > 5.0 else 'Light')

    def get_season(m):
        if m in [12,1,2]: return 0
        elif m in [3,4,5]: return 1
        elif m in [6,7,8,9]: return 2
        else: return 3

    def predict_bn(temp, wind, pres, rain):
        inf_e  = st.session_state['bn_inference']
        result = inf_e.query(
            variables=['humidity'],
            evidence={'air_temp': disc_temp(temp), 'windspeed': disc_wind(wind),
                      'pressure': disc_pres(pres), 'rainfall':  disc_rain(rain)},
            show_progress=False
        )
        probs = dict(zip(result.state_names['humidity'], result.values))
        pred  = max(probs, key=probs.get)
        return pred, probs

    def predict_rf(temp, wind, pres, rain, month, hour):
        rf_m  = st.session_state['rf_model']
        feat  = st.session_state['rf_feat_cols']
        season    = get_season(month)
        is_day    = 1 if 6 <= hour <= 18 else 0
        is_rain   = 1 if rain > 0.001 else 0
        temp_pres = temp * pres / 1000
        X = pd.DataFrame([[temp, wind, pres, rain, month, hour,
                           season, is_day, is_rain, temp_pres]], columns=feat)
        pred  = rf_m.predict(X)[0]
        proba = rf_m.predict_proba(X)[0]
        probs = dict(zip(rf_m.classes_, proba))
        return pred, probs

    tab1, tab2 = st.tabs(["🎛️ Single Prediction", "📋 Batch Prediction"])

    with tab1:
        st.markdown("### Enter Current Weather Conditions")
        col1, col2 = st.columns(2)
        with col1:
            air_temp  = st.slider("🌡️ Air Temperature (°C)",  -10.0, 50.0, 28.0, 0.5)
            windspeed = st.slider("💨 Wind Speed (m/s)",        0.0,  20.0,  0.5, 0.1)
            month_val = st.slider("📅 Month",                   1,    12,    6,   1)
        with col2:
            pressure  = st.slider("🌀 Pressure (hPa)",        950.0, 1020.0, 982.0, 0.5)
            rainfall  = st.slider("🌧️ Rainfall (mm/hr)",       0.0,   50.0,   0.0, 0.5)
            hour_val  = st.slider("🕐 Hour of Day",             0,    23,    12,   1,
                                  disabled=use_bn)

        if use_bn:
            st.markdown("### 📥 Input as seen by BN")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Temperature", disc_temp(air_temp))
            c2.metric("Wind",        disc_wind(windspeed))
            c3.metric("Pressure",    disc_pres(pressure))
            c4.metric("Rainfall",    disc_rain(rainfall))

        if st.button("🔮 Predict Humidity", type="primary", use_container_width=True):
            with st.spinner("Predicting..."):
                if use_bn:
                    pred, probs = predict_bn(air_temp, windspeed, pressure, rainfall)
                else:
                    pred, probs = predict_rf(air_temp, windspeed, pressure, rainfall, month_val, hour_val)

            info  = LABEL_MAP.get(pred, {'display': pred, 'emoji': '🌦️', 'color': '#2e86c1'})
            color = info['color']
            label = f"{info['emoji']} {info['display'].upper()}"
            algo_tag = "🔵 Bayesian Network" if use_bn else "🟢 Random Forest"

            st.markdown(f"""
            <div class='predict-result'
                 style='background:{color}22; border:3px solid {color}; color:{color};'>
                {algo_tag}<br>
                Predicted Humidity: <span style='font-size:34px; font-weight:900'>{label}</span>
            </div>
            """, unsafe_allow_html=True)

            cats       = ['Low','Medium','High']
            disp_cats  = [LABEL_MAP[c]['display'] for c in cats]
            vals       = [probs.get(c, 0) for c in cats]
            colors_bar = [LABEL_MAP[c]['color'] for c in cats]

            fig = go.Figure(go.Bar(
                x=disp_cats, y=vals,
                marker_color=colors_bar,
                text=[f"{v:.1%}" for v in vals],
                textposition='outside'
            ))
            fig.update_layout(
                title=f"Probability Distribution ({algo_tag})",
                yaxis_title="Probability", yaxis_range=[0, 1.15], height=350
            )
            st.plotly_chart(fig, use_container_width=True)

            gauge_val = {'Low': 20, 'Medium': 55, 'High': 90}.get(pred, 50)
            fig2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=gauge_val,
                title={"text": f"{info['emoji']} {info['display']}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": color},
                    "steps": [
                        {"range": [0,  40], "color": "#fdebd0"},
                        {"range": [40, 75], "color": "#d6eaf8"},
                        {"range": [75, 100],"color": "#d5f5e3"},
                    ],
                }
            ))
            fig2.update_layout(height=280)
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("### 📋 Full Probability Table")
            st.dataframe(pd.DataFrame({
                "Level":         cats,
                "Display":       [LABEL_MAP[c]['display'] for c in cats],
                "Probability":   [f"{probs.get(c,0):.4f}" for c in cats],
                "Percentage":    [f"{probs.get(c,0):.1%}"  for c in cats],
                "Result":        [f"{LABEL_MAP[c]['emoji']} ✅" if c == pred else "" for c in cats]
            }), use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### 📋 Batch Prediction — 6 Delhi Scenarios")

        scenarios = [
            {"Name":"🌞 Peak Summer (May)",      "air_temp":42,"windspeed":0.3,"pressure":993, "rainfall":0,  "month":5,  "hour":14},
            {"Name":"🌧️ Monsoon Peak (July)",     "air_temp":33,"windspeed":1.0,"pressure":968, "rainfall":20, "month":7,  "hour":10},
            {"Name":"❄️ Winter Morning (Jan)",    "air_temp":8, "windspeed":3.0,"pressure":1010,"rainfall":0,  "month":1,  "hour":8},
            {"Name":"🌸 Spring Evening (Mar)",    "air_temp":25,"windspeed":1.5,"pressure":982, "rainfall":0,  "month":3,  "hour":18},
            {"Name":"⛈️ Pre-Monsoon Storm (Jun)", "air_temp":36,"windspeed":0.8,"pressure":971, "rainfall":8,  "month":6,  "hour":15},
            {"Name":"🍂 Post-Monsoon (Oct)",      "air_temp":27,"windspeed":0.5,"pressure":987, "rainfall":2,  "month":10, "hour":12},
        ]
        st.dataframe(pd.DataFrame(scenarios), use_container_width=True, hide_index=True)

        if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
            results = []
            prog = st.progress(0)
            for i, sc in enumerate(scenarios):
                if use_bn:
                    pred, probs = predict_bn(sc['air_temp'], sc['windspeed'],
                                             sc['pressure'], sc['rainfall'])
                else:
                    pred, probs = predict_rf(sc['air_temp'], sc['windspeed'],
                                             sc['pressure'], sc['rainfall'],
                                             sc['month'], sc['hour'])
                b_info = LABEL_MAP.get(pred, {'display': pred, 'emoji': '🌦️'})
                results.append({
                    "Scenario":      sc['Name'],
                    "P(Dry)":        f"{probs.get('Low',0):.3f}",
                    "P(Moderate)":   f"{probs.get('Medium',0):.3f}",
                    "P(Humid)":      f"{probs.get('High',0):.3f}",
                    "🎯 Prediction": f"{b_info['emoji']} {b_info['display']}"
                })
                prog.progress((i+1)/len(scenarios))

            algo_tag = "Bayesian Network" if use_bn else "Random Forest"
            st.markdown(f"### ✅ Results using {algo_tag}")
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

            fig = go.Figure()
            for level, dname in [('Low','Dry'),('Medium','Moderate'),('High','Humid')]:
                fig.add_trace(go.Bar(
                    name=f"{LABEL_MAP[level]['emoji']} {dname}",
                    x=[r['Scenario'] for r in results],
                    y=[float(r[f'P({dname})']) for r in results],
                    marker_color=LABEL_MAP[level]['color']
                ))
            fig.update_layout(barmode='group',
                              title=f"Humidity Probabilities — {algo_tag}",
                              height=420, xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)

            csv = pd.DataFrame(results).to_csv(index=False)
            st.download_button("📥 Download Results as CSV", csv,
                               "batch_predictions.csv", "text/csv",
                               use_container_width=True)