import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

st.set_page_config(page_title="AI Predictive Maintenance", layout="wide")

# -----------------------------
# CSS (UI 개선)
# -----------------------------
st.markdown("""
<style>
[data-testid="stMetricValue"] {
    font-size: 24px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 데이터 로드
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("train_scaled_fd001.csv")

@st.cache_resource
def load_model_lstm():
    return load_model("lstm_model.keras")

data = load_data()
model = load_model_lstm()

feature_cols = [c for c in data.columns if c not in ['engine_id','cycle','max_cycle','RUL']]
sensor_cols = [c for c in feature_cols if "sensor_" in c]

# -----------------------------
# 제목
# -----------------------------
st.title("AI Predictive Maintenance System")

st.caption("LSTM 기반 항공기 엔진 Remaining Useful Life 예측")

# -----------------------------
# KPI 카드
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("데이터셋", "C-MAPSS FD001")
col2.metric("엔진 개수", data["engine_id"].nunique())
col3.metric("센서 수", len(sensor_cols))
col4.metric("AI 모델", "LSTM")

st.caption("NASA Turbofan Engine Degradation Dataset")

st.markdown("---")

# -----------------------------
# 모델 성능 비교
# -----------------------------
st.subheader("Model Performance (RMSE)")

results = pd.DataFrame({
    "Model":["RandomForest","XGBoost","LSTM"],
    "RMSE":[18.76,18.0,5.69]
})

colA,colB = st.columns(2)

with colA:
    st.dataframe(results,use_container_width=True)

with colB:
    fig,ax=plt.subplots()
    ax.bar(results["Model"],results["RMSE"])
    ax.set_ylabel("RMSE")
    ax.set_title("Model Comparison")
    st.pyplot(fig)

st.markdown("---")

# -----------------------------
# 엔진 선택
# -----------------------------
st.sidebar.header("엔진 선택")

engine_ids = sorted(data["engine_id"].unique())

engine_id = st.sidebar.selectbox(
    "Select Engine",
    engine_ids
)

engine_data = data[data["engine_id"]==engine_id]

seq_length = 30

# -----------------------------
# Cycle 선택
# -----------------------------
st.subheader(f"Engine {engine_id} Analysis")

cycle_select = st.slider(
    "Prediction Cycle",
    min_value=seq_length,
    max_value=int(engine_data["cycle"].max()),
    value=int(engine_data["cycle"].max())
)

engine_until = engine_data[engine_data["cycle"]<=cycle_select]

# -----------------------------
# 센서 선택
# -----------------------------
sensor = st.selectbox(
    "Sensor",
    sensor_cols
)

# -----------------------------
# 센서 그래프
# -----------------------------
fig1,ax1 = plt.subplots(figsize=(10,4))

ax1.plot(engine_data["cycle"],engine_data[sensor])

ax1.axvline(cycle_select,linestyle="--")

ax1.set_xlabel("Cycle")
ax1.set_ylabel(sensor)

ax1.set_title("Sensor Trend")

st.pyplot(fig1)

# -----------------------------
# RUL 그래프
# -----------------------------
fig2,ax2=plt.subplots(figsize=(10,4))

ax2.plot(engine_data["cycle"],engine_data["RUL"])

ax2.axvline(cycle_select,linestyle="--")

ax2.set_xlabel("Cycle")
ax2.set_ylabel("RUL")

ax2.set_title("True RUL (Remaining Useful Life)")

st.pyplot(fig2)

# -----------------------------
# LSTM 예측
# -----------------------------
st.subheader("RUL 예측 결과")

seq_input = engine_until[feature_cols].iloc[-seq_length:].values

seq_input = np.expand_dims(seq_input,axis=0)

pred_rul = model.predict(seq_input)[0][0]

true_rul = engine_until["RUL"].iloc[-1]

colX,colY,colZ = st.columns(3)

colX.metric("실제 RUL",round(true_rul,2))
colY.metric("예측 RUL",round(pred_rul,2))

risk = max(0,min(100,int((1-pred_rul/125)*100)))

colZ.metric("고장 위험도",f"{risk}%")

# -----------------------------
# 상태 메시지
# -----------------------------
if pred_rul < 20:
    st.error("고장 위험이 매우 높습니다. 즉시 점검이 필요합니다.")

elif pred_rul < 50:
    st.warning("성능 저하가 감지되었습니다. 유지보수를 권장합니다.")

else:
    st.success("현재 비교적 안정적인 상태입니다.")

# -----------------------------
# 예측 입력 구간
# -----------------------------
st.subheader("예측에 사용된 최근 30 Cycle 구간")

start = cycle_select - seq_length

window = engine_data[
    (engine_data["cycle"]>=start) &
    (engine_data["cycle"]<=cycle_select)
]

fig3,ax3=plt.subplots(figsize=(10,4))

ax3.plot(engine_data["cycle"],engine_data[sensor],alpha=0.3)

ax3.plot(window["cycle"],window[sensor],linewidth=3)

ax3.axvline(cycle_select,linestyle="--")

ax3.set_title("Last 30 Cycles Used for Prediction")

st.pyplot(fig3)