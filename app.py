import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- 1. AI 모델 (경계선 선수들을 더 촘촘하게 학습) ---
@st.cache_resource
def train_hof_realistic_model():
    # [Black, Gray, HOFm, HOFs, WAR, Peak, JAWS]
    X = np.array([
        [60, 300, 350, 75, 110.0, 60.0, 85.0], # 전설 (99%)
        [40, 185, 165, 50, 46.2, 41.2, 43.7],  # 디지 딘 (임팩트 극강이나 누적 부족 -> 60~70% 타겟)
        [15, 120, 150, 48, 50.0, 38.0, 44.0],  # 세페다 (임팩트형 합격 -> 50~60% 타겟)
        [10, 150, 80, 55, 75.0, 35.0, 55.0],   # 누적형 합격 (70%대)
        [5, 80, 70, 35, 55.0, 30.0, 42.5],     # 세이버형 불합격 (로프턴 등 -> 30~40% 타겟)
        [2, 40, 40, 25, 35.0, 25.0, 30.0]      # 일반 선수 (5% 미만)
    ])
    y = np.array([1, 1, 1, 1, 0, 0])
    # 규제(C)를 더 강화하여 특정 지표 하나로 확률이 튀는 것을 방지
    return LogisticRegression(class_weight='balanced', C=0.05, max_iter=2000).fit(X, y)

model = train_hof_realistic_model()

# --- 2. 기준 데이터 ---
STATS_AVG = {
    "타자": {"Black": 27, "Gray": 144, "HOFm": 100, "HOFs": 50, "WAR": 67.0, "Peak": 43.0, "JAWS": 55.0},
    "투수": {"Black": 40, "Gray": 185, "HOFm": 165, "HOFs": 50, "WAR": 73.0, "Peak": 50.0, "JAWS": 62.0}
}

# --- 3. UI 구성 ---
st.set_page_config(page_title="MLB HOF 현실 보정 분석기", layout="centered")
st.title("🏛️ MLB HOF 현실 보정 AI 분석기")

pos = st.radio("포지션", ["타자", "투수"], horizontal=True)
p_style = st.selectbox("선수 스타일", ["세이버메트릭스형", "누적 기록형", "임팩트/명성형"])
avg = STATS_AVG[pos]

st.divider()
c1, c2, c3 = st.columns(3)
with c1:
    black = st.number_input("Black Ink", value=float(avg['Black']))
    gray = st.number_input("Gray Ink", value=float(avg['Gray']))
with c2:
    hof_m = st.number_input("HOF Monitor", value=float(avg['HOFm']))
    hof_s = st.number_input("HOF Standards", value=float(avg['HOFs']))
with c3:
    c_war = st.number_input("Career WAR", value=float(avg['WAR']))
    p_war = st.number_input("7yr-Peak WAR", value=float(avg['Peak']))
    jaws = st.number_input("JAWS", value=float(avg['JAWS']))

if st.button("현실 보정 분석 실행"):
    input_data = np.array([[black, gray, hof_m, hof_s, c_war, p_war, jaws]])
    prob = model.predict_proba(input_data)[0, 1] * 100
    
    # [핵심] 득표율 시뮬레이션 보정 (하한선 필터링)
    # 아무리 임팩트가 좋아도 WAR가 평균의 70% 미만이면 감점
    war_penalty = 1.0
    if c_war < avg['WAR'] * 0.7:
        war_penalty = 0.8 # 20% 페널티
    
    if p_style == "세이버메트릭스형":
        score = (jaws / avg['JAWS'] * 50) + (p_war / avg['Peak'] * 30) + (hof_m / avg['HOFm'] * 20)
    elif p_style == "누적 기록형":
        score = (c_war / avg['WAR'] * 50) + (hof_s / avg['HOFs'] * 30) + (gray / avg['Gray'] * 20)
    else: # 임팩트/명성형
        score = (hof_m / avg['HOFm'] * 40) + (black / avg['Black'] * 30) + (p_war / avg['Peak'] * 30)

    # 최종 득표율 산출 (페널티 적용)
    est_vote = min(99.9, (score * 70 + 10) * war_penalty)

    st.divider()
    res_col1, res_col2 = st.columns(2)
    res_col1.metric("AI 헌액 확률", f"{prob:.1f}%")
    res_col2.metric("예상 최고 득표율", f"{est_vote:.1f}%")
    st.progress(prob / 100)
    
    if est_vote < 75 and prob > 80:
        st.warning("⚠️ **주의:** 임팩트는 압도적이나 누적 수치가 낮아 기자단 투표에서 고전할 수 있습니다.")
