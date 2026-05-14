import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- 1. AI 모델 학습 (유형별 특성 통합 학습) ---
@st.cache_resource
def train_hof_type_model():
    # [Black, Gray, HOFm, HOFs, WAR, Peak, JAWS]
    X = np.array([
        [60, 300, 350, 75, 110.0, 60.0, 85.0], # 전설 (모든 유형 합집합)
        [15, 120, 160, 48, 50.0, 38.0, 44.0],  # 임팩트형 합격 (세페다)
        [10, 150, 85, 60, 75.0, 35.0, 55.0],   # 누적형 합격 (서튼, 벨트레)
        [30, 160, 90, 45, 75.0, 55.0, 65.0],   # 세이버형 합격 (트라웃, 레인스)
        [5, 60, 50, 30, 45.0, 30.0, 37.5]      # 미달
    ])
    y = np.array([1, 1, 1, 1, 0])
    return LogisticRegression(class_weight='balanced', C=1.0).fit(X, y)

model = train_hof_type_model()

# --- 2. 기준 데이터 ---
STATS_AVG = {
    "타자": {"Black": 27, "Gray": 144, "HOFm": 100, "HOFs": 50, "WAR": 67.0, "Peak": 43.0, "JAWS": 55.0},
    "투수": {"Black": 40, "Gray": 185, "HOFm": 100, "HOFs": 50, "WAR": 73.0, "Peak": 50.0, "JAWS": 62.0}
}

# --- 3. UI 구성 ---
st.set_page_config(page_title="MLB HOF 유형별 분석기", layout="centered")
st.title("🏛️ MLB HOF 유형별 정밀 분석기")

# 상단 설정 (포지션 + 스타일)
col_p, col_s = st.columns(2)
with col_p:
    pos = st.radio("포지션", ["타자", "투수"], horizontal=True)
with col_s:
    p_style = st.selectbox("선수 스타일 선택", ["세이버메트릭스형", "누적 기록형", "임팩트/명성형"])

avg = STATS_AVG[pos]

# 지표 입력 (v2.0 레이아웃)
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

if st.button("유형별 정밀 분석 시작"):
    input_data = np.array([[black, gray, hof_m, hof_s, c_war, p_war, jaws]])
    
    # 1. AI 확률 계산
    prob = model.predict_proba(input_data)[0, 1] * 100
    
    # 2. 스타일별 가중치 적용 (득표율 시뮬레이션)
    if p_style == "세이버메트릭스형":
        # WAR, Peak, JAWS에 70% 비중
        score = (jaws / avg['JAWS'] * 50) + (p_war / avg['Peak'] * 20) + (hof_m / avg['HOFm'] * 30)
    elif p_style == "누적 기록형":
        # Career WAR, Standards, Gray Ink에 70% 비중
        score = (c_war / avg['WAR'] * 40) + (hof_s / avg['HOFs'] * 40) + (gray / avg['Gray'] * 20)
    else: # 임팩트/명성형
        # Black Ink, HOF Monitor에 70% 비중
        score = (hof_m / avg['HOFm'] * 50) + (black / avg['Black'] * 30) + (p_war / avg['Peak'] * 20)

    est_vote = min(99.9, score * 0.75 + 15)

    st.divider()
    res_col1, res_col2 = st.columns(2)
    res_col1.metric("AI 헌액 확률", f"{prob:.1f}%")
    res_col2.metric("예상 최고 득표율", f"{est_vote:.1f}%")
    st.progress(prob / 100)
    
    st.info(f"💡 현재 **[{p_style}]** 모드로 분석되었습니다. 해당 유형의 핵심 지표 가중치가 상향되었습니다.")
