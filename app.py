import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- 1. AI 모델 학습 (다양한 유형의 HOFer 학습) ---
@st.cache_resource
def train_hof_balanced_model():
    # [Black, Gray, HOFm, HOFs, WAR, Peak, JAWS]
    X = np.array([
        [60, 300, 350, 75, 110.0, 60.0, 85.0], # 1. 올타임 레전드 (누적+임팩트+세이버 완벽)
        [30, 160, 150, 55, 75.0, 45.0, 60.0],  # 2. 정석적인 헌액자 (모든 지표 평균 이상)
        [15, 120, 160, 48, 50.0, 38.0, 44.0],  # 3. 명성형 헌액자 (세페다 등: WAR 낮으나 HOFm 높음)
        [10, 130, 80, 55, 70.0, 35.0, 52.5],   # 4. 누적형 헌액자 (임팩트 낮으나 통산 WAR 높음)
        [12, 90, 90, 35, 62.0, 38.0, 50.0],    # 5. 세이버형 경계선 (WAR는 좋으나 수상 실적 부족)
        [5, 60, 50, 30, 45.0, 30.0, 37.5],     # 6. 미달자 (모든 지표 부족)
        [20, 100, 110, 40, 55.0, 42.0, 48.5]   # 7. 전성기형 경계선 (전성기는 좋으나 누적 부족)
    ])
    y = np.array([1, 1, 1, 1, 0, 0, 1]) # 5번 같은 경우 실제 투표에서 고전하는 경향 반영
    
    # 모델의 안정성을 위해 C값과 solver 최적화
    model = LogisticRegression(class_weight='balanced', C=0.2, max_iter=2000)
    model.fit(X, y)
    return model

model = train_hof_balanced_model()

# --- 2. 공식 기준 데이터 ---
STATS_AVG = {
    "타자": {"Black": 27, "Gray": 144, "HOFm": 100, "HOFs": 50, "WAR": 67.0, "Peak": 43.0, "JAWS": 55.0},
    "투수": {"Black": 40, "Gray": 185, "HOFm": 100, "HOFs": 50, "WAR": 73.0, "Peak": 50.0, "JAWS": 62.0}
}

# --- 3. UI (v2.0 UI 유지) ---
st.set_page_config(page_title="MLB HOF 통합 분석기 v3.3", layout="centered")
st.title("🏛️ MLB 명예의 전당 AI 진단기")

pos = st.radio("포지션", ["타자", "투수"], horizontal=True)
avg = STATS_AVG[pos]

tab1, tab2 = st.tabs(["🔍 정밀 진단", "📊 데이터 가이드"])

with tab1:
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

    if st.button("AI 분석 실행"):
        input_data = np.array([[black, gray, hof_m, hof_s, c_war, p_war, jaws]])
        
        # 확률 계산
        prob = model.predict_proba(input_data)[0, 1] * 100
        
        # 득표율 시뮬레이션 (3대 축 밸런스 점수)
        # 세이버(40%) + 명성(40%) + 누적(20%) 비율로 조정
        sabermetrics = (jaws / avg['JAWS']) * 40
        fame = (hof_m / avg['HOFm']) * 40
        longevity = (hof_s / avg['HOFs']) * 20
        
        est_vote = min(99.9, (sabermetrics + fame + longevity) / 100 * 75 + 10)

        st.divider()
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("AI 헌액 확률", f"{prob:.1f}%")
        res_col2.metric("예상 최고 득표율", f"{est_vote:.1f}%")
        st.progress(prob / 100)

        if est_vote >= 75:
            st.success("🏆 **입성 안정권:** 지표의 밸런스가 매우 좋습니다.")
        elif est_vote >= 50:
            st.warning("⚾ **경합 후보:** 한 분야는 강하지만 다른 분야 보완이 필요합니다.")
        else:
            st.error("❌ **입성 어려움:** 역대 헌액자 평균치에 미달합니다.")
