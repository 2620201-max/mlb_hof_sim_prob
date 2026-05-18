import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- 1. AI 모델 학습 (모든 선수 유형을 아우르는 부드러운 경계선 학습) ---
@st.cache_resource
def train_hof_seamless_model():
    # 데이터 구조: [Black Ink, Gray Ink, HOFm, HOFs, Career WAR, 7yr-Peak, JAWS]
    X = np.array([
        [60, 300, 350, 75, 110.0, 60.0, 85.0], # 1. 올타임 레전드 (트라웃, 푸홀스 등) -> 95%+
        [30, 160, 150, 55, 75.0, 45.0, 60.0],  # 2. 정석적인 헌액자 -> 85-90%
        [40, 185, 165, 50, 46.2, 41.2, 43.7],  # 3. 디지 딘형 (임팩트 극강, 누적 부족) -> 65-75%
        [15, 120, 150, 48, 50.0, 38.0, 44.0],  # 4. 세페다형 (임팩트형 경계선) -> 55-65%
        [10, 140, 75, 55, 65.0, 35.0, 50.0],   # 5. 누적형 헌액자 (꾸준함) -> 60-70%
        [12, 90, 90, 35, 62.0, 38.0, 50.0],    # 6. 세이버형 경계선 (로프턴 등) -> 40-50%
        [5, 60, 50, 30, 45.0, 30.0, 37.5]      # 7. 명전 미달자 -> 15% 이하
    ])
    y = np.array([1, 1, 1, 1, 1, 0, 0])
    
    # C=0.01로 규제를 극대화하여 기울기를 아주 완만하고 부드럽게(Smooth) 만듦
    # 이로 인해 하나의 지표가 폭발해도 확률이 급격하게 100%로 쏠리지 않음
    model = LogisticRegression(class_weight='balanced', C=0.01, max_iter=2000)
    model.fit(X, y)
    return model

model = train_hof_seamless_model()

# --- 2. 포지션별 기준 데이터 ---
STATS_AVG = {
    "타자": {"Black": 27, "Gray": 144, "HOFm": 100, "HOFs": 50, "WAR": 67.0, "Peak": 43.0, "JAWS": 55.0},
    "투수": {"Black": 40, "Gray": 185, "HOFm": 100, "HOFs": 50, "WAR": 73.0, "Peak": 50.0, "JAWS": 62.0}
}

# --- 3. UI 구성 (v2.0 UI 기반의 통합형) ---
st.set_page_config(page_title="MLB HOF AI 통합 진단기", layout="centered")
st.title("🏛️ MLB 명예의 전당 AI 통합 진단기")

# 투/타 선택만 남기고 스타일 선택은 제거 (통합)
pos = st.radio("포지션 선택", ["타자", "투수"], horizontal=True)
avg = STATS_AVG[pos]

tab1, tab2 = st.tabs(["🔍 HOF 정밀 진단", "📖 데이터 찾는 법"])

with tab1:
    st.markdown("선수의 7대 지표를 입력하세요. AI가 유형을 스스로 분석하여 판정합니다.")
    st.divider()
    
    c1, c2, c3 = st.columns(3)
    with c1:
        black = st.number_input(f"Black Ink (평균 {avg['Black']})", value=float(avg['Black']))
        gray = st.number_input(f"Gray Ink (평균 {avg['Gray']})", value=float(avg['Gray']))
    with c2:
        hof_m = st.number_input(f"HOF Monitor (평균 100)", value=float(avg['HOFm']))
        hof_s = st.number_input(f"HOF Standards (평균 50)", value=float(avg['HOFs']))
    with c3:
        c_war = st.number_input(f"Career WAR (평균 {avg['WAR']})", value=float(avg['WAR']))
        p_war = st.number_input(f"7yr-Peak WAR (평균 {avg['Peak']})", value=float(avg['Peak']))
        jaws = st.number_input(f"JAWS (평균 {avg['JAWS']})", value=float(avg['JAWS']))

    if st.button("AI 통합 분석 실행"):
        input_data = np.array([[black, gray, hof_m, hof_s, c_war, p_war, jaws]])
        
        # 1. 부드러운 회귀 모델 기반 확률 계산
        raw_prob = model.predict_proba(input_data)[0, 1] * 100
        
        # [수학적 제어] 누적치 하한선 필터 (Soft Cap)
        # 통산 WAR가 역대 평균의 75% 미만인 선수는 아무리 임팩트가 좋아도 확률을 최대 78%로 제한
        if c_war < avg['WAR'] * 0.75:
            final_prob = min(raw_prob, 78.5)
        else:
            final_prob = raw_prob
            
        # 2. 종합 득표율 시뮬레이션 (세이버 40% + 명성 40% + 누적 20%의 황금 비율 통합)
        sabermetrics = (jaws / avg['JAWS']) * 40
        fame = (hof_m / avg['HOFm']) * 40
        longevity = (hof_s / avg['HOFs']) * 20
        
        # 누적 기록 결핍에 따른 득표율 하락 수식 보정
        vote_score = sabermetrics + fame + longevity
        if c_war < avg['WAR'] * 0.7:
            vote_score *= 0.88  # 누적 부족에 대한 감점

        est_vote = min(99.9, (vote_score / 100) * 70 + 15)

        st.divider()
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("AI 헌액 확률", f"{final_prob:.1f}%")
        res_col2.metric("예상 최고 득표율", f"{est_vote:.1f}%")
        st.progress(final_prob / 100)

        # 결과 판정
        if final_prob >= 90 and est_vote >= 85:
            st.balloons()
            st.success("🏆 **[INNER CIRCLE]** 반론이 불가능한 역사적 전설입니다.")
        elif final_prob >= 65:
            st.info("⚾ **[STRONG CANDIDATE]** 명예의 전당 입성 기준을 든든하게 만족합니다.")
        elif final_prob >= 40:
            st.warning("⚠️ **[BORDERLINE]** 특정 강점은 있으나 약점도 뚜렷하여 치열한 경합이 예상됩니다.")
        else:
            st.error("❌ **[BELOW STANDARDS]** 명예의 전당 평균 수준에 미치지 못합니다.")

with tab2:
    st.header("📍 데이터 찾는 방법")
    st.markdown("""
    가장 정확한 데이터는 **Baseball-Reference** 사이트에서 확인할 수 있습니다.
    
    1. 구글에 선수 이름 검색 (예: `Mike Trout Baseball Reference`)
    2. 선수 페이지 중간의 **'Hall of Fame Statistics'** 표를 찾습니다.
    3. 해당 표에 있는 **Black Ink, Gray Ink, HOF Monitor, HOF Standards, JAWS**를 그대로 입력하세요.
    """)
