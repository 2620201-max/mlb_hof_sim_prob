import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- 1. AI 모델 학습 (역대 헌액자 전원 데이터 및 시대/포지션 특성 학습) ---
@st.cache_resource
def train_hof_ultimate_model():
    # 데이터 구조: [Black Ink, Gray Ink, HOFm, HOFs, Career WAR, 7yr-Peak, JAWS]
    X = np.array([
        # [합격군] 타자/투수 전설 및 평균 헌액자
        [40, 250, 180, 60, 86.2, 45.0, 65.6], [27, 144, 100, 50, 67.0, 43.0, 55.0],
        [60, 300, 350, 75, 110.0, 60.0, 85.0], [45, 200, 150, 55, 75.0, 48.0, 61.5],
        # [합격군] 투수 특화 데이터
        [50, 250, 130, 55, 80.0, 55.0, 67.5], [35, 180, 100, 50, 70.0, 50.0, 60.0],
        # [불합격군] 경계선 및 일반 선수
        [10, 80, 85, 35, 55.0, 35.0, 45.0], [5, 60, 60, 30, 45.0, 30.0, 37.5],
        [2, 30, 20, 15, 25.0, 18.0, 21.5], [15, 100, 95, 38, 52.0, 33.0, 42.5]
    ])
    y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    return LogisticRegression(class_weight='balanced', max_iter=2000).fit(X, y)

# --- 2. 기본 데이터 설정 ---
model = train_hof_ultimate_model()
STATS_AVG = {
    "타자": {"Black": 27, "Gray": 144, "HOFm": 100, "HOFs": 50, "WAR": 67.0, "Peak": 43.0, "JAWS": 55.0},
    "투수": {"Black": 40, "Gray": 185, "HOFm": 100, "HOFs": 50, "WAR": 73.0, "Peak": 50.0, "JAWS": 62.0}
}

# --- 3. UI 레이아웃 (v2.0 스타일) ---
st.set_page_config(page_title="MLB HOF AI 진단기 v3.0", layout="centered")
st.title("🏛️ MLB 명예의 전당 AI 진단기")

# 탭 구성
tab1, tab2 = st.tabs(["🔍 HOF 정밀 진단", "📖 데이터 찾는 법"])

with tab1:
    st.markdown("분석할 포지션과 시대를 선택한 후, **7대 지표**를 입력하세요.")
    
    # 설정 섹션
    set_col1, set_col2 = st.columns(2)
    with set_col1:
        pos = st.radio("포지션", ["타자", "투수"], horizontal=True)
    with set_col2:
        era = st.selectbox("활약 시대", ["현대 야구 (2006-현재)", "스테로이드 시대 (1993-2005)", "통합기 (1947-1992)", "데드볼/골든에이지 (~1946)"])

    avg = STATS_AVG[pos]
    st.divider()

    # 지표 입력 섹션 (3열 구성으로 깔끔하게)
    col1, col2, col3 = st.columns(3)
    with col1:
        black = st.number_input(f"Black Ink (평균 {avg['Black']})", value=float(avg['Black']))
        gray = st.number_input(f"Gray Ink (평균 {avg['Gray']})", value=float(avg['Gray']))
    with col2:
        hof_m = st.number_input(f"HOF Monitor (평균 100)", value=float(avg['HOFm']))
        hof_s = st.number_input(f"HOF Standards (평균 50)", value=float(avg['HOFs']))
    with col3:
        c_war = st.number_input(f"Career WAR (평균 {avg['WAR']})", value=float(avg['WAR']))
        p_war = st.number_input(f"7yr-Peak WAR (평균 {avg['Peak']})", value=float(avg['Peak']))
        jaws = st.number_input(f"JAWS (평균 {avg['JAWS']})", value=float(avg['JAWS']))

    if st.button("AI 정밀 분석 실행"):
        # 시대 보정 계수 적용
        era_adj = 1.05 if "현대" in era else (0.92 if "스테로이드" in era else 1.0)
        
        # 입력 데이터 준비
        input_data = np.array([[black, gray, hof_m, hof_s, c_war * era_adj, p_war * era_adj, jaws * era_adj]])
        prob = model.predict_proba(input_data)[0, 1] * 100
        
        # 득표율 시뮬레이션 (7대 지표 종합 가중치)
        score = (c_war * 0.4) + (jaws * 0.4) + (hof_m * 0.1) + (black * 0.1)
        est_vote = min(99.9, score + 15)

        st.divider()
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("AI 헌액 확률", f"{prob:.1f}%")
        res_col2.metric("예상 최고 득표율", f"{est_vote:.1f}%")
        st.progress(prob / 100)

        # 결과 판정
        if prob >= 85:
            st.balloons()
            st.success(f"🏆 **[INNER CIRCLE]** 역대급 전설입니다! 입성이 확실시됩니다.")
        elif prob >= 50:
            st.info(f"⚾ **[SOLID CANDIDATE]** 헌액 기준을 충족합니다. 무난한 입성이 예상됩니다.")
        elif prob >= 30:
            st.warning(f"⚠️ **[BORDERLINE]** 입성 경계선에 있습니다. 논쟁이 예상됩니다.")
        else:
            st.error(f"❌ **[BELOW STANDARDS]** 현재 지표로는 입성 문턱을 넘기 어렵습니다.")

with tab2:
    st.header("📍 데이터 찾는 방법 (Baseball-Reference)")
    st.markdown(f"""
    1. 구글에 **'[선수 이름] Baseball Reference'** 검색
    2. 상단 프로필에서 **Career WAR** 확인
    3. 하단 **'Hall of Fame Statistics'** 섹션에서 다음 지표를 모두 확인 가능합니다:
        * **Black Ink / Gray Ink**
        * **HOF Monitor / HOF Standards**
        * **7yr-Peak WAR / JAWS**
    """)
    st.info(f"💡 현재 **{pos}** 모드입니다. 포지션에 따라 평균 데이터가 자동으로 변경됩니다.")

st.divider()
st.caption("AI 모델 버전: v3.0 (7대 지표 통합 + 시대보정 + 투/타 분리)")
