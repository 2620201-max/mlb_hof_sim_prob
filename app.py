import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- 1. AI 모델 학습 (고정 데이터) ---
@st.cache_resource
def train_hof_model():
    # 학습 데이터: [HOF Monitor, WAR]
    X = np.array([
        [165, 71], [320, 106], [170, 80], [110, 75], 
        [95, 51], [85, 40], [50, 30], [250, 110], [200, 90]
    ])
    y = np.array([1, 1, 0, 0, 0, 0, 0, 1, 1]) # 1: 헌액, 0: 탈락
    model = LogisticRegression(class_weight='balanced').fit(X, y)
    return model

# --- 2. UI 구성 ---
st.set_page_config(page_title="MLB HOF 확률 진단기", layout="centered")
model = train_hof_model()

st.title("🏛️ MLB 명예의 전당 AI 진단기")
st.markdown("사이트 차단 문제로 인해 **직접 수치를 입력하는 방식**으로 긴급 변경되었습니다.")

# 사용자 입력 섹션
st.divider()
st.subheader("선수 지표 입력")
col1, col2 = st.columns(2)

with col1:
    war = st.number_input("Career WAR (예: 트라웃 86.2)", min_value=0.0, max_value=200.0, value=70.0, step=0.1)
with col2:
    hofm = st.number_input("HOF Monitor (예: 트라웃 178)", min_value=0.0, max_value=500.0, value=100.0, step=1.0)

# 분석 버튼
if st.button("AI 확률 분석 시작"):
    # AI 확률 계산
    prob = model.predict_proba([[hofm, war]])[0, 1] * 100
    
    # 득표율 시뮬레이션
    first_ballot = min(99.9, (hofm * 0.4) + (war * 0.3) + 15)

    st.divider()
    st.header("🔮 분석 결과")
    
    # 메트릭 표시
    c1, c2 = st.columns(2)
    c1.metric("입성 확률", f"{prob:.1f}%")
    c2.metric("예상 득표율", f"{first_ballot:.1f}%")

    st.progress(prob / 100)

    # 판정 메시지
    if prob >= 75:
        st.balloons()
        st.success("🏆 **[LOCK]** 이 선수는 명예의 전당 입성이 확실시됩니다!")
    elif prob >= 40:
        st.warning("⚾ **[BORDERLINE]** 입성 가능성이 있는 경계선 선수입니다.")
    else:
        st.error("❌ **[UNLIKELY]** 현재 지표로는 입성이 어렵습니다.")

    st.info(f"참고: 입력하신 WAR {war}와 HOF Monitor {hofm}을 기반으로 AI가 역대 헌액자 데이터와 대조한 결과입니다.")

# 지표 찾는 법 안내
st.divider()
with st.expander("💡 선수의 WAR와 HOF Monitor는 어디서 보나요?"):
    st.write("1. 구글에 '선수이름 + Baseball Reference' 검색")
    st.write("2. 해당 사이트의 'Leaderboards & Awards' 섹션에서 **Hall of Fame Monitor** 확인")
    st.write("3. 상단 메인 섹션에서 **WAR** 확인")
