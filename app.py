import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- 1. AI 모델 학습 (데이터 대폭 강화) ---
@st.cache_resource
def train_hof_model():
    # 학습 데이터: [HOF Monitor, WAR]
    X = np.array([
        # 헌액자군 (Elected)
        [170, 75], [200, 90], [300, 110], [150, 70], [130, 65], [180, 85], [400, 100],
        # 미헌액자군 (Not Elected)
        [90, 50], [70, 45], [110, 60], [80, 35], [50, 30], [30, 20], [100, 55], [120, 50]
    ])
    y = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # 모델 생성 (C값을 조절해 과적합 방지 및 확률 보정)
    model = LogisticRegression(class_weight='balanced', C=1.0)
    model.fit(X, y)
    return model

# --- 2. UI 구성 ---
st.set_page_config(page_title="MLB HOF AI 진단기", layout="centered")
model = train_hof_model()

st.title("🏛️ MLB 명예의 전당 AI 진단기")
st.markdown("트라웃 같은 전설적인 수치도 정확히 판정하도록 모델이 업데이트되었습니다.")

st.divider()
st.subheader("📊 선수 지표 입력")
col1, col2 = st.columns(2)

with col1:
    war = st.number_input("Career WAR (예: 86.2)", min_value=0.0, max_value=200.0, value=86.2, step=0.1)
with col2:
    hofm = st.number_input("HOF Monitor (예: 178)", min_value=0.0, max_value=500.0, value=178.0, step=1.0)

if st.button("AI 확률 분석 시작"):
    # 입력 데이터를 2차원 배열로 변환
    input_data = np.array([[hofm, war]])
    
    # 확률 계산
    prob = model.predict_proba(input_data)[0, 1] * 100
    
    # 득표율 시뮬레이션 (수식 보정)
    first_ballot = min(99.9, (hofm * 0.45) + (war * 0.3) + 10)

    st.divider()
    st.header("🔮 분석 결과")
    
    c1, c2 = st.columns(2)
    c1.metric("입성 확률", f"{prob:.1f}%")
    c2.metric("예상 득표율", f"{first_ballot:.1f}%")

    # 확률 바 표시
    st.progress(prob / 100)

    if prob >= 80:
        st.balloons()
        st.success(f"🏆 **[LOCK]** 이 선수는 무조건 갑니다! (확률: {prob:.1f}%)")
    elif prob >= 45:
        st.warning("⚾ **[BORDERLINE]** 아슬아슬한 경계선에 있습니다.")
    else:
        st.error("❌ **[UNLIKELY]** 입성 가능성이 매우 낮습니다.")

st.divider()
st.caption("※ 이 모델은 역대 명예의 전당 헌액자들의 평균치를 학습한 AI입니다.")
