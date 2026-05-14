import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- 1. AI 모델 학습 (투수/타자 통합 학습셋) ---
@st.cache_resource
def train_hof_dual_model():
    # 데이터: [Black, Gray, HOFm, HOFs, WAR, Peak, JAWS]
    # 타자(y=1)와 투수(y=1)의 다양한 합격 케이스 학습
    X = np.array([
        [27, 144, 100, 50, 67.0, 43.0, 55.0], # 타자 평균
        [40, 185, 100, 50, 73.0, 50.0, 62.0], # 투수 평균
        [60, 300, 350, 75, 110.0, 60.0, 85.0], # 레전드 타자
        [70, 350, 250, 80, 100.0, 55.0, 78.0], # 레전드 투수
        [15, 90, 110, 45, 55.0, 38.0, 46.0],  # 하한선 타자
        [20, 110, 90, 40, 60.0, 45.0, 52.0],  # 하한선 투수
        [5, 50, 50, 25, 40.0, 25.0, 32.5]     # 탈락자
    ])
    y = np.array([1, 1, 1, 1, 1, 1, 0])
    return LogisticRegression(class_weight='balanced').fit(X, y)

# --- 2. 데이터 세팅 ---
model = train_hof_dual_model()

# 공식 평균 데이터 (Baseball-Reference 기준)
STATS_AVG = {
    "타자": {"Black": 27, "Gray": 144, "HOFm": 100, "HOFs": 50, "WAR": 67.0, "Peak": 43.0, "JAWS": 55.0},
    "투수": {"Black": 40, "Gray": 185, "HOFm": 100, "HOFs": 50, "WAR": 73.0, "Peak": 50.0, "JAWS": 62.0}
}

# --- 3. UI ---
st.set_page_config(page_title="MLB HOF 투/타 분리 분석기", layout="wide")
st.title("🏛️ MLB HOF 투수/타자 정밀 분석기")

# 포지션 선택
pos = st.radio("분석할 포지션을 선택하세요", ["타자", "투수"], horizontal=True)
avg = STATS_AVG[pos]

tab1, tab2 = st.tabs(["🔍 정밀 진단", "📊 포지션별 평균"])

with tab1:
    st.info(f"현재 **[{pos}]** 모드입니다. 해당 포지션의 기준치로 분석합니다.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        black = st.number_input(f"Black Ink (평균 {avg['Black']})", 0, 150, avg['Black'])
        gray = st.number_input(f"Gray Ink (평균 {avg['Gray']})", 0, 600, avg['Gray'])
    with c2:
        hof_m = st.number_input(f"HOF Monitor (기준 100)", 0.0, 600.0, float(avg['HOFm']))
        hof_s = st.number_input(f"HOF Standards (기준 50)", 0.0, 100.0, float(avg['HOFs']))
    with c3:
        c_war = st.number_input(f"Career WAR (평균 {avg['WAR']})", 0.0, 200.0, avg['WAR'])
        p_war = st.number_input(f"7yr-Peak WAR (평균 {avg['Peak']})", 0.0, 100.0, avg['Peak'])
        jaws = st.number_input(f"JAWS (평균 {avg['JAWS']})", 0.0, 150.0, avg['JAWS'])

    if st.button(f"{pos} AI 정밀 판정 시작"):
        # 입력 데이터 준비
        input_data = np.array([[black, gray, hof_m, hof_s, c_war, p_war, jaws]])
        prob = model.predict_proba(input_data)[0, 1] * 100
        
        st.divider()
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric(f"{pos} 최종 헌액 확률", f"{prob:.1f}%")
            st.progress(prob / 100)
        
        with res_col2:
            # 비교 테이블
            comp_data = {
                "지표": ["Black Ink", "Gray Ink", "HOFm", "HOFs", "WAR", "Peak", "JAWS"],
                f"{pos} 평균": list(avg.values()),
                "입력값": [black, gray, hof_m, hof_s, c_war, p_war, jaws]
            }
            df = pd.DataFrame(comp_data)
            df["비교"] = df.apply(lambda x: "✅ 우세" if x["입력값"] >= x[f"{pos} 평균"] else "❌ 열세", axis=1)
            st.table(df)

with tab2:
    st.subheader(f"📊 {pos} 명예의 전당 입성자 평균 지표")
    st.write(f"투수와 타자는 입성 기준이 다릅니다. 아래는 Baseball-Reference에서 제공하는 공식 평균치입니다.")
    st.json(avg)
