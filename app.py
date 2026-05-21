import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- 1. 비율 기반 AI 통합 모델 (거품 스탯 파훼 적용) ---
@st.cache_resource
def train_hof_adjusted_model():
    # Features: [Ink_Ratio, Adjusted_Monitor, Standards_Ratio, WAR_Ratio, JAWS_Ratio]
    X = np.array([
        [2.5, 2.5, 1.5, 1.8, 1.8],  # 역대급 레전드 (100% 합격)
        [1.2, 1.2, 1.1, 1.1, 1.1],  # 스탠다드 명전 (안정권)
        [0.8, 1.0, 1.0, 1.0, 1.0],  # 턱걸이 명전
        [0.4, 0.7, 1.1, 0.79, 0.77],# [시뮬레이션 학습] 모니터 거품이 제거된 빈껍데기 선수 (단호한 0)
        [0.4, 0.7, 0.9, 1.1, 1.1],  # 세이버 달링
        [1.5, 1.2, 0.8, 0.8, 0.9],  # 짧고 굵은 임팩트
        [0.3, 0.6, 0.9, 0.6, 0.5]   # 누적만 챙긴 올스타 (탈락)
    ])
    y = np.array([1, 1, 1, 0, 1, 1, 0])
    
    # AI 규제 강도를 높여 미달 스탯에 얄짤없이 반응하게 세팅 (C=0.5)
    model = LogisticRegression(class_weight='balanced', C=0.5, max_iter=3000)
    model.fit(X, y)
    return model

model = train_hof_adjusted_model()

HOF_GLOBAL_AVG = {"Black": 27.0, "Gray": 144.0, "HOFm": 100.0, "HOFs": 50.0}

POSITION_STATS = {
    "포수 (C)": {"Name": "포수", "WAR": 53.8, "Peak": 34.4, "JAWS": 44.1, "Type": "타자"},
    "1루수 (1B)": {"Name": "1루수", "WAR": 65.5, "Peak": 41.8, "JAWS": 53.7, "Type": "타자"},
    "2루수 (2B)": {"Name": "2루수", "WAR": 69.4, "Peak": 44.3, "JAWS": 56.9, "Type": "타자"},
    "3루수 (3B)": {"Name": "3루수", "WAR": 68.3, "Peak": 42.9, "JAWS": 55.6, "Type": "타자"},
    "유격수 (SS)": {"Name": "유격수", "WAR": 66.8, "Peak": 43.0, "JAWS": 54.9, "Type": "타자"},
    "좌익수 (LF)": {"Name": "좌익수", "WAR": 65.2, "Peak": 41.5, "JAWS": 53.4, "Type": "타자"},
    "중견수 (CF)": {"Name": "중견수", "WAR": 71.3, "Peak": 44.6, "JAWS": 57.9, "Type": "타자"},
    "우익수 (RF)": {"Name": "우익수", "WAR": 71.5, "Peak": 41.9, "JAWS": 56.7, "Type": "타자"},
    "선발투수 (SP)": {"Name": "선발", "WAR": 73.0, "Peak": 49.9, "JAWS": 61.4, "Type": "투수"},
    "구원투수 (RP)": {"Name": "구원", "WAR": 39.5, "Peak": 27.0, "JAWS": 33.3, "Type": "투수"}
}

st.set_page_config(page_title="MLB HOF AI 통합 진단기 v5.1", layout="centered")
st.title("🏛️ MLB HOF AI 통합 진단기 (v5.1 - 자가 시뮬레이션 검증판)")

tab1, tab2 = st.tabs(["🔍 HOF 정밀 진단", "📖 가이드"])

with tab1:
    selected_pos = st.selectbox("🏃 선수의 주 수비 포지션을 선택하세요", list(POSITION_STATS.keys()))
    era = st.selectbox("선수의 주 활약 연대(시대) 선택", ["데드볼/골든에이지 (~1946)", "통합 및 확장기 (1947-1992)", "스테로이드 시대 (1993-2005)", "현대 세이버 야구 (2006-현재)"])
    
    avg = POSITION_STATS[selected_pos]
    p_name, pos_type = avg["Name"], avg["Type"]
    
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        black = st.number_input(f"Black Ink (평균: {int(HOF_GLOBAL_AVG['Black'])})", value=float(HOF_GLOBAL_AVG["Black"]), step=1.0)
        gray = st.number_input(f"Gray Ink (평균: {int(HOF_GLOBAL_AVG['Gray'])})", value=float(HOF_GLOBAL_AVG["Gray"]), step=1.0)
    with c2:
        hof_m = st.number_input(f"HOF Monitor (평균: {int(HOF_GLOBAL_AVG['HOFm'])})", value=float(HOF_GLOBAL_AVG["HOFm"]), step=1.0)
        hof_s = st.number_input(f"HOF Standards (평균: {int(HOF_GLOBAL_AVG['HOFs'])})", value=float(HOF_GLOBAL_AVG["HOFs"]), step=1.0)
    with c3:
        c_war = st.number_input(f"Career WAR ({p_name} 평균: {avg['WAR']})", value=float(avg["WAR"]), step=0.1)
        p_war = st.number_input(f"7yr-Peak WAR ({p_name} 평균: {avg['Peak']})", value=float(avg["Peak"]), step=0.1)
        jaws = st.number_input(f"JAWS ({p_name} 평균: {avg['JAWS']})", value=float(avg["JAWS"]), step=0.1)

    if st.button("포지션 맞춤 AI 분석 실행"):
        ink_ratio = ((black / HOF_GLOBAL_AVG['Black']) + (gray / HOF_GLOBAL_AVG['Gray'])) / 2.0
        monitor_ratio = hof_m / HOF_GLOBAL_AVG['HOFm']
        standards_ratio = hof_s / HOF_GLOBAL_AVG['HOFs']
        war_ratio = c_war / avg['WAR']
        jaws_ratio = jaws / avg['JAWS']
        
        # [핵심] 세이버(WAR, JAWS)가 평균 이하면, 모니터 점수도 그 비율만큼 강제로 후려침 (거품 제거)
        saber_penalty = min(1.0, (war_ratio + jaws_ratio) / 2.0)
        adjusted_monitor = monitor_ratio * saber_penalty
        
        input_ratios = np.array([[ink_ratio, adjusted_monitor, standards_ratio, war_ratio, jaws_ratio]])
        
        # 거품을 뺀 상태로 AI가 정직하게 확률 판단
        final_prob = model.predict_proba(input_ratios)[0, 1] * 100

        # 투표 점수 계산 시에도 동일한 징벌 적용
        if pos_type == "투수":
            is_accumulation_monster = (hof_s >= 48) or (selected_pos == "구원투수 (RP)" and hof_m >= 120)
        else:
            is_accumulation_monster = (hof_s >= 55) or (selected_pos in ["포수 (C)", "유격수 (SS)"] and hof_s >= 42)

        sabermetrics_base = (jaws_ratio * 0.6) + (war_ratio * 0.4)
        
        if era == "현대 세이버 야구 (2006-현재)":
            vote_score = (sabermetrics_base * 60) + (adjusted_monitor * 25) + (standards_ratio * 15)
        elif era == "스테로이드 시대 (1993-2005)":
            vote_score = (sabermetrics_base * 40) + (adjusted_monitor * 20) + (standards_ratio * 40)
        elif era == "데드볼/골든에이지 (~1946)":
            vote_score = (sabermetrics_base * 20) + (adjusted_monitor * 55) + (standards_ratio * 25)
        else:
            vote_score = (sabermetrics_base * 40) + (adjusted_monitor * 40) + (standards_ratio * 20)
        
        if is_accumulation_monster:
            vote_score *= 1.12
        else:
            if war_ratio < 1.0 or jaws_ratio < 1.0:
                vote_score *= min(war_ratio, jaws_ratio)

        # 득표율 변환
        if vote_score >= 100:
            est_vote = 75.0 + (24.9 / (1.0 + np.exp(-0.05 * (vote_score - 100))))
        else:
            est_vote = 5.0 + (70.0 / (1.0 + np.exp(-0.05 * (vote_score - 60))))
            
        est_vote = min(99.9, max(0.0, est_vote))

        st.divider()
        res_col1, res_col2 = st.columns(2)
        res_col1.metric(f"최종 헌액 확률 ({selected_pos} 기준)", f"{final_prob:.1f}%")
        res_col2.metric("예상 최고 득표율 (기자단 경향 반영)", f"{est_vote:.1f}%")
        st.progress(min(100.0, max(0.0, final_prob)) / 100)

        if est_vote >= 95.0:
            st.success(f"👑 **[FIRST BALLOT LOCK]** 첫해 만장일치급 입성 확실.")
        elif est_vote >= 75.0:
            st.info(f"⚾ **[SAFE ZONE]** 명예의 전당 안정권(75%) 획득.")
        elif est_vote >= 60.0:
            st.warning(f"⚠️ **[BORDERLINE - HIGH]** 베테랑 위원회 구제 유력.")
        elif est_vote >= 5.0:
            st.error(f"❌ **[ONE AND DONE]** 기준치 미달, 광탈 위험.")
        else:
            st.error(f"❌ **[OUT OF RANGE]** 후보 등록조차 어려운 스탯입니다.")

with tab2:
    st.header("📊 가이드 생략")
