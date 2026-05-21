import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- 1. AI 통합 모델 학습 ---
@st.cache_resource
def train_hof_ultimate_model():
    X = np.array([
        [120, 450, 500, 95, 140.0, 70.0, 105.0], # 1. 지구 파괴급 신 (루스, 존슨, 메이스)
        [60, 280, 320, 75, 100.0, 58.0, 79.0],  # 2. 올타임 슈퍼 레전드 (푸홀스, 마덕스)
        [40, 200, 200, 60, 80.0, 48.0, 64.0],   # 3. 확실한 First Ballot (트라웃, 벨트레)
        [25, 140, 130, 50, 65.0, 40.0, 52.5],   # 4. 정석적인 헌액자
        [40, 185, 165, 50, 46.2, 41.2, 43.7],   # 5. 디지 딘형 (임팩트형)
        [12, 120, 110, 45, 55.0, 36.0, 45.5],   # 6. 세페다형 / 세이버 경계선
        [5, 60, 50, 30, 40.0, 28.0, 34.0]       # 7. 명전 미달자
    ])
    y = np.array([1, 1, 1, 1, 1, 0, 0])
    
    model = LogisticRegression(class_weight='balanced', C=0.005, max_iter=3000)
    model.fit(X, y)
    return model

model = train_hof_ultimate_model()

# --- 2. 명예의 전당 통계 전체 평균값 정의 ---
HOF_GLOBAL_AVG = {
    "Black": 27,   
    "Gray": 144,   
    "HOFm": 100,   
    "HOFs": 50     
}

# AI 모델의 기저 기준 스케일 (기존 v4.4 타자/투수 표준 베이스라인)
MODEL_BASE_WAR = {"타자": 67.0, "투수": 73.0}
MODEL_BASE_PEAK = {"타자": 43.0, "투수": 50.0}
MODEL_BASE_JAWS = {"타자": 55.0, "투수": 62.0}

# --- 3. 명예의 전당 입성자 실제 포지션별 세이버메트릭스 데이터셋 ---
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

# --- 4. UI 구성 ---
st.set_page_config(page_title="MLB HOF AI 통합 진단기 v4.92", layout="centered")
st.title("🏛️ MLB HOF AI 통합 진단기 (v4.92 - 상위권 득표율 교정판)")

tab1, tab2 = st.tabs(["🔍 HOF 정밀 진단", "📖 가이드 (데이터 검색 및 시대 설명)"])

with tab1:
    selected_pos = st.selectbox(
        "🏃 선수의 주 수비 포지션을 선택하세요", 
        list(POSITION_STATS.keys())
    )
    
    era = st.selectbox(
        "선수의 주 활약 연대(시대) 선택",
        ["데드볼/골든에이지 (~1946)", "통합 및 확장기 (1947-1992)", "스테로이드 시대 (1993-2005)", "현대 세이버 야구 (2006-현재)"]
    )
    
    avg = POSITION_STATS[selected_pos]
    p_name = avg["Name"]   
    pos_type = avg["Type"] 
    
    st.divider()
    
    c1, c2, c3 = st.columns(3)
    with c1:
        black = st.number_input(f"Black Ink (전체 평균: {HOF_GLOBAL_AVG['Black']})", value=float(HOF_GLOBAL_AVG["Black"]), step=1.0)
        gray = st.number_input(f"Gray Ink (전체 평균: {HOF_GLOBAL_AVG['Gray']})", value=float(HOF_GLOBAL_AVG["Gray"]), step=1.0)
    with c2:
        hof_m = st.number_input(f"HOF Monitor (전체 평균: {HOF_GLOBAL_AVG['HOFm']})", value=float(HOF_GLOBAL_AVG["HOFm"]), step=1.0)
        hof_s = st.number_input(f"HOF Standards (전체 평균: {HOF_GLOBAL_AVG['HOFs']})", value=float(HOF_GLOBAL_AVG["HOFs"]), step=1.0)
    with c3:
        c_war = st.number_input(f"Career WAR ({p_name} 평균: {avg['WAR']})", value=float(avg["WAR"]), step=0.1)
        p_war = st.number_input(f"7yr-Peak WAR ({p_name} 평균: {avg['Peak']})", value=float(avg["Peak"]), step=0.1)
        jaws = st.number_input(f"JAWS ({p_name} 평균: {avg['JAWS']})", value=float(avg["JAWS"]), step=0.1)

    if st.button("포지션 맞춤 AI 분석 실행"):
        scaled_c_war = (c_war / avg['WAR']) * MODEL_BASE_WAR[pos_type]
        scaled_p_war = (p_war / avg['Peak']) * MODEL_BASE_PEAK[pos_type]
        scaled_jaws = (jaws / avg['JAWS']) * MODEL_BASE_JAWS[pos_type]
        
        input_data = np.array([[black, gray, hof_m, hof_s, scaled_c_war, scaled_p_war, scaled_jaws]])
        
        raw_prob = model.predict_proba(input_data)[0, 1] * 100
        
        if pos_type == "투수":
            is_accumulation_monster = (hof_s >= 48) or (selected_pos == "구원투수 (RP)" and hof_m >= 120)
            war_threshold = avg['WAR'] * 0.75 if selected_pos == "선발투수 (SP)" else avg['WAR'] * 0.65
        else:
            is_accumulation_monster = (hof_s >= 55) or (selected_pos in ["포수 (C)", "유격수 (SS)"] and hof_s >= 42)
            war_threshold = avg['WAR'] * 0.75
            
        if c_war < war_threshold and not is_accumulation_monster:
            final_prob = min(raw_prob, 75.0)
        else:
            final_prob = raw_prob
            
        if era == "현대 세이버 야구 (2006-현재)":
            sabermetrics = (jaws / avg['JAWS']) * 60
            fame = (hof_m / HOF_GLOBAL_AVG['HOFm']) * 25
            longevity = (hof_s / HOF_GLOBAL_AVG['HOFs']) * 15
        elif era == "스테로이드 시대 (1993-2005)":
            sabermetrics = (jaws / avg['JAWS']) * 40
            fame = (hof_m / HOF_GLOBAL_AVG['HOFm']) * 20
            longevity = (hof_s / HOF_GLOBAL_AVG['HOFs']) * 40
        elif era == "데드볼/골든에이지 (~1946)":
            sabermetrics = (jaws / avg['JAWS']) * 20
            fame = (hof_m / HOF_GLOBAL_AVG['HOFm']) * 55
            longevity = (hof_s / HOF_GLOBAL_AVG['HOFs']) * 25
        else:
            sabermetrics = (jaws / avg['JAWS']) * 40
            fame = (hof_m / HOF_GLOBAL_AVG['HOFm']) * 40
            longevity = (hof_s / HOF_GLOBAL_AVG['HOFs']) * 20
        
        vote_score = sabermetrics + fame + longevity
        
        if is_accumulation_monster:
            vote_score *= 1.12
        elif c_war < avg['WAR'] * 0.7:
            vote_score *= 0.85

        # ------------------ [핵심 교정 수식 변경] ------------------
        # 선형 수식을 버리고, 상위권으로 갈수록 압축 저항이 발생하는 시그모이드 감쇠 곡선 적용
        if vote_score >= 100:
            # 평균 이상 구간: 75%에서 출발하여 vote_score가 대폭 높아야 95%~99%에 도달하도록 감쇠 조정
            est_vote = 75.0 + (24.9 / (1.0 + np.exp(-0.05 * (vote_score - 100))))
        else:
            # 평균 미만 구간: 하락 곡선을 자연스럽게 연결
            est_vote = 5.0 + (70.0 / (1.0 + np.exp(-0.05 * (vote_score - 60))))
            
        est_vote = min(99.9, max(0.0, est_vote))
        # --------------------------------------------------------

        st.divider()
        res_col1, res_col2 = st.columns(2)
        res_col1.metric(f"최종 헌액 확률 ({selected_pos} 기준)", f"{final_prob:.1f}%")
        res_col2.metric("예상 최고 득표율 (기자단 경향 반영)", f"{est_vote:.1f}%")
        st.progress(final_prob / 100)

        if est_vote >= 95.0:
            st.balloons()
            st.success(f"👑 **[FIRST BALLOT LOCK]** 명전 통합 통계 및 해당 포지션 가치를 완벽하게 장악한 만장일치급 레전드입니다. 첫해 입성 확실.")
        elif est_vote >= 85.0:
            st.balloons()
            st.success(f"🏆 **[HOF ELECT]** 해당 포지션의 최상위권 스타로 투표 첫해 혹은 초반 기수에 여유롭게 합격합니다.")
        elif est_vote >= 75.0:
            st.info(f"⚾ **[SAFE ZONE]** 명예의 전당 안정권(75%)을 획득했습니다. 쿠퍼스타운 입성이 확실합니다.")
        elif est_vote >= 60.0:
            st.warning(f"⚠️ **[BORDERLINE - HIGH]** 누적 가치 혹은 세이버메트릭스 비율이 컷에 살짝 미달하나 장기 투표 혹은 베테랑 위원회 구제권입니다.")
        elif est_vote >= 40.0:
            st.warning(f"🤔 **[BORDERLINE - LOW]** 투표 유지선(5%)은 지키겠으나 매년 투표 마감 때마다 논쟁이 폭발할 잔류 그룹입니다.")
        elif est_vote >= 5.0:
            st.error(f"❌ **[ONE AND DONE]** 시대를 호령한 훌륭한 올스타 선수지만, 명예의 전당 기준 통계에는 미달하여 첫해 탈락 위험이 큽니다.")
        else:
            st.error(f"❌ **[OUT OF RANGE]** 명전 투표 후보 리스트에 등록되는 것조차 어려운 하위 스탯입니다.")

with tab2:
    st.header("🔍 1. 데이터 검색 3단계 가이드")
    st.markdown("""
    모든 스탯은 미국의 권위 있는 야구 통계 사이트인 **Baseball-Reference**에서 1분 만에 찾을 수 있습니다.
    
    * **1단계 (구글링):** 구글에 `[선수 영문 이름] + baseball reference` 검색 후 접속 (예: *C.C. Sabathia baseball reference*)
    * **2단계 (스크롤):** 메인 기록 테이블들을 지나 아래쪽 **Leaderboards & Awards** 섹션으로 이동합니다.
    * **3단계 (매칭):** **[Hall of Fame Statistics]** 테이블에 있는 7가지 항목을 그대로 입력창에 채워 넣습니다.
    """)
    
    st.divider()
    st.header("📊 2. 연대별 기자단 투표 성향 및 예외 조항")
    st.markdown("""
    * **누적 스탯 프리미엄 (사바시아 조항):** 현대 야구에서 투수의 분업화 등으로 인해 WAR 손해를 보더라도, **HOF Standards(누적 점수)가 압도적인 선수(투수 48점 이상 / 타자 55점 이상 / 포수 및 유격수 42점 이상)**는 3,000탈삼진, 250승, 혹은 포지션 누적 금자탑을 쌓은 것으로 간주하여 **최종 득표율에 1.12배 보너스 가중치**를 부여하고 비율스탯 감점을 면제합니다.
    * **명전 전용 통계 (`Black/Gray Ink`, `Monitor`, `Standards`):** 포지션 수비 부담과 무관하게 타이틀 획득, 탑텐 랭크, 통산 올스타 선정 횟수 등을 다루므로 **명전 입성자 전체 평균값**을 일괄 적용하여 변별력을 높였습니다.
    * **세이버메트릭스 통계 (`WAR`, `Peak`, `JAWS`):** 포지션별 밸런스 붕괴를 막기 위해 **실제 수비 위치별 평균 합격 스탯**을 타겟팅하여 작동합니다.
    """)
