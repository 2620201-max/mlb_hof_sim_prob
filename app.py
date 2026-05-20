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
# 타자/투수 전체를 아우르는 HOF 통계의 기준 표준값
HOF_GLOBAL_AVG = {
    "Black": 27,   # 명전 입성자 전체 평균 Black Ink
    "Gray": 144,   # 명전 입성자 전체 평균 Gray Ink
    "HOFm": 100,   # 기준점 100
    "HOFs": 50     # 기준점 50
}

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
st.set_page_config(page_title="MLB HOF AI 통합 진단기 v4.8", layout="centered")
st.title("🏛️ MLB HOF AI 통합 진단기 (v4.8 - HOF 잉크 전체평균 고정형)")

tab1, tab2 = st.tabs(["🔍 HOF 정밀 진단", "📖 포지션별 데이터 가이드"])

with tab1:
    selected_pos = st.selectbox(
        "🏃 선수의 주 수비 포지션을 선택하세요", 
        list(POSITION_STATS.keys())
    )
    
    era = st.selectbox(
        "선수의 주 활약 연대(시대) 선택",
        ["데드볼/골든에이지 (~1946)", "통합 및 확장기 (1947-1992)", "스테로이드 시대 (1993-2005)", "현대 세이버 야구 (2006-현재)"]
    )
    
    # 데이터 매칭
    avg = POSITION_STATS[selected_pos]
    p_name = avg["Name"]   # 포지션명 요약
    pos_type = avg["Type"] # 타자 / 투수 구분
    
    st.divider()
    
    # 값 입력 칸 구성 (HOF 통계는 전체 평균 고정 / WAR 계열은 포지션 평균 반영)
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
        input_data = np.array([[black, gray, hof_m, hof_s, c_war, p_war, jaws]])
        
        # 1. AI 헌액 확률 계산
        raw_prob = model.predict_proba(input_data)[0, 1] * 100
        
        # [누적 보정 프리미엄 규칙]
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
            
        # 2. 시대별 투표 기자단 성향 반영
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
        
        # 누적 페널티 조건 우회
        if is_accumulation_monster:
            vote_score *= 1.12
        elif c_war < avg['WAR'] * 0.7:
            vote_score *= 0.85

        est_vote = min(99.9, (vote_score / 100) * 65 + 15)

        # 3. 결과 출력
        st.divider()
        res_col1, res_col2 = st.columns(2)
        res_col1.metric(f"최종 헌액 확률 ({selected_pos} 기준)", f"{final_prob:.1f}%")
        res_col2.metric("예상 최고 득표율 (기자단 경향 반영)", f"{est_vote:.1f}%")
        st.progress(final_prob / 100)

        if est_vote >= 95.0:
            st.balloons()
            st.success(f"👑 **[FIRST BALLOT LOCK]** 명전 통합 통계 및 해당 포지션 가치를 완벽하게 장악한 만장일치급 레전드입니다.")
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
    st.header("📊 명예의 전당 통계 설계 안내")
    st.markdown("""
    * **명전 전용 통계 (`Black/Gray Ink`, `Monitor`, `Standards`):** 포지션 수비 부담과 무관하게 타이틀 획득, 탑텐 랭크, 통산 올스타 선정 횟수 등을 다루므로 **명전 입성자 전체 평균값**을 일괄 적용하여 변별력을 높였습니다.
    * **세이버메트릭스 통계 (`WAR`, `Peak`, `JAWS`):** 포지션별 밸런스 붕괴를 막기 위해 **실제 수비 위치별 평균 합격 스탯**을 타겟팅하여 작동합니다.
    """)
