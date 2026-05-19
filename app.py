import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- 1. AI 통합 모델 학습 (v4.4의 단단한 변별력 알고리즘 유지) ---
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

# --- 2. [진짜 야구 포지션] 명예의 전당 입성자 실제 포지션별 평균 통계 데이터셋 ---
POSITION_STATS = {
    "포수 (C)": {"Black": 13, "Gray": 92, "HOFm": 90, "HOFs": 44, "WAR": 53.8, "Peak": 34.4, "JAWS": 44.1, "Type": "타자"},
    "1루수 (1B)": {"Black": 32, "Gray": 154, "HOFm": 115, "HOFs": 54, "WAR": 65.5, "Peak": 41.8, "JAWS": 53.7, "Type": "타자"},
    "2루수 (2B)": {"Black": 20, "Gray": 128, "HOFm": 105, "HOFs": 49, "WAR": 69.4, "Peak": 44.3, "JAWS": 56.9, "Type": "타자"},
    "3루수 (3B)": {"Black": 21, "Gray": 122, "HOFm": 98, "HOFs": 51, "WAR": 68.3, "Peak": 42.9, "JAWS": 55.6, "Type": "타자"},
    "유격수 (SS)": {"Black": 13, "Gray": 109, "HOFm": 87, "HOFs": 43, "WAR": 66.8, "Peak": 43.0, "JAWS": 54.9, "Type": "타자"},
    "좌익수 (LF)": {"Black": 32, "Gray": 156, "HOFm": 110, "HOFs": 53, "WAR": 65.2, "Peak": 41.5, "JAWS": 53.4, "Type": "타자"},
    "중견수 (CF)": {"Black": 31, "Gray": 147, "HOFm": 120, "HOFs": 55, "WAR": 71.3, "Peak": 44.6, "JAWS": 57.9, "Type": "타자"},
    "우익수 (RF)": {"Black": 33, "Gray": 161, "HOFm": 125, "HOFs": 56, "WAR": 71.5, "Peak": 41.9, "JAWS": 56.7, "Type": "타자"},
    "선발투수 (SP)": {"Black": 44, "Gray": 196, "HOFm": 115, "HOFs": 52, "WAR": 73.0, "Peak": 49.9, "JAWS": 61.4, "Type": "투수"},
    "구원투수 (RP)": {"Black": 8, "Gray": 55, "HOFm": 72, "HOFs": 32, "WAR": 39.5, "Peak": 27.0, "JAWS": 33.3, "Type": "투수"}
}

# --- 3. UI 구성 ---
st.set_page_config(page_title="MLB 포지션별 HOF AI 진단기", layout="centered")
st.title("🏛️ MLB HOF AI 통합 진단기 (v4.6 - 리얼 포지션별 연동판)")

tab1, tab2 = st.tabs(["🔍 수비 포지션별 HOF 정밀 진단", "📖 포지션별 데이터 가이드"])

with tab1:
    # 사용자님이 원하시던 '진짜' 수비 포지션 셀렉트 박스
    selected_pos = st.selectbox(
        "🏃 선수의 주 수비 포지션을 선택하세요", 
        list(POSITION_STATS.keys())
    )
    
    era = st.selectbox(
        "선수의 주 활약 연대(시대) 선택",
        ["데드볼/골든에이지 (~1946)", "통합 및 확장기 (1947-1992)", "스테로이드 시대 (1993-2005)", "현대 세이버 야구 (2006-현재)"]
    )
    
    # 선택된 포지션의 공식 평균 수치를 변수에 즉시 바인딩
    avg = POSITION_STATS[selected_pos]
    pos_type = avg["Type"]  # 타자 또는 투수인지 판별
    
    st.caption(f"💡 **{selected_pos}** 가 선택되었습니다. 아래 스탯 입력창의 초기값이 해당 포지션의 실제 명전 평균값으로 세팅됩니다.")
    st.divider()
    
    # 값 입력 칸 (포지션 선택에 따라 완벽하게 실시간 변동)
    c1, c2, c3 = st.columns(3)
    with c1:
        black = st.number_input("Black Ink", value=float(avg["Black"]), step=1.0)
        gray = st.number_input("Gray Ink", value=float(avg["Gray"]), step=1.0)
    with c2:
        hof_m = st.number_input("HOF Monitor", value=float(avg["HOFm"]), step=1.0)
        hof_s = st.number_input("HOF Standards", value=float(avg["HOFs"]), step=1.0)
    with c3:
        c_war = st.number_input("Career WAR", value=float(avg["WAR"]), step=0.1)
        p_war = st.number_input("7yr-Peak WAR", value=float(avg["Peak"]), step=0.1)
        jaws = st.number_input("JAWS", value=float(avg["JAWS"]), step=0.1)

    if st.button("포지션 맞춤 AI 분석 실행"):
        input_data = np.array([[black, gray, hof_m, hof_s, c_war, p_war, jaws]])
        
        # 1. AI 헌액 확률 계산
        raw_prob = model.predict_proba(input_data)[0, 1] * 100
        
        # [누적 보정 프리미엄 규칙] 포지션 성격에 맞춤화
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
            
        # 2. 시대별 투표 기자단 성향 반영 (득표율 수식 - 포지션별 분리 적용)
        if era == "현대 세이버 야구 (2006-현재)":
            sabermetrics = (jaws / avg['JAWS']) * 60
            fame = (hof_m / avg['HOFm']) * 25
            longevity = (hof_s / avg['HOFs']) * 15
        elif era == "스테로이드 시대 (1993-2005)":
            sabermetrics = (jaws / avg['JAWS']) * 40
            fame = (hof_m / avg['HOFm']) * 20
            longevity = (hof_s / avg['HOFs']) * 40
        elif era == "데드볼/골든에이지 (~1946)":
            sabermetrics = (jaws / avg['JAWS']) * 20
            fame = (hof_m / avg['HOFm']) * 55
            longevity = (hof_s / avg['HOFs']) * 25
        else:
            sabermetrics = (jaws / avg['JAWS']) * 40
            fame = (hof_m / avg['HOFm']) * 40
            longevity = (hof_s / avg['HOFs']) * 20
        
        vote_score = sabermetrics + fame + longevity
        
        # 누적 페널티 조건 및 보너스 가중치 우회
        if is_accumulation_monster:
            vote_score *= 1.12  # 누적 백골전사 프리미엄 12% 가산
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
            st.success(f"👑 **[FIRST BALLOT LOCK]** 해당 포지션 역사상 만장일치 수준을 논할 전설입니다. 첫해 입성 확실.")
        elif est_vote >= 85.0:
            st.balloons()
            st.success(f"🏆 **[HOF ELECT]** 해당 포지션의 상징적인 선수로 투표 초반 기수에 압도적 표수로 입성합니다.")
        elif est_vote >= 75.0:
            st.info(f"⚾ **[SAFE ZONE]** 명예의 전당 헌액 커트라인(75%)을 넘겼습니다. 쿠퍼스타운 입성이 확실시됩니다.")
        elif est_vote >= 60.0:
            st.warning(f"⚠️ **[BORDERLINE - HIGH]** 수비 포지션 평균엔 살짝 미치지 못하지만, 추후 베테랑 위원회 등으로 구제될 확률이 높습니다.")
        elif est_vote >= 40.0:
            st.warning(f"🤔 **[BORDERLINE - LOW]** 매년 투표 후보직(5%)은 유지하되, 기자단 투표 창구에서 치열한 키보드 배틀이 벌어질 보더라인 라인입니다.")
        elif est_vote >= 5.0:
            st.error(f"❌ **[ONE AND DONE]** 한 시대를 풍미한 포지션 스타(Very Good)지만, 명전 기준에는 미달하여 첫해 광속 탈락할 위험이 큽니다.")
        else:
            st.error(f"❌ **[OUT OF RANGE]** 명예의 전당 투표 후보 명단(Ballot)에 들어가는 것 자체가 불가능한 스탯입니다.")

with tab2:
    st.header("📊 실제 메이저리그 명예의 전당 포지션별 특징 설명")
    st.markdown("""
    * **포수 (C) 및 유격수 (SS):** 체력 소모가 극심하고 수비 기여도가 높기 때문에, 통산 Career WAR 가 기준선(**53.8~66.8**)보다 약간 낮아도 HOF Monitor나 Standards 수치가 높으면 강력하게 우대받습니다.
    * **1루수 (1B) 및 우익수 (RF):** 공격 위주의 포지션이므로 평균 요구 WAR(**65.5~71.5**)와 HOF Monitor 스탯 요구치(**115~125**)가 타 포지션에 비해 현격히 높습니다. 슬러거 기준이 까다롭습니다.
    * **구원투수 (RP):** 누적 이닝의 한계로 Career WAR 평균(**39.5**)은 매우 낮지만, 세이브왕 타이틀이나 임팩트를 측정하는 HOF Monitor 수치가 합격을 가르는 핵심 열쇠가 됩니다.
    """)
