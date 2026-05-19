import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- 1. AI 통합 모델 학습 (v4.3의 단단한 상단 변별력 유지) ---
@st.cache_resource
def train_hof_ultimate_model():
    X = np.array([
        [120, 450, 500, 95, 140.0, 70.0, 105.0], # 1. 지구 파괴급 신 (루스, 존슨, 메이스)
        [60, 280, 320, 75, 100.0, 58.0, 79.0],  # 2. 올타임 슈퍼 레전드 (푸홀스, 마드록스)
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

# --- 2. 기본 포지션별 기준 통계 ---
STATS_AVG = {
    "타자": {"Black": 27, "Gray": 144, "HOFm": 100, "HOFs": 50, "WAR": 67.0, "Peak": 43.0, "JAWS": 55.0},
    "투수": {"Black": 40, "Gray": 185, "HOFm": 100, "HOFs": 50, "WAR": 73.0, "Peak": 50.0, "JAWS": 62.0}
}

# --- 3. UI 구성 ---
st.set_page_config(page_title="MLB HOF AI 통합 진단기", layout="centered")
st.title("🏛️ MLB HOF AI 통합 진단기 (v4.4 - 누적 가치 반영)")

tab1, tab2 = st.tabs(["🔍 HOF 정밀 진단", "📖 가이드 (데이터 검색 및 시대 설명)"])

with tab1:
    col_pos, col_era = st.columns(2)
    with col_pos:
        pos = st.radio("포지션 선택", ["타자", "투수"], horizontal=True)
    with col_era:
        era = st.selectbox(
            "선수의 주 활약 연대(시대) 선택",
            ["데드볼/골든에이지 (~1946)", "통합 및 확장기 (1947-1992)", "스테로이드 시대 (1993-2005)", "현대 세이버 야구 (2006-현재)"]
        )
    
    avg = STATS_AVG[pos]
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
        
        # 1. AI 헌액 확률 계산
        raw_prob = model.predict_proba(input_data)[0, 1] * 100
        
        # [누적 보정 프리미엄] 사바시아처럼 WAR는 낮지만 HOF Standards(누적치)가 평균(50)을 한참 상회하는 투수 구제
        is_accumulation_monster = (pos == "투수" and hof_s >= 48) or (pos == "타자" and hof_s >= 55)
        
        if c_war < avg['WAR'] * 0.75 and not is_accumulation_monster:
            final_prob = min(raw_prob, 75.0)
        else:
            final_prob = raw_prob
            
        # 2. 시대별 투표 기자단 성향 반영 (득표율 수식)
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
        
        # 누적 페널티 조건 우회 (누적 몬스터는 감점 면제 및 보너스 가중치 부여)
        if is_accumulation_monster:
            vote_score *= 1.12  # 사바시아 보너스 (3,000K, 250승 프리미엄 적용)
        elif c_war < avg['WAR'] * 0.7:
            vote_score *= 0.85

        est_vote = min(99.9, (vote_score / 100) * 65 + 15)

        # 3. 결과 출력
        st.divider()
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("최종 헌액 확률 (수학적 고정)", f"{final_prob:.1f}%")
        res_col2.metric("예상 최고 득표율 (기자단 성향 반영)", f"{est_vote:.1f}%")
        st.progress(final_prob / 100)

        if est_vote >= 95.0:
            st.balloons()
            st.success(f"👑 **[FIRST BALLOT LOCK]** 만장일치를 논할 수준의 역대급 전설입니다. 첫해 입성이 100% 확실합니다.")
        elif est_vote >= 85.0:
            st.balloons()
            st.success(f"🏆 **[HOF ELECT]** 투표 첫해 혹은 초반 기수에 압도적인 표수로 여유롭게 입성할 선수입니다.")
        elif est_vote >= 75.0:
            st.info(f"⚾ **[SAFE ZONE]** 명예의 전당 기준선(75%)을 넘겼습니다. 안정적으로 쿠퍼스타운행 티켓을 따냅니다.")
        elif est_vote >= 60.0:
            st.warning(f"⚠️ **[BORDERLINE - HIGH]** 입성 컷에는 살짝 미달하지만, 투표 기수가 지남에 따라 재평가되어 추후 입성할 가능성이 매우 높습니다 (세페다, 디지 딘 코스).")
        elif est_vote >= 40.0:
            st.warning(f"🤔 **[BORDERLINE - LOW]** 매년 투표 후보에는 남겠지만(5% 유지), 기자들 사이에서 치열한 키보드 배틀이 벌어지며 장기 잔류할 상입니다.")
        elif est_vote >= 5.0:
            st.error(f"❌ **[ONE AND DONE]** 메이저리그를 풍미한 훌륭한 선수(Very Good)지만, 명전 투표에서는 첫해 5% 미만으로 광속 탈락할 위험이 큽니다.")
        else:
            st.error(f"❌ **[OUT OF RANGE]** 명예의 전당 투표 후보(Ballot) 자격을 얻는 것조차 쉽지 않은 스탯입니다.")

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
    * **누적 스탯 프리미엄 (사바시아 조항):** * 현대 야구에서 투수의 분업화로 인해 WAR 손해를 보더라도, **HOF Standards(누적 점수)가 압도적인 선수(투수 48점 이상)**는 3,000탈삼진/250승 등의 금자탑을 쌓은 것으로 간주하여 **최종 득표율에 1.12배 보너스 가중치**를 부여하고 소프트캡 감점을 면제합니다.
    """)
