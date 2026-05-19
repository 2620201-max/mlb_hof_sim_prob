import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- 1. AI 통합 모델 학습 (최상위권 인플레이션 차단 튜닝) ---
@st.cache_resource
def train_hof_ultimate_model():
    # 데이터 구조: [Black Ink, Gray Ink, HOFm, HOFs, Career WAR, 7yr-Peak, JAWS]
    X = np.array([
        [120, 450, 500, 95, 140.0, 70.0, 105.0], # 1. 지구 파괴급 신 (루스, 사이영, 메이스) -> **여기를 넣어야 겨우 98~99%가 나옵니다**
        [60, 280, 320, 75, 100.0, 58.0, 79.0],  # 2. 올타임 슈퍼 레전드 (푸홀스, 로드리게스 등) -> 90%대 초중반 타겟
        [40, 200, 200, 60, 80.0, 48.0, 64.0],   # 3. 확실한 First Ballot (트라웃, 벨트레 등) -> **이제 여기가 80%대로 통제됩니다**
        [25, 140, 130, 50, 65.0, 40.0, 52.5],   # 4. 정석적인 헌액자 -> 65~75%
        [40, 185, 165, 50, 46.2, 41.2, 43.7],   # 5. 디지 딘형 (임팩트 극강, 누적 부족) -> 55~65%
        [12, 120, 110, 45, 55.0, 36.0, 45.5],   # 6. 세페다형 / 세이버 경계선 -> 40~50%
        [5, 60, 50, 30, 40.0, 28.0, 34.0]       # 7. 명전 미달자 -> 10% 이하
    ])
    y = np.array([1, 1, 1, 1, 1, 0, 0])
    
    # C=0.005로 규제를 극대화하여 그래프 기울기를 엄청나게 완만하게 만듦 (Smooth 확률 분산)
    # 웬만한 스탯 상승으로는 확률이 쉽게 뻥튀기되지 않는 구조
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
st.title("🏛️ MLB HOF AI 통합 진단기 (v4.3 - 철벽 밸런스)")

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
    
    # 지표 입력
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
        
        # 1. 헌액 확률 계산 (기울기가 완만해져 최상위권 뻥튀기 전면 차단)
        raw_prob = model.predict_proba(input_data)[0, 1] * 100
        
        # 소프트 캡 (누적 WAR 결핍 제어)
        if c_war < avg['WAR'] * 0.75:
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
        if c_war < avg['WAR'] * 0.7:
            vote_score *= 0.85 # 감점 폭 미세 상향

        # 최고 득표율 기준선도 최상위권 변별력을 위해 70% 비율로 압축
        est_vote = min(99.9, (vote_score / 100) * 65 + 15)

        # 3. 결과 출력
        st.divider()
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("최종 헌액 확률 (수학적 고정)", f"{final_prob:.1f}%")
        res_col2.metric("예상 최고 득표율 (기자단 성향 반영)", f"{est_vote:.1f}%")
        st.progress(final_prob / 100)

        # 득표율 기준 판정 메세지팩
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
    
    * **1단계 (구글링):** 구글에 `[선수 영문 이름] + baseball reference` 검색 후 접속 (예: *Dizzy Dean baseball reference*)
    * **2단계 (스크롤):** 메인 기록 테이블들을 지나 아래쪽 **Leaderboards & Awards** 섹션으로 이동합니다.
    * **3단계 (매칭):** **[Hall of Fame Statistics]** 테이블에 있는 7가지 항목을 그대로 입력창에 채워 넣습니다.
    """)
    st.divider()
    st.header("📊 2. 연대별 기자단 투표 성향 (가중치 원리)")
    st.markdown("""
    본 프로그램은 회귀 스탯을 강제로 주작하지 않고, **선택한 시대의 실제 기자단 투표 메커니즘**을 가중치 비율로 계산합니다.
    
    * **데드볼/골든에이지 (~1946):** 클래식 명성 및 블랙잉크(타이틀) 비중 **55%** 폭등. 세이버메트릭스 비중 20% 축소.
    * **통합 및 확장기 (1947-1992):** 세이버 40% / 명성 40% / 누적 20%의 가장 표준적인 밸런스 투표.
    * **스테로이드 시대 (1993-2005):** 약물 인플레로 인해 HOF Monitor(명성) 반영 비율을 **20%**로 대폭 제한하고 누적과 비율을 엄격히 검증.
    * **현대 세이버 야구 (2006-현재):** 전통적 명성 스탯을 불신하는 시대. **JAWS(세이버 지표) 반영 비율을 60%까지 극대화**하여 이닝이 부족한 현대 투수/타자 구제.
    """)
