import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- 1. AI 통합 모델 학습 (정직한 데이터, 정교한 변별력) ---
@st.cache_resource
def train_hof_ultimate_model():
    # 데이터 구조: [Black Ink, Gray Ink, HOFm, HOFs, Career WAR, 7yr-Peak, JAWS]
    X = np.array([
        [70, 320, 380, 80, 115.0, 62.0, 88.5],  # 1. 신계 레전드 (루스, 에런, 푸홀스) -> 99% 만점 기준
        [45, 220, 240, 62, 85.0, 52.0, 68.5],   # 2. 현실판 탑클래스 (트라웃, 벨트레) -> 91~95% (변별력 유지)
        [30, 160, 150, 55, 75.0, 45.0, 60.0],  # 3. 정석적인 헌액자 -> 80-85%
        [40, 185, 165, 50, 46.2, 41.2, 43.7],  # 4. 디지 딘형 (임팩트 극강, 누적 부족) -> 65-75%
        [15, 120, 150, 48, 50.0, 38.0, 44.0],  # 5. 세페다형 (임팩트형 경계선) -> 55-65%
        [10, 140, 75, 55, 65.0, 35.0, 50.0],   # 6. 누적형 헌액자 (꾸준함) -> 60-70%
        [12, 90, 90, 35, 62.0, 38.0, 50.0],    # 7. 세이버형 경계선 (로프턴 등) -> 40-50%
        [5, 60, 50, 30, 45.0, 30.0, 37.5]      # 8. 명전 미달자 -> 15% 이하
    ])
    y = np.array([1, 1, 1, 1, 1, 1, 0, 0])
    
    # C=0.08로 격차를 확실히 인지하게 고정 (시대에 따라 흔들리지 않는 절대 기준)
    model = LogisticRegression(class_weight='balanced', C=0.08, max_iter=2000)
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
st.title("🏛️ MLB HOF AI 통합 진단기 (v4.2)")

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
    
    # 지표 입력 (유저가 입력한 값 그대로 모델에 들어감)
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
        # 입력 데이터 그대로 원본 어레이 생성 (데이터 왜곡 없음)
        input_data = np.array([[black, gray, hof_m, hof_s, c_war, p_war, jaws]])
        
        # 1. 고정된 회귀 모델을 통한 수학적 확률 계산
        raw_prob = model.predict_proba(input_data)[0, 1] * 100
        
        # 물리적 소프트 캡 (누적 WAR가 평균의 75% 미만이면 최대 78.5% 제한)
        if c_war < avg['WAR'] * 0.75:
            final_prob = min(raw_prob, 78.5)
        else:
            final_prob = raw_prob
            
        # 2. [핵심] 시대별 투표 기자단의 '성향(가중치)' 스위칭 로직
        # 인풋 데이터를 건드리는 게 아니라, 최종 점수를 내는 비율을 바꿈!
        if era == "현대 세이버 야구 (2006-현재)":
            # 현대 기자단: 세이버메트릭스(JAWS)를 무려 60% 반영, 클래식 명성(HOFm)은 축소
            sabermetrics = (jaws / avg['JAWS']) * 60
            fame = (hof_m / avg['HOFm']) * 25
            longevity = (hof_s / avg['HOFs']) * 15
        elif era == "스테로이드 시대 (1993-2005)":
            # 스테로이드 시절: 약물 홈런 버블로 인해 HOFm 기준을 엄격하게 잡음 (세이버와 누적 위주)
            sabermetrics = (jaws / avg['JAWS']) * 40
            fame = (hof_m / avg['HOFm']) * 20  # 명성 반영률 유일하게 축소
            longevity = (hof_s / avg['HOFs']) * 40
        elif era == "데드볼/골든에이지 (~1946)":
            # 옛날 야구: 무조건 임팩트, 타이틀 갯수(HOFm), 블랭인크가 장땡
            sabermetrics = (jaws / avg['JAWS']) * 20
            fame = (hof_m / avg['HOFm']) * 55  # 전통적 클래식 명성 55% 폭등
            longevity = (hof_s / avg['HOFs']) * 25
        else:
            # 통합기 표준 세팅 (40 / 40 / 20)
            sabermetrics = (jaws / avg['JAWS']) * 40
            fame = (hof_m / avg['HOFm']) * 40
            longevity = (hof_s / avg['HOFs']) * 20
        
        # 누적 하한선에 따른 감점 시스템 지표 고정
        vote_score = sabermetrics + fame + longevity
        if c_war < avg['WAR'] * 0.7:
            vote_score *= 0.88

        # 최종 득표율 산출
        est_vote = min(99.9, (vote_score / 100) * 70 + 15)

        # 3. 결과 출력
        st.divider()
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("AI 헌액 확률 (수학적 고정)", f"{final_prob:.1f}%")
        res_col2.metric("예상 최고 득표율 (기자단 성향 반영)", f"{est_vote:.1f}%")
        st.progress(final_prob / 100)

        # 득표율 기준 판정 메세지팩 완벽 유지
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
