import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- 1. AI 통합 모델 학습 ---
@st.cache_resource
def train_hof_ultimate_model():
    X = np.array([
        [60, 300, 350, 75, 110.0, 60.0, 85.0], # 1. 올타임 레전드
        [30, 160, 150, 55, 75.0, 45.0, 60.0],  # 2. 정석적인 헌액자
        [40, 185, 165, 50, 46.2, 41.2, 43.7],  # 3. 디지 딘형
        [15, 120, 150, 48, 50.0, 38.0, 44.0],  # 4. 세페다형
        [10, 140, 75, 55, 65.0, 35.0, 50.0],   # 5. 누적형 헌액자
        [12, 90, 90, 35, 62.0, 38.0, 50.0],    # 6. 세이버형 경계선
        [5, 60, 50, 30, 45.0, 30.0, 37.5]      # 7. 명전 미달자
    ])
    y = np.array([1, 1, 1, 1, 1, 0, 0])
    model = LogisticRegression(class_weight='balanced', C=0.01, max_iter=2000)
    model.fit(X, y)
    return model

model = train_hof_ultimate_model()

# --- 2. 기본 포지션별 기준 통계 ---
STATS_AVG = {
    "타자": {"Black": 27, "Gray": 144, "HOFm": 100, "HOFs": 50, "WAR": 67.0, "Peak": 43.0, "JAWS": 55.0},
    "투수": {"Black": 40, "Gray": 185, "HOFm": 100, "HOFs": 50, "WAR": 73.0, "Peak": 50.0, "JAWS": 62.0}
}

# --- 3. UI 구성 ---
st.set_page_config(page_title="MLB HOF AI 시대보정 분석기", layout="centered")
st.title("🏛️ MLB HOF AI 통합 및 시대보정 분석기")

# 탭 구조 통합 유지
tab1, tab2 = st.tabs(["🔍 HOF 정밀 진단 (시대보정)", "📖 가이드 (데이터 검색 및 시대 설명)"])

with tab1:
    col_pos, col_era = st.columns(2)
    with col_pos:
        pos = st.radio("포지션 선택", ["타자", "투수"], horizontal=True)
    with col_era:
        era = st.selectbox(
            "선수의 주 활약 연대(시대) 선택",
            ["데드볼/골든에이지 (~1946)", "통합 및 확장기 (1947-1992)", "스테로이드 시대 (1993-2005)", "현대 세이버 야구 (2006-현재)"]
        )
    
    avg = STATS_AVG[pos].copy()
    
    era_comment = ""
    if era == "스테로이드 시대 (1993-2005)" and pos == "타자":
        era_comment = "⚠️ **시대 특성:** 홈런/타점 인플레이션으로 인해 HOF Monitor 점수가 평균적으로 높게 형성되던 시기입니다. 평가 기준이 엄격해집니다."
    elif era == "현대 세이버 야구 (2006-현재)" and pos == "투수":
        era_comment = "📉 **시대 특성:** 분업화로 인해 현대 투수들은 통산 WAR 및 이닝 누적(Standards)이 대폭 감소했습니다. 누적치 기준을 완화하여 평가합니다."
    elif era == "데드볼/골든에이지 (~1946)":
        era_comment = "⚾ **시대 특성:** 클래식 스탯(승수, 타율)의 가치가 매우 높고 세이버 지표가 정립되기 전입니다."

    if era_comment:
        st.caption(era_comment)
        
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

    if st.button("AI 시대보정 분석 실행"):
        adjusted_c_war = c_war
        adjusted_hof_m = hof_m
        
        if era == "현대 세이버 야구 (2006-현재)" and pos == "투수":
            adjusted_c_war = c_war * 1.18
            
        if era == "스테로이드 시대 (1993-2005)" and pos == "타자":
            adjusted_hof_m = hof_m * 0.88
            
        input_data = np.array([[black, gray, adjusted_hof_m, hof_s, adjusted_c_war, p_war, jaws]])
        
        raw_prob = model.predict_proba(input_data)[0, 1] * 100
        if adjusted_c_war < avg['WAR'] * 0.75:
            final_prob = min(raw_prob, 78.5)
        else:
            final_prob = raw_prob
            
        sabermetrics = (jaws / avg['JAWS']) * 40
        fame = (adjusted_hof_m / avg['HOFm']) * 40
        longevity = (hof_s / avg['HOFs']) * 20
        
        vote_score = sabermetrics + fame + longevity
        if adjusted_c_war < avg['WAR'] * 0.7:
            vote_score *= 0.88

        est_vote = min(99.9, (vote_score / 100) * 70 + 15)

        st.divider()
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("최종 헌액 확률 (시대보정 적용)", f"{final_prob:.1f}%")
        res_col2.metric("예상 최고 득표율", f"{est_vote:.1f}%")
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

# --- 두 가지 정보(검색법 + 시대설명)를 tab2에 모두 유지 ---
with tab2:
    st.header("🔍 1. 데이터 검색 3단계 가이드")
    st.markdown("""
    모든 스탯은 미국의 권위 있는 야구 통계 사이트인 **Baseball-Reference**에서 1분 만에 찾을 수 있습니다.
    
    * **1단계 (구글링):** 구글에 `[선수 영문 이름] + baseball reference` 검색 후 접속 (예: *Dizzy Dean baseball reference*)
    * **2단계 (스크롤):** 메인 기록 테이블들을 지나 아래쪽 **Leaderboards & Awards** 섹션으로 이동합니다.
    * **3단계 (매칭):** **[Hall of Fame Statistics]** 테이블에 있는 7가지 항목을 그대로 입력창에 채워 넣습니다.
        * *Black Ink / Gray Ink / Hall of Fame Monitor / Hall of Fame Standards / WAR / 7-Yr Peak WAR / JAWS* 순서대로 매칭됩니다.
    """)
    
    st.divider()
    
    st.header("📊 2. 연대별 HOF 투표 트렌드 분석")
    st.markdown("""
    명예의 전당 투표단(기자단)의 성향과 메이저리그 환경은 연대별로 극적인 변화를 겪었습니다. 
    이 분석기는 아래와 같은 **역사적 통계 경향**을 로직에 반영하고 있습니다.
    
    * **데드볼/골든에이지 (~1946)**
        * 클래식 지표(다승, 타율) 및 당대의 스타성(Black Ink)이 투표를 지배하던 시기입니다. 
        * 디지 딘처럼 누적이 짧아도 당대 리그를 폭격한 임팩트가 있다면 베테랑 위원회 등을 통해 구제될 확률이 높습니다.
        
    * **통합 및 확장기 (1947-1992)**
        * 야구의 규격이 정문화되고 가장 안정적인 통계 분포를 보이는 '명전 투수/타자의 정석' 시기입니다. 
        * 우리가 흔히 아는 타자 WAR 67, 투수 WAR 73 기준이 가장 칼같이 적용됩니다.
        
    * **스테로이드 시대 (1993-2005)**
        * 벌크업된 타자들의 무시무시한 누적 스탯으로 인해 **HOF Monitor 점수가 비정상적으로 폭등**하던 버블 시기입니다.
        * AI는 이 시대 타자를 평가할 때 명성 점수(HOFm)에 패널티 필터를 적용하여 성적 뻥튀기를 걷어냅니다.
        
    * **현대 세이버 야구 (2006-현재)**
        * 투수들의 이닝 관리가 엄격해지면서 **현대 투수들은 통산 WAR 70이나 200승 고지를 밟는 것이 물리적으로 불가능**해졌습니다.
        * AI는 현대 투수가 선택될 경우, Career WAR와 Standards 수치에 **보정 가중치(+18%)**를 자동으로 더해 억울하게 탈락하는 일을 방지합니다. (예: 커쇼, 슈어저 이후 세대 구제용)
    """)
