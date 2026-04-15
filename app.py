import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import requests
from bs4 import BeautifulSoup, Comment
import re

# --- 1. 모델 학습 데이터 및 엔진 세팅 ---
@st.cache_resource
def train_hof_model():
    # 헌액자(1)와 미헌액자(0) 균형을 맞춘 훈련 데이터
    data = [
        ("Derek Jeter", 165.0, 71.3, 1), ("Greg Maddux", 320.0, 106.6, 1),
        ("Mike Mussina", 135.0, 82.8, 1), ("Pedro Martinez", 195.0, 83.9, 1),
        ("Tony Gwynn", 180.0, 69.2, 1), ("Chipper Jones", 185.0, 85.3, 1),
        ("Scott Rolen", 150.0, 70.1, 1), ("Todd Helton", 175.0, 61.8, 1),
        ("Curt Schilling", 170.0, 80.5, 0), ("Lou Whitaker", 110.0, 75.1, 0),
        ("Bobby Grich", 95.0, 71.1, 0), ("Kenny Lofton", 105.0, 68.4, 0),
        ("Jim Edmonds", 90.0, 60.4, 0), ("Bernie Williams", 85.0, 49.6, 0),
        ("Johan Santana", 85.0, 51.7, 0), ("Dustin Pedroia", 95.0, 51.9, 0)
    ]
    df = pd.DataFrame(data, columns=["name", "HOFm", "WAR", "elected"])
    model = LogisticRegression(class_weight='balanced', C=0.5)
    model.fit(df[["HOFm", "WAR"]], df["elected"])
    return model

# --- 2. 실시간 데이터 스크래핑 함수 ---
def fetch_player_stats(query_name):
    query = query_name.replace(" ", "+")
    url = f"https://www.baseball-reference.com/search/search.fcgi?search={query}"
    res = requests.get(url, timeout=10)
    
    # 주석 내부 데이터까지 파싱
    soup = BeautifulSoup(res.text, "html.parser")
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    full_soup = BeautifulSoup(res.text + "".join(comments), "html.parser")

    def get_stat(label):
        tag = full_soup.find(lambda t: t.name == "strong" and label in t.text)
        if tag:
            text = tag.parent.get_text()
            match = re.search(r"(\d+\.\d+|\d+)", text)
            return float(match.group(1)) if match else 0.0
        return 0.0

    hofm = get_stat("Hall of Fame Monitor")
    war = get_stat("WAR")
    p_name = full_soup.find("h1").text.strip() if full_soup.find("h1") else query_name
    return p_name, hofm, war

# --- 3. Streamlit UI 레이아웃 ---
st.set_page_config(page_title="MLB HOF 확률 진단기", layout="centered")

# 모델 준비
model = train_hof_model()

st.title("🏛️ MLB HOF AI 확률 진단기")
st.markdown("선수의 이름을 입력하면 AI가 명예의 전당 헌액 확률을 진단합니다.")

# 입력 섹션
player_input = st.text_input("선수 영문 이름 입력 (예: Buster Posey, Ichiro Suzuki)", "")

if player_input:
    with st.spinner("데이터 분석 중..."):
        try:
            name, hofm, war = fetch_player_stats(player_input)
            
            # 확률 계산
            prob = model.predict_proba([[hofm, war]])[0, 1]
            prob_pct = prob * 100

            st.divider()
            
            # 결과 표시
            st.header(f"⚾ 분석 결과: {name}")
            col1, col2, col3 = st.columns(3)
            col1.metric("HOF Monitor", f"{hofm}")
            col2.metric("Career WAR", f"{war}")
            col3.metric("AI 확률", f"{prob_pct:.1f}%")

            # 판정 로직
            if prob >= 0.75:
                st.success(f"🏆 **판정: 헌액 유력 (LOCK)**")
                st.balloons()
            elif prob >= 0.4:
                st.warning(f"⚾ **판정: 경계선 (Borderline)**")
            else:
                st.error(f"❌ **판정: 입성 불투명 (Low Probability)**")
            
            st.info(f"지표 가이드: {name} 선수는 성적 기반 AI 모델에서 {prob_pct:.1f}%의 일치율을 보였습니다.")

        except Exception as e:
            st.error("데이터를 불러오지 못했습니다. 이름을 다시 확인해 주세요.")

else:
    st.write("---")
    st.caption("※ 본 엔진은 약물/도박 등 외적 논란을 배제한 통계 중심 모델입니다.")
