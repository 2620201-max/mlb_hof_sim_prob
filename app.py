import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup, Comment
import re
from sklearn.linear_model import LogisticRegression

# --- 1. AI 모델 학습 (확률 계산용) ---
@st.cache_resource
def train_hof_model():
    # 학습 데이터: [HOF Monitor, WAR]
    X = np.array([[165, 71], [320, 106], [170, 80], [110, 75], [95, 51], [85, 40], [50, 30]])
    y = np.array([1, 1, 0, 0, 0, 0, 0]) # 1: 헌액, 0: 탈락
    model = LogisticRegression(class_weight='balanced').fit(X, y)
    return model

# --- 2. 데이터 수집 함수 (Albert Pujols 등 검색 에러 해결판) ---
def fetch_player_all_in_one(query_name):
    query = query_name.replace(" ", "+")
    search_url = f"https://www.baseball-reference.com/search/search.fcgi?search={query}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    res = requests.get(search_url, headers=headers, timeout=10)
    final_url = res.url

    # 검색 결과 목록 페이지일 경우 첫 번째 선수로 이동
    if "search.fcgi" in final_url:
        soup = BeautifulSoup(res.text, "html.parser")
        search_results = soup.find("div", id="players")
        if search_results and search_results.find("a"):
            player_link = search_results.find("a")['href']
            final_url = f"https://www.baseball-reference.com{player_link}"
            res = requests.get(final_url, headers=headers, timeout=10)

    # 주석 포함 전체 데이터 파싱
    full_soup = BeautifulSoup(res.text + "".join(BeautifulSoup(res.text, "html.parser").find_all(string=lambda text: isinstance(text, Comment))), "html.parser")

    p_name = full_soup.find("h1").get_text(strip=True) if full_soup.find("h1") else query_name
    
    def get_stat(label):
        tag = full_soup.find(string=re.compile(label))
        if tag:
            match = re.search(r"(\d+\.\d+|\d+)", tag.parent.get_text())
            return float(match.group(1)) if match else 0.0
        return 0.0

    return p_name, get_stat("Hall of Fame Monitor"), get_stat("WAR")

# --- 3. UI 레이아웃 ---
st.set_page_config(page_title="MLB HOF AI 분석기", layout="centered")
model = train_hof_model()

st.title("🏛️ MLB HOF AI 통합 진단기")
st.markdown("선수 이름을 입력하면 **실시간 데이터**로 확률과 득표율을 분석합니다.")

name_input = st.text_input("영문 이름 입력 (예: Albert Pujols, Mike Trout)", "")

if name_input:
    with st.spinner("쿠퍼스타운에서 데이터를 가져오는 중..."):
        try:
            name, hofm, war = fetch_player_all_in_one(name_input)
            
            if hofm == 0 and war == 0:
                st.warning("데이터를 찾지 못했습니다. 성과 이름을 정확히 입력해주세요.")
            else:
                st.success(f"분석 완료: {name}")
                
                # 결과 카드
                col1, col2 = st.columns(2)
                col1.metric("HOF Monitor", hofm)
                col2.metric("Career WAR", war)

                # 1. AI 확률 계산
                prob = model.predict_proba([[hofm, war]])[0, 1] * 100
                
                # 2. 득표율 시뮬레이션 (간이 공식)
                first_ballot = min(99.2, (hofm * 0.4) + (war * 0.3) + 15)

                st.divider()
                st.subheader("🔮 AI 예측 결과")
                
                st.write(f"**명예의 전당 헌액 확률: {prob:.1f}%**")
                st.progress(prob / 100)
                
                st.write(f"**첫해 예상 득표율: {first_ballot:.1f}%**")
                st.progress(first_ballot / 100)

                # 판정 메시지
                if prob >= 80:
                    st.balloons()
                    st.info("🏆 이 선수는 '확정적(LOCK)'입니다. 첫 투표 입성이 유력합니다.")
                elif prob >= 40:
                    st.warning("⚾ 경계선에 있는 선수입니다. 기자단의 재평가가 중요합니다.")
                else:
                    st.error("❌ 현재 지표로는 입성이 쉽지 않아 보입니다.")

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
