import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup, Comment
import re
from sklearn.linear_model import LogisticRegression

# --- 1. 모델 학습 ---
@st.cache_resource
def train_hof_model():
    X = np.array([[165, 71], [320, 106], [170, 80], [110, 75], [95, 51], [85, 40], [50, 30]])
    y = np.array([1, 1, 0, 0, 0, 0, 0])
    return LogisticRegression(class_weight='balanced').fit(X, y)

# --- 2. 데이터 수집 (검색 목록 강제 돌파 버전) ---
def fetch_player_data(query_name):
    query = query_name.strip().replace(" ", "+")
    search_url = f"https://www.baseball-reference.com/search/search.fcgi?search={query}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    res = requests.get(search_url, headers=headers, timeout=10)
    
    # [핵심] 만약 검색 결과 목록 페이지가 떴다면 (URL에 search.fcgi가 그대로 남아있다면)
    if "search.fcgi" in res.url:
        soup = BeautifulSoup(res.text, "html.parser")
        # 'search_results' 또는 'players' ID를 가진 영역에서 첫 번째 결과 추출
        search_div = soup.find("div", id="players") or soup.find("div", class_="search-item-url")
        if search_div and search_div.find("a"):
            first_link = search_div.find("a")['href']
            # 만약 링크가 상대 경로라면 절대 경로로 변환
            target_url = first_link if first_link.startswith("http") else f"https://www.baseball-reference.com{first_link}"
            res = requests.get(target_url, headers=headers, timeout=10)

    # 데이터 파싱
    full_html = res.text + "".join(BeautifulSoup(res.text, "html.parser").find_all(string=lambda text: isinstance(text, Comment)))
    full_soup = BeautifulSoup(full_html, "html.parser")

    # 이름, HOFm, WAR 추출
    p_name = full_soup.find("h1").get_text(strip=True) if full_soup.find("h1") else query_name
    
    def extract(label):
        tag = full_soup.find(string=re.compile(label))
        if tag:
            match = re.search(r"(\d+\.\d+|\d+)", tag.parent.get_text())
            return float(match.group(1)) if match else 0.0
        return 0.0

    return p_name, extract("Hall of Fame Monitor"), extract("WAR")

# --- 3. UI ---
st.title("🏛️ MLB HOF AI 통합 진단기")
model = train_hof_model()

name_input = st.text_input("영문 이름을 입력하세요 (예: Mike Trout)", "")

if name_input:
    with st.spinner("쿠퍼스타운에서 데이터를 찾는 중..."):
        try:
            name, hofm, war = fetch_player_data(name_input)
            
            # 둘 다 0이면 정말 못 찾은 것
            if hofm == 0 and war == 0:
                st.warning(f"'{name_input}'의 핵심 데이터를 찾지 못했습니다. Mike Trout처럼 성과 이름을 모두 입력해 보세요.")
            else:
                st.success(f"데이터 로드 완료: {name}")
                col1, col2 = st.columns(2)
                col1.metric("HOF Monitor", hofm)
                col2.metric("Career WAR", war)

                prob = model.predict_proba([[hofm, war]])[0, 1] * 100
                st.write(f"### 🔮 AI 분석 헌액 확률: {prob:.1f}%")
                st.progress(prob / 100)

        except Exception as e:
            st.error(f"사이트 접속 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")
