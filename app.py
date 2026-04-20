import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
import re

# --- 1. 데이터 수집 함수 (검색 결과가 여러 명일 때 대비) ---
def fetch_player_final(query_name):
    query = query_name.replace(" ", "+")
    # 검색을 시도합니다.
    search_url = f"https://www.baseball-reference.com/search/search.fcgi?search={query}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    res = requests.get(search_url, headers=headers, timeout=10)
    
    # 만약 검색 결과가 여러 명이라서 목록 페이지가 떴다면, 첫 번째 선수를 클릭해서 들어갑니다.
    if "search.fcgi" in res.url:
        soup = BeautifulSoup(res.text, "html.parser")
        search_results = soup.find("div", id="players")
        if search_results:
            first_link = search_results.find("a")['href']
            final_url = f"https://www.baseball-reference.com{first_link}"
            res = requests.get(final_url, headers=headers, timeout=10)
    
    # 주석 데이터를 포함한 전체 데이터 파싱
    full_html = res.text + "".join(BeautifulSoup(res.text, "html.parser").find_all(string=lambda text: isinstance(text, Comment)))
    full_soup = BeautifulSoup(full_html, "html.parser")

    # 이름, HOFm, WAR 추출
    p_name = full_soup.find("h1").text.strip() if full_soup.find("h1") else query_name
    
    def extract_stat(label):
        # 텍스트 기반으로 수치 추출
        target = full_soup.find(string=re.compile(label))
        if target:
            text = target.parent.get_text()
            match = re.search(r"(\d+\.\d+|\d+)", text)
            return float(match.group(1)) if match else 0.0
        return 0.0

    war = extract_stat("WAR")
    hofm = extract_stat("Hall of Fame Monitor")
    
    return p_name, hofm, war

# --- 2. 실행부 ---
st.title("🏛️ MLB HOF 데이터 로봇")

name_input = st.text_input("선수 영문 이름을 입력 (예: Mike Trout)", "")

if name_input:
    with st.spinner("데이터 기차 타러 가는 중..."):
        try:
            name, hofm, war = fetch_player_final(name_input)
            
            if hofm == 0 and war == 0:
                st.warning(f"'{name_input}'의 데이터를 찾지 못했습니다. 전체 이름을 정확히 적어주세요 (예: Albert Pujols).")
            else:
                st.success(f"데이터 로드 완료: {name}")
                st.metric("Career WAR", war)
                st.metric("HOF Monitor", hofm)
                
                # 확률 계산 (간이 로직)
                prob = min(99.9, (hofm * 0.4) + (war * 0.4))
                st.write(f"### 🔮 예상 헌액 확률: {prob:.1f}%")
                st.progress(prob / 100)
                
        except Exception as e:
            st.error("앗! 사이트 접속 중에 문제가 생겼어요. 다시 시도해 주세요.")
