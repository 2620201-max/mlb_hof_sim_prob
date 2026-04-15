import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
import re

# --- 1. 데이터 수집 함수 (더 강력한 버전) ---
def fetch_player_data_fix(query_name):
    query = query_name.replace(" ", "+")
    url = f"https://www.baseball-reference.com/search/search.fcgi?search={query}"
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    res = requests.get(url, headers=headers, timeout=10)
    
    # 1. 주석 데이터 강제 결합 (많은 지표가 주석 안에 숨어있음)
    soup = BeautifulSoup(res.text, "html.parser")
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    full_html = res.text + "".join(comments)
    full_soup = BeautifulSoup(full_html, "html.parser")

    # 2. 이름 추출
    name_tag = full_soup.find("h1")
    p_name = name_tag.text.strip() if name_tag else query_name

    # 3. WAR 및 HOFm 추출 (정규식 강화)
    def extract_val(label):
        # "Hall of Fame Monitor", "WAR" 등의 텍스트 바로 뒤의 숫자를 찾음
        target = full_soup.find(string=re.compile(label))
        if target:
            parent_text = target.parent.get_text()
            # 숫자(소수점 포함)만 골라내기
            match = re.search(r"(\d+\.\d+|\d+)", parent_text)
            if match:
                return float(match.group(1))
        return 0.0

    war = extract_val("WAR")
    hofm = extract_val("Hall of Fame Monitor")
    
    return p_name, hofm, war

# --- 2. 득표율 시뮬레이션 로직 ---
def get_shares(hofm, war):
    # 트라웃급 지표면 무조건 높은 점수
    first = (hofm * 0.4) + (war * 0.3) + 20
    final = first + (10 if war > 60 else 5)
    return min(99.9, max(1.0, first)), min(99.9, max(1.0, final))

# --- 3. UI 구성 ---
st.title("🏛️ MLB HOF 통합 진단기")

name_input = st.text_input("선수 영문 이름을 입력하세요", placeholder="Mike Trout")

if name_input:
    with st.spinner("데이터 분석 중..."):
        try:
            name, hof_score, war_score = fetch_player_data_fix(name_input)
            
            # 둘 다 0이면 검색 실패로 간주
            if hof_score == 0 and war_score == 0:
                st.error("데이터를 찾을 수 없습니다. 철자를 확인하거나 다른 선수를 입력해 주세요.")
            else:
                f_share, l_share = get_shares(hof_score, war_score)
                
                st.header(f"⚾ {name}")
                c1, c2 = st.columns(2)
                c1.metric("HOF Monitor", hof_score)
                c2.metric("Career WAR", war_score)
                
                st.subheader("🗳️ 예상 득표율")
                st.write(f"1년 차: {f_share:.1f}%")
                st.progress(f_share / 100)
                st.write(f"최종 예상: {l_share:.1f}%")
                st.progress(l_share / 100)
                
        except Exception as e:
            st.error(f"오류 발생: {e}")
