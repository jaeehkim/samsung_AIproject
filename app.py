"""
==============================
[스트림릿 실습 가이드]

- 1~6번 섹션: 데이터 처리, 점수 계산, LLM 호출 등
    → "백엔드/로직 영역"(수정 X)
    
- 7번 섹션: Streamlit 화면 구성(UI)
    → "프론트/UI 영역" (핵심 수정 포인트)
       (제목, 설명, 탭 구성, 표시되는 컬럼, 버튼/라디오 등)
       
==============================
"""

import os
from pathlib import Path
import math
import base64  # ✅ 추가

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# RAG 모듈 import 상단
from rag_module import (
    get_rag_system,
    extract_file_urls,
    display_source_documents
)
from streamlit_calendar import calendar  # ✅ 추가

primaryColor = "#0c4da2"
backgroundColor = "#f5f7fb"
secondaryBackgroundColor = "#ffffff"

# ==========================
# 0. Pretendard 폰트 적용
# ==========================
pretendard_css = """
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

html, body, [class*="css"] {
    font-family: 'Pretendard', sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}
</style>
"""
st.markdown(pretendard_css, unsafe_allow_html=True)


# ==========================================
# 1. 경로 / 상수 설정 (수정 X)
# ==========================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

DEPT_FILE = DATA_DIR / "부서별 역량 및 관심분야 예시.csv"
API_FILE = DATA_DIR / "사업공고 API,크롤링.csv"

# 🔹 신규: 입찰공고 원천 데이터 파일
BID_FILE = DATA_DIR / "입찰공고서비스_re.csv"

# 부서 키워드를 만들 때 참고할 컬럼 목록
KEYWORD_SOURCE_COLS = ["핵심역량", "관심지원분야", "주요키워드"]


# ==========================================
# 2. 공통 유틸 함수 (수정 X)
# ==========================================
def clean_csv(path: Path) -> pd.DataFrame:
    """
    0행에 컬럼명이 있고, 1행부터 실제 데이터가 있는 CSV를
    깔끔하게 읽어오는 함수.
    """
    df_raw = pd.read_csv(path, header=None)
    df_raw.columns = df_raw.iloc[0]  # 0행 → 컬럼명
    df = df_raw.iloc[1:].reset_index(drop=True)  # 1행부터 데이터
    return df


def split_keywords(text: str):
    """
    여러 종류의 구분자( / , ; · ㆍ | )를 기준으로
    문자열을 '키워드 리스트'로 쪼개주는 함수.
    """
    if pd.isna(text):
        return []

    # 여러 구분자를 하나의 구분자(|)로 통일
    for sep in ["/", ",", ";", "·", "ㆍ", "|"]:
        text = text.replace(sep, "|")

    # 공백 제거 + 빈 문자열 제거
    parts = [p.strip() for p in text.split("|") if p.strip()]
    return parts


# ==========================================
# 3. 데이터 로딩 함수 (수정 X)
# ==========================================
@st.cache_data
def load_department_profiles() -> pd.DataFrame:
    """부서 프로필 CSV 로딩"""
    return clean_csv(DEPT_FILE)


@st.cache_data
def load_api_sources() -> pd.DataFrame:
    """사업공고 API/크롤링 소스 메타데이터 CSV 로딩"""
    return clean_csv(API_FILE)


# ==========================================
# 4. 부서별 키워드 생성 (IDF 기반) ✅
# ==========================================
def build_department_keywords(df: pd.DataFrame) -> pd.DataFrame:
    """
    부서별로 검색용 키워드 리스트(검색키워드)를 생성하고
    문자열 버전(검색키워드_문자열)까지 추가.
    IDF(Inverse Document Frequency) 방식으로 부서별 특화 키워드에 가중치 부여.
    """

    def row_to_keywords(row):
        texts = []
        # 지정된 컬럼들에서 텍스트 모으기
        for col in KEYWORD_SOURCE_COLS:
            if col in row and pd.notna(row[col]):
                texts.append(str(row[col]))
        if not texts:
            return []
        # 여러 컬럼을 하나로 합친 후, 다시 다양한 구분자 처리
        combined = " / ".join(texts)
        for sep in ["/", ",", "·", "ㆍ", "|"]:
            combined = combined.replace(sep, "|")
        parts = [p.strip() for p in combined.split("|") if p.strip()]
        # 중복 제거
        seen = set()
        keywords = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                keywords.append(p)
        return keywords
    
    # 1단계: 기본 키워드 추출
    df = df.copy()
    df["검색키워드"] = df.apply(row_to_keywords, axis=1)
    
    # 2단계: IDF 계산 - 부서별 특화도 측정
    keyword_dept_count = {}
    total_depts = len(df)
    
    for keywords in df["검색키워드"]:
        unique_keywords = set(keywords)
        for kw in unique_keywords:
            keyword_dept_count[kw] = keyword_dept_count.get(kw, 0) + 1
    
    # IDF 점수 계산
    keyword_idf = {
        kw: math.log(total_depts / count) 
        for kw, count in keyword_dept_count.items()
    }
    
    # 3단계: 키워드 재정렬 - IDF 높은 순 (특화 키워드 우선)
    def reorder_by_idf(keywords):
        if not keywords:
            return []
        sorted_kw = sorted(keywords, key=lambda k: keyword_idf.get(k, 0), reverse=True)
        return sorted_kw
    
    df["검색키워드"] = df["검색키워드"].apply(reorder_by_idf)
    
    # 4단계: 문자열 변환
    df["검색키워드_문자열"] = df["검색키워드"].apply(
        lambda ks: ", ".join(ks) if isinstance(ks, list) else ""
    )
    
    # (옵션) 키워드별 IDF 점수 리스트
    df["키워드_IDF점수"] = df["검색키워드"].apply(
        lambda ks: [round(keyword_idf.get(k, 0), 2) for k in ks] if isinstance(ks, list) else []
    )
    
    return df


# ==========================================
# 5. 사업-부서 매칭 점수 계산 (겹치는 키워드 개수) ✅
# ==========================================
def score_projects_for_department(
    dept_keywords: list[str],
    projects_df: pd.DataFrame
) -> pd.DataFrame:
    """
    부서의 검색키워드 리스트와 각 사업공고의 키워드리스트를 비교해
    매칭 개수를 점수로 계산하고, 정렬된 DataFrame을 반환.
    """
    if not isinstance(dept_keywords, list):
        dept_keywords = []

    # 소문자 세트로 변환해서 비교 (대소문자 섞여도 안전하게)
    dept_set = {str(k).strip().lower() for k in dept_keywords if str(k).strip()}

    def compute_score(row: pd.Series):
        proj_kws = row.get("키워드리스트", [])
        proj_set = {str(k).strip().lower() for k in proj_kws if str(k).strip()}

        overlap = dept_set & proj_set
        score = len(overlap)
        return score, list(overlap)

    scored_df = projects_df.copy()
    scored_df[["매칭점수", "매칭키워드"]] = scored_df.apply(
        lambda r: pd.Series(compute_score(r)), axis=1
    )

    # 점수 높은 순으로 정렬
    scored_df = scored_df.sort_values(by="매칭점수", ascending=False).reset_index(drop=True)
    return scored_df


# ==========================================
# 6. LLM 초기화 (수정 X)
# ==========================================
@st.cache_resource
def get_llm():
    """
    .env 파일에 설정된 환경변수를 사용해 LLM 초기화.
    OPENAI_API_KEY, LLM_BASE_URL이 없으면 Streamlit 경고 표시.
    """
    load_dotenv(".env", override=True)

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")

    if not api_key or not base_url:
        st.warning(".env에 OPENAI_API_KEY 또는 LLM_BASE_URL 이 설정되어 있지 않습니다.")
        return None

    llm = ChatOpenAI(
        model="openai/gpt-5",  # 엘리스 환경에 맞는 모델명 사용
        openai_api_key=api_key,
        base_url=base_url,
        temperature=1
    )
    return llm


# ==========================================
# 6-1. 제안서 프롬프트 생성
# ==========================================
def build_proposal_prompt(dept_row: pd.Series, project_row: pd.Series) -> str:
    """
    부서 정보 + 사업공고 정보를 받아
    제안서 작성을 위한 LLM 프롬프트 문자열 생성.
    """
    # ----- 부서 정보 -----
    dept_name = dept_row.get("부서명", "")
    dept_core = dept_row.get("핵심역량", "")
    dept_interest = dept_row.get("관심지원분야", "")
    dept_keywords = dept_row.get("검색키워드", [])
    dept_role = dept_row.get("참여 가능 역할", "")
    dept_recent = dept_row.get("최근수행사업 예시", "")

    # ----- 사업 정보 -----
    proj_title = project_row.get("사업명", "")
    proj_desc = project_row.get("사업설명", "")
    proj_field = project_row.get("분야", "")
    proj_keywords = project_row.get("주요키워드", "")
    proj_ministry = project_row.get("주관부처", "")
    proj_deadline = project_row.get("마감일", "")

    prompt = f"""
당신은 대형 건설사의 사업기획 담당자로서, 정부지원사업 제안서를 작성하는 역할을 맡고 있습니다.
아래 정보를 바탕으로, 해당 부서가 이 사업에 지원하기 위한 제안서 초안을 한국어로 작성하세요.

[부서 정보]
- 부서명: {dept_name}
- 핵심역량: {dept_core}
- 관심 지원분야: {dept_interest}
- 검색 키워드: {", ".join(dept_keywords) if isinstance(dept_keywords, list) else dept_keywords}
- 참여 가능 역할: {dept_role}
- 최근 수행사업 예시: {dept_recent}

[사업공고 정보]
- 사업명: {proj_title}
- 주관부처: {proj_ministry}
- 분야: {proj_field}
- 마감일: {proj_deadline}
- 주요 키워드: {proj_keywords}
- 사업 설명: {proj_desc}

[작성 요구사항]
1. 전체 분량은 A4 기준 1페이지 정도로 작성합니다.
2. 아래 섹션 구조를 따릅니다.
   1) 사업 개요 요약
   2) 우리 부서의 참여 배경 및 필요성
   3) 우리 부서의 강점 및 차별화 포인트
   4) 수행 내용 및 추진 전략 (간단한 단계 구조)
   5) 기대 효과 (정량/정성적 효과 위주)
3. 실제 공공기관에 제출할 수 있을 만큼 자연스럽고 격식 있는 문체로 작성합니다.
4. 글머리표를 적절히 활용해 가독성을 높여 작성합니다.

위 조건을 만족하는 제안서 초안을 작성하세요.
"""
    return prompt


# ==========================================
# 7. Streamlit UI
# ==========================================

# 7-1. 페이지 기본 설정 & 헤더 --------------------
st.set_page_config(
    page_title="「지원사업 제안서 Agent」",
    layout="wide",
)

# 🔹 메인 KV 배너 / 로고 이미지 처리
def img_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

kv_path = DATA_DIR / "배너_fin.png"
logo_path = DATA_DIR / "samsung_logo2.png"

kv_b64 = img_to_base64(kv_path)
logo_b64 = img_to_base64(logo_path)

# 메인 테마 컬러
PRIMARY_COLOR = "#0c4da2"
PRIMARY_COLOR_DARK = "#093570"
PRIMARY_COLOR_LIGHT = "#e3edf9"
samsung_blue = PRIMARY_COLOR  # 슬라이더에서도 동일 컬러 사용

# ---------------- 기본 테마 스타일 ----------------
st.markdown(
    f"""
    <style>
        :root {{
            --primary-color: {PRIMARY_COLOR};
        }}

        .stApp {{
            background-color: #f5f7fb;
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: {PRIMARY_COLOR};
        }}
        p, label, span {{
            color: #1e293b;
        }}

        a, a:visited {{
            color: {PRIMARY_COLOR};
        }}
        a:hover {{
            color: {PRIMARY_COLOR_DARK};
        }}

        /* ---------------- 버튼 ---------------- */
        .stButton > button {{
            background-color: {PRIMARY_COLOR} !important;
            color: #ffffff !important;               /* 버튼 기본 텍스트 색 */
            border-radius: 6px !important;
            border: 1px solid {PRIMARY_COLOR} !important;
            padding: 0.4rem 0.9rem !important;
        }}

        /* 버튼 안쪽에 들어가는 모든 텍스트 요소도 흰색으로 고정 */
        .stButton > button * {{
            color: #ffffff !important;
        }}

        .stButton > button:hover {{
            background-color: {PRIMARY_COLOR_DARK} !important;
            border-color: {PRIMARY_COLOR_DARK} !important;
            color: #ffffff !important;
        }}

        /* 🔹 메인 상단 탭만 pill 형태 (main-tabs-wrapper 안에 있는 탭) */
        .main-tabs-wrapper .stTabs {{
            margin-top: 1.2rem;
        }}
        .main-tabs-wrapper .stTabs [data-baseweb="tab-list"] {{
            background-color: #ffffff;
            padding: 0.6rem;
            border-radius: 999px;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
            gap: 0.4rem;
            border: 1px solid #e2e8f0;
        }}
        .main-tabs-wrapper .stTabs button[data-baseweb="tab"] {{
            font-weight: 500;
            font-size: 0.95rem;
            padding: 0.4rem 1.3rem !important;
            border-radius: 999px !important;
            color: #64748b !important;
            background-color: transparent !important;
            border: none !important;
            border-bottom: none !important;
            transition: background-color 0.15s ease, 
                        color 0.15s ease, 
                        box-shadow 0.15s ease,
                        transform 0.08s ease;
        }}
        .main-tabs-wrapper .stTabs button[data-baseweb="tab"]:hover {{
            background-color: #f1f5f9 !important;
            color: #0f172a !important;
        }}
        .main-tabs-wrapper .stTabs button[data-baseweb="tab"][aria-selected="true"] {{
            background-color: {PRIMARY_COLOR} !important;
            color: #ffffff !important;
            box-shadow: 0 6px 14px rgba(15, 23, 42, 0.18);
            transform: translateY(-1px);
        }}
        .main-tabs-wrapper .stTabs [data-baseweb="tab-highlight"] {{
            display: none !important;
        }}

        /* 🔹 나머지 탭(내부 탭들: 부서/소스 데이터, RAG 안쪽 등)은 기본 스타일 유지 */
        .stTabs:not(.main-tabs-wrapper .stTabs) [data-baseweb="tab-list"] {{
            /* 별도 스타일 안 줘서 기본 Streamlit 느낌 그대로 사용 */
        }}

        div[data-baseweb="select"] > div {{
            border-radius: 6px !important;
        }}
        input, textarea {{
            border-radius: 4px !important;
        }}

        thead tr th {{
            background-color: {PRIMARY_COLOR_LIGHT} !important;
            color: #111827 !important;
        }}

        .kv-header {{
            position: relative;
            width: 100%;
            padding: 40px;
            border-radius: 10px;
            background-image: url("data:image/png;base64,{kv_b64}");
            background-size: cover;
            background-position: center;
            display: flex;
            justify-content: space-between;
            align-items: center;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        }}

        .kv-header::before {{
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(
                120deg,
                rgba(12,77,162,0.88),
                rgba(12,77,162,0.55),
                rgba(15,23,42,0.35)
            );
            z-index: 1;
        }}

        .kv-header > * {{
            position: relative;
            z-index: 2;
        }}

        .kv-logo {{
            position: absolute;
            top: 20px;
            right: 20px;
            z-index: 3;
        }}

        .kv-logo img {{
            width: 160px;
            border-radius: 0 !important;
        }}

        .title-45 {{
            font-size: 45px;
            font-weight: 700;
            color: #ffffff;
            line-height: 1.4;
        }}

        .text-20 {{
            font-size: 20px;
            font-weight: 400;
            color: #ffffff;
            line-height: 1.6;
        }}

        .feature-title-20 {{
            font-size: 20px;
            font-weight: 700;
            color: #ffffff;
            margin-top: 20px;
            line-height: 1.5;
        }}

        .feature-desc-18 {{
            font-size: 18px;
            font-weight: 400;
            color: #ffffff;
            margin-bottom: 20px;
            line-height: 1.5;
        }}

        .indent-40 {{ padding-left: 40px; }}
        .indent-60 {{ padding-left: 60px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- 슬라이더 전체 커스텀 (thumb + 트랙) ----------------
st.markdown(
    f"""
    <style>
        /* 슬라이더 바 색상 변경 */
        div.stSlider > div[data-baseweb="slider"] > div > div {{
            background: {samsung_blue} !important;
        }}
        /* 슬라이더 손잡이(동그라미) 색상 변경 */
        div.stSlider > div[data-baseweb="slider"] > div > div[role="slider"] {{
            background-color: {samsung_blue} !important;
            border-color: {samsung_blue} !important;
        }}
        /* (선택사항) 슬라이더 값 텍스트 색상 변경 */
        div[data-testid="stThumbValue"] {{
            color: {samsung_blue} !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# 🧩 헤더 HTML 블록
st.markdown(
    f"""
    <div class="kv-header">
        <!-- 전체 텍스트 블록 -->
        <div>
            <!-- 제목 (45px) -->
            <div class="title-45">
                「사업경쟁력 강화 및 사업개발 제안 Agent」  WE:
            </div>
            <br>
            <!-- 소개문 (20px) -->
            <div class="text-20">
                바쁜 건설 현장과 치열한 입찰 경쟁 속에서, 우리 직원들의 업무 효율을 극대화하고 
                수주 경쟁력을 높여줄 건설 전문 AI 입찰 제안서 에이전트를 소개합니다.
            </div>
            <!-- 기능 1 -->
            <div class="feature-title-20 indent-40">
                💡 전국의 입찰 정보를 한눈에, 자동 수집 및 분류
            </div>
            <div class="feature-desc-18 indent-60">
                국내에서 발생하는 공공 및 민간의 모든 입찰 사업 정보를 실시간으로, 그리고 자동으로 수집합니다.
            </div>
            <!-- 기능 2 -->
            <div class="feature-title-20 indent-40">
                💡 혁신적인 제안서 초안 자동 생성
            </div>
            <div class="feature-desc-18 indent-60">
                목록에서 원하는 입찰 사업을 선택하기만 하면, 해당 사업의 요구 사항,
                발주처의 특성, 과거 수주 성공 사례 등의 데이터를 분석하여
                입찰 제안서 초안을 자동으로 생성합니다.
            </div>
            <!-- 마지막 설명 -->
            <div class="text-20">
                이 에이전트는 직원들이 불필요한 반복 업무에서 벗어나,
                수주 경쟁률을 높이는 핵심 역량 강화에 집중할 수 있도록 돕는 강력한 파트너입니다.<br>
                👉 지금 바로 사업부를 선택하거나 최신 입찰 목록을 확인해보세요.
            </div>
        </div>
        <!-- 오른쪽 로고 영역 -->
        <div class="kv-logo">
            <img src="data:image/png;base64,{logo_b64}">
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ✅ 제안서 탭으로 점프 제어용 플래그
if "jump_to_proposal" not in st.session_state:
    st.session_state["jump_to_proposal"] = False

# ✅ RAG 탭으로 점프 제어용 플래그
if "jump_to_rag" not in st.session_state:
    st.session_state["jump_to_rag"] = False

# 탭 자동 전환 스크립트
if st.session_state["jump_to_proposal"]:
    st.session_state["jump_to_proposal"] = False
    st.markdown(
        """
        <script>
        const tabButtons = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
        if (tabButtons && tabButtons.length > 0) {
            const target = Array.from(tabButtons).find(
                el => el.innerText.includes('4) 제안서 초안 생성')
            );
            if (target) {
                target.click();
            }
        }
        </script>
        """,
        unsafe_allow_html=True,
    )

if st.session_state["jump_to_rag"]:
    st.session_state["jump_to_rag"] = False
    st.markdown(
        """
        <script>
        const tabButtons = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
        if (tabButtons && tabButtons.length > 0) {
            const target = Array.from(tabButtons).find(
                el => el.innerText.includes('5) RAG 기반 문서 내용 질의하기')
            );
            if (target) {
                target.click();
            }
        }
        </script>
        """,
        unsafe_allow_html=True,
    )

# 7-2. 데이터 로딩 ------------------------
dept_df = load_department_profiles()
api_df = load_api_sources()
dept_df = build_department_keywords(dept_df)

# 🔹 입찰공고 원천 데이터 로딩
projects_raw = pd.read_csv(BID_FILE)

# 🔹 이 앱에서 사용할 컬럼명으로 매핑
projects_df = projects_raw.rename(
    columns={
        "입찰공고번호": "공고ID",
        "입찰공고명": "사업명",
        "수요기관명": "주관부처",
        "입찰마감일시": "마감일",
        "주공종명": "분야",
        "입찰공고상세URL": "공고링크",
    }
)

# 🔹 '주요키워드' 생성
def make_project_keywords(row):
    fields = []
    for col in ["사업명", "분야", "공사현장지역명"]:
        val = row.get(col)
        if val is None:
            continue
        val_str = str(val).strip()
        if val_str:
            fields.append(val_str)
    return " / ".join(fields)

projects_df["주요키워드"] = projects_df.apply(make_project_keywords, axis=1)

# 🔹 '키워드리스트' 생성
projects_df["키워드리스트"] = projects_df["주요키워드"].apply(split_keywords)

# 🔹 LLM 초기화
llm = get_llm()

# 부서 목록 / 기본값
dept_names = dept_df["부서명"].dropna().tolist()
DEFAULT_DEPT_NAME = dept_names[0] if dept_names else None


# ==========================================
# 7-4. 메인 영역: 탭 레이아웃
# ==========================================
# 🔹 메인 탭만 따로 감싸는 래퍼
tab_calendar, tab3, tab4, tab5, tab_data = st.tabs(
    [
        "0) 홈 · 대시보드",
        "3) 추천 사업공고",
        "4) 제안서 초안 생성",
        "5) RAG 기반 문서 내용 질의하기",
        "6) 부서/소스 데이터 상세 보기",
    ]
)

# --------------------------
# [Tab 0] 홈 · 대시보드
# --------------------------
with tab_calendar:
    st.markdown("### 📊 전체 사업공고 대시보드")
    st.caption("이번 달 마감 현황과 D-Day 리스트를 한눈에 확인합니다.")

    today = pd.Timestamp.today().normalize()

    proj_df_for_calendar = projects_df.copy()
    proj_df_for_calendar["마감일_dt"] = pd.to_datetime(
        proj_df_for_calendar["마감일"], errors="coerce"
    )
    proj_df_for_calendar = proj_df_for_calendar.dropna(subset=["마감일_dt"])

    this_month_mask = (
        (proj_df_for_calendar["마감일_dt"].dt.year == today.year)
        & (proj_df_for_calendar["마감일_dt"].dt.month == today.month)
    )
    this_month_count = int(this_month_mask.sum())

    # 이번 달 마감 공고의 추정가격 총액
    if "추정가격" in proj_df_for_calendar.columns:
        est_series = (
            proj_df_for_calendar.loc[this_month_mask, "추정가격"]
            .astype(str)
            .str.replace(",", "")
            .str.replace(" ", "")
        )
        est_values = pd.to_numeric(est_series, errors="coerce")
        this_month_est_sum = est_values.sum(skipna=True)
    else:
        this_month_est_sum = 0

    this_month_est_sum_label = f"{int(this_month_est_sum):,} 원" if this_month_est_sum else "0 원"

    dday_series = (proj_df_for_calendar["마감일_dt"] - today).dt.days
    upcoming_10_mask = (dday_series >= 0) & (dday_series <= 10)
    upcoming_10_count = int(upcoming_10_mask.sum())

    month_label = f"{today.year}년 {today.month}월"

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div style="
                background-color:#f8fafc;
                padding:16px 20px;
                border-radius:12px;
                border:1px solid #e2e8f0;
            ">
                <div style="font-size:0.9rem; color:#64748b;">
                    이번 달 마감 공고 수 ({month_label})
                </div>
                <div style="font-size:2rem; font-weight:700; margin-top:4px; color:#0f172a;">
                    {this_month_count}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div style="
                background-color:#FFD6D6;
                padding:16px 20px;
                border-radius:12px;
                border:1px solid #e2e8f0;
            ">
                <div style="font-size:0.9rem; color:#854d0e;">
                    앞으로 10일 이내 마감 예정
                </div>
                <div style="font-size:2rem; font-weight:700; margin-top:4px; color:#713f12;">
                    {upcoming_10_count}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div style="
                background-color:#f1f5f9;
                padding:16px 20px;
                border-radius:12px;
                border:1px solid #cbd5f5;
            ">
                <div style="font-size:0.9rem; color:#475569;">
                    이번 달 마감 공고 추정가격 합계 ({month_label})
                </div>
                <div style="font-size:2rem; font-weight:700; margin-top:4px; color:#0f172a;">
                    {this_month_est_sum_label}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    left, right = st.columns([1.1, 1])

    # ===== 왼쪽: 캘린더 =====
    with left:
        st.markdown("#### 📅 마감일 캘린더")

        calendar_events = []
        for _, row in proj_df_for_calendar.iterrows():
            date_str = row["마감일_dt"].strftime("%Y-%m-%d")
            title = f"{row.get('사업명', '')}"

            event = {
                "title": title,
                "start": date_str,
                "color": "#93c5fd",
                "textColor": "black",
            }
            calendar_events.append(event)

        calendar_options = {
            "initialView": "dayGridMonth",
            "locale": "ko",
            "height": 850,
            "contentHeight": 850,
            "expandRows": True,
            "dayMaxEventRows": 5,
            "headerToolbar": {
                "left": "prev,next today",
                "center": "title",
                "right": "dayGridMonth,dayGridWeek,dayGridDay",
            },
        }

        calendar(
            events=calendar_events,
            options=calendar_options,
            key="project_deadline_calendar",
        )

    # ===== 오른쪽: 전체 공고 현황 + 행 클릭 상세 =====
    with right:
        st.markdown("#### 📋 전체 공고 리스트")

        df_table = proj_df_for_calendar.copy()
        df_table["Dday_num"] = (df_table["마감일_dt"] - today).dt.days

        def format_dday(n):
            if pd.isna(n):
                return ""
            n = int(n)
            if n > 0:
                return f"D-{n}"
            elif n == 0:
                return "D-Day"
            else:
                return f"D+{abs(n)}"

        df_table["디데이"] = df_table["Dday_num"].apply(format_dday)

        df_table["is_past"] = df_table["Dday_num"] < 0
        df_table["abs_days"] = df_table["Dday_num"].abs()
        df_table = df_table.sort_values(
            ["is_past", "abs_days"], ascending=[True, True]
        )

        search_query = st.text_input(
            "사업명으로 검색",
            value="",
            placeholder="사업명에 포함된 키워드를 입력하세요.",
        )

        if search_query:
            df_view = df_table[
                df_table["사업명"].astype(str).str.contains(
                    search_query, case=False, na=False
                )
            ]
        else:
            df_view = df_table

        desired_cols = ["디데이", "공고ID", "사업명", "주관부처", "마감일"]
        cols_to_show = [c for c in desired_cols if c in df_view.columns]

        df_view_display = df_view[cols_to_show].reset_index(drop=True)

        soon_mask = (df_view["Dday_num"] >= 0) & (df_view["Dday_num"] <= 10)
        soon_mask = soon_mask.reset_index(drop=True)

        def highlight_soon(row):
            if soon_mask.iloc[row.name]:
                return ["background-color: #FFD6D6"] * len(row)
            return [""] * len(row)

        styled = df_view_display.style.apply(highlight_soon, axis=1)

        # ✅ 여기서는 st.dataframe + on_select 사용 (버전 호환)
        event = st.dataframe(
            styled,
            use_container_width=True,
            height=480,
            on_select="rerun",
            selection_mode="single-row",
            key="home_table",
        )

        selected_rows = event.selection.rows

        st.markdown("")

        if selected_rows:
            row_idx = selected_rows[0]
            selected_row = df_view_display.iloc[row_idx]
            selected_id = df_view.iloc[row_idx]["공고ID"]
            selected_id = str(selected_id)

            st.markdown(f"##### 📄 선택한 공고 상세정보")

            raw_row = None
            if "입찰공고번호" in projects_raw.columns:
                tmp = projects_raw[projects_raw["입찰공고번호"].astype(str) == selected_id]

                if len(tmp) > 0:
                    raw_row = tmp.iloc[0]

            if raw_row is not None:
                key_cols_candidate = [
                    "입찰공고번호",
                    "입찰공고명",
                    "수요기관명",
                    "입찰공고등록일시",
                    "입찰마감일시",
                    "계약방법명",
                    "입찰방식명",
                    "공사현장지역명",
                    "추정가격",
                    "기초금액",
                    "입찰공고상세URL",
                ]
                key_cols = [c for c in key_cols_candidate if c in projects_raw.columns]

                detail_df = (
                    pd.DataFrame(
                        {
                            "항목": key_cols,
                            "값": [raw_row[c] for c in key_cols],
                        }
                    )
                    if key_cols
                    else pd.DataFrame(columns=["항목", "값"])
                )

                st.data_editor(
                    detail_df,
                    use_container_width=True,
                    hide_index=True,
                    disabled=True,
                )

                if "입찰공고상세URL" in projects_raw.columns:
                    url = raw_row.get("입찰공고상세URL", "")
                    if isinstance(url, str) and url.strip():
                        st.markdown(
                            f"[🔗 나라장터 상세 페이지 열기]({url})",
                            unsafe_allow_html=True,
                        )

            else:
                st.info("선택한 공고의 원본 상세정보를 찾을 수 없습니다.")
        else:
            st.caption("리스트에서 공고를 클릭하면 아래에 상세 정보가 표시됩니다.")


# --------------------------
# [Tab 3] 추천 결과
# --------------------------
with tab3:
    st.markdown("### 🎯 부서별 추천 사업공고")
    st.caption("부서 역량·관심 키워드 기반으로, 참여 적합도가 높은 사업공고를 추천합니다.")

    setting_col, main_col = st.columns([1, 3])

    # 왼쪽: 부서 선택 + Top N
    with setting_col:
        st.markdown("#### 1단계. 부서 선택")

        dept_select_options = ["(부서를 선택하세요)"] + dept_names

        prev_dept = st.session_state.get("selected_dept_name")
        if prev_dept and prev_dept in dept_names:
            default_idx = dept_names.index(prev_dept) + 1
        else:
            default_idx = 0

        selected_label = st.selectbox(
            "추천을 조회할 부서를 선택하세요",
            dept_select_options,
            index=default_idx,
            key="recommend_dept_select",
        )

        if selected_label != "(부서를 선택하세요)":
            selected_dept_name = selected_label
            # 전역 부서 상태 동기화 (Tab4/5 기본값으로 사용)
            st.session_state["selected_dept_name"] = selected_dept_name
            st.session_state["proposal_dept_select"] = selected_dept_name
            st.session_state["rag_dept_select"] = selected_dept_name
        else:
            selected_dept_name = None

        st.markdown("#### 2단계. 추천 개수 설정")

        top_n = st.slider(
            "표시할 추천 개수 (Top N)",
            min_value=1,
            max_value=10,
            value=st.session_state.get("selected_top_n", 5),
            step=1,
            key="selected_top_n",
        )

    # 오른쪽: 추천 리스트
    with main_col:
        if not selected_dept_name:
            st.info("왼쪽에서 부서를 선택하면 추천 결과가 표시됩니다.")
        else:
            if selected_dept_name in dept_df["부서명"].values:
                dept_row = dept_df[dept_df["부서명"] == selected_dept_name].iloc[0]
            else:
                dept_row = dept_df.iloc[0]
                selected_dept_name = dept_row.get("부서명", "")
                st.session_state["selected_dept_name"] = selected_dept_name

            dept_keywords = dept_row.get("검색키워드", [])

            scored_projects_df = score_projects_for_department(dept_keywords, projects_df)
            total_cnt = len(projects_df)
            matched_cnt = (scored_projects_df["매칭점수"] > 0).sum()

            top_scored = scored_projects_df.head(top_n).copy()

            today_rec = pd.Timestamp.today().normalize()
            top_scored["마감일_dt"] = pd.to_datetime(
                top_scored["마감일"], errors="coerce"
            )
            top_scored["Dday_num"] = (top_scored["마감일_dt"] - today_rec).dt.days

            def format_dday(n):
                if pd.isna(n):
                    return ""
                n = int(n)
                if n > 0:
                    return f"D-{n}"
                elif n == 0:
                    return "D-Day"
                else:
                    return f"D+{abs(n)}"

            top_scored["디데이"] = top_scored["Dday_num"].apply(format_dday)

            st.markdown(
                f"""
                <div style="
                    padding:10px 14px;
                    border-radius:10px;
                    background-color:#f8fafc;
                    border:1px solid #e2e8f0;
                    margin-bottom:10px;
                    font-size:0.95rem;
                ">
                    전체 후보 사업 수: <b>{total_cnt}개</b><br/>
                    부서 키워드와 1개 이상 매칭된 사업 수: <b>{matched_cnt}개</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### 추천 Top 리스트")

            display_cols = [
                "디데이",
                "공고ID",
                "사업명",
                "주관부처",
                "공사현장지역명",
                "추정가격",
                "마감일",
                "매칭점수",
                "매칭키워드",
            ]
            cols_to_show = [c for c in display_cols if c in top_scored.columns]

            df_view_display = top_scored[cols_to_show].reset_index(drop=True)

            soon_mask = (
                (top_scored["Dday_num"] >= 0)
                & (top_scored["Dday_num"] <= 10)
            )
            soon_mask = soon_mask.reset_index(drop=True)

            def highlight_soon(row):
                if soon_mask.iloc[row.name]:
                    return ["background-color: #fef9c3"] * len(row)
                return [""] * len(row)

            styled = df_view_display.style.apply(highlight_soon, axis=1)

            # ✅ 선택이 필요한 추천 리스트 → st.dataframe 사용
            event = st.dataframe(
                styled,
                use_container_width=True,
                height=380,
                on_select="rerun",
                selection_mode="single-row",
                key="recommend_table",
            )

            st.markdown("---")

            selected_rows = event.selection.rows

            if selected_rows:
                row_idx = selected_rows[0]
                selected_row = df_view_display.iloc[row_idx]
                selected_id = str(selected_row["공고ID"])

                st.markdown(f"##### 📄 선택한 추천 공고 상세")

                # 👉 여기서만 부서/공고를 세션에 저장하는 버튼 두 개
                st.markdown("")
                col_p, col_r = st.columns(2)

                with col_p:
                    st.markdown("**1) 바로 제안서 작성으로 보내기**")
                    if st.button(
                        "🧠 이 공고로 제안서 초안 생성",
                        key=f"go_proposal_{selected_id}",
                        use_container_width=True,
                        type="primary",
                    ):
                        st.session_state["selected_project_id"] = selected_id
                        st.session_state["selected_dept_name"] = selected_dept_name
                        st.session_state["jump_to_proposal"] = True
                        st.rerun()

                with col_r:
                    st.markdown("**2) 공고 문서 내용 먼저 살펴보기**")
                    if st.button(
                        "📚 이 공고 문서 RAG로 열기",
                        key=f"go_rag_{selected_id}",
                        use_container_width=True,
                        type="secondary",
                    ):
                        st.session_state["selected_project_id"] = selected_id
                        st.session_state["selected_dept_name"] = selected_dept_name
                        st.session_state["jump_to_rag"] = True
                        st.rerun()

                # 원본 CSV 상세
                raw_row = None
                if "입찰공고번호" in projects_raw.columns:
                    tmp = projects_raw[projects_raw["입찰공고번호"].astype(str) == selected_id]
                    if len(tmp) > 0:
                        raw_row = tmp.iloc[0]

                if raw_row is not None:
                    rows = []
                    for col in projects_raw.columns:
                        val = raw_row.get(col)
                        if pd.isna(val):
                            continue
                        if isinstance(val, str) and val.strip() == "":
                            continue
                        rows.append({"항목": col, "값": val})

                    if rows:
                        detail_df = pd.DataFrame(rows)
                        st.data_editor(
                            detail_df.set_index("항목"),
                            use_container_width=True,
                            disabled=True,
                        )
                    else:
                        st.caption("표시할 상세 정보가 없습니다.")
                else:
                    st.info("선택한 공고의 원본 상세정보를 찾을 수 없습니다.")
            else:
                st.caption("추천 리스트에서 공고를 클릭하면 아래에 상세 정보와 다음 액션 버튼이 표시됩니다.")


# --------------------------
# [Tab 4] 제안서 초안 생성
# --------------------------
with tab4:
    st.markdown("### 🧠 제안서 초안 자동 생성")
    st.caption("선택한 부서·사업 기준으로, 제출용 제안서 초안을 AI가 자동으로 작성합니다.")

    if llm is None:
        st.error("LLM이 초기화되지 않았습니다. .env에 OPENAI_API_KEY, LLM_BASE_URL을 설정해 주세요.")
    elif not dept_names:
        st.warning("부서 프로필 데이터가 없습니다.")
    else:
        # ==============================
        # 부서 선택
        # ==============================
        st.markdown("#### 1단계. 제안서를 준비할 부서 선택")

        dept_options = ["(부서를 선택하세요)"] + dept_names

        # 전역 부서 선택값 → 이 탭의 selectbox 상태에 동기화
        global_dept = st.session_state.get("selected_dept_name")
        if global_dept and global_dept in dept_names:
            st.session_state["proposal_dept_select"] = global_dept

        dept_label = st.selectbox(
            "부서 선택",
            dept_options,
            key="proposal_dept_select",
        )

        if dept_label != "(부서를 선택하세요)":
            selected_dept_name = dept_label
            st.session_state["selected_dept_name"] = selected_dept_name
        else:
            selected_dept_name = None

        # ==============================
        # 사업 선택 (검색 + 선택)
        # ==============================
        st.markdown("#### 2단계. 제안서를 작성할 사업 선택")

        project_row = None
        selected_project_id = st.session_state.get("selected_project_id")

        if not selected_dept_name:
            st.info("먼저 제안서를 작성할 부서를 선택해 주세요.")
        else:
            # 선택된 부서 기준으로 매칭 사업 리스트 계산
            dept_row = dept_df[dept_df["부서명"] == selected_dept_name].iloc[0]
            dept_keywords = dept_row.get("검색키워드", [])

            scored_for_proposal = score_projects_for_department(dept_keywords, projects_df)

            # 검색 입력: 공고번호 / 사업명
            search_query = st.text_input(
                "사업 검색 (공고ID 또는 사업명)",
                value="",
                placeholder="예: 2024-000123, 스마트시티, 공동주택 등",
                key="proposal_project_search",
            )

            base_df = scored_for_proposal.copy()

            if search_query:
                q = str(search_query).strip()
                mask = (
                    base_df["공고ID"].astype(str).str.contains(q, case=False, na=False)
                    | base_df["사업명"].astype(str).str.contains(q, case=False, na=False)
                )
                filtered = base_df[mask]
            else:
                # 검색어가 없으면 상위 N개만 보여주기 (예: 50개)
                filtered = base_df.head(50)

            if filtered.empty:
                st.warning("검색 조건에 맞는 사업이 없습니다. 검색어를 변경해 보세요.")
            else:
                placeholder = "(사업을 선택하세요)"
                project_options_real = [
                    f"{str(row['공고ID'])} | {row['사업명']} (매칭점수: {row['매칭점수']})"
                    for _, row in filtered.iterrows()
                ]
                project_options = [placeholder] + project_options_real

                default_idx = 0
                if selected_project_id is not None:
                    selected_project_id = str(selected_project_id)
                    if selected_project_id in filtered["공고ID"].astype(str).values:
                        default_proj_label = next(
                            (
                                opt
                                for opt in project_options_real
                                if opt.startswith(f"{selected_project_id} |")
                            ),
                            None,
                        )
                        if default_proj_label:
                            default_idx = project_options.index(default_proj_label)

                project_label = st.selectbox(
                    "제안서를 작성할 사업 선택",
                    project_options,
                    index=default_idx,
                    key="proposal_project_select",
                )

                if project_label != placeholder:
                    selected_project_id = project_label.split(" | ")[0]  # 공고ID 문자열
                    st.session_state["selected_project_id"] = selected_project_id

                    project_row = filtered[
                        filtered["공고ID"].astype(str) == selected_project_id
                    ].iloc[0]
                else:
                    selected_project_id = None
                    project_row = None

        # ==============================
        # 제안서 생성 영역
        # ==============================
        if not selected_dept_name or project_row is None:
            st.markdown("---")
            st.info("부서와 사업을 모두 선택하면 제안서 초안 생성 옵션이 활성화됩니다.")
        else:
            st.markdown("---")

            st.markdown(
                f"""
                <div style="
                    padding: 14px 18px;
                    border-radius: 10px;
                    background-color: #fffbeb;
                    border: 1px solid #fed7aa;
                    margin-bottom: 12px;
                    line-height: 1.5;
                    font-size:0.95rem;
                ">
                    <b>{selected_dept_name}</b> 부서 기준으로<br/>
                    <b>[{project_row.get('공고ID', '')}] {project_row.get('사업명', '')}</b><br/>
                    사업에 대한 제안서 초안을 생성합니다.<br/><br/>
                    아래에서 작성 스타일을 선택한 후 <b>제안서 초안 생성</b> 버튼을 눌러 주세요.
                </div>
                """,
                unsafe_allow_html=True,
            )

            style = st.radio(
                "작성 스타일",
                ["기본(격식 있는 보고서)", "조금 더 간결하게", "요약본(핵심만)"],
                index=0,
                horizontal=True,
            )

            generate = st.button("🚀 제안서 초안 생성", use_container_width=True, type="primary")

            if generate:
                with st.spinner("제안서 초안을 생성 중입니다..."):
                    base_prompt = build_proposal_prompt(dept_row, project_row)

                    if style == "조금 더 간결하게":
                        base_prompt += "\n추가 지시사항: 전체 분량을 줄이고, 문장을 간결하게 작성하세요.\n"
                    elif style == "요약본(핵심만)":
                        base_prompt += "\n추가 지시사항: A4 1/2 페이지 이내 분량으로 핵심 내용만 요약해 작성하세요.\n"

                    response = llm.invoke(base_prompt)
                    proposal_text = (
                        response.content if hasattr(response, "content") else str(response)
                    )

                st.markdown("#### ✅ 생성된 제안서 초안")
                st.markdown(proposal_text)

                with st.expander("📋 복사용 원문 보기"):
                    st.text_area(
                        "아래 내용을 복사해 내부 양식/한글 문서 등에 붙여넣어 활용하세요.",
                        proposal_text,
                        height=400,
                    )


# ========================================
# [Tab 5] RAG 기반 문서 조회 (LLM 스마트 질문 생성)
# ========================================
with tab5:
    st.markdown("### 📚 공고 문서 RAG 기반 분석")
    st.caption("선택한 사업공고의 첨부 문서를 벡터화하여, 자연어로 질의응답할 수 있습니다.")

    api_key = os.getenv("OPENAI_API_KEY")
    embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
    llm_base_url = os.getenv("LLM_BASE_URL")

    # RAG 시스템 초기화 (3개 인자 전달!)
    rag_system = None
    if api_key and llm_base_url:
        try:
            rag_system = get_rag_system(api_key, llm_base_url, embedding_base_url)
        except Exception as e:
            st.error(f"RAG 시스템 초기화 실패: {str(e)}")

    if not api_key or not embedding_base_url:
        st.error("🔑 RAG 기능을 사용하려면 .env에 OPENAI_API_KEY, EMBEDDING_BASE_URL, LLM_BASE_URL을 설정해주세요.")
    elif rag_system is None:
        st.error("RAG 시스템 초기화에 실패했습니다.")
    else:
        # ==============================
        # 부서 선택
        # ==============================
        st.markdown("#### 1단계. 기준 부서 선택")

        dept_options = ["(부서를 선택하세요)"] + dept_names

        # 전역 부서 선택값 → 이 탭의 selectbox 상태에 동기화
        global_dept = st.session_state.get("selected_dept_name")
        if global_dept and global_dept in dept_names:
            st.session_state["rag_dept_select"] = global_dept

        dept_label = st.selectbox(
            "부서를 선택하세요",
            dept_options,
            key="rag_dept_select",
        )

        if dept_label != "(부서를 선택하세요)":
            selected_dept_name = dept_label
            st.session_state["selected_dept_name"] = selected_dept_name
        else:
            selected_dept_name = None

        # ==============================
        # 사업 선택 (제안서 탭과 같은 UX로)
        # ==============================
        st.markdown("#### 2단계. 문서를 분석할 사업 선택")

        project_row = None
        selected_project_id = st.session_state.get("selected_project_id")

        if not selected_dept_name:
            st.info("먼저 기준이 될 부서를 선택해 주세요.")
        else:
            dept_row = dept_df[dept_df["부서명"] == selected_dept_name].iloc[0]
            dept_keywords = dept_row.get("검색키워드", [])

            scored_for_proposal = score_projects_for_department(dept_keywords, projects_df)

            search_query = st.text_input(
                "사업 검색 (공고ID 또는 사업명)",
                value="",
                placeholder="예: 2024-000123, 스마트시티, 공동주택 등",
                key="rag_project_search",
            )

            base_df = scored_for_proposal.copy()

            if search_query:
                q = str(search_query).strip()
                mask = (
                    base_df["공고ID"].astype(str).str.contains(q, case=False, na=False)
                    | base_df["사업명"].astype(str).str.contains(q, case=False, na=False)
                )
                filtered = base_df[mask]
            else:
                filtered = base_df.head(50)

            if filtered.empty:
                st.warning("검색 조건에 맞는 사업이 없습니다. 검색어를 변경해 보세요.")
            else:
                placeholder = "(사업을 선택하세요)"
                project_options_real = [
                    f"{str(row['공고ID'])} | {row['사업명']} (매칭점수: {row['매칭점수']})"
                    for _, row in filtered.iterrows()
                ]
                project_options = [placeholder] + project_options_real

                default_idx = 0
                if selected_project_id is not None:
                    selected_project_id = str(selected_project_id)
                    if selected_project_id in filtered["공고ID"].astype(str).values:
                        default_proj_label = next(
                            (
                                opt
                                for opt in project_options_real
                                if opt.startswith(f"{selected_project_id} |")
                            ),
                            None,
                        )
                        if default_proj_label:
                            default_idx = project_options.index(default_proj_label)

                project_label = st.selectbox(
                    "문서를 조회할 사업 선택",
                    project_options,
                    index=default_idx,
                    key="rag_project_select",
                )

                if project_label != placeholder:
                    selected_project_id = project_label.split(" | ")[0]
                    st.session_state["selected_project_id"] = selected_project_id

                    project_row = filtered[
                        filtered["공고ID"].astype(str) == selected_project_id
                    ].iloc[0]
                else:
                    selected_project_id = None
                    project_row = None

        # ==============================
        # RAG 본 기능 (문서 로드 + 질의응답)
        # ==============================
        if project_row is None or not selected_dept_name:
            st.markdown("---")
            st.info("부서와 사업을 모두 선택하면 문서 조회 및 질의응답 기능이 활성화됩니다.")
        else:
            st.markdown("#### 3단계. 선택된 사업 정보")
            st.markdown(f"- **사업명**: {project_row.get('사업명', '')}")
            st.markdown(f"- **공고ID**: {project_row.get('공고ID', '')}")
            st.markdown(f"- **주관부처**: {project_row.get('주관부처', '')}")
            
            st.markdown("---")
            
            # 파일 URL 추출
            file_urls = extract_file_urls(project_row)
            
            if not file_urls:
                st.warning("⚠️ 이 사업에는 첨부된 문서가 없습니다.")
            else:
                st.markdown("#### 4단계. 첨부 문서 확인 및 벡터 DB 구축")
                st.info("💡 PDF, HWP, DOCX, TXT 등 주요 문서 포맷을 지원합니다.")

                file_rows = []
                for idx, url in enumerate(file_urls, 1):
                    file_ext = Path(url).suffix.upper() or "알 수 없음"
                    file_rows.append(
                        {
                            "No.": idx,
                            "파일유형": file_ext,
                            "URL": url,
                        }
                    )
                file_df = pd.DataFrame(file_rows)
                st.data_editor(file_df, use_container_width=True, disabled=True, hide_index=True)
                
                with st.expander("벡터 DB 구축 옵션", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        chunk_size = st.slider("청크 크기 (characters)", 500, 2000, 800, 100)
                    with col2:
                        chunk_overlap = st.slider("청크 오버랩 (characters)", 0, 300, 150, 50)
                    
                    if st.button("📥 문서 로드 및 벡터 DB 구축", type="primary"):
                        with st.spinner("첨부 문서를 다운로드하고 벡터 DB를 구축 중입니다... (HWP 파일은 다소 시간이 걸릴 수 있습니다)"):
                            documents = rag_system.load_documents_from_urls(file_urls)
                            
                            if documents:
                                rag_system.build_vectorstore(documents, chunk_size, chunk_overlap)
                                st.session_state['vectorstore_ready'] = True
                                st.session_state['current_project_id'] = str(project_row.get('공고ID', ''))
                                st.session_state['chat_history'] = []
                                st.session_state['smart_questions_generated'] = False
                            else:
                                st.error("문서 로드에 실패했습니다.")
                                st.session_state['vectorstore_ready'] = False
            
            # 문서 검색/질의응답 섹션
            if st.session_state.get('vectorstore_ready', False) and \
               st.session_state.get('current_project_id') == str(project_row.get('공고ID', '')):
                
                st.markdown("---")
                st.markdown("#### 5단계. 문서 기반 질의응답 / 검색")
                
                search_tab1, search_tab2 = st.tabs(["💬 AI 챗봇 질의응답", "🔍 키워드 유사도 검색"])
                
                # ===========================
                # [검색 탭 1] AI 챗봇
                # ===========================
                with search_tab1:
                    st.markdown("##### AI에게 문서 내용 질문하기")
                    
                    if 'chat_history' not in st.session_state:
                        st.session_state['chat_history'] = []
                    
                    # 스마트 질문 생성 (최초 1회)
                    if not st.session_state.get('smart_questions_generated', False):
                        with st.spinner("문서 내용을 분석하여 추천 질문을 생성 중입니다..."):
                            smart_questions = rag_system.generate_smart_questions(num_questions=4)
                            st.session_state['smart_questions'] = smart_questions
                            st.session_state['smart_questions_generated'] = True
                    
                    # 추천 질문
                    if 'smart_questions' in st.session_state and st.session_state['smart_questions']:
                        st.markdown("**💡 이 문서에 대해 이렇게 물어볼 수 있어요**")
                        example_questions = st.session_state['smart_questions']
                        
                        cols = st.columns(2)
                        for idx, example_q in enumerate(example_questions):
                            with cols[idx % 2]:
                                if st.button(example_q, key=f"smart_q_{idx}"):
                                    st.session_state['chat_history'].append({
                                        'role': 'user',
                                        'content': example_q
                                    })
                                    
                                    with st.spinner("문서를 기반으로 답변을 생성 중입니다..."):
                                        answer, source_docs = rag_system.query(example_q)
                                        
                                        st.session_state['chat_history'].append({
                                            'role': 'assistant',
                                            'content': answer,
                                            'sources': source_docs
                                        })
                                    
                                    st.rerun()
                    else:
                        st.markdown("**예시 질문**")
                        example_questions = [
                            "이 사업의 주요 내용을 요약해 주세요.",
                            "지원 대상과 지원 규모는 어떻게 되나요?",
                        ]
                        
                        for idx, example_q in enumerate(example_questions):
                            if st.button(example_q, key=f"fallback_q_{idx}"):
                                st.session_state['chat_history'].append({
                                    'role': 'user',
                                    'content': example_q
                                })
                                
                                with st.spinner("문서를 기반으로 답변을 생성 중입니다..."):
                                    answer, source_docs = rag_system.query(example_q)
                                    
                                    st.session_state['chat_history'].append({
                                        'role': 'assistant',
                                        'content': answer,
                                        'sources': source_docs
                                    })
                                
                                st.rerun()
                    
                    st.markdown("---")
                    
                    # 대화 히스토리 표시
                    if st.session_state.get('chat_history'):
                        st.markdown("##### 대화 내역")
                        
                        for message in st.session_state['chat_history']:
                            if message['role'] == 'user':
                                with st.chat_message("user"):
                                    st.markdown(message['content'])
                            else:
                                with st.chat_message("assistant"):
                                    st.markdown(message['content'])
                                    
                                    if 'sources' in message and message['sources']:
                                        with st.expander(f"📚 참조 문서/페이지 ({len(message['sources'])}개)"):
                                            for doc_idx, doc in enumerate(message['sources'], 1):
                                                st.markdown(f"**[{doc_idx}] 출처:** {doc.metadata.get('source_url', 'N/A')[:250]}...")
                                                st.markdown(f"**페이지:** {doc.metadata.get('page', 'N/A')}")
                                                content_preview = doc.page_content[:250] + "..." if len(doc.page_content) > 250 else doc.page_content
                                                st.text(content_preview)
                                                st.markdown("---")
                        
                        col1, col2 = st.columns([3, 1])
                        with col2:
                            if st.button("대화 내용 초기화", key="clear_chat"):
                                st.session_state['chat_history'] = []
                                st.rerun()
                    
                    st.markdown("---")
                    
                    user_question = st.chat_input("문서 내용에 대해 자유롭게 질문해 보세요.")
                    
                    if user_question:
                        st.session_state['chat_history'].append({
                            'role': 'user',
                            'content': user_question
                        })
                        
                        with st.spinner("문서를 분석하여 답변을 생성 중입니다..."):
                            answer, source_docs = rag_system.query(user_question)
                            
                            st.session_state['chat_history'].append({
                                'role': 'assistant',
                                'content': answer,
                                'sources': source_docs
                            })
                        
                        st.rerun()

                # ===========================
                # [검색 탭 2] 유사도 검색
                # ===========================
                with search_tab2:
                    st.markdown("##### 키워드 중심 유사도 검색")
                    search_query = st.text_input(
                        "검색 키워드를 입력하세요",
                        placeholder="예: 지원 대상, 평가 기준, 신청 방법",
                        key="search_query"
                    )
                    search_k = st.slider("표시할 문서 조각 개수", 1, 10, 5, key="search_k")
                    
                    if st.button("🔍 문서 내 유사 내용 검색", key="search_btn"):
                        if search_query.strip():
                            with st.spinner("키워드와 유사한 문서 내용을 검색 중입니다..."):
                                results = rag_system.similarity_search(search_query, k=search_k)
                                
                                if results:
                                    st.success(f"✅ {len(results)}개의 관련 문서 조각을 찾았습니다.")
                                    
                                    for idx, (doc, score) in enumerate(results, 1):
                                        with st.expander(f"📄 결과 {idx} (유사도: {score:.4f})"):
                                            st.markdown(f"**출처:** {doc.metadata.get('source_url', 'N/A')}")
                                            st.markdown(f"**페이지:** {doc.metadata.get('page', 'N/A')}")
                                            st.markdown("**내용:**")
                                            st.text(doc.page_content)
                                else:
                                    st.warning("검색 결과가 없습니다.")
                        else:
                            st.warning("검색 키워드를 입력해 주세요.")
            else:
                st.info("먼저 상단의 '문서 로드 및 벡터 DB 구축'을 완료한 뒤, 하단 검색 기능을 사용할 수 있습니다.")


# -----------------------------------------
# [Tab 6] 부서/소스 데이터 상세 보기
# -----------------------------------------
with tab_data:
    st.markdown("### 🗂 부서/소스 데이터 상세 보기")
    st.caption("추천·제안서·RAG 기능의 기반이 되는 부서 프로필과 사업공고 소스 메타데이터를 확인합니다.")

    tab_dept, tab_api = st.tabs(["부서 프로필 데이터", "사업공고 소스 메타데이터"])

    # ===== 부서 프로필 데이터 탭 =====
    with tab_dept:
        st.markdown("#### 부서별 역량 및 검색 키워드")
        st.caption("부서별 역량/관심분야 데이터와, 추천 로직에 사용하는 검색 키워드를 확인할 수 있습니다.")

        # ✅ CSV 특성에 맞춰 표시용 DF 정리 (NaN 컬럼 제거)
        dept_display_df = dept_df.copy()
        dept_display_df = dept_display_df.loc[:, ~dept_display_df.columns.isna()]
        dept_display_df = dept_display_df.dropna(axis=1, how="all")

        # 1) 전체 부서 테이블 (행 선택)
        st.markdown("**전체 부서 목록**")
        st.caption("아래 표에서 부서 행을 클릭하면, 아래에 해당 부서의 상세 정보가 표시됩니다.")

        dept_event = st.dataframe(
            dept_display_df,
            use_container_width=True,
            height=500,
            on_select="rerun",
            selection_mode="single-row",
            key="dept_table",
        )

        selected_rows = dept_event.selection.rows

        # 2) 선택된 부서 상세 정보 (표 아래에 표시)
        st.markdown("---")
        st.markdown("**선택된 부서 상세 정보**")

        if selected_rows:
            row_idx = selected_rows[0]
            dept_row = dept_display_df.iloc[row_idx]

            st.markdown(f"- **부서명:** {dept_row.get('부서명', '')}")
            st.markdown(f"- **소속부문:** {dept_row.get('소속부문', '')}")
            st.markdown(f"- **핵심역량:** {dept_row.get('핵심역량', '')}")
            st.markdown(f"- **관심지원분야:** {dept_row.get('관심지원분야', '')}")
            st.markdown(f"- **참여 가능 역할:** {dept_row.get('참여 가능 역할', '')}")

            st.markdown("")
            st.markdown("**검색 키워드 (IDF 기준 중요도 순)**")
            st.write(dept_row.get("검색키워드", []))

            with st.expander("JSON 형태로 전체 프로필 확인"):
                st.json(
                    {
                        "부서명": dept_row.get("부서명", ""),
                        "소속부문": dept_row.get("소속부문", ""),
                        "핵심역량": dept_row.get("핵심역량", ""),
                        "관심지원분야": dept_row.get("관심지원분야", ""),
                        "주요키워드": dept_row.get("주요키워드", ""),
                        "지원사업 형태 선호": dept_row.get("지원사업 형태 선호", ""),
                        "참여 가능 역할": dept_row.get("참여 가능 역할", ""),
                        "관심지역": dept_row.get("관심지역", ""),
                        "예산 선호규모": dept_row.get("예산 선호규모", ""),
                        "최근수행사업 예시": dept_row.get("최근수행사업 예시", ""),
                        "제외항목": dept_row.get("제외항목", ""),
                        "검색키워드": dept_row.get("검색키워드", []),
                        "키워드_IDF점수": dept_row.get("키워드_IDF점수", []),
                    }
                )
        else:
            st.info("위 표에서 부서를 선택하면 이 영역에 상세 정보가 표시됩니다.")

    # ===== 사업공고 소스 메타데이터 탭 =====
    with tab_api:
        st.markdown("#### 사업공고 API / 크롤링 소스 메타데이터")
        st.caption("각 사업공고가 어떤 API/크롤링 소스에서 수집되었는지에 대한 데이터입니다.")
        st.data_editor(
            api_df,
            use_container_width=True,
            height=600,
            disabled=True,
        )
