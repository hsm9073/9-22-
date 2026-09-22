import streamlit as st
import pandas as pd


DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .stApp {
        background: #f7f9fc;
    }

    [data-testid="stSidebar"] {
        background: #111827;
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    .hero {
        background: linear-gradient(135deg, #0f766e 0%, #2563eb 100%);
        padding: 34px 38px;
        border-radius: 22px;
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 12px 30px rgba(37, 99, 235, 0.18);
    }

    .hero-icon {
        font-size: 48px;
        margin-bottom: 8px;
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1.5px;
    }

    .hero-subtitle {
        font-size: 16px;
        opacity: 0.9;
        margin-top: 10px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 750;
        color: #111827;
        margin: 30px 0 12px 0;
    }

    .section-desc {
        color: #6b7280;
        margin-bottom: 18px;
    }

    .source-box {
        background: #eef6ff;
        border-left: 5px solid #2563eb;
        padding: 18px 20px;
        border-radius: 12px;
        color: #334155;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b;
    }

    div[data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 800;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 13px;
        padding: 35px 0 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


df = load_data()


korean_meanings = {
    "id": "",
    "gender": "",
    "age": "",
    "hypertension": "",
    "heart_disease": "",
    "ever_married": "",
    "work_type": "",
    "Residence_type": "",
    "avg_glucose_level": "",
    "bmi": "",
    "smoking_status": "",
    "stroke": "",
}

DATA_SOURCE_TEXT = ""


with st.sidebar:
    st.markdown("## 🧠 뇌졸중 예측 실습실")
    st.caption("Stroke Data Explorer")

    st.divider()

    st.markdown("### 📌 현재 데이터")
    st.write(f"**{len(df):,}명**의 데이터를 분석합니다.")
    st.write(f"**{len(df.columns)}개**의 변수가 있습니다.")

    st.divider()

    st.markdown("### 🗂️ 페이지")
    st.markdown("🏠 **데이터 소개**")
    st.markdown("🔎 **탐색**")
    st.markdown("🤖 **분류 모델**")

    st.divider()

    st.caption("교재와 함께 데이터를 직접 탐색해 보세요.")


st.markdown(
    """
    <div class="hero">
        <div class="hero-icon">🧠</div>
        <div class="hero-title">뇌졸중 예측 실습실</div>
        <div class="hero-subtitle">
            데이터를 살펴보고 뇌졸중과 관련된 특징을 탐색하는 데이터 분석 실습 공간
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="section-title">📊 데이터 한눈에 보기</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-desc">
        사람들의 인구통계학적 정보와 건강 관련 정보를 바탕으로
        뇌졸중 발생 여부를 살펴볼 수 있는 데이터입니다.
    </div>
    """,
    unsafe_allow_html=True,
)


total_people = len(df)
column_count = len(df.columns)
stroke_count = int(df["stroke"].sum())
stroke_ratio = df["stroke"].mean() * 100

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("👥 전체 사람 수", f"{total_people:,}명")

with c2:
    st.metric("🧩 데이터 열", f"{column_count}개")

with c3:
    st.metric("🩺 뇌졸중 발생", f"{stroke_count:,}명")

with c4:
    st.metric("📈 뇌졸중 비율", f"{stroke_ratio:.2f}%")


st.markdown(
    '<div class="section-title">📚 데이터 열 정보</div>',
    unsafe_allow_html=True,
)

column_info = []

for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        value_type = "숫자형"
    else:
        value_type = "범주형"

    column_info.append(
        {
            "열 이름": column,
            "우리말 뜻": korean_meanings[column],
            "값의 종류": value_type,
            "빈 값 개수": int(df[column].isna().sum()),
        }
    )

column_info_df = pd.DataFrame(column_info)

st.dataframe(
    column_info_df,
    use_container_width=True,
    hide_index=True,
    height=480,
)


st.markdown(
    '<div class="section-title">🔍 데이터 미리보기</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-desc">원본 데이터의 처음 다섯 행입니다.</div>',
    unsafe_allow_html=True,
)

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True,
)


st.markdown(
    '<div class="section-title">🔗 데이터 출처</div>',
    unsafe_allow_html=True,
)

if DATA_SOURCE_TEXT:
    st.markdown(
        f'<div class="source-box">{DATA_SOURCE_TEXT}</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="source-box">
            📖 교재에 있는 데이터 출처 내용을 여기에 입력하세요.
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    '<div class="footer">🧠 Stroke Data Explorer · 데이터 분석 실습</div>',
    unsafe_allow_html=True,
)
