import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split


DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


# 페이지 설정
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)


# 데이터 불러오기
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


df = load_data()


# ---------------------------------------------------------
# 교재를 보고 직접 채워 넣는 부분
# ---------------------------------------------------------

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

# 데이터 출처 설명을 교재를 보고 입력하세요.
DATA_SOURCE_TEXT = ""


# ---------------------------------------------------------
# 화면
# ---------------------------------------------------------

st.title("🧠 뇌졸중 예측 실습실")

st.write(
    """
    이 데이터는 사람들의 인구통계학적 정보와 건강 관련 정보를 이용하여
    뇌졸중(stroke) 발생 여부를 살펴볼 수 있는 데이터입니다.

    각 사람에 대한 나이, 고혈압 여부, 심장질환 여부, 평균 혈당 수치,
    BMI, 흡연 상태 등의 정보와 뇌졸중 발생 여부가 포함되어 있습니다.
    """
)


# 핵심 통계
total_people = len(df)
column_count = len(df.columns)
stroke_count = int(df["stroke"].sum())
stroke_ratio = stroke_count / total_people * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("전체 사람 수", f"{total_people:,}명")

with col2:
    st.metric("열 개수", f"{column_count}개")

with col3:
    st.metric("stroke가 1인 사람 수", f"{stroke_count:,}명")

with col4:
    st.metric("stroke가 1인 비율", f"{stroke_ratio:.2f}%")


st.divider()


# ---------------------------------------------------------
# 열 정보 표
# ---------------------------------------------------------

st.subheader("데이터 열 정보")

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
    hide_index=True
)


# ---------------------------------------------------------
# 처음 다섯 줄
# ---------------------------------------------------------

st.subheader("데이터 처음 5줄")

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# 데이터 출처
# ---------------------------------------------------------

st.divider()

st.subheader("데이터 출처")

if DATA_SOURCE_TEXT:
    st.write(DATA_SOURCE_TEXT)
else:
    st.info("교재의 데이터 출처 내용을 여기에 입력하세요.")


# ---------------------------------------------------------
# 참고: 실습에서 사용할 수 있도록 데이터 확인
# ---------------------------------------------------------

with st.expander("실습용 데이터 기본 정보"):
    st.write(f"- 데이터 크기: {df.shape[0]:,}행 × {df.shape[1]}열")
    st.write(f"- BMI 빈 값: {df['bmi'].isna().sum():,}개")
