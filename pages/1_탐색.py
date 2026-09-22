import streamlit as st
import pandas as pd
import plotly.express as px


DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


st.set_page_config(
    page_title="탐색 | 뇌졸중 예측 실습실",
    page_icon="🔎",
    layout="wide",
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

    .page-header {
        background: linear-gradient(135deg, #172554 0%, #2563eb 100%);
        padding: 30px 36px;
        border-radius: 22px;
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 12px 30px rgba(37, 99, 235, 0.18);
    }

    .page-icon {
        font-size: 42px;
    }

    .page-title {
        font-size: 34px;
        font-weight: 800;
        margin-top: 4px;
    }

    .page-description {
        opacity: 0.88;
        margin-top: 7px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 750;
        color: #111827;
        margin: 30px 0 5px 0;
    }

    .section-description {
        color: #64748b;
        margin-bottom: 16px;
    }

    .insight-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 20px 22px;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
        margin-top: 14px;
    }

    .insight-title {
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 5px;
    }

    .insight-text {
        color: #64748b;
        font-size: 14px;
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


with st.sidebar:
    st.markdown("## 🧠 뇌졸중 예측 실습실")
    st.caption("Stroke Data Explorer")

    st.divider()

    st.markdown("### 📌 데이터")
    st.write(f"**{len(df):,}명**")
    st.write(f"**{len(df.columns)}개 변수**")

    st.divider()

    st.markdown("### 🗂️ 현재 페이지")
    st.markdown("🔎 **탐색**")

    st.divider()

    st.caption("그래프를 통해 데이터의 특징을 찾아보세요.")


st.markdown(
    """
    <div class="page-header">
        <div class="page-icon">🔎</div>
        <div class="page-title">데이터 탐색</div>
        <div class="page-description">
            그래프와 표를 이용해 뇌졸중 데이터의 특징을 직접 발견해 봅니다.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="section-title">📊 숫자 데이터의 분포</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">나이와 평균 혈당이 어떻게 분포되어 있는지 확인합니다.</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이 분포",
        color_discrete_sequence=["#2563eb"],
    )

    fig_age.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수",
        bargap=0.06,
        template="plotly_white",
        height=400,
        title_font_size=18,
    )

    st.plotly_chart(
        fig_age,
        use_container_width=True,
        config={"displayModeBar": False},
    )

with col2:
    fig_glucose = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        color_discrete_sequence=["#0f766e"],
    )

    fig_glucose.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수",
        bargap=0.06,
        template="plotly_white",
        height=400,
        title_font_size=18,
    )

    st.plotly_chart(
        fig_glucose,
        use_container_width=True,
        config={"displayModeBar": False},
    )


st.markdown(
    '<div class="section-title">🩺 뇌졸중 여부에 따른 비교</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">뇌졸중이 있는 그룹과 없는 그룹의 나이와 평균 혈당을 비교합니다.</div>',
    unsafe_allow_html=True,
)

plot_df = df.copy()

plot_df["stroke_group"] = plot_df["stroke"].map(
    {
        0: "뇌졸중 없음",
        1: "뇌졸중 있음",
    }
)

col1, col2 = st.columns(2)

with col1:
    fig_age_box = px.box(
        plot_df,
        x="stroke_group",
        y="age",
        color="stroke_group",
        title="나이 비교",
        color_discrete_map={
            "뇌졸중 없음": "#93c5fd",
            "뇌졸중 있음": "#ef4444",
        },
    )

    fig_age_box.update_layout(
        xaxis_title="",
        yaxis_title="나이",
        template="plotly_white",
        height=430,
        showlegend=False,
        title_font_size=18,
    )

    st.plotly_chart(
        fig_age_box,
        use_container_width=True,
        config={"displayModeBar": False},
    )

with col2:
    fig_glucose_box = px.box(
        plot_df,
        x="stroke_group",
        y="avg_glucose_level",
        color="stroke_group",
        title="평균 혈당 비교",
        color_discrete_map={
            "뇌졸중 없음": "#93c5fd",
            "뇌졸중 있음": "#ef4444",
        },
    )

    fig_glucose_box.update_layout(
        xaxis_title="",
        yaxis_title="평균 혈당",
        template="plotly_white",
        height=430,
        showlegend=False,
        title_font_size=18,
    )

    st.plotly_chart(
        fig_glucose_box,
        use_container_width=True,
        config={"displayModeBar": False},
    )


age_means = (
    plot_df.groupby("stroke_group")["age"]
    .mean()
    .rename("나이 평균")
)

glucose_means = (
    plot_df.groupby("stroke_group")["avg_glucose_level"]
    .mean()
    .rename("평균 혈당 평균")
)

mean_table = pd.concat(
    [age_means, glucose_means],
    axis=1,
).reset_index()

mean_table = mean_table.rename(
    columns={"stroke_group": "뇌졸중 여부"}
)

mean_table["나이 평균"] = mean_table["나이 평균"].round(2)
mean_table["평균 혈당 평균"] = mean_table["평균 혈당 평균"].round(2)

st.markdown(
    '<div class="insight-card">'
    '<div class="insight-title">📋 그룹별 평균값</div>'
    '<div class="insight-text">두 그룹의 평균 나이와 평균 혈당입니다.</div>'
    '</div>',
    unsafe_allow_html=True,
)

st.dataframe(
    mean_table,
    use_container_width=True,
    hide_index=True,
)


st.markdown(
    '<div class="section-title">❤️ 건강 상태와 뇌졸중</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">고혈압과 심장병 여부에 따라 뇌졸중 비율을 비교합니다.</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    hypertension_rate = (
        df.groupby("hypertension")["stroke"]
        .mean()
        .mul(100)
        .reset_index()
    )

    hypertension_rate["hypertension"] = hypertension_rate["hypertension"].map(
        {
            0: "고혈압 없음",
            1: "고혈압 있음",
        }
    )

    fig_hypertension = px.bar(
        hypertension_rate,
        x="hypertension",
        y="stroke",
        text="stroke",
        title="고혈압 여부에 따른 뇌졸중 비율",
        color="hypertension",
        color_discrete_sequence=["#93c5fd", "#2563eb"],
    )

    fig_hypertension.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
    )

    fig_hypertension.update_layout(
        xaxis_title="",
        yaxis_title="뇌졸중 비율 (%)",
        yaxis_range=[
            0,
            max(hypertension_rate["stroke"]) * 1.25,
        ],
        template="plotly_white",
        height=430,
        showlegend=False,
        title_font_size=17,
    )

    st.plotly_chart(
        fig_hypertension,
        use_container_width=True,
        config={"displayModeBar": False},
    )

with col2:
    heart_rate = (
        df.groupby("heart_disease")["stroke"]
        .mean()
        .mul(100)
        .reset_index()
    )

    heart_rate["heart_disease"] = heart_rate["heart_disease"].map(
        {
            0: "심장병 없음",
            1: "심장병 있음",
        }
    )

    fig_heart = px.bar(
        heart_rate,
        x="heart_disease",
        y="stroke",
        text="stroke",
        title="심장병 여부에 따른 뇌졸중 비율",
        color="heart_disease",
        color_discrete_sequence=["#a7f3d0", "#0f766e"],
    )

    fig_heart.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
    )

    fig_heart.update_layout(
        xaxis_title="",
        yaxis_title="뇌졸중 비율 (%)",
        yaxis_range=[
            0,
            max(heart_rate["stroke"]) * 1.25,
        ],
        template="plotly_white",
        height=430,
        showlegend=False,
        title_font_size=17,
    )

    st.plotly_chart(
        fig_heart,
        use_container_width=True,
        config={"displayModeBar": False},
    )


st.markdown(
    '<div class="section-title">🧬 BMI 빈 값 살펴보기</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">BMI가 기록되지 않은 사람들과 전체 데이터의 뇌졸중 비율을 비교합니다.</div>',
    unsafe_allow_html=True,
)

missing_bmi_df = df[df["bmi"].isna()]

missing_bmi_rate = missing_bmi_df["stroke"].mean() * 100
overall_stroke_rate = df["stroke"].mean() * 100

bmi_rate_table = pd.DataFrame(
    {
        "구분": [
            "BMI가 비어 있는 사람",
            "전체 사람",
        ],
        "사람 수": [
            len(missing_bmi_df),
            len(df),
        ],
        "뇌졸중 비율 (%)": [
            missing_bmi_rate,
            overall_stroke_rate,
        ],
    }
)

bmi_rate_table["뇌졸중 비율 (%)"] = (
    bmi_rate_table["뇌졸중 비율 (%)"].round(2)
)

st.dataframe(
    bmi_rate_table,
    use_container_width=True,
    hide_index=True,
)


st.markdown(
    '<div class="section-title">🚬 흡연 상태</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">흡연 상태별로 몇 명의 사람이 데이터에 포함되어 있는지 확인합니다.</div>',
    unsafe_allow_html=True,
)

smoking_count = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .reset_index()
)

smoking_count.columns = [
    "흡연 상태",
    "사람 수",
]

smoking_count["흡연 상태"] = smoking_count["흡연 상태"].fillna(
    "빈 값"
)

st.dataframe(
    smoking_count,
    use_container_width=True,
    hide_index=True,
)


st.markdown(
    '<div class="footer">🧠 뇌졸중 예측 실습실 · 데이터 탐색 페이지</div>',
    unsafe_allow_html=True,
)
