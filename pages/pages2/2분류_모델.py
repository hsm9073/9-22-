import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

RANDOM_STATE = 42

FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}


st.set_page_config(
    page_title="분류 모델 | 뇌졸중 예측 실습실",
    page_icon="🤖",
    layout="wide",
)


# ---------------------------------------------------------
# 스타일
# ---------------------------------------------------------

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
        background: linear-gradient(135deg, #312e81 0%, #2563eb 55%, #0f766e 100%);
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
        opacity: 0.9;
        margin-top: 7px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 750;
        color: #111827;
        margin: 32px 0 6px 0;
    }

    .section-description {
        color: #64748b;
        margin-bottom: 16px;
    }

    .model-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 22px;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
    }

    .model-name {
        font-size: 20px;
        font-weight: 750;
        color: #172554;
        margin-bottom: 5px;
    }

    .model-description {
        color: #64748b;
        font-size: 14px;
    }

    .small-note {
        color: #64748b;
        font-size: 13px;
        margin-top: 8px;
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


# ---------------------------------------------------------
# 데이터
# ---------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


df = load_data()


# ---------------------------------------------------------
# 사이드바
# ---------------------------------------------------------

with st.sidebar:
    st.markdown("## 🧠 뇌졸중 예측 실습실")
    st.caption("Stroke Data Explorer")

    st.divider()

    st.markdown("### 📌 데이터")
    st.write(f"**{len(df):,}명**")
    st.write(f"**{len(df.columns)}개 변수**")

    st.divider()

    st.markdown("### 🗂️ 현재 페이지")
    st.markdown("🤖 **분류 모델**")

    st.divider()

    st.caption("분류 모델이 어떻게 답을 만드는지 살펴보세요.")


# ---------------------------------------------------------
# 헤더
# ---------------------------------------------------------

st.markdown(
    """
    <div class="page-header">
        <div class="page-icon">🤖</div>
        <div class="page-title">분류 모델</div>
        <div class="page-description">
            나이와 건강 정보를 이용해 뇌졸중 여부를 분류하는 두 가지 모델을 비교합니다.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 입력 속성 선택
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">🎯 입력으로 사용할 속성 고르기</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-description">
        뇌졸중을 예측할 때 사용할 속성을 선택하세요.
        처음에는 체질량지수(bmi)를 제외한 네 가지가 선택되어 있습니다.
    </div>
    """,
    unsafe_allow_html=True,
)

feature_options = [
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease",
]

selected_features = st.multiselect(
    "사용할 속성",
    options=feature_options,
    default=[
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease",
    ],
    format_func=lambda x: FEATURE_NAMES[x],
)


if len(selected_features) < 2:
    st.warning("입력 속성은 두 개 이상 골라 주세요.")
    st.stop()


selected_feature_labels = [
    FEATURE_NAMES[feature]
    for feature in selected_features
]

st.info(
    "선택한 속성: " + " · ".join(selected_feature_labels)
)


# ---------------------------------------------------------
# 번호 순 정렬 + 10명 단위 테스트 고정
# ---------------------------------------------------------

sorted_df = df.sort_values("id").reset_index(drop=True)

test_indices = []

for start in range(0, len(sorted_df), 10):
    group_indices = list(range(start, min(start + 10, len(sorted_df))))

    # 각 10명 묶음의 앞 3명을 테스트용으로 고정
    test_indices.extend(group_indices[:3])


test_indices = test_indices[:1533]

test_mask = sorted_df.index.isin(test_indices)

test_df = sorted_df.loc[test_mask].copy()
train_pool_df = sorted_df.loc[~test_mask].copy()


# ---------------------------------------------------------
# BMI 중앙값 처리
# ---------------------------------------------------------

if "bmi" in selected_features:
    train_bmi_median = train_pool_df["bmi"].median()

    train_pool_df["bmi"] = train_pool_df["bmi"].fillna(
        train_bmi_median
    )

    test_df["bmi"] = test_df["bmi"].fillna(
        train_bmi_median
    )

    st.caption(
        f"체질량지수를 선택했으므로 훈련용 데이터에서 계산한 중앙값 "
        f"{train_bmi_median:.2f}으로 빈 값을 채웠습니다."
    )


# ---------------------------------------------------------
# 훈련용 크기 맞추기
# ---------------------------------------------------------

positive_train = train_pool_df[
    train_pool_df["stroke"] == 1
].copy()

negative_train = train_pool_df[
    train_pool_df["stroke"] == 0
].copy()

minority_count = min(
    len(positive_train),
    len(negative_train),
)

positive_train = positive_train.sample(
    n=minority_count,
    random_state=RANDOM_STATE,
)

negative_train = negative_train.sample(
    n=minority_count,
    random_state=RANDOM_STATE,
)

balanced_train_df = pd.concat(
    [positive_train, negative_train]
).sort_values("id").reset_index(drop=True)


X_train = balanced_train_df[selected_features]
y_train = balanced_train_df["stroke"]

X_test = test_df[selected_features]
y_test = test_df["stroke"]


# ---------------------------------------------------------
# 모델
# ---------------------------------------------------------

logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE,
)

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
)

logistic_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)


logistic_train_accuracy = logistic_model.score(
    X_train,
    y_train,
)

logistic_test_accuracy = logistic_model.score(
    X_test,
    y_test,
)

tree_train_accuracy = tree_model.score(
    X_train,
    y_train,
)

tree_test_accuracy = tree_model.score(
    X_test,
    y_test,
)


# ---------------------------------------------------------
# 기준 모델
# ---------------------------------------------------------

majority_class = y_train.value_counts().idxmax()

baseline_train_accuracy = (
    y_train == majority_class
).mean()

baseline_test_accuracy = (
    y_test == majority_class
).mean()


# ---------------------------------------------------------
# 모델 정확도 카드
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">🏆 모델 정확도</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-description">
        테스트 데이터의 정확도를 크게 표시하고, 같은 카드 안에 훈련 데이터와 테스트 데이터를 함께 표시합니다.
    </div>
    """,
    unsafe_allow_html=True,
)

card1, card2, card3 = st.columns(3)


with card1:
    st.markdown(
        """
        <div class="model-card">
            <div class="model-name">
                로지스틱 회귀(확률로 답하는 모델)
            </div>
            <div class="model-description">
                뇌졸중일 확률을 계산한 뒤 0.5를 기준으로 나눕니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.metric(
        "테스트 데이터 정확도",
        f"{logistic_test_accuracy * 100:.2f}%",
    )

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "훈련",
            f"{logistic_train_accuracy * 100:.2f}%",
        )

    with c2:
        st.metric(
            "테스트",
            f"{logistic_test_accuracy * 100:.2f}%",
        )


with card2:
    st.markdown(
        """
        <div class="model-card">
            <div class="model-name">
                의사결정트리(질문으로 답하는 모델)
            </div>
            <div class="model-description">
                속성에 관한 질문을 최대 세 번까지 하며 답을 찾아갑니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.metric(
        "테스트 데이터 정확도",
        f"{tree_test_accuracy * 100:.2f}%",
    )

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "훈련",
            f"{tree_train_accuracy * 100:.2f}%",
        )

    with c2:
        st.metric(
            "테스트",
            f"{tree_test_accuracy * 100:.2f}%",
        )


with card3:
    st.markdown(
        """
        <div class="model-card">
            <div class="model-name">
                입력을 하나도 보지 않는 기준 모델
            </div>
            <div class="model-description">
                훈련용 데이터에서 많은 쪽으로만 계속 답합니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.metric(
        "테스트 데이터 정확도",
        f"{baseline_test_accuracy * 100:.2f}%",
    )

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "훈련",
            f"{baseline_train_accuracy * 100:.2f}%",
        )

    with c2:
        st.metric(
            "테스트",
            f"{baseline_test_accuracy * 100:.2f}%",
        )


st.caption(
    f"훈련용으로는 크기를 맞춘 {len(balanced_train_df):,}명, "
    f"테스트용으로는 고정한 {len(test_df):,}명을 사용했습니다."
)


# ---------------------------------------------------------
# 산점도
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">🗺️ 두 속성으로 보는 분류 결과</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-description">
        선택한 속성 가운데 두 개를 골라 평면에 나타냅니다.
        점은 테스트 데이터이며, 배경은 의사결정트리가 나눈 영역입니다.
        검은 선은 로지스틱 회귀가 확률 0.5에서 나누는 경계입니다.
    </div>
    """,
    unsafe_allow_html=True,
)


axis_col1, axis_col2 = st.columns(2)

with axis_col1:
    x_feature = st.selectbox(
        "가로축",
        selected_features,
        format_func=lambda x: FEATURE_NAMES[x],
    )

remaining_features = [
    feature
    for feature in selected_features
    if feature != x_feature
]

with axis_col2:
    y_feature = st.selectbox(
        "세로축",
        remaining_features,
        format_func=lambda x: FEATURE_NAMES[x],
    )


# 두 축이 아닌 변수의 중앙값
fixed_values = {}

for feature in selected_features:
    if feature not in [x_feature, y_feature]:
        fixed_values[feature] = X_train[feature].median()


if fixed_values:
    fixed_text = " · ".join(
        [
            f"{FEATURE_NAMES[feature]} = {value:.2f}"
            if isinstance(value, float)
            else f"{FEATURE_NAMES[feature]} = {value}"
            for feature, value in fixed_values.items()
        ]
    )

    st.info(
        "그림의 두 축이 아닌 속성은 훈련용 데이터의 중앙값으로 고정했습니다: "
        + fixed_text
    )


# ---------------------------------------------------------
# 그림용 범위
# ---------------------------------------------------------

x_values = X_train[x_feature]
y_values = X_train[y_feature]

x_min = float(x_values.min())
x_max = float(x_values.max())
y_min = float(y_values.min())
y_max = float(y_values.max())

x_padding = (x_max - x_min) * 0.08
y_padding = (y_max - y_min) * 0.08

if x_padding == 0:
    x_padding = 1

if y_padding == 0:
    y_padding = 1

plot_x_min = x_min - x_padding
plot_x_max = x_max + x_padding
plot_y_min = y_min - y_padding
plot_y_max = y_max + y_padding


# ---------------------------------------------------------
# 격자 만들기
# ---------------------------------------------------------

grid_size = 80

x_grid = [
    plot_x_min
    + (plot_x_max - plot_x_min) * i / (grid_size - 1)
    for i in range(grid_size)
]

y_grid = [
    plot_y_min
    + (plot_y_max - plot_y_min) * i / (grid_size - 1)
    for i in range(grid_size)
]

grid_rows = []

for y_value in y_grid:
    for x_value in x_grid:
        row = {}
