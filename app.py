
import streamlit as st
import pandas as pd
import plotly.express as px

# ============================================================
# AMAZON REVIEW INTELLIGENCE
# Big Data Analytics Dashboard - Version 2.0
# ============================================================

st.set_page_config(
    page_title="Amazon Review Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- DATA --------------------

@st.cache_data
def load_data():
    summary = pd.read_csv("dashboard_summary.csv")
    products = pd.read_csv("product_summary.csv")
    return summary, products


summary, products = load_data()

summary["Year"] = pd.to_numeric(
    summary["Year"], errors="coerce"
)
summary["Score"] = pd.to_numeric(
    summary["Score"], errors="coerce"
)
summary["review_count"] = pd.to_numeric(
    summary["review_count"], errors="coerce"
).fillna(0)

# -------------------- SIDEBAR --------------------

st.sidebar.title("📊 Amazon Intelligence")
st.sidebar.caption("Big Data Analytics Dashboard")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Product Analytics",
        "Machine Learning Insights"
    ]
)

st.sidebar.divider()

# ============================================================
# PAGE 1 - EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    st.title("📊 Executive Overview")
    st.caption(
        "Amazon Fine Food Reviews | Business Intelligence"
    )

    years = sorted(
        summary["Year"].dropna().astype(int).unique()
    )

    year_range = st.sidebar.select_slider(
        "Select Year Range",
        options=years,
        value=(years[0], years[-1])
    )

    filtered = summary[
        summary["Year"].between(
            year_range[0], year_range[1]
        )
    ].copy()

    total_reviews = int(
        filtered["review_count"].sum()
    )

    positive_reviews = int(
        filtered.loc[
            filtered["Sentiment"] == "Positive",
            "review_count"
        ].sum()
    )

    avg_rating = (
        (
            filtered["Score"] *
            filtered["review_count"]
        ).sum() / total_reviews
        if total_reviews else 0
    )

    positive_pct = (
        positive_reviews / total_reviews * 100
        if total_reviews else 0
    )

    total_products = len(products)

    # KPI CARDS
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Reviews",
        f"{total_reviews:,}"
    )
    c2.metric(
        "Total Products (All Years)",
        f"{total_products:,}"
    )
    c3.metric(
        "Average Rating",
        f"{avg_rating:.2f} / 5"
    )
    c4.metric(
        "Positive Reviews",
        f"{positive_pct:.1f}%"
    )

    st.divider()

    # SENTIMENT + RATING
    col1, col2 = st.columns(2)

    with col1:
        sentiment = (
            filtered.groupby(
                "Sentiment", as_index=False
            )["review_count"].sum()
        )

        fig_sentiment = px.pie(
            sentiment,
            names="Sentiment",
            values="review_count",
            hole=0.55,
            title="Sentiment Distribution",
            color="Sentiment",
            color_discrete_map={
                "Positive": "#22C55E",
                "Neutral": "#F59E0B",
                "Negative": "#EF4444"
            }
        )

        st.plotly_chart(
            fig_sentiment,
            use_container_width=True
        )

    with col2:
        ratings = (
            filtered.groupby(
                "Score", as_index=False
            )["review_count"].sum()
        )

        fig_ratings = px.bar(
            ratings,
            x="Score",
            y="review_count",
            title="Rating Distribution",
            text_auto=True,
            labels={
                "Score": "Rating",
                "review_count": "Number of Reviews"
            },
            color_discrete_sequence=["#6366F1"]
        )

        fig_ratings.update_layout(
            xaxis=dict(
                tickmode="linear",
                dtick=1
            )
        )

        st.plotly_chart(
            fig_ratings,
            use_container_width=True
        )

    # YEARLY TREND
    yearly = (
        filtered.groupby(
            "Year", as_index=False
        )["review_count"].sum()
        .sort_values("Year")
    )

    fig_yearly = px.line(
        yearly,
        x="Year",
        y="review_count",
        markers=True,
        title="Review Trends by Year",
        labels={
            "Year": "Year",
            "review_count": "Number of Reviews"
        }
    )

    fig_yearly.update_traces(
        line=dict(width=3),
        marker=dict(size=8)
    )

    st.plotly_chart(
        fig_yearly,
        use_container_width=True
    )

    st.info(
        "Sentiment is derived from review ratings: "
        "1-2 = Negative, 3 = Neutral, 4-5 = Positive. "
        "Product count represents all years."
    )

# ============================================================
# PAGE 2 - PRODUCT ANALYTICS
# ============================================================

elif page == "Product Analytics":

    st.title("📦 Product Analytics")
    st.caption(
        "Product Performance | All Available Years"
    )

    product_data = products.copy()

    product_data["negative_pct"] = (
        product_data["negative_count"] /
        product_data["review_count"].replace(0, float("nan"))
        * 100
    ).fillna(0)

    total_products = len(product_data)
    avg_product_rating = (
        (
            product_data["average_rating"] *
            product_data["review_count"]
        ).sum() /
        product_data["review_count"].sum()
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Products",
        f"{total_products:,}"
    )
    c2.metric(
        "Overall Average Rating",
        f"{avg_product_rating:.2f} / 5"
    )
    c3.metric(
        "Total Product Reviews",
        f"{int(product_data['review_count'].sum()):,}"
    )

    st.divider()

    min_reviews = st.slider(
        "Minimum Number of Reviews",
        min_value=1,
        max_value=100,
        value=10
    )

    filtered_products = product_data[
        product_data["review_count"] >= min_reviews
    ].copy()

    top_products = filtered_products.nlargest(
        15, "review_count"
    )

    fig_top = px.bar(
        top_products,
        x="review_count",
        y="ProductId",
        orientation="h",
        color="average_rating",
        color_continuous_scale="Viridis",
        title="Top 15 Most Reviewed Products",
        labels={
            "review_count": "Number of Reviews",
            "ProductId": "Product ID",
            "average_rating": "Average Rating"
        }
    )

    fig_top.update_layout(
        yaxis=dict(
            categoryorder="total ascending"
        )
    )

    st.plotly_chart(
        fig_top,
        use_container_width=True
    )

    st.subheader("Products with High Negative Review Rates")

    risk_products = (
        filtered_products
        .sort_values(
            ["negative_pct", "review_count"],
            ascending=[False, False]
        )
        .head(15)
    )

    fig_risk = px.bar(
        risk_products,
        x="negative_pct",
        y="ProductId",
        orientation="h",
        color="negative_pct",
        color_continuous_scale="Reds",
        title="Top 15 Products by Negative Review Percentage",
        labels={
            "negative_pct": "Negative Reviews (%)",
            "ProductId": "Product ID"
        }
    )

    fig_risk.update_layout(
        yaxis=dict(
            categoryorder="total ascending"
        )
    )

    st.plotly_chart(
        fig_risk,
        use_container_width=True
    )

    st.subheader("Product Performance Table")

    st.dataframe(
        filtered_products.sort_values(
            "review_count",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Product metrics cover all years. "
        "Products with few reviews may have unstable "
        "negative review percentages."
    )

# ============================================================
# PAGE 3 - MACHINE LEARNING INSIGHTS
# ============================================================

elif page == "Machine Learning Insights":

    st.title("🧠 Machine Learning Insights")
    st.caption(
        "Sentiment Classification | Spark MLlib vs Scikit-learn"
    )

    st.info(
        "Results below are from the BAB V comparison "
        "experiment using a sample of 14,697 reviews "
        "(11,727 training and 2,970 testing). "
        "They are separate from the main full-data "
        "Spark MLlib evaluation."
    )

    metrics = pd.DataFrame({
        "Model": [
            "Spark MLlib",
            "Scikit-learn"
        ],
        "Accuracy": [80.24, 79.83],
        "Macro Precision": [58.95, 57.59],
        "Macro Recall": [60.03, 56.48],
        "Macro F1": [59.42, 56.98],
        "Weighted F1": [80.66, 79.72],
        "Training Time (s)": [27.53, 3.76],
        "Prediction Time (s)": [2.61, 0.20]
    })

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Best Accuracy",
        "80.24%",
        "Spark MLlib"
    )
    c2.metric(
        "Best Macro F1",
        "59.42%",
        "Spark MLlib"
    )
    c3.metric(
        "Fastest Training",
        "3.76 s",
        "Scikit-learn"
    )

    st.divider()

    selected_metrics = st.multiselect(
        "Select Evaluation Metrics",
        [
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1",
            "Weighted F1"
        ],
        default=[
            "Accuracy",
            "Macro F1",
            "Weighted F1"
        ]
    )

    if selected_metrics:

        plot_data = metrics.melt(
            id_vars="Model",
            value_vars=selected_metrics,
            var_name="Metric",
            value_name="Score"
        )

        fig_metrics = px.bar(
            plot_data,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            text_auto=".2f",
            title="Model Performance Comparison",
            labels={
                "Score": "Score (%)"
            }
        )

        fig_metrics.update_layout(
            yaxis_range=[0, 105]
        )

        st.plotly_chart(
            fig_metrics,
            use_container_width=True
        )

    st.subheader("Computational Performance")

    time_data = metrics.melt(
        id_vars="Model",
        value_vars=[
            "Training Time (s)",
            "Prediction Time (s)"
        ],
        var_name="Operation",
        value_name="Seconds"
    )

    fig_time = px.bar(
        time_data,
        x="Operation",
        y="Seconds",
        color="Model",
        barmode="group",
        text_auto=".2f",
        title="Training and Prediction Time"
    )

    st.plotly_chart(
        fig_time,
        use_container_width=True
    )

    st.subheader("Detailed Evaluation Results")

    st.dataframe(
        metrics,
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Main Spark MLlib Experiment")

    main_c1, main_c2 = st.columns(2)

    main_c1.metric(
        "Main Model Accuracy",
        "81.84%"
    )
    main_c2.metric(
        "Main Model Macro F1",
        "67.63%"
    )

    st.caption(
        "Main model: 568,453 cleaned observations, "
        "113,576 test observations. "
        "Results are not directly interchangeable "
        "with the smaller comparison experiment."
    )

    st.subheader("Key Findings")

    st.markdown("""
    - **Spark MLlib** achieved slightly higher accuracy
      and macro F1-score on the comparison sample.
    - **Scikit-learn** trained and predicted faster in
      the measured experiment.
    - **Macro F1-score** helps evaluate performance
      across Negative, Neutral, and Positive classes.
    - Results are specific to these model configurations
      and datasets, not proof that one framework is
      universally superior.
    """)

# -------------------- FOOTER --------------------

st.divider()
st.caption(
    "Amazon Review Intelligence | "
    "Big Data Analytics Project | "
    "Data: Amazon Fine Food Reviews"
)
