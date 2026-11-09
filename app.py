
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Amazon Review Intelligence",
    page_icon="📊",
    layout="wide"
)

# ---------- DATA LOADING ----------
@st.cache_data
def load_data():
    summary = pd.read_csv("dashboard_summary.csv")
    products = pd.read_csv("product_summary.csv")
    return summary, products

summary, products = load_data()

# ---------- SIDEBAR ----------
st.sidebar.title("📊 Amazon Intelligence")
st.sidebar.caption("Big Data Analytics Dashboard")

page = st.sidebar.radio(
    "Navigation",
    ["Executive Overview", "Product Analytics"]
)

years = sorted(summary["Year"].dropna().astype(int).unique())

selected_years = st.sidebar.multiselect(
    "Select Year",
    years,
    default=years
)

filtered = summary[
    summary["Year"].isin(selected_years)
].copy()

# ---------- EXECUTIVE OVERVIEW ----------
if page == "Executive Overview":

    st.title("📊 Executive Overview")
    st.caption("Amazon Fine Food Reviews | Business Intelligence")

    total_reviews = int(filtered["review_count"].sum())

    positive_reviews = int(
        filtered.loc[
            filtered["Sentiment"] == "Positive",
            "review_count"
        ].sum()
    )

    avg_rating = (
        (filtered["Score"] * filtered["review_count"]).sum()
        / total_reviews
        if total_reviews else 0
    )

    positive_pct = (
        positive_reviews / total_reviews * 100
        if total_reviews else 0
    )

    # Total products is computed globally.
    # The year filter does not apply to this KPI.
    total_products = len(products)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Reviews", f"{total_reviews:,}")
    c2.metric("Total Products (All Years)", f"{total_products:,}")
    c3.metric("Average Rating", f"{avg_rating:.2f} / 5")
    c4.metric("Positive Reviews", f"{positive_pct:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        sentiment = (
            filtered.groupby("Sentiment", as_index=False)
            ["review_count"].sum()
        )

        fig = px.pie(
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

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        ratings = (
            filtered.groupby("Score", as_index=False)
            ["review_count"].sum()
        )

        fig = px.bar(
            ratings,
            x="Score",
            y="review_count",
            title="Rating Distribution",
            labels={
                "Score": "Rating",
                "review_count": "Number of Reviews"
            },
            color_discrete_sequence=["#6366F1"]
        )

        st.plotly_chart(fig, use_container_width=True)

    yearly = (
        filtered.groupby("Year", as_index=False)
        ["review_count"].sum()
        .sort_values("Year")
    )

    fig = px.line(
        yearly,
        x="Year",
        y="review_count",
        markers=True,
        title="Review Trends by Year",
        labels={"review_count": "Number of Reviews"}
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------- PRODUCT ANALYTICS ----------
elif page == "Product Analytics":

    st.title("📦 Product Analytics")
    st.caption("Product performance across all years")

    min_reviews = st.slider(
        "Minimum Number of Reviews",
        min_value=1,
        max_value=100,
        value=10
    )

    product_data = products[
        products["review_count"] >= min_reviews
    ].copy()

    top_products = product_data.nlargest(
        15, "review_count"
    )

    fig = px.bar(
        top_products,
        x="review_count",
        y="ProductId",
        orientation="h",
        title="Top 15 Most Reviewed Products",
        color="average_rating",
        color_continuous_scale="Viridis"
    )

    fig.update_layout(yaxis={"categoryorder": "total ascending"})

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Product Performance Table")

    product_data["negative_pct"] = (
        product_data["negative_count"]
        / product_data["review_count"] * 100
    ).round(2)

    st.dataframe(
        product_data.sort_values(
            "review_count", ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Product statistics cover all years. "
        "The year filter applies only to Executive Overview."
    )
