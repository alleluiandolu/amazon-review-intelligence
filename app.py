
import re
from pathlib import Path


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx


# ============================================================
# AMAZON REVIEW INTELLIGENCE - VERSION 4.0
# Big Data Analytics | Amazon Fine Food Reviews
# ============================================================

st.set_page_config(
    page_title="Amazon Review Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).resolve().parent

COLORS = {
    "Positive": "#22C55E",
    "Neutral": "#F59E0B",
    "Negative": "#EF4444"
}


# ============================================================
# DATA HELPERS
# ============================================================

def find_file(filename):
    for folder in [BASE, BASE / "data"]:
        exact = folder / filename
        if exact.is_file():
            return exact

        if not folder.is_dir():
            continue

        target = Path(filename)
        pattern = (
            re.escape(target.stem)
            + r"(?:\s*\(\d+\))?"
            + re.escape(target.suffix)
        )

        for path in sorted(folder.iterdir()):
            if path.is_file() and re.fullmatch(
                pattern, path.name, re.IGNORECASE
            ):
                return path

    return None


@st.cache_data
def read_csv(path):
    return pd.read_csv(path)


def load_file(filename, required=True):
    path = find_file(filename)

    if path is None:
        if required:
            st.error(f"Missing required file: {filename}")
            st.stop()
        return None

    try:
        return read_csv(str(path)).copy()
    except Exception as exc:
        st.error(f"Cannot read {filename}: {exc}")
        if required:
            st.stop()
        return None


def sentiment_clean(series):
    mapping = {
        "positive": "Positive",
        "positif": "Positive",
        "negative": "Negative",
        "negatif": "Negative",
        "neutral": "Neutral",
        "netral": "Neutral"
    }
    return (
        series.astype(str).str.strip().str.lower()
        .map(mapping).fillna("Unknown")
    )


def sentiment_chart(df, value_col, title):
    grouped = (
        df.groupby("Sentiment", as_index=False)[value_col]
        .sum()
    )
    return px.pie(
        grouped,
        names="Sentiment",
        values=value_col,
        color="Sentiment",
        color_discrete_map=COLORS,
        hole=0.55,
        title=title
    )


def weighted_rating(df):
    count = df["review_count"].sum()
    if count == 0:
        return 0.0
    return float(
        (df["Score"] * df["review_count"]).sum() / count
    )


def render_network(graph, title, height=620):
    if graph.number_of_nodes() == 0:
        st.warning("No graph nodes available.")
        return

    pos = nx.spring_layout(
        graph, seed=42, iterations=80, weight=None
    )

    edge_x, edge_y = [], []
    for u, v in graph.edges():
        edge_x.extend([pos[u][0], pos[v][0], None])
        edge_y.extend([pos[u][1], pos[v][1], None])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="#64748B", width=0.8),
        hoverinfo="skip",
        showlegend=False
    ))

    node_x, node_y = [], []
    labels, hover, sizes, colors = [], [], [], []

    for node, attrs in graph.nodes(data=True):
        kind = attrs.get("kind", "Keyword")
        label = attrs.get("label", str(node))

        node_x.append(pos[node][0])
        node_y.append(pos[node][1])
        labels.append("" if kind == "Review" else label)
        hover.append(
            f"{kind}: {label}<br>Degree: {graph.degree(node)}"
        )

        if kind == "Product":
            colors.append("#A78BFA")
            sizes.append(19)
        elif kind == "Review":
            colors.append("#38BDF8")
            sizes.append(7)
        elif kind == "Sentiment":
            colors.append(COLORS.get(label, "#94A3B8"))
            sizes.append(28)
        else:
            colors.append("#A78BFA")
            sizes.append(13)

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=labels,
        textposition="top center",
        hovertext=hover,
        hoverinfo="text",
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(width=0.5, color="white")
        ),
        showlegend=False
    ))

    fig.update_layout(
        title=title,
        height=height,
        template="plotly_dark",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=60, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)


def pick_column(df, names):
    lookup = {
        str(c).strip().lower(): c
        for c in df.columns
    }
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


# ============================================================
# LOAD DATA
# ============================================================

summary = load_file("dashboard_summary.csv")
products = load_file("product_summary.csv")
reviews = load_file("review_sample.csv.gz")

for col in ["Year", "Score", "review_count"]:
    summary[col] = pd.to_numeric(
        summary[col], errors="coerce"
    )

summary["review_count"] = (
    summary["review_count"].fillna(0)
)
summary["Sentiment"] = sentiment_clean(
    summary["Sentiment"]
)

for col in [
    "review_count", "average_rating",
    "negative_count"
]:
    products[col] = pd.to_numeric(
        products[col], errors="coerce"
    ).fillna(0)

products["ProductId"] = (
    products["ProductId"].astype(str)
)

reviews["Score"] = pd.to_numeric(
    reviews["Score"], errors="coerce"
)
reviews["Year"] = pd.to_numeric(
    reviews["Year"], errors="coerce"
)
reviews["Text"] = reviews["Text"].fillna("")
reviews["Sentiment"] = sentiment_clean(
    reviews["Sentiment"]
)

graph_data = load_file(
    "graph_review_sample.csv", required=False
)
keyword_data = load_file(
    "top_words_by_sentiment.csv", required=False
)


# ============================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ============================================================

def page_overview():
    st.title("📊 Executive Overview")
    st.caption("Amazon Fine Food Reviews | Business Intelligence")

    years = sorted(
        summary["Year"].dropna().astype(int).unique()
    )

    if not years:
        st.warning("No valid year values found.")
        return

    selected = st.sidebar.select_slider(
        "Select Year Range",
        options=years,
        value=(years[0], years[-1])
    )

    df = summary[
        summary["Year"].between(*selected)
    ].copy()

    total = int(df["review_count"].sum())
    positive = df.loc[
        df["Sentiment"] == "Positive", "review_count"
    ].sum()

    positive_pct = positive / total * 100 if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", f"{total:,}")
    c2.metric("Products (All Years)", f"{len(products):,}")
    c3.metric("Average Rating", f"{weighted_rating(df):.2f}/5")
    c4.metric("Positive Reviews", f"{positive_pct:.1f}%")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.plotly_chart(
            sentiment_chart(
                df, "review_count", "Sentiment Distribution"
            ),
            use_container_width=True
        )

    with right:
        ratings = df.groupby(
            "Score", as_index=False
        )["review_count"].sum()

        fig = px.bar(
            ratings,
            x="Score",
            y="review_count",
            title="Rating Distribution",
            text_auto=".2s"
        )
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, use_container_width=True)

    yearly = df.groupby(
        "Year", as_index=False
    )["review_count"].sum().sort_values("Year")

    fig = px.line(
        yearly,
        x="Year",
        y="review_count",
        markers=True,
        title="Review Volume by Year"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Sentiment labels are derived from ratings: "
        "1–2 Negative, 3 Neutral, 4–5 Positive. "
        "The product count covers all years."
    )


# ============================================================
# PAGE 2: PRODUCT ANALYTICS
# ============================================================

def page_products():
    st.title("📦 Product Analytics")
    st.caption("Product Performance | All Available Years")

    df = products.copy()

    df["negative_pct"] = np.where(
        df["review_count"] > 0,
        df["negative_count"] /
        df["review_count"] * 100,
        0
    )

    total = df["review_count"].sum()
    avg = (
        (df["average_rating"] * df["review_count"]).sum()
        / total if total else 0
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Products", f"{len(df):,}")
    c2.metric("Reviews", f"{int(total):,}")
    c3.metric("Average Rating", f"{avg:.2f}/5")

    st.divider()

    minimum = st.slider(
        "Minimum Reviews per Product",
        1, 100, 10
    )

    filtered = df[
        df["review_count"] >= minimum
    ].copy()

    if filtered.empty:
        st.warning("No products match this threshold.")
        return

    top = filtered.nlargest(15, "review_count")

    fig = px.bar(
        top,
        x="review_count",
        y="ProductId",
        color="average_rating",
        color_continuous_scale="Viridis",
        orientation="h",
        title="Top 15 Most Reviewed Products"
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )
    st.plotly_chart(fig, use_container_width=True)

    risk = filtered.sort_values(
        ["negative_pct", "review_count"],
        ascending=[False, False]
    ).head(15)

    fig = px.bar(
        risk,
        x="negative_pct",
        y="ProductId",
        color="negative_pct",
        color_continuous_scale="Reds",
        orientation="h",
        title="Products with Highest Negative Review Percentage"
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Product Performance Table")
    st.dataframe(
        filtered.sort_values(
            "review_count", ascending=False
        ),
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# PAGE 3: MACHINE LEARNING INSIGHTS
# ============================================================

def page_ml():
    st.title("🧠 Machine Learning Insights")
    st.caption("BAB V | Spark MLlib vs Scikit-learn")

    st.info(
        "Comparison experiment: 14,697 reviews "
        "(11,727 training / 2,970 testing). "
        "Separate from the main full-data experiment."
    )

    metrics = pd.DataFrame({
        "Model": ["Spark MLlib", "Scikit-learn"],
        "Accuracy": [80.24, 79.83],
        "Macro Precision": [58.95, 57.59],
        "Macro Recall": [60.03, 56.48],
        "Macro F1": [59.42, 56.98],
        "Weighted F1": [80.66, 79.72],
        "Training Time (s)": [27.53, 3.76],
        "Prediction Time (s)": [2.61, 0.20]
    })

    c1, c2, c3 = st.columns(3)
    c1.metric("Best Accuracy", "80.24%")
    c2.metric("Best Macro F1", "59.42%")
    c3.metric("Fastest Training", "3.76 s")

    st.divider()

    choices = [
        "Accuracy", "Macro Precision",
        "Macro Recall", "Macro F1", "Weighted F1"
    ]

    selected = st.multiselect(
        "Evaluation Metrics",
        choices,
        default=["Accuracy", "Macro F1", "Weighted F1"]
    )

    if selected:
        long = metrics.melt(
            id_vars="Model",
            value_vars=selected,
            var_name="Metric",
            value_name="Score"
        )

        fig = px.bar(
            long,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            text_auto=".2f",
            title="Classification Performance (%)"
        )
        fig.update_yaxes(range=[0, 105])
        st.plotly_chart(fig, use_container_width=True)

    time_df = metrics.melt(
        id_vars="Model",
        value_vars=[
            "Training Time (s)", "Prediction Time (s)"
        ],
        var_name="Operation",
        value_name="Seconds"
    )

    fig = px.bar(
        time_df,
        x="Operation",
        y="Seconds",
        color="Model",
        barmode="group",
        text_auto=".2f",
        title="Computational Performance"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed Comparison")
    st.dataframe(
        metrics, hide_index=True, use_container_width=True
    )

    st.subheader("Main Spark MLlib Experiment")
    a, b, c = st.columns(3)
    a.metric("Accuracy", "81.84%")
    b.metric("Macro F1", "67.63%")
    c.metric("Weighted F1", "83.73%")

    st.caption(
        "Main experiment: 568,453 cleaned observations, "
        "113,576 test observations. "
        "Metrics are not directly interchangeable "
        "with the comparison sample."
    )

    st.markdown(
        "**Interpretation:** Spark MLlib performed slightly "
        "better on accuracy and macro F1 in the comparison "
        "sample, while Scikit-learn ran faster. "
        "Neutral sentiment remains more difficult to classify."
    )


# ============================================================
# PAGE 4: GRAPH ANALYTICS
# ============================================================

def page_graph():
    st.title("🕸️ Graph Analytics")
    st.caption("BAB VI | Network-Based Review Exploration")

    with st.expander("🔧 Graph File Diagnostics"):
        for filename in [
            "graph_review_sample.csv",
            "top_words_by_sentiment.csv"
        ]:
            path = find_file(filename)
            st.write(
                filename,
                "— " + str(path) if path else "— NOT FOUND"
            )

    st.subheader("Product–Review–Sentiment Network")

    if graph_data is not None:
        source = graph_data.copy()
        st.success(
            "Loaded BAB VI file: graph_review_sample.csv"
        )
    else:
        st.warning(
            "BAB VI graph file not found. "
            "Showing an exploratory dashboard sample instead."
        )
        source = reviews.rename(columns={
            "ProductId": "product_id",
            "Id": "review_id",
            "Sentiment": "sentiment"
        }).copy()

    product_col = pick_column(
        source, ["product_id", "ProductId"]
    )
    review_col = pick_column(
        source, ["review_id", "Id"]
    )
    sentiment_col = pick_column(
        source, ["sentiment", "label"]
    )

    if not all([product_col, review_col, sentiment_col]):
        st.error("Graph CSV columns are not recognized.")
    else:
        source = source.rename(columns={
            product_col: "product_id",
            review_col: "review_id",
            sentiment_col: "sentiment"
        })

        source = source.dropna(
            subset=["product_id", "review_id"]
        ).copy()

        source["sentiment"] = sentiment_clean(
            source["sentiment"]
        )
        source["product_id"] = (
            source["product_id"].astype(str)
        )

        st.caption(
            f"Source: {len(source):,} review observations."
        )

        product_list = (
            source["product_id"]
            .value_counts()
            .index.tolist()
        )

        if product_list:
            n_products = st.slider(
                "Number of Products",
                1, min(20, len(product_list)),
                min(5, len(product_list))
            )
            n_reviews = st.slider(
                "Maximum Reviews per Product",
                1, 10, 5
            )

            subset = source[
                source["product_id"].isin(
                    product_list[:n_products]
                )
            ].groupby(
                "product_id", group_keys=False
            ).head(n_reviews)

            graph = nx.Graph()

            for idx, row in subset.iterrows():
                p = "P:" + str(row["product_id"])
                r = "R:" + str(row["review_id"]) + ":" + str(idx)
                s = "S:" + str(row["sentiment"])

                graph.add_node(
                    p, kind="Product",
                    label=str(row["product_id"])
                )
                graph.add_node(
                    r, kind="Review",
                    label=str(row["review_id"])
                )
                graph.add_node(
                    s, kind="Sentiment",
                    label=str(row["sentiment"])
                )
                graph.add_edge(p, r)
                graph.add_edge(r, s)

            a, b, c = st.columns(3)
            a.metric("Nodes", graph.number_of_nodes())
            b.metric("Edges", graph.number_of_edges())
            c.metric(
                "Connected Components",
                nx.number_connected_components(graph)
            )

            render_network(
                graph,
                "Interactive Product–Review–Sentiment Network"
            )

            counts = subset["sentiment"].value_counts()
            fig = px.bar(
                x=counts.index,
                y=counts.values,
                labels={"x": "Sentiment", "y": "Reviews"},
                title="Sentiment in Displayed Network"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Network metrics reflect the selected subset. "
                "Sentiment comes from review ratings, "
                "not model predictions."
            )

    st.divider()
    st.subheader("Sentiment–Keyword Network")

    if keyword_data is None:
        st.warning(
            "top_words_by_sentiment.csv was not found."
        )
        return

    st.success(
        "Loaded keyword file: top_words_by_sentiment.csv"
    )

    words = keyword_data.copy()

    sc = pick_column(
        words, ["sentiment", "label", "category"]
    )
    wc = pick_column(
        words, ["word", "keyword", "term", "token"]
    )
    cc = pick_column(
        words, ["count", "frequency", "freq", "weight"]
    )

    if not sc or not wc:
        st.error("Keyword columns are not recognized.")
        st.dataframe(words.head())
        return

    words[sc] = sentiment_clean(words[sc])
    words = words.dropna(subset=[wc]).copy()
    words[wc] = words[wc].astype(str).str.lower().str.strip()

    if cc:
        words[cc] = pd.to_numeric(
            words[cc], errors="coerce"
        ).fillna(0)
        words = words.sort_values(cc, ascending=False)

    words = words.groupby(
        sc, group_keys=False
    ).head(9)

    graph = nx.Graph()

    for _, row in words.iterrows():
        sentiment = str(row[sc])
        word = str(row[wc])

        s = "S:" + sentiment
        w = "W:" + word

        graph.add_node(
            s, kind="Sentiment", label=sentiment
        )
        graph.add_node(
            w, kind="Keyword", label=word
        )
        graph.add_edge(
            s, w,
            weight=float(row[cc]) if cc else 1
        )

    a, b, c = st.columns(3)
    a.metric("Nodes", graph.number_of_nodes())
    b.metric("Edges", graph.number_of_edges())
    c.metric(
        "Unique Keywords",
        sum(
            attrs.get("kind") == "Keyword"
            for _, attrs in graph.nodes(data=True)
        )
    )

    render_network(
        graph, "Interactive Sentiment–Keyword Network"
    )

    st.caption(
        "Edges represent keyword associations. "
        "Frequencies are stored as edge weights, "
        "but line widths are not scaled by frequency. "
        "Associations do not establish causation."
    )

    st.subheader("Keyword Frequency Table")
    st.dataframe(
        words, hide_index=True, use_container_width=True
    )


# ============================================================
# PAGE 5: REVIEW EXPLORER
# ============================================================

def page_reviews():
    st.title("🔎 Review Explorer")
    st.caption("Search and Explore Customer Reviews")

    st.info(
        f"Exploring {len(reviews):,} sampled reviews, "
        "not the full dataset."
    )

    a, b, c = st.columns(3)

    with a:
        sentiments = st.multiselect(
            "Sentiment",
            ["Positive", "Neutral", "Negative"],
            default=["Positive", "Neutral", "Negative"]
        )

    with b:
        ratings = st.multiselect(
            "Rating",
            [1, 2, 3, 4, 5],
            default=[1, 2, 3, 4, 5]
        )

    with c:
        query = st.text_input(
            "Search Review Text",
            placeholder="coffee, taste, delicious"
        )

    df = reviews[
        reviews["Sentiment"].isin(sentiments)
        & reviews["Score"].isin(ratings)
    ].copy()

    if query.strip():
        df = df[
            df["Text"].str.contains(
                query.strip(),
                case=False,
                na=False,
                regex=False
            )
        ]

    st.metric("Matching Reviews", f"{len(df):,}")

    left, right = st.columns(2)

    with left:
        counts = df["Sentiment"].value_counts()
        if not counts.empty:
            fig = px.pie(
                names=counts.index,
                values=counts.values,
                hole=0.5,
                title="Filtered Sentiment Distribution",
                color=counts.index,
                color_discrete_map=COLORS
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        counts = df["Score"].value_counts().sort_index()
        if not counts.empty:
            fig = px.bar(
                x=counts.index,
                y=counts.values,
                labels={"x": "Rating", "y": "Count"},
                title="Filtered Rating Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

    columns = [
        "Id", "ProductId", "Score",
        "Sentiment", "Year", "Text"
    ]
    available = [c for c in columns if c in df.columns]

    st.subheader("Customer Review Records")
    st.dataframe(
        df[available],
        hide_index=True,
        use_container_width=True,
        height=450
    )

    st.download_button(
        "⬇️ Download Filtered Reviews",
        data=df[available].to_csv(index=False).encode("utf-8"),
        file_name="filtered_amazon_reviews.csv",
        mime="text/csv"
    )


# ============================================================
# PAGE 6: BUSINESS INTELLIGENCE INSIGHTS
# ============================================================

def page_business():
    st.title("💼 Business Intelligence Insights")
    st.caption(
        "Data-Driven Recommendations | Customer & Product Strategy"
    )

    total = summary["review_count"].sum()

    sentiment_totals = (
        summary.groupby("Sentiment")["review_count"]
        .sum()
    )

    positive = sentiment_totals.get("Positive", 0)
    negative = sentiment_totals.get("Negative", 0)
    neutral = sentiment_totals.get("Neutral", 0)

    pos_pct = positive / total * 100 if total else 0
    neg_pct = negative / total * 100 if total else 0

    # ---------- CUSTOMER SATISFACTION ----------

    st.subheader("1. Customer Satisfaction Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Reviews", f"{int(total):,}")
    c2.metric("Positive Share", f"{pos_pct:.1f}%")
    c3.metric("Negative Share", f"{neg_pct:.1f}%")
    c4.metric(
        "Average Rating",
        f"{weighted_rating(summary):.2f}/5"
    )

    left, right = st.columns(2)

    with left:
        st.plotly_chart(
            sentiment_chart(
                summary,
                "review_count",
                "Overall Customer Sentiment"
            ),
            use_container_width=True
        )

    with right:
        breakdown = pd.DataFrame({
            "Sentiment": [
                "Positive", "Neutral", "Negative"
            ],
            "Reviews": [
                positive, neutral, negative
            ]
        })

        fig = px.bar(
            breakdown,
            x="Sentiment",
            y="Reviews",
            color="Sentiment",
            color_discrete_map=COLORS,
            title="Customer Sentiment Breakdown",
            text_auto=".2s"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---------- PRODUCT RISK ----------

    st.divider()
    st.subheader("2. Product Risk Screening")

    st.write(
        "Identify products with a high proportion of "
        "negative reviews. This is a screening indicator, "
        "not proof of a product defect."
    )

    threshold = st.slider(
        "Minimum Reviews for Risk Screening",
        min_value=10,
        max_value=200,
        value=50,
        step=10
    )

    risk = products[
        products["review_count"] >= threshold
    ].copy()

    risk["negative_pct"] = np.where(
        risk["review_count"] > 0,
        risk["negative_count"] /
        risk["review_count"] * 100,
        0
    )

    risk = risk.sort_values(
        ["negative_pct", "review_count"],
        ascending=[False, False]
    )

    if risk.empty:
        st.warning(
            "No products match the minimum review threshold."
        )
    else:
        top_risk = risk.head(10)

        fig = px.bar(
            top_risk,
            x="negative_pct",
            y="ProductId",
            orientation="h",
            color="negative_pct",
            color_continuous_scale="Reds",
            title="Top 10 Products by Negative Review Share",
            labels={
                "negative_pct": "Negative Reviews (%)"
            }
        )

        fig.update_layout(
            yaxis={"categoryorder": "total ascending"}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            top_risk[[
                "ProductId",
                "review_count",
                "negative_count",
                "negative_pct",
                "average_rating"
            ]],
            hide_index=True,
            use_container_width=True
        )

    # ---------- YEARLY TREND ----------

    st.divider()
    st.subheader("3. Review Activity Trends")

    yearly = (
        summary.groupby(
            "Year", as_index=False
        )["review_count"].sum()
        .dropna(subset=["Year"])
        .sort_values("Year")
    )

    fig = px.area(
        yearly,
        x="Year",
        y="review_count",
        markers=True,
        title="Annual Review Activity"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Review activity measures observed review volume, "
        "not product sales or revenue. "
        "Yearly totals should not be interpreted as "
        "sales growth."
    )

    # ---------- RECOMMENDATIONS ----------

    st.divider()
    st.subheader("4. Business Recommendations")

    st.markdown(
        "**Customer experience monitoring**"
    )
    st.write(
        f"Negative reviews account for {neg_pct:.1f}% "
        "of the rating-derived sentiment distribution. "
        "Review their text to identify recurring complaints "
        "before deciding on corrective actions."
    )

    st.markdown(
        "**Prioritize product investigations**"
    )

    if not risk.empty:
        highest = risk.iloc[0]
        st.write(
            f"Product `{highest['ProductId']}` has "
            f"{highest['negative_pct']:.1f}% negative reviews "
            f"among {int(highest['review_count']):,} reviews. "
            "It is a candidate for further investigation "
            "under the selected minimum-review threshold."
        )
    else:
        st.write(
            "Lower the minimum review threshold "
            "to identify candidates for investigation."
        )

    st.markdown(
        "**Improve sentiment classification**"
    )
    st.write(
        "The BAB V experiment showed weaker performance "
        "for the Neutral class. Consider additional "
        "class-level evaluation and data quality checks "
        "before using predictions for operational decisions."
    )

    st.markdown(
        "**Combine multiple analytical perspectives**"
    )
    st.write(
        "Use product metrics for prioritization, "
        "review text for qualitative investigation, "
        "and graph networks to explore relationships. "
        "These analyses are complementary rather than "
        "proof of causal relationships."
    )

    # ---------- EXECUTIVE SUMMARY ----------

    st.divider()
    st.subheader("5. Executive Summary")

    st.markdown(
        f"""
        The Amazon Fine Food Reviews dataset contains
        **{int(total):,} reviews** across
        **{len(products):,} product IDs**.

        Rating-derived sentiment indicates that
        **{pos_pct:.1f}%** of reviews are positive and
        **{neg_pct:.1f}%** are negative.

        The main Spark MLlib classification experiment
        achieved **81.84% accuracy**, while the
        separate framework comparison showed
        **80.24% accuracy for Spark MLlib** and
        **79.83% for Scikit-learn**.

        Product-level risk screening and graph analytics
        provide additional ways to investigate customer
        experience and prioritize further analysis.
        """
    )

    st.caption(
        "Recommendations are analytical suggestions, "
        "not validated causal findings or evidence "
        "of actual financial performance."
    )


# ============================================================
# NAVIGATION / APPLICATION ENTRY POINT
# ============================================================

PAGES = {
    "Executive Overview": page_overview,
    "Product Analytics": page_products,
    "Machine Learning Insights": page_ml,
    "Graph Analytics": page_graph,
    "Review Explorer": page_reviews,
    "Business Intelligence Insights": page_business
}

# SIDEBAR
logo_path = Path(__file__).resolve().parent / "logo.png"

if logo_path.exists():
    st.sidebar.image(str(logo_path), width=165)

st.sidebar.markdown("### Amazon Review Intelligence")
st.sidebar.caption("Big Data Analytics Dashboard")


st.markdown("### Welcome to Amazon Review Intelligence")
st.write(
    "Explore customer reviews, product performance, "
    "sentiment classification, graph analytics, and "
    "business intelligence insights through an "
    "interactive analytics dashboard."
)

st.sidebar.divider()



page = st.sidebar.radio(
    "Navigation",
    list(PAGES.keys())
)

st.sidebar.divider()
st.sidebar.caption("Amazon Fine Food Reviews")
st.sidebar.caption("Version 4.0 | Six-Page Dashboard")

PAGES[page]()

st.divider()
st.caption(
    "Amazon Review Intelligence | "
    "Big Data Analytics | Version 4.0"
)
