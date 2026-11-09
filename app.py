
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# ============================================================
# AMAZON REVIEW INTELLIGENCE - VERSION 3.0
# Big Data Analytics | Amazon Fine Food Reviews
# ============================================================

st.set_page_config(
    page_title="Amazon Review Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLORS = {
    "Positive": "#22C55E",
    "Neutral": "#F59E0B",
    "Negative": "#EF4444"
}

# ---------------- DATA LOADING ----------------

@st.cache_data
def load_data():
    summary = pd.read_csv("dashboard_summary.csv")
    products = pd.read_csv("product_summary.csv")
    reviews = pd.read_csv("review_sample.csv.gz")

    for col in ["Year", "Score", "review_count"]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(
                summary[col], errors="coerce"
            )

    reviews["Score"] = pd.to_numeric(
        reviews["Score"], errors="coerce"
    )
    reviews["Year"] = pd.to_numeric(
        reviews["Year"], errors="coerce"
    )
    reviews["Text"] = reviews["Text"].fillna("")

    return summary, products, reviews


summary, products, reviews = load_data()


@st.cache_data
def load_graph_data():
    if os.path.exists("graph_review_sample.csv"):
        return pd.read_csv("graph_review_sample.csv")
    return None


@st.cache_data
def load_keyword_data():
    if os.path.exists("top_words_by_sentiment.csv"):
        return pd.read_csv("top_words_by_sentiment.csv")
    return None


# ---------------- SIDEBAR ----------------

st.sidebar.title("📊 Amazon Intelligence")
st.sidebar.caption("Big Data Analytics Dashboard")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Product Analytics",
        "Machine Learning Insights",
        "Graph Analytics",
        "Review Explorer"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Amazon Fine Food Reviews")
st.sidebar.caption("Dashboard Version 3.0")

# ============================================================
# PAGE 1 - EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    st.title("📊 Executive Overview")
    st.caption("Customer Review & Sentiment Analytics")

    years = sorted(
        summary["Year"].dropna().astype(int).unique()
    )

    year_range = st.sidebar.select_slider(
        "Select Year Range",
        options=years,
        value=(years[0], years[-1])
    )

    filtered = summary[
        summary["Year"].between(*year_range)
    ].copy()

    total = int(filtered["review_count"].sum())

    positive = int(
        filtered.loc[
            filtered["Sentiment"] == "Positive",
            "review_count"
        ].sum()
    )

    avg_rating = (
        (filtered["Score"] * filtered["review_count"]).sum()
        / total if total else 0
    )

    positive_pct = positive / total * 100 if total else 0

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Reviews", f"{total:,}")
    c2.metric("Total Products (All Years)", f"{len(products):,}")
    c3.metric("Average Rating", f"{avg_rating:.2f} / 5")
    c4.metric("Positive Reviews", f"{positive_pct:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        sentiment = filtered.groupby(
            "Sentiment", as_index=False
        )["review_count"].sum()

        fig = px.pie(
            sentiment,
            names="Sentiment",
            values="review_count",
            hole=0.55,
            color="Sentiment",
            color_discrete_map=COLORS,
            title="Sentiment Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        rating = filtered.groupby(
            "Score", as_index=False
        )["review_count"].sum()

        fig = px.bar(
            rating,
            x="Score",
            y="review_count",
            title="Rating Distribution",
            text_auto=".2s",
            color_discrete_sequence=["#6366F1"]
        )
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, use_container_width=True)

    yearly = filtered.groupby(
        "Year", as_index=False
    )["review_count"].sum()

    fig = px.line(
        yearly.sort_values("Year"),
        x="Year",
        y="review_count",
        markers=True,
        title="Review Trends by Year"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Sentiment labels are derived from ratings: "
        "1-2 Negative, 3 Neutral, 4-5 Positive. "
        "The product count covers all years."
    )

# ============================================================
# PAGE 2 - PRODUCT ANALYTICS
# ============================================================

elif page == "Product Analytics":

    st.title("📦 Product Analytics")
    st.caption("Product Performance & Customer Satisfaction")

    data = products.copy()

    data["negative_pct"] = (
        data["negative_count"] /
        data["review_count"].replace(0, np.nan) * 100
    ).fillna(0)

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Products", f"{len(data):,}")
    c2.metric(
        "Total Reviews",
        f"{int(data['review_count'].sum()):,}"
    )

    overall_rating = (
        (data["average_rating"] * data["review_count"]).sum()
        / data["review_count"].sum()
    )
    c3.metric("Average Rating", f"{overall_rating:.2f} / 5")

    st.divider()

    min_reviews = st.slider(
        "Minimum Reviews per Product",
        min_value=1,
        max_value=100,
        value=10
    )

    filtered = data[
        data["review_count"] >= min_reviews
    ].copy()

    top = filtered.nlargest(15, "review_count")

    fig = px.bar(
        top,
        x="review_count",
        y="ProductId",
        orientation="h",
        color="average_rating",
        color_continuous_scale="Viridis",
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
        orientation="h",
        color="negative_pct",
        color_continuous_scale="Reds",
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
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Product metrics use all years. "
        "Negative percentages can be unstable "
        "for products with small review counts."
    )

# ============================================================
# PAGE 3 - MACHINE LEARNING INSIGHTS
# ============================================================

elif page == "Machine Learning Insights":

    st.title("🧠 Machine Learning Insights")
    st.caption("Spark MLlib vs Scikit-learn")

    st.info(
        "BAB V comparison experiment: 14,697 reviews "
        "(11,727 train / 2,970 test). "
        "Separate from the main Spark MLlib experiment."
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
    c1.metric("Best Accuracy", "80.24%", "Spark MLlib")
    c2.metric("Best Macro F1", "59.42%", "Spark MLlib")
    c3.metric("Fastest Training", "3.76 s", "Scikit-learn")

    st.divider()

    metric_cols = [
        "Accuracy", "Macro Precision", "Macro Recall",
        "Macro F1", "Weighted F1"
    ]

    selected = st.multiselect(
        "Evaluation Metrics",
        metric_cols,
        default=["Accuracy", "Macro F1", "Weighted F1"]
    )

    if selected:
        plot_data = metrics.melt(
            id_vars="Model",
            value_vars=selected,
            var_name="Metric",
            value_name="Score"
        )

        fig = px.bar(
            plot_data,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            text_auto=".2f",
            title="Model Performance Comparison"
        )
        fig.update_yaxes(range=[0, 105])
        st.plotly_chart(fig, use_container_width=True)

    time_data = metrics.melt(
        id_vars="Model",
        value_vars=[
            "Training Time (s)",
            "Prediction Time (s)"
        ],
        var_name="Operation",
        value_name="Seconds"
    )

    fig = px.bar(
        time_data,
        x="Operation",
        y="Seconds",
        color="Model",
        barmode="group",
        text_auto=".2f",
        title="Computational Performance"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed Model Evaluation")
    st.dataframe(
        metrics, use_container_width=True, hide_index=True
    )

    st.subheader("Main Spark MLlib Experiment")
    a, b = st.columns(2)
    a.metric("Main Model Accuracy", "81.84%")
    b.metric("Main Model Macro F1", "67.63%")

    st.caption(
        "Main experiment: 568,453 cleaned observations, "
        "113,576 test observations."
    )

    st.markdown("""
    **Key Findings**

    - Spark MLlib achieved slightly higher accuracy and
      macro F1 in the comparison experiment.
    - Scikit-learn required less training and prediction time.
    - Neutral sentiment remains challenging to classify.
    - Results are specific to the tested configurations.
    """)

# ============================================================
# PAGE 4 - GRAPH ANALYTICS
# ============================================================

elif page == "Graph Analytics":

    st.title("🕸️ Graph Analytics")
    st.caption("Product–Review–Sentiment Network")

    graph_df = load_graph_data()

    if graph_df is not None:
        st.success(
            "Using graph_review_sample.csv from BAB VI."
        )
        graph_source = graph_df.copy()
        product_col = "product_id"
        review_col = "review_id"
        sentiment_col = "sentiment"
    else:
        st.warning(
            "BAB VI graph sample is not uploaded yet. "
            "Showing an exploratory graph using the "
            "5,000-review dashboard sample instead. "
            "This is NOT the original BAB VI network."
        )
        graph_source = reviews.copy()
        product_col = "ProductId"
        review_col = "Id"
        sentiment_col = "Sentiment"

    graph_source[sentiment_col] = (
        graph_source[sentiment_col]
        .astype(str).str.strip().str.title()
    )

    product_counts = (
        graph_source[product_col]
        .value_counts()
    )

    available_products = product_counts.index.tolist()

    max_products = min(10, len(available_products))

    if max_products == 0:
        st.warning("No graph observations available.")
    else:
        n_products = st.slider(
            "Number of Products to Display",
            1, max_products, min(5, max_products)
        )

        n_reviews = st.slider(
            "Maximum Reviews per Product",
            1, 10, 5
        )

        selected_products = available_products[:n_products]

        graph_subset = (
            graph_source[
                graph_source[product_col].isin(
                    selected_products
                )
            ]
            .groupby(product_col, group_keys=False)
            .head(n_reviews)
            .reset_index(drop=True)
        )

        G = nx.Graph()

        for idx, row in graph_subset.iterrows():
            product = "P:" + str(row[product_col])
            review = "R:" + str(idx)
            sentiment = "S:" + str(row[sentiment_col])

            G.add_node(product, node_type="Product")
            G.add_node(review, node_type="Review")
            G.add_node(
                sentiment,
                node_type="Sentiment",
                sentiment=row[sentiment_col]
            )

            G.add_edge(product, review)
            G.add_edge(review, sentiment)

        c1, c2, c3 = st.columns(3)
        c1.metric("Nodes", G.number_of_nodes())
        c2.metric("Edges", G.number_of_edges())
        c3.metric(
            "Connected Components",
            nx.number_connected_components(G)
        )

        pos = nx.spring_layout(
            G, seed=42, iterations=60
        )

        edge_x, edge_y = [], []

        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=0.7, color="#64748B"),
            hoverinfo="skip",
            showlegend=False
        ))

        node_colors = []
        node_sizes = []
        node_labels = []
        hover_labels = []

        for node, attrs in G.nodes(data=True):
            node_type = attrs["node_type"]

            if node_type == "Product":
                color = "#A78BFA"
                size = 19
                label = node[2:]
            elif node_type == "Review":
                color = "#38BDF8"
                size = 7
                label = ""
            else:
                sentiment = attrs["sentiment"]
                color = COLORS.get(sentiment, "#94A3B8")
                size = 30
                label = sentiment

            node_colors.append(color)
            node_sizes.append(size)
            node_labels.append(label)
            hover_labels.append(
                f"{node_type}: {node[2:]}"
            )

        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in G.nodes()],
            y=[pos[n][1] for n in G.nodes()],
            mode="markers+text",
            text=node_labels,
            textposition="top center",
            hovertext=hover_labels,
            hoverinfo="text",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1, color="#FFFFFF")
            ),
            showlegend=False
        ))

        fig.update_layout(
            title="Interactive Product–Review–Sentiment Network",
            height=650,
            template="plotly_dark",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=10, r=10, t=60, b=10)
        )

        st.plotly_chart(
            fig, use_container_width=True
        )

        st.caption(
            "Purple = Product, Blue = Review, "
            "Green = Positive, Yellow = Neutral, "
            "Red = Negative. "
            "Nodes support hover, zoom and pan."
        )

        st.subheader("Sentiment in Displayed Graph")

        graph_sentiment = (
            graph_subset[sentiment_col]
            .value_counts()
            .rename_axis("Sentiment")
            .reset_index(name="Reviews")
        )

        fig = px.bar(
            graph_sentiment,
            x="Sentiment",
            y="Reviews",
            color="Sentiment",
            color_discrete_map=COLORS
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Sentiment–Keyword Network")

    keyword_df = load_keyword_data()

    if keyword_df is None:
        st.info(
            "Upload top_words_by_sentiment.csv from BAB VI "
            "to activate the actual sentiment–keyword graph."
        )
    else:
        st.write("Keyword dataset preview:")
        st.dataframe(
            keyword_df.head(20),
            use_container_width=True,
            hide_index=True
        )

        # Column names may differ between notebooks.
        # Only draw when the expected columns are present.
        columns_lower = {
            str(c).lower().strip(): c
            for c in keyword_df.columns
        }

        sentiment_key = next(
            (columns_lower[k] for k in
             ["sentiment", "label", "category"]
             if k in columns_lower),
            None
        )
        word_key = next(
            (columns_lower[k] for k in
             ["word", "keyword", "term"]
             if k in columns_lower),
            None
        )
        count_key = next(
            (columns_lower[k] for k in
             ["count", "frequency", "freq"]
             if k in columns_lower),
            None
        )

        if sentiment_key and word_key:
            words = keyword_df.copy()
            words[sentiment_key] = (
                words[sentiment_key]
                .astype(str).str.title()
            )

            if count_key:
                words[count_key] = pd.to_numeric(
                    words[count_key], errors="coerce"
                ).fillna(0)
                words = (
                    words.sort_values(
                        count_key, ascending=False
                    )
                    .groupby(sentiment_key)
                    .head(9)
                )
            else:
                words = words.groupby(
                    sentiment_key
                ).head(9)

            KG = nx.Graph()

            for _, row in words.iterrows():
                sentiment = str(row[sentiment_key])
                word = str(row[word_key])
                KG.add_node(
                    "S:" + sentiment,
                    node_type="Sentiment"
                )
                KG.add_node(
                    "W:" + word,
                    node_type="Keyword"
                )
                KG.add_edge(
                    "S:" + sentiment,
                    "W:" + word
                )

            kpos = nx.spring_layout(
                KG, seed=42, iterations=100
            )

            fig_k = go.Figure()

            ex, ey = [], []
            for u, v in KG.edges():
                ex += [kpos[u][0], kpos[v][0], None]
                ey += [kpos[u][1], kpos[v][1], None]

            fig_k.add_trace(go.Scatter(
                x=ex, y=ey,
                mode="lines",
                line=dict(color="#64748B", width=1),
                hoverinfo="skip",
                showlegend=False
            ))

            fig_k.add_trace(go.Scatter(
                x=[kpos[n][0] for n in KG.nodes()],
                y=[kpos[n][1] for n in KG.nodes()],
                mode="markers+text",
                text=[n[2:] for n in KG.nodes()],
                textposition="top center",
                marker=dict(
                    size=[
                        28 if n.startswith("S:") else 13
                        for n in KG.nodes()
                    ],
                    color=[
                        COLORS.get(n[2:], "#38BDF8")
                        if n.startswith("S:")
                        else "#A78BFA"
                        for n in KG.nodes()
                    ]
                ),
                hoverinfo="text",
                showlegend=False
            ))

            fig_k.update_layout(
                title="Sentiment–Keyword Network",
                height=600,
                template="plotly_dark",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )

            st.plotly_chart(
                fig_k, use_container_width=True
            )

            st.caption(
                "Edges indicate keyword association "
                "with sentiment categories. This view "
                "does not represent causal relationships."
            )
        else:
            st.warning(
                "Keyword file needs a sentiment/label "
                "column and a word/keyword column."
            )

# ============================================================
# PAGE 5 - REVIEW EXPLORER
# ============================================================

elif page == "Review Explorer":

    st.title("🔎 Review Explorer")
    st.caption("Search and Explore Customer Reviews")

    st.info(
        "This page uses a reproducible sample of "
        "up to 5,000 reviews, not the full dataset."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        selected_sentiments = st.multiselect(
            "Sentiment",
            ["Positive", "Neutral", "Negative"],
            default=["Positive", "Neutral", "Negative"]
        )

    with c2:
        selected_ratings = st.multiselect(
            "Rating",
            [1, 2, 3, 4, 5],
            default=[1, 2, 3, 4, 5]
        )

    with c3:
        search_text = st.text_input(
            "Search Review Text",
            placeholder="e.g. coffee, taste, delicious"
        )

    filtered_reviews = reviews[
        reviews["Sentiment"].isin(selected_sentiments)
        & reviews["Score"].isin(selected_ratings)
    ].copy()

    if search_text.strip():
        filtered_reviews = filtered_reviews[
            filtered_reviews["Text"].str.contains(
                search_text.strip(),
                case=False,
                na=False,
                regex=False
            )
        ]

    st.metric(
        "Matching Reviews",
        f"{len(filtered_reviews):,}"
    )

    col1, col2 = st.columns(2)

    with col1:
        sentiment_counts = (
            filtered_reviews["Sentiment"]
            .value_counts()
            .rename_axis("Sentiment")
            .reset_index(name="Count")
        )

        if not sentiment_counts.empty:
            fig = px.pie(
                sentiment_counts,
                names="Sentiment",
                values="Count",
                hole=0.5,
                color="Sentiment",
                color_discrete_map=COLORS,
                title="Filtered Sentiment Distribution"
            )
            st.plotly_chart(
                fig, use_container_width=True
            )

    with col2:
        rating_counts = (
            filtered_reviews["Score"]
            .value_counts()
            .sort_index()
            .rename_axis("Rating")
            .reset_index(name="Count")
        )

        if not rating_counts.empty:
            fig = px.bar(
                rating_counts,
                x="Rating",
                y="Count",
                title="Filtered Rating Distribution"
            )
            st.plotly_chart(
                fig, use_container_width=True
            )

    st.subheader("Customer Review Records")

    display_cols = [
        "Id", "ProductId", "Score",
        "Sentiment", "Year", "Text"
    ]

    st.dataframe(
        filtered_reviews[display_cols],
        use_container_width=True,
        hide_index=True,
        height=450
    )

    csv = filtered_reviews[
        display_cols
    ].to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Filtered Reviews",
        data=csv,
        file_name="filtered_amazon_reviews.csv",
        mime="text/csv"
    )

    st.caption(
        "Review text is truncated to 300 characters "
        "during data preparation. "
        "The exported file contains only filtered "
        "records from the review sample."
    )

# ---------------- FOOTER ----------------

st.divider()
st.caption(
    "Amazon Review Intelligence | "
    "Big Data Analytics | "
    "Amazon Fine Food Reviews | Version 3.0"
)
