
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pathlib import Path
import re

# ============================================================
# AMAZON REVIEW INTELLIGENCE - VERSION 3.1
# Big Data Analytics Dashboard
# ============================================================

st.set_page_config(
    page_title="Amazon Review Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

COLORS = {
    "Positive": "#22C55E",
    "Neutral": "#F59E0B",
    "Negative": "#EF4444"
}


# ============================================================
# FILE HANDLING
# ============================================================

def find_file(filename):
    """
    Search in repository root, data folder,
    and common duplicate download names.
    """
    target = Path(filename)
    candidates = [
        BASE_DIR / filename,
        BASE_DIR / "data" / filename
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # Allow names such as graph_review_sample(1).csv
    folders = [BASE_DIR, BASE_DIR / "data"]

    for folder in folders:
        if not folder.is_dir():
            continue

        for file in sorted(folder.iterdir()):
            if not file.is_file():
                continue

            pattern = (
                re.escape(target.stem)
                + r"(?:\s*\(\d+\))?"
                + re.escape(target.suffix)
            )

            if re.fullmatch(
                pattern, file.name, flags=re.IGNORECASE
            ):
                return file

    return None


@st.cache_data
def read_csv_file(path):
    return pd.read_csv(path)


def load_required(filename):
    path = find_file(filename)

    if path is None:
        st.error(f"Required file not found: {filename}")
        st.stop()

    return read_csv_file(str(path))


def load_optional(filename):
    path = find_file(filename)

    if path is None:
        return None, None

    try:
        return read_csv_file(str(path)), path
    except Exception as error:
        st.error(
            f"Could not read {path.name}: {error}"
        )
        return None, path


def normalize_sentiment(series):
    mapping = {
        "positive": "Positive",
        "positif": "Positive",
        "negative": "Negative",
        "negatif": "Negative",
        "neutral": "Neutral",
        "netral": "Neutral"
    }

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna("Unknown")
    )


# ============================================================
# LOAD MAIN DATA
# ============================================================

summary = load_required("dashboard_summary.csv")
products = load_required("product_summary.csv")
reviews = load_required("review_sample.csv.gz")

summary["Year"] = pd.to_numeric(
    summary["Year"], errors="coerce"
)

summary["Score"] = pd.to_numeric(
    summary["Score"], errors="coerce"
)

summary["review_count"] = pd.to_numeric(
    summary["review_count"], errors="coerce"
).fillna(0)

summary["Sentiment"] = normalize_sentiment(
    summary["Sentiment"]
)

products["review_count"] = pd.to_numeric(
    products["review_count"], errors="coerce"
).fillna(0)

products["average_rating"] = pd.to_numeric(
    products["average_rating"], errors="coerce"
)

for col in [
    "positive_count", "neutral_count", "negative_count"
]:
    products[col] = pd.to_numeric(
        products[col], errors="coerce"
    ).fillna(0)

reviews["Score"] = pd.to_numeric(
    reviews["Score"], errors="coerce"
)

reviews["Year"] = pd.to_numeric(
    reviews["Year"], errors="coerce"
)

reviews["Text"] = reviews["Text"].fillna("")

reviews["Sentiment"] = normalize_sentiment(
    reviews["Sentiment"]
)


# ============================================================
# SIDEBAR
# ============================================================

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
st.sidebar.caption("Version 3.1 | Graph Fix")


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
        (
            filtered["Score"] *
            filtered["review_count"]
        ).sum() / total
        if total else 0
    )

    positive_pct = (
        positive / total * 100
        if total else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Reviews", f"{total:,}")
    c2.metric(
        "Total Products (All Years)",
        f"{len(products):,}"
    )
    c3.metric("Average Rating", f"{avg_rating:.2f} / 5")
    c4.metric("Positive Reviews", f"{positive_pct:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        sentiment = (
            filtered.groupby(
                "Sentiment", as_index=False
            )["review_count"].sum()
        )

        fig = px.pie(
            sentiment,
            names="Sentiment",
            values="review_count",
            hole=0.55,
            color="Sentiment",
            color_discrete_map=COLORS,
            title="Sentiment Distribution"
        )

        st.plotly_chart(
            fig, use_container_width=True
        )

    with col2:
        rating = (
            filtered.groupby(
                "Score", as_index=False
            )["review_count"].sum()
        )

        fig = px.bar(
            rating,
            x="Score",
            y="review_count",
            title="Rating Distribution",
            text_auto=".2s",
            color_discrete_sequence=["#6366F1"]
        )

        fig.update_xaxes(dtick=1)

        st.plotly_chart(
            fig, use_container_width=True
        )

    yearly = (
        filtered.groupby(
            "Year", as_index=False
        )["review_count"].sum()
        .sort_values("Year")
    )

    fig = px.line(
        yearly,
        x="Year",
        y="review_count",
        markers=True,
        title="Review Trends by Year"
    )

    st.plotly_chart(
        fig, use_container_width=True
    )

    st.info(
        "Sentiment is derived from review ratings: "
        "1–2 Negative, 3 Neutral, 4–5 Positive. "
        "Product count covers all years."
    )


# ============================================================
# PAGE 2 - PRODUCT ANALYTICS
# ============================================================

elif page == "Product Analytics":

    st.title("📦 Product Analytics")
    st.caption("Product Performance & Satisfaction")

    data = products.copy()

    data["negative_pct"] = (
        data["negative_count"] /
        data["review_count"].replace(0, np.nan) * 100
    ).fillna(0)

    total_reviews = int(data["review_count"].sum())

    overall_rating = (
        (
            data["average_rating"] *
            data["review_count"]
        ).sum() / total_reviews
        if total_reviews else 0
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Products", f"{len(data):,}")
    c2.metric("Total Reviews", f"{total_reviews:,}")
    c3.metric(
        "Average Rating",
        f"{overall_rating:.2f} / 5"
    )

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

    top = filtered.nlargest(
        15, "review_count"
    )

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

    st.plotly_chart(
        fig, use_container_width=True
    )

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

    st.plotly_chart(
        fig, use_container_width=True
    )

    st.subheader("Product Performance Table")

    st.dataframe(
        filtered.sort_values(
            "review_count",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Product metrics cover all years. "
        "Use the minimum review filter to avoid "
        "overinterpreting products with few reviews."
    )


# ============================================================
# PAGE 3 - MACHINE LEARNING INSIGHTS
# ============================================================

elif page == "Machine Learning Insights":

    st.title("🧠 Machine Learning Insights")
    st.caption("Spark MLlib vs Scikit-learn")

    st.info(
        "BAB V comparison: 14,697 reviews, "
        "11,727 training and 2,970 testing. "
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

    c1.metric(
        "Best Accuracy", "80.24%",
        "Spark MLlib", delta_color="off"
    )

    c2.metric(
        "Best Macro F1", "59.42%",
        "Spark MLlib", delta_color="off"
    )

    c3.metric(
        "Fastest Training", "3.76 s",
        "Scikit-learn", delta_color="off"
    )

    st.divider()

    metric_cols = [
        "Accuracy", "Macro Precision",
        "Macro Recall", "Macro F1", "Weighted F1"
    ]

    selected = st.multiselect(
        "Evaluation Metrics",
        metric_cols,
        default=[
            "Accuracy", "Macro F1", "Weighted F1"
        ]
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

        st.plotly_chart(
            fig, use_container_width=True
        )

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

    st.plotly_chart(
        fig, use_container_width=True
    )

    st.subheader("Detailed Evaluation")

    st.dataframe(
        metrics,
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Main Spark MLlib Experiment")

    a, b = st.columns(2)

    a.metric("Main Accuracy", "81.84%")
    b.metric("Main Macro F1", "67.63%")

    st.caption(
        "Main experiment: 568,453 cleaned observations, "
        "113,576 test observations."
    )

    st.markdown("""
    **Key Findings**

    - Spark MLlib achieved slightly higher accuracy
      and macro F1 in the comparison experiment.
    - Scikit-learn trained and predicted faster.
    - Neutral sentiment was harder to classify.
    - Results apply to the evaluated configurations.
    """)


# ============================================================
# GRAPH VISUALIZATION HELPER
# ============================================================

def draw_network(
    graph,
    title,
    height=650,
    seed=42
):
    if graph.number_of_nodes() == 0:
        st.warning("No graph nodes available.")
        return

    pos = nx.spring_layout(
        graph,
        seed=seed,
        iterations=80,
        weight=None
    )

    edge_x = []
    edge_y = []

    for u, v in graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(
            color="#64748B",
            width=0.8
        ),
        hoverinfo="skip",
        showlegend=False
    ))

    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_labels = []
    node_hover = []

    for node, attrs in graph.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        node_type = attrs.get(
            "node_type", "Unknown"
        )

        label = attrs.get(
            "label", str(node)
        )

        if node_type == "Product":
            color = "#A78BFA"
            size = 20
        elif node_type == "Review":
            color = "#38BDF8"
            size = 8
        elif node_type == "Sentiment":
            color = COLORS.get(
                label, "#94A3B8"
            )
            size = 30
        else:
            color = "#A78BFA"
            size = 14

        node_colors.append(color)
        node_sizes.append(size)

        node_labels.append(
            "" if node_type == "Review" else label
        )

        node_hover.append(
            f"{node_type}: {label}"
            f"<br>Degree: {graph.degree(node)}"
        )

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_labels,
        textposition="top center",
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(
                width=1,
                color="#FFFFFF"
            )
        ),
        showlegend=False
    ))

    fig.update_layout(
        title=title,
        height=height,
        template="plotly_dark",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(
            l=10, r=10, t=60, b=10
        ),
        dragmode="pan"
    )

    st.plotly_chart(
        fig, use_container_width=True
    )


# ============================================================
# PAGE 4 - GRAPH ANALYTICS
# ============================================================

elif page == "Graph Analytics":

    st.title("🕸️ Graph Analytics")
    st.caption(
        "Network Analysis | BAB VI Integration"
    )

    graph_df, graph_path = load_optional(
        "graph_review_sample.csv"
    )

    keyword_df, keyword_path = load_optional(
        "top_words_by_sentiment.csv"
    )

    # ---------- FILE DIAGNOSTICS ----------

    with st.expander(
        "🔧 Graph File Diagnostics",
        expanded=(
            graph_df is None or keyword_df is None
        )
    ):
        st.write("Application directory:", str(BASE_DIR))

        st.write("Graph review file:")
        st.write(
            str(graph_path)
            if graph_path else "NOT FOUND"
        )

        st.write("Keyword file:")
        st.write(
            str(keyword_path)
            if keyword_path else "NOT FOUND"
        )

        available_files = sorted(
            str(p.relative_to(BASE_DIR))
            for p in BASE_DIR.rglob("*")
            if p.is_file()
            and p.suffix.lower() in [
                ".csv", ".gz"
            ]
        )

        st.write("Available CSV files:")
        st.code(
            "\n".join(available_files)
            if available_files else "No CSV files found."
        )

        if graph_df is not None:
            st.write(
                "Graph columns:",
                graph_df.columns.tolist()
            )

        if keyword_df is not None:
            st.write(
                "Keyword columns:",
                keyword_df.columns.tolist()
            )

    # ---------- PRODUCT REVIEW NETWORK ----------

    st.subheader(
        "Product–Review–Sentiment Network"
    )

    if graph_df is not None:

        st.success(
            f"Loaded BAB VI file: {graph_path.name}"
        )

        source = graph_df.copy()

        source.columns = (
            source.columns
            .astype(str)
            .str.strip()
        )

        aliases = {
            "product_id": [
                "product_id", "ProductId", "productid"
            ],
            "review_id": [
                "review_id", "Id", "reviewid"
            ],
            "sentiment": [
                "sentiment", "Sentiment", "label"
            ]
        }

        found = {}

        for key, options in aliases.items():
            found[key] = next(
                (
                    col for col in source.columns
                    if col.lower() in [
                        x.lower() for x in options
                    ]
                ),
                None
            )

        if all(found.values()):
            source = source.rename(
                columns={
                    found["product_id"]: "product_id",
                    found["review_id"]: "review_id",
                    found["sentiment"]: "sentiment"
                }
            )

            source["sentiment"] = (
                normalize_sentiment(
                    source["sentiment"]
                )
            )

            source = source.dropna(
                subset=["product_id", "review_id"]
            )

            st.caption(
                f"Source: {len(source):,} observations "
                "from the BAB VI graph sample."
            )
        else:
            st.error(
                "The graph CSV is missing required "
                "product, review, or sentiment columns."
            )
            source = pd.DataFrame()

    else:
        st.warning(
            "BAB VI graph file was not found. "
            "Using an exploratory sample instead. "
            "This is NOT the original BAB VI graph."
        )

        source = reviews.rename(
            columns={
                "ProductId": "product_id",
                "Id": "review_id",
                "Sentiment": "sentiment"
            }
        ).copy()

    if not source.empty:

        product_counts = (
            source["product_id"]
            .astype(str)
            .value_counts()
        )

        available_products = (
            product_counts.index.tolist()
        )

        max_products = min(
            20, len(available_products)
        )

        n_products = st.slider(
            "Number of Products",
            min_value=1,
            max_value=max_products,
            value=min(5, max_products)
        )

        n_reviews = st.slider(
            "Maximum Reviews per Product",
            min_value=1,
            max_value=10,
            value=5
        )

        selected_products = (
            available_products[:n_products]
        )

        subset = source[
            source["product_id"]
            .astype(str)
            .isin(selected_products)
        ].copy()

        subset = (
            subset.groupby(
                "product_id",
                group_keys=False
            )
            .head(n_reviews)
        )

        G = nx.Graph()

        for idx, row in subset.iterrows():

            product_id = str(
                row["product_id"]
            )

            review_id = str(
                row["review_id"]
            )

            sentiment = str(
                row["sentiment"]
            )

            p_node = "P:" + product_id
            r_node = "R:" + review_id
            s_node = "S:" + sentiment

            G.add_node(
                p_node,
                node_type="Product",
                label=product_id
            )

            G.add_node(
                r_node,
                node_type="Review",
                label=review_id
            )

            G.add_node(
                s_node,
                node_type="Sentiment",
                label=sentiment
            )

            G.add_edge(p_node, r_node)
            G.add_edge(r_node, s_node)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Nodes", G.number_of_nodes()
        )

        c2.metric(
            "Edges", G.number_of_edges()
        )

        c3.metric(
            "Connected Components",
            nx.number_connected_components(G)
        )

        draw_network(
            G,
            "Interactive Product–Review–Sentiment Network"
        )

        st.caption(
            "Purple = Product, Blue = Review. "
            "Sentiment nodes are green, yellow, or red. "
            "Graph statistics depend on the selected "
            "products and review limits."
        )

        sentiment_counts = (
            subset["sentiment"]
            .value_counts()
            .rename_axis("Sentiment")
            .reset_index(name="Reviews")
        )

        fig = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Reviews",
            color="Sentiment",
            color_discrete_map=COLORS,
            title="Sentiment Distribution in Displayed Network"
        )

        st.plotly_chart(
            fig, use_container_width=True
        )

    # ---------- KEYWORD NETWORK ----------

    st.divider()
    st.subheader("Sentiment–Keyword Network")

    if keyword_df is None:

        st.warning(
            "top_words_by_sentiment.csv was not found. "
            "Check the diagnostics section above."
        )

    else:

        st.success(
            f"Loaded keyword file: {keyword_path.name}"
        )

        words = keyword_df.copy()
        words.columns = (
            words.columns.astype(str).str.strip()
        )

        def match_column(options):
            return next(
                (
                    col for col in words.columns
                    if col.lower() in options
                ),
                None
            )

        sentiment_col = match_column([
            "sentiment", "label", "category"
        ])

        word_col = match_column([
            "word", "keyword", "term", "token"
        ])

        count_col = match_column([
            "count", "frequency", "freq",
            "word_count", "weight"
        ])

        if sentiment_col is None or word_col is None:

            st.error(
                "Keyword file needs sentiment and "
                "word columns. Check diagnostics."
            )

            st.dataframe(
                words.head(10),
                hide_index=True
            )

        else:

            words[sentiment_col] = (
                normalize_sentiment(
                    words[sentiment_col]
                )
            )

            words = words.dropna(
                subset=[word_col]
            )

            words[word_col] = (
                words[word_col]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            if count_col is not None:

                words[count_col] = pd.to_numeric(
                    words[count_col],
                    errors="coerce"
                ).fillna(0)

                words = words.sort_values(
                    count_col,
                    ascending=False
                )

            words = (
                words.groupby(
                    sentiment_col,
                    group_keys=False
                )
                .head(9)
            )

            KG = nx.Graph()

            for _, row in words.iterrows():

                sentiment = str(
                    row[sentiment_col]
                )

                word = str(
                    row[word_col]
                )

                s_node = "S:" + sentiment
                w_node = "W:" + word

                KG.add_node(
                    s_node,
                    node_type="Sentiment",
                    label=sentiment
                )

                KG.add_node(
                    w_node,
                    node_type="Keyword",
                    label=word
                )

                weight = (
                    float(row[count_col])
                    if count_col else 1
                )

                KG.add_edge(
                    s_node,
                    w_node,
                    weight=weight
                )

            k1, k2, k3 = st.columns(3)

            k1.metric(
                "Keyword Graph Nodes",
                KG.number_of_nodes()
            )

            k2.metric(
                "Keyword Graph Edges",
                KG.number_of_edges()
            )

            k3.metric(
                "Unique Keywords",
                sum(
                    1 for _, attrs in KG.nodes(data=True)
                    if attrs["node_type"] == "Keyword"
                )
            )

            draw_network(
                KG,
                "Interactive Sentiment–Keyword Network",
                height=600
            )

            st.caption(
                "Edges represent keyword association "
                "with rating-derived sentiment classes. "
                "Edge weights are stored when a frequency "
                "column exists, but line widths are "
                "not scaled by weight in this view."
            )

            st.subheader("Keyword Frequency Table")

            st.dataframe(
                words,
                hide_index=True,
                use_container_width=True
            )


# ============================================================
# PAGE 5 - REVIEW EXPLORER
# ============================================================

elif page == "Review Explorer":

    st.title("🔎 Review Explorer")
    st.caption("Interactive Customer Review Search")

    st.info(
        "This page uses a reproducible sample "
        "of up to 5,000 reviews, not the full dataset."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        sentiments = st.multiselect(
            "Sentiment",
            ["Positive", "Neutral", "Negative"],
            default=[
                "Positive", "Neutral", "Negative"
            ]
        )

    with c2:
        ratings = st.multiselect(
            "Rating",
            [1, 2, 3, 4, 5],
            default=[1, 2, 3, 4, 5]
        )

    with c3:
        search_text = st.text_input(
            "Search Review Text",
            placeholder="coffee, taste, delicious"
        )

    filtered = reviews[
        reviews["Sentiment"].isin(sentiments)
        & reviews["Score"].isin(ratings)
    ].copy()

    if search_text.strip():

        filtered = filtered[
            filtered["Text"].str.contains(
                search_text.strip(),
                case=False,
                na=False,
                regex=False
            )
        ]

    st.metric(
        "Matching Reviews",
        f"{len(filtered):,}"
    )

    col1, col2 = st.columns(2)

    with col1:

        sentiment_counts = (
            filtered["Sentiment"]
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
            filtered["Score"]
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
        filtered[display_cols],
        use_container_width=True,
        hide_index=True,
        height=450
    )

    csv = (
        filtered[display_cols]
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "⬇️ Download Filtered Reviews",
        data=csv,
        file_name="filtered_amazon_reviews.csv",
        mime="text/csv"
    )

    st.caption(
        "Review text was truncated to 300 characters "
        "during data preparation. Export includes "
        "only records from the dashboard sample."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Amazon Review Intelligence | "
    "Big Data Analytics | Version 3.1"
)
