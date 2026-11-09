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

# SIDEBAR: logo, welcome, then navigation
with st.sidebar:
    logo_path = BASE / "logo.png"
    if logo_path.is_file():
        st.image(str(logo_path), width=185)

    st.markdown("### Amazon Review Intelligence")
    st.caption("Big Data Analytics Dashboard")
    st.markdown("**Welcome to Amazon Review Intelligence**")
    st.caption(
        "Explore customer sentiment, product performance, "
        "machine learning, and graph analytics."
    )
    st.divider()
    page = st.radio("Navigation", list(PAGES.keys()))
    st.divider()
    st.caption("Amazon Fine Food Reviews")
    st.caption("Version 4.1 | Academic Edition")

PAGES[page]()

st.divider()
st.caption(
    "Amazon Review Intelligence | "
    "Big Data Analytics | Version 4.1"
)
