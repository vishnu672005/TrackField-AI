import os
import streamlit as st
import full_analysis


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Track & Field AI",
    page_icon="🏃",
    layout="wide"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    text-align: center;
    padding: 2rem 1rem 1.5rem 1rem;
    margin-bottom: 1.5rem;
    background: #171a22;
    border-radius: 14px;
}

.hero-title {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}

.hero-subtitle {
    font-size: 1.25rem;
    color: #b8c0cc;
    margin-bottom: 0.8rem;
}

.hero-description {
    font-size: 1rem;
    color: #8f98a6;
    max-width: 750px;
    margin: auto;
    line-height: 1.6;
}

.upload-title {
    font-size: 1.4rem;
    font-weight: 600;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HERO / HEADER
# ============================================================

st.markdown("""
<div class="hero">
<div class="hero-title">🏃 Track &amp; Field AI</div>

<div class="hero-subtitle">
AI-Powered Sports Performance &amp; Biomechanical Analysis
</div>

<div class="hero-description">
Upload a Track &amp; Field video and let AI identify the
sporting event, analyse movement patterns, and highlight
potentially injury-prone biomechanical indicators.
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# VIDEO UPLOAD
# ============================================================

st.markdown(
    '<div class="upload-title">🎥 Upload Your Video</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a Track & Field video",
    type=["mp4", "avi", "mov", "mkv"]
)


if uploaded_file is not None:

    # --------------------------------------------------------
    # Video preview
    # --------------------------------------------------------

    left, center, right = st.columns([1, 2, 1])

    with center:
        st.video(uploaded_file)

    st.write("")

    # --------------------------------------------------------
    # Analyze button
    # --------------------------------------------------------

    if st.button("🔍 Analyze Video"):

        st.info(
            "Analysis started. This may take a few minutes "
            "because the model is running on CPU."
        )

        # ----------------------------------------------------
        # Save uploaded video
        # ----------------------------------------------------

        os.makedirs("uploads", exist_ok=True)

        video_path = os.path.join(
            "uploads",
            uploaded_file.name
        )

        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # ----------------------------------------------------
        # Run complete analysis
        # ----------------------------------------------------

        results = full_analysis.predict_and_analyse(
            video_path
        )

        # ====================================================
        # ANALYSIS SUMMARY
        # ====================================================

        st.subheader("Analysis Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Detected Event",
                results["event"]
            )

        with col2:
            st.metric(
                "Confidence",
                f'{results["confidence"]:.1f}%'
            )

        with col3:
            st.metric(
                "Risk Score",
                f'{results["risk_score"]}/100'
            )

        with col4:
            st.metric(
                "Risk Flags",
                len(results["findings"])
            )

        st.success(
            "Analysis completed successfully! 🎉"
        )

        # ====================================================
        # BIOMECHANICAL RISK INDICATORS
        # ====================================================

        if results["findings"]:

            st.subheader(
                "⚠️ Biomechanical Risk Indicators"
            )

            for finding in results["findings"]:

                st.warning(
                    f'**{finding["severity"]} — '
                    f'{finding["body_part"]}**: '
                    f'{finding["description"]}'
                )

        # ====================================================
        # ANALYSIS RESULTS
        # ====================================================

        st.header("📊 Analysis Results")

        event_report = (
            "outputs/event_classification.png"
        )

        injury_report = (
            "outputs/injury_risk_report.png"
        )

        # ----------------------------------------------------
        # Result tabs
        # ----------------------------------------------------

        tab1, tab2 = st.tabs([
            "📈 Event Classification",
            "🦵 Biomechanical Risk Analysis"
        ])

        # ----------------------------------------------------
        # Event Classification
        # ----------------------------------------------------

        with tab1:

            if os.path.exists(event_report):

                st.image(
                    event_report,
                    width=900
                )

            else:

                st.warning(
                    "Event classification report was not found."
                )

        # ----------------------------------------------------
        # Biomechanical Risk Analysis
        # ----------------------------------------------------

        with tab2:

            if os.path.exists(injury_report):

                st.image(
                    injury_report,
                    width=900
                )

            else:

                st.warning(
                    "Biomechanical risk report was not found."
                )

        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.info(
            "The results are biomechanical indicators "
            "and are not a medical diagnosis."
        )