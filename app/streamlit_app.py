"""Streamlit demo for the Media Metadata Intelligence System."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metadata_intelligence.api.dependencies import (  # noqa: E402
    get_search_resources,
    get_tagging_resources,
)
from metadata_intelligence.demo import (  # noqa: E402
    artifact_status,
    convert_predicted_tags_to_dataframe,
    convert_similar_content_to_dataframe,
    default_threshold,
    generate_demo_report,
)


st.set_page_config(
    page_title="Media Metadata Intelligence System",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading tag prediction artifacts...")
def load_tagging_resources_cached():
    return get_tagging_resources()


@st.cache_resource(show_spinner="Loading semantic search artifacts...")
def load_search_resources_cached():
    return get_search_resources()


def main() -> None:
    st.title("Media Metadata Intelligence System")
    st.write(
        "Predict multi-label story tags from a movie title and synopsis, then "
        "retrieve semantically similar movies from the MPST dataset."
    )

    status = artifact_status()
    with st.sidebar:
        st.header("System Info")
        st.write(f"Tagging model available: {status['tagging_model_available']}")
        st.write(f"Search index available: {status['search_index_available']}")
        st.caption(
            "If artifacts are missing, run `python scripts/train_baseline.py` "
            "and `python scripts/build_search_index.py`."
        )

    with st.form("intelligence_form"):
        title = st.text_input("Movie title", value="Example Film")
        synopsis = st.text_area(
            "Synopsis",
            value=(
                "A detective investigates a violent murder while uncovering "
                "a dark family secret."
            ),
            height=180,
        )
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            threshold = st.slider(
                "Tag threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(default_threshold()),
                step=0.01,
            )
        with col_b:
            top_k_similar = st.slider(
                "Top-k similar movies",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
            )
        with col_c:
            limit_tags = st.checkbox("Limit number of tags", value=False)
            top_k_tags = st.slider(
                "Top-k tags",
                min_value=1,
                max_value=20,
                value=10,
                step=1,
                disabled=not limit_tags,
            )

        submitted = st.form_submit_button("Generate Intelligence Report")

    if not submitted:
        st.info("Enter a title and synopsis, then generate an intelligence report.")
        return

    if not title.strip() or not synopsis.strip():
        st.error("Title and synopsis are required.")
        return

    try:
        tagging_resources = load_tagging_resources_cached()
        search_resources = load_search_resources_cached()
    except RuntimeError as error:
        st.error(str(error))
        return

    report = generate_demo_report(
        title=title,
        synopsis=synopsis,
        threshold=threshold,
        top_k_similar=top_k_similar,
        tagging_resources=tagging_resources,
        search_resources=search_resources,
        top_k_tags=top_k_tags if limit_tags else None,
    )

    st.subheader("Predicted Tags")
    tags_df = convert_predicted_tags_to_dataframe(report["predicted_tags"])
    if tags_df.empty:
        st.warning("No tags passed the current threshold. Try lowering the threshold.")
    else:
        st.dataframe(
            tags_df.style.format({"confidence": "{:.1%}"}),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Similar Content")
    similar_df = convert_similar_content_to_dataframe(report["similar_content"])
    st.dataframe(
        similar_df.style.format({"similarity_score": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Structured Metadata JSON")
    st.json(report)

    st.subheader("System Info")
    system_info = {
        **report["metadata"],
        **artifact_status(),
    }
    st.json(system_info)


if __name__ == "__main__":
    main()
