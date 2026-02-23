import streamlit as st
from app.pages import show_preview_page, show_batch_page, show_about_page, show_api_page

st.set_page_config(page_title="MedScreen", page_icon="🩺", layout="wide")

if "slice_idx" not in st.session_state:
    st.session_state.slice_idx = 0

with st.sidebar:
    st.title("🩺 MedScreen")

    page = st.radio(
        "Navigation",
        ["About", "Study Preview", "Batch Processing", "API"],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("""
        **Links:**
        - [GitHub](https://github.com/AlekseiLugovoi/medscreen)
        - [Demo Data](https://drive.google.com/drive/folders/1ChmkPR-5OwZB8Ub9h23VuHoOiA2hX-gx?usp=sharing)
    """)

if page == "About":
    show_about_page()
elif page == "Study Preview":
    show_preview_page()
elif page == "Batch Processing":
    show_batch_page()
elif page == "API":
    show_api_page()
