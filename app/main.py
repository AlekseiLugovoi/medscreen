# app/main.py

import streamlit as st
from pages import show_preview_page, show_batch_page, show_about_page

st.set_page_config(page_title="MedScreen", page_icon="🩺", layout="wide")

# Инициализация session_state
if "slice_idx" not in st.session_state:
    st.session_state.slice_idx = 0

# --- Сайдбар с radio buttons ---
with st.sidebar:
    st.title("🩺 MedScreen")
    
    page = st.radio(
        "Режим работы",
        ["Превью файла", "Пакетная обработка", "О проекте"],
        label_visibility="collapsed"
    )
    
    st.divider()
    # ИСПРАВЛЕНО: Убрана версия
    st.markdown("""
        🔗Полезные ссылки:
        - [Репозиторий проекта](https://github.com/AlekseiLugovoi/medscreen)
        - Демо модели
        - Веса модели
        - Презентация
    """)

# --- Отрисовка выбранной страницы ---
if page == "Превью файла":
    show_preview_page()
elif page == "Пакетная обработка":
    show_batch_page()
elif page == "О проекте":
    show_about_page()