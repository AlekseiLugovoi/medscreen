# app/main.py

import streamlit as st
from pages import show_preview_page, show_batch_page, show_about_page, show_api_page

st.set_page_config(page_title="MedScreen", page_icon="🩺", layout="wide")

# Инициализация session_state
if "slice_idx" not in st.session_state:
    st.session_state.slice_idx = 0

# --- Сайдбар с radio buttons ---
with st.sidebar:
    st.title("🩺 MedScreen")
    
    page = st.radio(
        "Режим работы",
        ["О проекте", "Превью исследования", "Пакетная обработка", "API-интерфейс"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    st.markdown("""
        **Полезные ссылки:**
        - [Репозиторий проекта](https://github.com/AlekseiLugovoi/medscreen)
        - [Демо данные](https://disk.yandex.ru/d/2ddI6aLMkoIYrA)
        - [Презентация](https://disk.yandex.ru/d/LpKu44Kq0Xa_0w)
    """)

# --- Отрисовка выбранной страницы ---
if page == "О проекте":
    show_about_page()
elif page == "Превью исследования":
    show_preview_page()
elif page == "Пакетная обработка":
    show_batch_page()
elif page == "API-интерфейс":
    show_api_page()
