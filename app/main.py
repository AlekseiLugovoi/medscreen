# app/main.py

import streamlit as st
<<<<<<< HEAD
from pages import show_preview_page, show_batch_page, show_about_page
=======
from pages import show_preview_page, show_batch_page, show_about_page, show_api_page
>>>>>>> 80cab49 (Refactoring)

st.set_page_config(page_title="MedScreen", page_icon="🩺", layout="wide")

# Инициализация session_state
if "slice_idx" not in st.session_state:
    st.session_state.slice_idx = 0

# --- Сайдбар с radio buttons ---
with st.sidebar:
    st.title("🩺 MedScreen")
    
    page = st.radio(
        "Режим работы",
<<<<<<< HEAD
        ["Превью исследования", "Пакетная обработка", "О проекте"],
=======
        ["О проекте", "Превью исследования", "Пакетная обработка", "API-интерфейс"],
>>>>>>> 80cab49 (Refactoring)
        label_visibility="collapsed"
    )
    
    st.divider()
    
    st.markdown("""
        **Полезные ссылки:**
        - [Репозиторий проекта](https://github.com/AlekseiLugovoi/medscreen)
        - [Демо данные](https://disk.yandex.ru/d/2ddI6aLMkoIYrA)
<<<<<<< HEAD
        - [Презентация](https://disk.yandex.ru/d/2ddI6aLMkoIYrA)
    """)

# --- Отрисовка выбранной страницы ---
if page == "Превью исследования":
    show_preview_page()
elif page == "Пакетная обработка":
    show_batch_page()
elif page == "О проекте":
    show_about_page()
=======
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
>>>>>>> 80cab49 (Refactoring)
