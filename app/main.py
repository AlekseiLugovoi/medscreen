import io
from PIL import Image, ImageDraw
import streamlit as st

st.set_page_config(page_title="MedScreen", page_icon="🩺", layout="wide")

# --- Утилиты ---
PREVIEW_W, PREVIEW_H = 448, 448

def letterbox(img: Image.Image, w=PREVIEW_W, h=PREVIEW_H, fill=(20, 20, 20)):
    """Вписывает изображение в рамку с сохранением пропорций."""
    iw, ih = img.size
    s = min(w / iw, h / ih)
    nw, nh = int(iw * s), int(ih * s)
    resized_img = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), fill)
    canvas.paste(resized_img, ((w - nw) // 2, (h - nh) // 2))
    return canvas

def mock_detect(img: Image.Image):
    """Заглушка: рисует рамку на изображении."""
    out = letterbox(img).copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    box = (int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8))
    d.rectangle(box, outline=(255, 64, 64), width=6)
    return out, "Патология", 0.73

# --- Инициализация состояния ---
if "files" not in st.session_state:
    st.session_state.files = []
    st.session_state.idx = 0
    st.session_state.result_img = None
    st.session_state.result_label = None
    st.session_state.result_score = None

# --- Функции-колбэки ---
def clear_results():
    """Сбрасывает результаты анализа."""
    st.session_state.result_img = None
    st.session_state.result_label = None
    st.session_state.result_score = None

def handle_upload():
    st.session_state.files = st.session_state.uploader
    st.session_state.idx = 0
    clear_results()

def clear_all():
    st.session_state.files = []
    st.session_state.idx = 0
    clear_results()

def prev_image():
    st.session_state.idx -= 1
    clear_results()

def next_image():
    st.session_state.idx += 1
    clear_results()

def analyze_image():
    img_bytes = st.session_state.files[st.session_state.idx].getvalue()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    det_img, label, score = mock_detect(img)
    st.session_state.result_img = det_img
    st.session_state.result_label = label
    st.session_state.result_score = score

# --- Логика страниц ---
def show_check_page():
    """Отрисовывает главную страницу 'Проверка'."""
    has_files = len(st.session_state.files) > 0
    col1, col2 = st.columns(2, gap="large")

    with col1:
        subheader_text = "Предпросмотр"
        if has_files:
            subheader_text += f" ({st.session_state.idx + 1} / {len(st.session_state.files)})"
        st.subheader(subheader_text)

        with st.container(border=True):
            if has_files:
                current_file = st.session_state.files[st.session_state.idx]
                img = Image.open(current_file).convert("RGB")
                st.image(letterbox(img), caption=current_file.name)
            else:
                st.info("Загрузите изображения в панели слева.")
        
        if has_files:
            btn_cols = st.columns([1, 1, 3])
            btn_cols[0].button("◀ Пред.", on_click=prev_image, disabled=(st.session_state.idx == 0), use_container_width=True)
            btn_cols[1].button("След. ▶", on_click=next_image, disabled=(st.session_state.idx >= len(st.session_state.files) - 1), use_container_width=True)
            btn_cols[2].button("Анализировать", type="primary", on_click=analyze_image, use_container_width=True)

    with col2:
        st.subheader("Результат анализа")
        with st.container(border=True):
            if st.session_state.result_img:
                cap = f"{st.session_state.result_label} (уверенность: {st.session_state.result_score:.2f})"
                st.image(st.session_state.result_img, caption=cap)
            else:
                st.info("Здесь появится результат после нажатия кнопки «Анализировать».")

def show_about_page():
    """Отрисовывает страницу 'О проекте'."""
    st.subheader("О проекте MedScreen")
    st.markdown("""
        **MedScreen** — это прототип для автоматической сортировки медицинских изображений.
        - **Текущая версия** реализует UI, пакетную загрузку и предпросмотр.
        - **Детекция** является заглушкой и имитирует результат работы модели.
    """)
    st.warning("> ⚠️ **Внимание:** Прототип не предназначен для клинического применения.")

# --- Основной код приложения ---
with st.sidebar:
    st.title("🩺 MedScreen")
    page = st.radio("Меню", ["Проверка", "О проекте"], label_visibility="collapsed")
    st.divider()

    if page == "Проверка":
        st.file_uploader(
            "Загрузите PNG/JPG",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="uploader",
            on_change=handle_upload
        )
        st.button("Очистить всё", on_click=clear_all, disabled=not st.session_state.files)

if page == "Проверка":
    show_check_page()
elif page == "О проекте":
    show_about_page()