# app/pages.py

import streamlit as st
import pydicom
import pandas as pd
import io
from utils import get_meta, to_uint8, normalize_dicom_and_get_frames

# --- Функции-колбэки ---
def on_file_upload():
    st.session_state.slice_idx = 0
    if 'result_image' in st.session_state: del st.session_state.result_image

def prev_slice():
    if st.session_state.slice_idx > 0: st.session_state.slice_idx -= 1

def next_slice():
    if st.session_state.slice_idx < len(st.session_state.frames) - 1: st.session_state.slice_idx += 1

def analyze_mock():
    st.session_state.result_image = st.session_state.frames[st.session_state.slice_idx]

# --- Функции отрисовки страниц ---

def show_preview_page():
    st.title("🔬 Предпросмотр DICOM файла")
    col_upload, _ = st.columns(2);
    with col_upload:
        uploaded_file = st.file_uploader("Загрузите DICOM файл", type=None, on_change=on_file_upload)
        if uploaded_file:
            with st.expander("Показать метаданные DICOM"):
                try: ds = pydicom.dcmread(io.BytesIO(uploaded_file.getvalue()), force=True); st.json(get_meta(ds))
                except Exception: st.error("Не удалось извлечь метаданные.")
    
    if uploaded_file:
        try:
            st.divider(); ds = pydicom.dcmread(io.BytesIO(uploaded_file.getvalue()), force=True); ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian; st.session_state.frames = normalize_dicom_and_get_frames(ds)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader(f"Предпросмотр ({st.session_state.slice_idx + 1}/{len(st.session_state.frames)})")
                with st.container(border=True):
                    st.image(to_uint8(st.session_state.frames[st.session_state.slice_idx]), caption=uploaded_file.name, use_container_width=True)
                if len(st.session_state.frames) > 1:
                    btn_cols = st.columns([1, 3, 1]); btn_cols[0].button("◀", on_click=prev_slice, use_container_width=True); btn_cols[2].button("▶", on_click=next_slice, use_container_width=True); btn_cols[1].slider("Срез", 0, len(st.session_state.frames) - 1, key="slice_idx", label_visibility="collapsed")
                st.button("Анализировать", type="primary", use_container_width=True, on_click=analyze_mock)
            
            # ИСПРАВЛЕНО: Возвращаем st.container(border=True)
            if 'result_image' in st.session_state:
                with col2:
                    st.subheader("Результат")
                    with st.container(border=True): # <--- ВОТ ЗДЕСЬ
                        st.image(to_uint8(st.session_state.result_image), caption="Результат (заглушка)", use_container_width=True)
                with col3:
                    st.subheader("Заключение")
                    with st.container(border=True): # <--- И ВОТ ЗДЕСЬ
                        st.info("Здесь появится текстовое заключение модели (заглушка).")

        except Exception as e: st.error(f"Не удалось обработать DICOM файл: {e}")

def show_batch_page():
    st.title("📦 Пакетная обработка DICOM")
    
    col1, _ = st.columns(2)
    with col1:
        uploaded_files = st.file_uploader(
            "Загрузите DICOM файлы", type=None, accept_multiple_files=True, key="batch_uploader"
        )
        
        if uploaded_files:
            total_size_mb = sum(f.size for f in uploaded_files) / (1024 * 1024)
            st.info(f"Загружено файлов: {len(uploaded_files)} (Общий объем: {total_size_mb:.2f} МБ)")
            
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("Сформировать CSV", type="primary", use_container_width=True):
                    csv_data = []; fieldnames = ['filename', 'modality', 'pixel_spacing', 'slice_thickness', 'num_frames']
                    progress_bar = st.progress(0, text="Идет обработка...")
                    for i, file in enumerate(uploaded_files):
                        try:
                            ds = pydicom.dcmread(io.BytesIO(file.getvalue()), force=True); frames = normalize_dicom_and_get_frames(ds)
                            csv_data.append({'filename': file.name, 'modality': getattr(ds, "Modality", "N/A"), 'pixel_spacing': str(getattr(ds, "PixelSpacing", "N/A")), 'slice_thickness': str(getattr(ds, "SliceThickness", "N/A")), 'num_frames': len(frames)})
                        except Exception: pass
                        progress_bar.progress((i + 1) / len(uploaded_files), text=f"Обработка {file.name}")
                    progress_bar.empty(); st.session_state.result_df = pd.DataFrame(csv_data, columns=fieldnames)
            
            with btn_col2:
                if 'result_df' in st.session_state:
                    csv_string = st.session_state.result_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Скачать CSV", csv_string, "data_info.csv", "text/csv", use_container_width=True)
            
    if 'result_df' in st.session_state and not st.session_state.result_df.empty:
        st.divider()
        st.dataframe(st.session_state.result_df.style.set_properties(**{'text-align': 'left'}))

def show_about_page():
    st.title("ℹ️ О проекте")
    st.markdown("...") # Ваш текст