import streamlit as st
import pandas as pd
from utils import (
    process_uploaded_file, validate_series, 
    run_pathology_inference, model, create_gif_from_frames, 
    precompute_uint8_frames
)

# Окна визуализации для КТ (Center, Width)
CT_WINDOWS = {
    "Легочное (Lung)": (-600, 1500),
    "Мягкотканное (Soft Tissue)": (40, 400),
    "Костное (Bone)": (400, 1800),
}

def reset_session_state():
    """Сбрасывает состояние сессии при загрузке нового файла."""
    st.session_state.clear()

def show_preview_page():
    st.title("🔬 Превью исследования")

    # Создаем две колонки для управления и для будущих элементов
    col1, col2 = st.columns([1, 1])

    with col1:
        # --- ЭТАП 1: Загрузка файла ---
        st.subheader("Шаг 1: Загрузите файл")
        uploaded_file = st.file_uploader(
            "Загрузите файл исследования (.zip, .dcm, .nii, .nii.gz)",
            type=["zip", "dcm", "nii", "gz"],
            on_change=reset_session_state,
            label_visibility="collapsed"
        )

        if not uploaded_file:
            return

        # --- ЭТАП 2: Обработка данных ---
        st.subheader("Шаг 2: Обработайте данные")
        if 'processed_data' not in st.session_state:
            if st.button("Обработать файл", type="primary"):
                # Используем st.status для отображения прогресса
                with st.status("Идет обработка...", expanded=True) as status:
                    data = process_uploaded_file(uploaded_file, status)
                    if data:
                        st.session_state.processed_data = data
                        # Сохраняем статус для отображения в expander
                        st.session_state.processing_status = "complete"
                        status.update(label="Обработка завершена!", state="complete", expanded=False)
                    else:
                        st.session_state.processing_status = "error"
                        status.update(label="Ошибка обработки!", state="error", expanded=True)
                st.rerun()
            return
        
        # Показываем свернутый лог после обработки
        if st.session_state.get("processing_status") == "complete":
            with st.expander("Лог обработки", expanded=False):
                st.success("Обработка успешно завершена.")
        elif st.session_state.get("processing_status") == "error":
            with st.expander("Лог обработки", expanded=True):
                st.error("В процессе обработки произошла ошибка.")


        # --- ЭТАП 3: Проверка и визуализация ---
        st.subheader("Шаг 3: Проверьте данные и запустите анализ")
        
        series_uids = list(st.session_state.processed_data.keys())
        selected_uid = st.selectbox(
            "Выберите серию для анализа:", series_uids, key='series_uid',
            format_func=lambda uid: f"Серия ...{uid[-12:]}"
        ) if len(series_uids) > 1 else series_uids[0]
        if not selected_uid: selected_uid = series_uids[0]
        
        series = st.session_state.processed_data[selected_uid]
        meta = series["meta"]

        # Чек-лист и метаданные в скрытых блоках
        with st.expander("Чек-лист валидации"):
            validation_results = validate_series(meta)
            validation_results[3]['status'] = len(series_uids) == 1
            all_valid = all(check['status'] for check in validation_results)
            for check in validation_results:
                st.caption(f"{'✅' if check['status'] else '❌'} {check['text']}")
            if all_valid: st.success("Все проверки пройдены!") 
            else: st.warning("Найдены несоответствия.")

        with st.expander("Метаданные"):
            st.json(meta)

        # Управление визуализацией
        if meta.get("Modality") == "CT":
            st.subheader("Окно визуализации")
            with st.container(border=True):
                viz_cols = st.columns([3, 1])
                with viz_cols[0]:
                    st.selectbox("Окно:", options=list(CT_WINDOWS.keys()), key='window_selection_temp', label_visibility="collapsed")
                with viz_cols[1]:
                    if st.button("Показать", type="primary", use_container_width=True):
                        st.session_state.show_visualization = True
                        st.session_state.active_window_name = st.session_state.window_selection_temp
        else:
            if 'show_visualization' not in st.session_state:
                st.session_state.show_visualization = True
                st.session_state.active_window_name = "NIfTI"

    # Блок визуализации (вне колонок, чтобы был на всю ширину)
    if st.session_state.get('show_visualization'):
        num_frames = meta['num_frames']
        if 'slice_idx' not in st.session_state: st.session_state.slice_idx = 0
        active_window = st.session_state.get('active_window_name', list(CT_WINDOWS.keys())[0])

        uint8_frames = precompute_uint8_frames(selected_uid, active_window, CT_WINDOWS)
        gif_bytes = create_gif_from_frames(selected_uid, active_window, CT_WINDOWS)

        vis_col1, vis_col2, vis_col3 = st.columns(3)
        with vis_col1:
            st.subheader("Анимация")
            st.image(gif_bytes, use_container_width=True)
        with vis_col2:
            st.subheader("Предпросмотр среза")
            st.image(uint8_frames[st.session_state.slice_idx], use_container_width=True)
            st.slider("Срез", 0, num_frames - 1, key='slice_idx', label_visibility="collapsed")
            st.caption(f"Показан: {st.session_state.slice_idx + 1}/{num_frames}")
        with vis_col3:
            st.subheader("Найденные патологии")
            if 'pathology_indices' not in st.session_state:
                if st.button("Найти патологии", use_container_width=True):
                    predictions = run_pathology_inference(model, series["frames"])
                    st.session_state.pathology_indices = [i for i, pred in enumerate(predictions) if pred]
                    st.rerun()
                st.info("Нажмите кнопку для запуска ML-анализа.")
            elif not st.session_state.pathology_indices:
                st.success("Патологий не найдено.", icon="✅")
            else:
                pathology_indices = st.session_state.pathology_indices
                st.error(f"Найдено на {len(pathology_indices)} срезах:", icon="⚠️")
                slice_numbers_str = ", ".join([str(i + 1) for i in pathology_indices])
                st.markdown("**Номера срезов:**")
                st.text(slice_numbers_str)

def show_batch_page():
    st.title("📦 Пакетная обработка")

    # Создаем две колонки, чтобы ограничить ширину виджетов
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_files = st.file_uploader(
            "Загрузите файлы (.zip, .dcm, .nii, .nii.gz)",
            type=["zip", "dcm", "nii", "gz"], accept_multiple_files=True,
            label_visibility="collapsed"
        )

        if not uploaded_files: return

        # Создаем колонки для кнопок
        button_col1, button_col2 = st.columns(2)

        with button_col1:
            if st.button("Обработать и сформировать CSV", type="primary", use_container_width=True):
                csv_data = []
                fieldnames = [
                    'archive_name', 'series_uid', 'source_format', 'modality', 
                    'is_valid', 'orientation', 'missing_slices', 'num_frames',
                    'has_pathology', 'pathology_slice_count'
                ]
                
                progress_bar = st.progress(0, "Начало обработки...")
                for i, file in enumerate(uploaded_files):
                    progress_text = f"Анализ файла {i+1}/{len(uploaded_files)}: {file.name}..."
                    progress_bar.progress((i) / len(uploaded_files), text=progress_text)
                    
                    class DummyStatus:
                        def write(self, *args, **kwargs): pass
                    
                    dummy_status = DummyStatus()
                    
                    from utils import _process_dicom_zip, _process_multiframe_dicom, _process_nifti_file
                    filename = file.name.lower()
                    series_data = {}
                    if filename.endswith(".zip"): series_data = _process_dicom_zip(file, dummy_status)
                    elif filename.endswith(".dcm"): series_data = _process_multiframe_dicom(file, dummy_status)
                    else: series_data = _process_nifti_file(file, dummy_status)

                    if not series_data:
                        csv_data.append({'archive_name': file.name, 'is_valid': False})
                        continue
                    
                    for series_uid, data in series_data.items():
                        meta = data['meta']
                        validation_checks = validate_series(meta)
                        validation_checks[3]['status'] = len(series_data) == 1
                        is_valid = all(check['status'] for check in validation_checks)

                        predictions = run_pathology_inference(model, data['frames'])
                        pathology_count = sum(predictions)

                        csv_data.append({
                            'archive_name': file.name, 'series_uid': series_uid,
                            'source_format': meta.get('SourceFormat', 'N/A'),
                            'modality': meta.get('Modality', 'N/A'),
                            'is_valid': is_valid,
                            'orientation': meta.get('orientation', 'N/A'),
                            'missing_slices': meta.get('missing_slices', 'N/A'),
                            'num_frames': meta.get('num_frames', 'N/A'),
                            'has_pathology': pathology_count > 0,
                            'pathology_slice_count': pathology_count
                        })
                
                progress_bar.progress(1.0, text="Обработка завершена!")
                st.session_state.result_df = pd.DataFrame(csv_data, columns=fieldnames).fillna("N/A")
                st.rerun()

        # Кнопка скачивания появляется во второй колонке, если есть результат
        if 'result_df' in st.session_state:
            with button_col2:
                csv_string = st.session_state.result_df.to_csv(index=False).encode('utf-8')
                st.download_button("Скачать CSV", csv_string, file_name="batch_report.csv", mime="text/csv", use_container_width=True)

    # Таблица с результатами отображается под колонками на всю ширину
    if 'result_df' in st.session_state:
        st.divider()
        st.subheader("Результаты обработки")
        st.dataframe(st.session_state.result_df)

def show_about_page():
    st.title("ℹ️ О проекте")
    st.markdown("""
        **MedScreen** — это интерактивный инструмент для предобработки и анализа медицинских изображений.
        
        ### Ключевые технологии:
        - **Streamlit:** Создание интерактивного веб-интерфейса.
        - **Pydicom, Nibabel:** Чтение и парсинг DICOM и NIfTI файлов.
        - **Numpy:** Обработка изображений и манипуляции с пиксельными данными.
        - **Pandas:** Работа с табличными данными и генерация CSV.
        - **Imageio:** Создание анимированных GIF.
    """)
    st.warning("> ⚠️ **Внимание:** Прототип не предназначен для клинического применения.")