import streamlit as st
import pandas as pd
<<<<<<< HEAD
from utils import (
    process_uploaded_file, validate_series, 
    run_pathology_inference, model, create_gif_from_frames, 
    precompute_uint8_frames
)
=======
from file_io import parse_zip_archive
from data_validation import validate_series
from visualization import prepare_frames_for_display, create_gif
from ml_processing import get_model, run_pathology_inference
>>>>>>> 80cab49 (Refactoring)

# Окна визуализации для КТ (Center, Width)
CT_WINDOWS = {
    "Легочное (Lung)": (-600, 1500),
    "Мягкотканное (Soft Tissue)": (40, 400),
<<<<<<< HEAD
    "Костное (Bone)": (400, 1800),
=======
    "Костное (Bone)": (700, 1500),
>>>>>>> 80cab49 (Refactoring)
}

def reset_session_state():
    """Сбрасывает состояние сессии при загрузке нового файла."""
    st.session_state.clear()

<<<<<<< HEAD
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
=======

def show_about_page():
    st.title("ℹ️ О проекте")
    st.markdown("""
        **MedScreen** — это интерактивный инструмент для предобработки и анализа медицинских изображений. Приложение имеет несколько режимов работы:
        
        - **Превью исследования:** Интерактивный анализ одного исследования. Позволяет детально изучить срезы, метаданные и результаты ML-модели.
        - **Пакетная обработка:** Автоматическая обработка нескольких архивов с исследованиями и формирование сводного CSV-отчета.
        - **API-интерфейс:** Демонстрация того, как можно было бы интегрировать сервис в автоматизированные пайплайны.

        ### Ключевые технологии:
        - **Streamlit:** Создание интерактивного веб-интерфейса.
        - **Pydicom, Nibabel, Pillow:** Чтение и парсинг форматов DICOM, NIfTI и изображений.
        - **Numpy:** Обработка изображений и манипуляции с пиксельными данными.
        - **Imageio:** Создание анимированных GIF.
        - **PyTorch & Transformers:** Работа ML-модели.
    """)
    st.warning("> ⚠️ **Внимание:** Прототип не предназначен для клинического применения.")


def show_preview_page():
    st.title("🔬 Превью исследования")

    # --- Левая колонка для управления ---
    col1, _ = st.columns([1, 1])

    with col1:
        # --- ЭТАП 1: Загрузка файла ---
        st.subheader("Шаг 1: Загрузите ZIP-архив")
        
        uploaded_file = st.file_uploader(
            "Загрузите исследование в ZIP-архиве",
            type=["zip"],
>>>>>>> 80cab49 (Refactoring)
            on_change=reset_session_state,
            label_visibility="collapsed"
        )

<<<<<<< HEAD
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
=======
        with st.expander("Требования к содержимому архива"):
            st.markdown("""
            Архив должен содержать **одно исследование** в одном из следующих форматов:
            - **Серия DICOM:** множество файлов (часто с расширением `.dcm` или без него)
            - **Многокадровый DICOM:** один `.dcm` файл, содержащий все срезы
            - **NIfTI:** один файл (`.nii` или `.nii.gz`)
            - **Серия изображений:** множество файлов (`.png`, `.jpg`)
            """)

        if not uploaded_file:
            return

        if 'processed_data' not in st.session_state:
            with st.spinner("Идет обработка..."):
                data, error_message = parse_zip_archive(uploaded_file)
                if error_message:
                    st.error(f"Ошибка чтения архива: {error_message}")
                    return
                st.session_state.processed_data = data
                st.rerun()

        # --- ЭТАП 2: Валидация данных ---
        st.subheader("Шаг 2: Валидация данных")
        
        # Берем первую найденную серию
        series_uid = list(st.session_state.processed_data.keys())[0]
        series_data = st.session_state.processed_data[series_uid]
        meta = series_data["meta"]

        with st.expander("Чек-лист валидации"):
            validation_results = validate_series(meta)
            all_valid = all(check['status'] for check in validation_results)
            for check in validation_results:
                st.caption(f"{'✅' if check['status'] else '❌'} {check['check']}: {check['message']}")
            if all_valid: st.success("Все проверки пройдены!")
            else: st.warning("Найдены несоответствия. Результаты могут быть неточными.")

        with st.expander("Метаданные"):
            st.json({k: str(v) for k, v in meta.items()})

        # --- ЭТАП 3: Визуализация и анализ ---
        st.subheader("Шаг 3: Визуализация и анализ")
        
        viz_container = st.container(border=True)
        with viz_container:
            viz_cols = st.columns([3, 1])
            with viz_cols[0]:
                if meta.get("Modality") == "CT":
                    st.selectbox("Окно визуализации:", options=list(CT_WINDOWS.keys()), key='window_name_temp', label_visibility="collapsed")
                else:
                    st.session_state.window_name_temp = "Default"
                    st.text_input("Окно визуализации:", "Default", disabled=True, label_visibility="collapsed")
            
            with viz_cols[1]:
                if st.button("Показать", type="primary", use_container_width=True):
                    st.session_state.show_visualization = True
                    st.session_state.active_window_name = st.session_state.window_name_temp


    # --- Блок визуализации (отображается после нажатия кнопки) ---
    if st.session_state.get('show_visualization'):
        active_window = st.session_state.get('active_window_name')
        
        display_frames = prepare_frames_for_display(series_data, active_window, CT_WINDOWS)
        gif_bytes = create_gif(display_frames)
        num_frames = len(display_frames)

        if 'slice_idx' not in st.session_state:
            st.session_state.slice_idx = num_frames // 2

        vis_col1, vis_col2, vis_col3 = st.columns(3)

        with vis_col1:
            st.subheader("Анимация")
            st.image(gif_bytes, use_container_width=True)

        with vis_col2:
            st.subheader("Предпросмотр среза")
            st.image(display_frames[st.session_state.slice_idx], use_container_width=True)
            st.slider("Срез", 0, num_frames - 1, key='slice_idx', label_visibility="collapsed")
            st.caption(f"Показан срез: {st.session_state.slice_idx + 1} / {num_frames}")

        with vis_col3:
            st.subheader("Найденные патологии")
            if 'pathology_indices' not in st.session_state:
                if st.button("Найти патологии", type="primary", use_container_width=True):
                    model = get_model()
                    predictions = run_pathology_inference(model, series_data["frames"])
>>>>>>> 80cab49 (Refactoring)
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
<<<<<<< HEAD
                st.text(slice_numbers_str)
=======
                st.text_area("Срезы с патологией", value=slice_numbers_str, height=100, disabled=True)

>>>>>>> 80cab49 (Refactoring)

def show_batch_page():
    st.title("📦 Пакетная обработка")

<<<<<<< HEAD
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
=======
    # Ограничиваем ширину виджетов
    col1, _ = st.columns([1, 1])

    with col1:
        uploaded_files = st.file_uploader(
            "Загрузите один или несколько ZIP-архивов",
            type=["zip"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        with st.expander("Требования к содержимому архивов"):
            st.markdown("""
            Можно загрузить **один или несколько** ZIP-архивов. 
            
            Каждый архив должен содержать **одно исследование** в одном из следующих форматов:
            - **Серия DICOM:** множество файлов (часто с расширением `.dcm` или без него).
            - **Многокадровый DICOM:** один `.dcm` файл, содержащий все срезы.
            - **NIfTI:** один файл (`.nii` или `.nii.gz`).
            - **Серия изображений:** множество файлов (`.png`, `.jpg`).
            """)

        if not uploaded_files:
            return

        # Колонки для кнопок
>>>>>>> 80cab49 (Refactoring)
        button_col1, button_col2 = st.columns(2)

        with button_col1:
            if st.button("Обработать и сформировать CSV", type="primary", use_container_width=True):
                csv_data = []
                fieldnames = [
<<<<<<< HEAD
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
=======
                    'archive_name', 'series_uid', 'source_format', 'modality',
                    'is_valid', 'body_part', 'orientation', 'num_frames',
                    'has_pathology', 'pathology_slice_count'
                ]
                
                model = get_model()
                progress_bar = st.progress(0, "Начало обработки...")
                
                for i, file in enumerate(uploaded_files):
                    progress_text = f"Анализ файла {i+1}/{len(uploaded_files)}: {file.name}..."
                    progress_bar.progress(i / len(uploaded_files), text=progress_text)
                    
                    series_data, error_message = parse_zip_archive(file)

                    if not series_data or error_message:
                        csv_data.append({'archive_name': file.name, 'is_valid': False, 'series_uid': error_message or "Parsing error"})
>>>>>>> 80cab49 (Refactoring)
                        continue
                    
                    for series_uid, data in series_data.items():
                        meta = data['meta']
                        validation_checks = validate_series(meta)
<<<<<<< HEAD
                        validation_checks[3]['status'] = len(series_data) == 1
=======
>>>>>>> 80cab49 (Refactoring)
                        is_valid = all(check['status'] for check in validation_checks)

                        predictions = run_pathology_inference(model, data['frames'])
                        pathology_count = sum(predictions)

                        csv_data.append({
<<<<<<< HEAD
                            'archive_name': file.name, 'series_uid': series_uid,
                            'source_format': meta.get('SourceFormat', 'N/A'),
                            'modality': meta.get('Modality', 'N/A'),
                            'is_valid': is_valid,
                            'orientation': meta.get('orientation', 'N/A'),
                            'missing_slices': meta.get('missing_slices', 'N/A'),
=======
                            'archive_name': file.name,
                            'series_uid': series_uid,
                            'source_format': meta.get('SourceFormat', 'N/A'),
                            'modality': meta.get('Modality', 'N/A'),
                            'is_valid': is_valid,
                            'body_part': meta.get('BodyPartExamined', 'N/A'),
                            'orientation': meta.get('orientation', 'N/A'),
>>>>>>> 80cab49 (Refactoring)
                            'num_frames': meta.get('num_frames', 'N/A'),
                            'has_pathology': pathology_count > 0,
                            'pathology_slice_count': pathology_count
                        })
                
                progress_bar.progress(1.0, text="Обработка завершена!")
                st.session_state.result_df = pd.DataFrame(csv_data, columns=fieldnames).fillna("N/A")
                st.rerun()

<<<<<<< HEAD
        # Кнопка скачивания появляется во второй колонке, если есть результат
        if 'result_df' in st.session_state:
            with button_col2:
                csv_string = st.session_state.result_df.to_csv(index=False).encode('utf-8')
                st.download_button("Скачать CSV", csv_string, file_name="batch_report.csv", mime="text/csv", use_container_width=True)

    # Таблица с результатами отображается под колонками на всю ширину
=======
        if 'result_df' in st.session_state:
            with button_col2:
                csv_string = st.session_state.result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Скачать CSV",
                    csv_string,
                    file_name="batch_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )

>>>>>>> 80cab49 (Refactoring)
    if 'result_df' in st.session_state:
        st.divider()
        st.subheader("Результаты обработки")
        st.dataframe(st.session_state.result_df)

<<<<<<< HEAD
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
=======

def show_api_page():
    st.title("🤖 API-интерфейс [В разработке]")
    st.info("Эта страница — демонстрация API. Для локального запуска используйте `localhost:8000`.")

    st.markdown("""
    Сервис предоставляет REST API для автоматической загрузки и обработки исследований.
    Взаимодействие выглядит следующим образом:
    """)

    st.subheader("Загрузка архива на обработку")
    st.markdown("Отправка POST-запроса с ZIP-архивом. Сервис обрабатывает его и возвращает результат в формате JSON.")
    st.code("""
# curl -X POST "http://localhost:8000/api/v1/upload" \\
#      -H "Content-Type: multipart/form-data" \\
#      -F "file=@/path/to/your/study.zip"
    """, language="bash")

    st.markdown("Полная документация API с возможностью отправки тестовых запросов доступна по адресу [http://localhost:8000/docs](http://localhost:8000/docs).")
>>>>>>> 80cab49 (Refactoring)
