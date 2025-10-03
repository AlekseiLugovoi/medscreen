import streamlit as st
import pandas as pd
import time
from file_io import parse_zip_archive
from data_validation import validate_series
from visualization import prepare_frames_for_display, create_gif
from ml_processing import get_model, run_pathology_inference

# Окна визуализации для КТ (Center, Width)
CT_WINDOWS = {
    "Легочное (Lung)": (-600, 1500),
    "Мягкотканное (Soft Tissue)": (40, 400),
    "Костное (Bone)": (700, 1500),
}

def reset_session_state():
    """Сбрасывает состояние сессии при загрузке нового файла."""
    st.session_state.clear()


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
            on_change=reset_session_state,
            label_visibility="collapsed"
        )

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
            if 'pathology_results' not in st.session_state:
                if st.button("Найти патологии", type="primary", use_container_width=True):
                    model = get_model()
                    results = run_pathology_inference(model, series_data["frames"])
                    
                    # --- ИСПРАВЛЕНИЕ: Добавляем ту же логику, что и в пакетной обработке ---
                    preds = results.get('preds', [])
                    raw_probs = results.get('raw_probs', [])
                    st.session_state.has_pathology_flag = any(preds)
                    st.session_state.max_prob = max(raw_probs) if raw_probs else 0.0
                    # --- Конец исправления ---

                    st.session_state.pathology_results = results
                    st.rerun()
                st.info("Нажмите кнопку для запуска ML-анализа.")
            
            else:
                # --- ИСПРАВЛЕНИЕ: Используем сохраненные флаги ---
                has_pathology = st.session_state.get('has_pathology_flag', False)
                max_prob = st.session_state.get('max_prob', 0.0)

                if not has_pathology:
                    st.success("Патологий не найдено.", icon="✅")
                    st.caption(f"Максимальная вероятность: {max_prob:.4f}")
                else:
                    pathology_indices = [i for i, pred in enumerate(st.session_state.pathology_results['preds']) if pred]
                    st.error(f"Найдено на {len(pathology_indices)} срезах:", icon="⚠️")
                    st.caption(f"Максимальная вероятность: {max_prob:.4f}")
                    slice_numbers_str = ", ".join([str(i + 1) for i in pathology_indices])
                    st.markdown("**Номера срезов:**")
                    st.text_area("Срезы с патологией", value=slice_numbers_str, height=100, disabled=True)


def show_batch_page():
    st.title("📦 Пакетная обработка")

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
        button_col1, button_col2 = st.columns(2)

        with button_col1:
            if st.button("Обработать и сформировать CSV", type="primary", use_container_width=True):
                csv_data = []
                # --- ИСПРАВЛЕНИЕ: Упрощаем колонки времени ---
                fieldnames = [
                    'archive_name', 'series_uid', 'source_format', 'modality',
                    'body_part', 'orientation', 'num_frames',
                    'is_valid', 'has_pathology', 'pred_pathology', 
                    'processing_time' # <-- Оставляем только одну колонку времени
                ]
                
                model = get_model()
                progress_bar = st.progress(0, "Начало обработки...")
                
                for i, file in enumerate(uploaded_files):
                    progress_text = f"Анализ файла {i+1}/{len(uploaded_files)}: {file.name}..."
                    progress_bar.progress(i / len(uploaded_files), text=progress_text)
                    
                    series_data, error_message = parse_zip_archive(file)

                    if not series_data or error_message:
                        csv_data.append({'archive_name': file.name, 'is_valid': False, 'series_uid': error_message or "Parsing error"})
                        continue
                    
                    for series_uid, data in series_data.items():
                        meta = data['meta']
                        validation_checks = validate_series(meta)
                        is_valid = all(check['status'] for check in validation_checks)

                        if is_valid and len(data['frames']) > 0:
                            inference_results = run_pathology_inference(model, data['frames'])
                            has_pathology_flag = inference_results.get('study_has_pathology', False)
                            final_prob = inference_results.get('study_prob_pathology', 0.0)
                            ml_time = inference_results.get('study_processing_time', 0.0)
                        else:
                            has_pathology_flag = False
                            final_prob = 0.0
                            ml_time = 0.0

                        csv_data.append({
                            'archive_name': file.name,
                            'series_uid': series_uid,
                            'source_format': meta.get('SourceFormat', 'N/A'),
                            'modality': meta.get('Modality', 'N/A'),
                            'body_part': meta.get('BodyPartExamined', 'N/A'),
                            'orientation': meta.get('orientation', 'N/A'),
                            'num_frames': meta.get('num_frames', 'N/A'),
                            'is_valid': is_valid,
                            'has_pathology': has_pathology_flag,
                            'pred_pathology': f"{final_prob:.4f}",
                            'ml_processing_time': f"{ml_time:.2f}s"
                        })
                
                progress_bar.progress(1.0, text="Обработка завершена!")
                st.session_state.result_df = pd.DataFrame(csv_data, columns=fieldnames).fillna("N/A")
                st.rerun()

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

    if 'result_df' in st.session_state:
        st.divider()
        st.subheader("Результаты обработки")
        st.dataframe(st.session_state.result_df)


def show_api_page():
    st.title("🤖 API-интерфейс")
    st.info("Для локального запуска API доступен по адресу `http://localhost:8502`.")

    st.markdown("""
    Сервис предоставляет REST API для автоматической пакетной обработки исследований.
    Это позволяет интегрировать `medscreen` в автоматизированные пайплайны.
    """)

    st.subheader("Пример использования")
    st.markdown("""
    Вы можете отправить один или несколько ZIP-архивов на эндпоинт `/api/v1/upload`. 
    Сервис обработает их и вернет **CSV-файл** с результатами, аналогичный тому, что формируется в режиме "Пакетная обработка".
    """)
    
    st.code("""
# Пример отправки двух архивов и сохранения результата в report.csv
curl -X POST "http://localhost:8502/api/v1/upload" \\
     -H "Content-Type: multipart/form-data" \\
     -F "files=@/путь/к/вашему/study1.zip" \\
     -F "files=@/путь/к/вашему/study2.zip" \\
     --output report.csv
    """, language="bash")

    st.subheader("Документация")
    st.markdown("Полная интерактивная документация API (Swagger) с возможностью отправки тестовых запросов доступна по адресу [http://localhost:8502/docs](http://localhost:8502/docs).")
