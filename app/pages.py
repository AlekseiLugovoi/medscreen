import streamlit as st
import pandas as pd

from app.file_io import parse_zip_archive
from app.data_validation import validate_series
from app.visualization import prepare_frames_for_display, create_gif
from app.ml_processing import run_inference, run_inference_batch, check_api_health

# CT visualization windows (Center, Width)
CT_WINDOWS = {
    "Lung": (-600, 1500),
    "Soft Tissue": (40, 400),
    "Bone": (700, 1500),
}


def reset_session_state():
    """Reset session state when a new file is uploaded."""
    st.session_state.clear()


def show_about_page():
    st.title("About")
    st.markdown("""
        **MedScreen** is a chest CT pathology screening tool powered by
        [MedGemma](https://ai.google.dev/gemma/docs/medgemma) vision-language model.

        The application provides several modes of operation:

        - **Study Preview:** Interactive analysis of a single CT study.
          Upload a ZIP archive, inspect slices, metadata, and run ML screening.
        - **Batch Processing:** Automatic processing of multiple studies
          with a downloadable CSV report.
        - **API:** REST endpoint for integration into automated pipelines.

        ### Key Technologies
        - **Streamlit** — interactive web interface
        - **vLLM + MedGemma** — ML inference with structured output
        - **FastAPI** — REST API backend
        - **Pydicom, Nibabel** — DICOM and NIfTI parsing
    """)


def show_preview_page():
    st.title("Study Preview")

    col1, _ = st.columns([1, 1])

    with col1:
        # --- Step 1: Upload ---
        st.subheader("Step 1: Upload ZIP Archive")

        uploaded_file = st.file_uploader(
            "Upload a CT study in a ZIP archive",
            type=["zip"],
            on_change=reset_session_state,
            label_visibility="collapsed",
        )

        with st.expander("Supported formats"):
            st.markdown("""
            Upload a ZIP with one study. Format is detected automatically:
            - **DICOM** — `.dcm` files from one series (or one multi-frame `.dcm`)
            - **NIfTI** — single `.nii` / `.nii.gz` file
            - **Images** — `.png` / `.jpg` files (treated as one series)
            """)

        if not uploaded_file:
            return

        if "file_content" not in st.session_state:
            st.session_state.file_content = uploaded_file.getvalue()

        if "processed_data" not in st.session_state:
            with st.spinner("Processing and preparing visualization..."):
                data, error_message = parse_zip_archive(st.session_state.file_content)
                if error_message:
                    st.error(f"Error reading archive: {error_message}")
                    del st.session_state.file_content
                    return
                st.session_state.processed_data = data

                # Precompute frames and GIF for instant display
                uid = list(data.keys())[0]
                sdata = data[uid]
                default_window = "Lung" if sdata["meta"].get("Modality") == "CT" else "Default"
                frames = prepare_frames_for_display(sdata, uid, default_window, CT_WINDOWS)
                st.session_state.display_frames = frames
                st.session_state.gif_bytes = create_gif(frames, uid, default_window)
                st.session_state.active_window_name = default_window
                st.session_state.show_visualization = True
                st.rerun()

        # --- Step 2: Validation ---
        st.subheader("Step 2: Data Validation")

        series_uid = list(st.session_state.processed_data.keys())[0]
        series_data = st.session_state.processed_data[series_uid]
        meta = series_data["meta"]

        with st.expander("Validation checklist"):
            validation_results = validate_series(meta)
            all_valid = all(check["status"] for check in validation_results)
            for check in validation_results:
                icon = "✅" if check["status"] else "❌"
                st.caption(f"{icon} {check['check']}: {check['message']}")
            if all_valid:
                st.success("All checks passed!")
            else:
                st.warning("Some checks failed. Results may be inaccurate.")

        with st.expander("Metadata"):
            st.json({k: str(v) for k, v in meta.items()})

        # --- Step 3: Visualization & Screening ---
        st.subheader("Step 3: Visualization & Screening")

        viz_container = st.container(border=True)
        with viz_container:
            viz_cols = st.columns([3, 1])
            with viz_cols[0]:
                if meta.get("Modality") == "CT":
                    st.selectbox(
                        "Visualization window:",
                        options=list(CT_WINDOWS.keys()),
                        key="window_name_temp",
                        label_visibility="collapsed",
                    )
                else:
                    st.session_state.window_name_temp = "Default"
                    st.text_input(
                        "Visualization window:", "Default",
                        disabled=True, label_visibility="collapsed",
                    )

            with viz_cols[1]:
                if st.button("Show", type="primary", use_container_width=True):
                    new_window = st.session_state.window_name_temp
                    if new_window != st.session_state.get("active_window_name"):
                        frames = prepare_frames_for_display(
                            series_data, series_uid, new_window, CT_WINDOWS)
                        st.session_state.display_frames = frames
                        st.session_state.gif_bytes = create_gif(
                            frames, series_uid, new_window)
                        st.session_state.active_window_name = new_window
                    st.session_state.show_visualization = True
                    st.rerun()

    # --- Visualization block (reads precomputed data from session_state) ---
    if st.session_state.get("show_visualization"):
        display_frames = st.session_state.display_frames
        gif_bytes = st.session_state.gif_bytes
        num_frames = len(display_frames)

        if "slice_idx" not in st.session_state:
            st.session_state.slice_idx = num_frames // 2

        vis_col1, vis_col2, vis_col3 = st.columns(3)

        with vis_col1:
            st.subheader("Animation")
            st.image(gif_bytes, use_container_width=True)

        with vis_col2:
            st.subheader("Slice Preview")
            st.image(display_frames[st.session_state.slice_idx], use_container_width=True)
            st.slider("Slice", 0, num_frames - 1, key="slice_idx", label_visibility="collapsed")
            st.caption(f"Slice: {st.session_state.slice_idx + 1} / {num_frames}")

        with vis_col3:
            st.subheader("Pathology Screening")

            if not check_api_health():
                st.error("API backend is not available. Make sure it is running.")
                return

            if "screening_results" not in st.session_state:
                if st.button("Run Screening", type="primary", use_container_width=True):
                    result = run_inference(
                        st.session_state.file_content,
                        filename=uploaded_file.name,
                    )
                    if result is None:
                        st.error("Inference failed. Make sure the ZIP contains a valid CT study (.nii.gz, DICOM, or images).")
                    else:
                        st.session_state.screening_results = result
                        st.rerun()
                st.info("Click the button to run ML screening.")
            else:
                _render_screening_results(st.session_state.screening_results)


def _render_screening_results(result: dict):
    """Render screening results."""
    verdict = result.get("verdict")

    if not result.get("is_valid"):
        st.warning("Study did not pass validation.")
        return

    if verdict == "NORMAL":
        st.success("No pathology detected ✅")
    else:
        st.error("Pathology signs detected ❌")

    # Details
    window_verdicts = result.get("window_verdicts", [])
    n_abn = window_verdicts.count("ABNORMAL")
    nw = len(window_verdicts)

    st.caption(
        f"Analyzed: {result.get('slices_processed', '?')}/{result.get('total_slices', '?')} slices "
        f"({result.get('coverage_pct', 0):.0f}%)"
    )
    st.caption(f"Inference time: {result.get('inference_time', '?')}")

    if n_abn > 0:
        window_reasonings = result.get("window_reasonings", [])
        window_slices = result.get("window_slices", [])
        with st.expander(f"Details: {n_abn}/{nw} windows with pathology", expanded=True):
            for v, r, slices in zip(window_verdicts, window_reasonings, window_slices):
                if v == "ABNORMAL":
                    st.caption(f"Slices {slices[0]}-{slices[1]}: {r}")


def show_batch_page():
    st.title("Batch Processing")

    col1, _ = st.columns([1, 1])

    with col1:
        uploaded_files = st.file_uploader(
            "Upload one or more ZIP archives",
            type=["zip"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        with st.expander("Supported formats"):
            st.markdown("""
            Upload one or more ZIP archives. Format is detected automatically:
            - **DICOM** — `.dcm` files from one series (or one multi-frame `.dcm`)
            - **NIfTI** — one or more `.nii` / `.nii.gz` files (each screened separately)
            - **Images** — `.png` / `.jpg` files (treated as one series)
            """)

        if not uploaded_files:
            return

        if not check_api_health():
            st.error("API backend is not available. Make sure it is running.")
            return

        button_col1, button_col2 = st.columns(2)

        with button_col1:
            if st.button("Process & Generate CSV", type="primary", use_container_width=True):
                csv_data = []
                progress_bar = st.progress(0, "Starting...")

                for i, file in enumerate(uploaded_files):
                    progress_bar.progress(
                        i / len(uploaded_files),
                        text=f"Processing {i+1}/{len(uploaded_files)}: {file.name}...",
                    )

                    results = run_inference_batch(file.getvalue(), filename=file.name)

                    if results:
                        for result in results:
                            csv_data.append({
                                "archive_name": result.get("archive_name", file.name),
                                "series_uid": result.get("series_uid", "N/A"),
                                "source_format": result.get("source_format", "N/A"),
                                "modality": result.get("modality", "N/A"),
                                "body_part": result.get("body_part", "N/A"),
                                "num_frames": result.get("num_frames", 0),
                                "is_valid": result.get("is_valid", False),
                                "verdict": result.get("verdict", "N/A"),
                                "abnormal_ratio": f"{result.get('abnormal_ratio', 0):.0%}" if result.get("abnormal_ratio") is not None else "N/A",
                                "inference_time": result.get("inference_time", "N/A"),
                            })
                    else:
                        csv_data.append({
                            "archive_name": file.name,
                            "is_valid": False,
                            "verdict": "ERROR",
                        })

                progress_bar.progress(1.0, text="Done!")
                st.session_state.result_df = pd.DataFrame(csv_data).fillna("N/A")
                st.rerun()

        if "result_df" in st.session_state:
            with button_col2:
                csv_string = st.session_state.result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    csv_string,
                    file_name="batch_report.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    # Show results below controls (outside col1 to use full width)
    if "result_df" in st.session_state:
        st.divider()
        st.subheader("Results")
        st.dataframe(st.session_state.result_df)


def show_api_page():
    st.title("API")

    api_available = check_api_health()
    if api_available:
        st.success("API backend is running.")
    else:
        st.error("API backend is not available.")

    st.markdown("""
    The service provides a REST API for automated CT screening.
    This allows integrating MedScreen into automated pipelines.
    """)

    st.subheader("Usage Example")

    st.code("""
curl -X POST http://localhost:8502/process \\
     -F "files=@normal_dicom_chest.zip" \\
     -F "files=@abnormal_nifti_covid_ct4.zip"
    """, language="bash")

    st.subheader("Documentation")
    st.markdown(
        "Interactive API docs (Swagger) are available at "
        "[http://localhost:8502/docs](http://localhost:8502/docs)."
    )
