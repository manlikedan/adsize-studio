from pathlib import Path
import tempfile
import zipfile

import streamlit as st

from resize_jcb_images import process_zip, targets


APP_NAME = "AdSize Studio"

st.set_page_config(page_title=APP_NAME, page_icon="ADS", layout="centered")


def parse_sizes(raw: str) -> list[tuple[int, int]]:
    parsed = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        cleaned = line.strip().lower().replace(" ", "")
        if not cleaned:
            continue

        if "x" not in cleaned:
            raise ValueError(f"Line {line_number}: use WIDTHxHEIGHT, for example 1200x628.")

        width_raw, height_raw = cleaned.split("x", 1)
        width = int(width_raw)
        height = int(height_raw)

        if width <= 0 or height <= 0:
            raise ValueError(f"Line {line_number}: width and height must be greater than zero.")

        parsed.append((width, height))

    if not parsed:
        raise ValueError("Add at least one output size.")

    return parsed


default_sizes = "\n".join(f"{width}x{height}" for width, height in targets)

st.title(APP_NAME)
st.caption("Resize product images into Google Ads-ready creative sizes from one organised zip.")

uploaded_zip = st.file_uploader("Product image zip", type=["zip"])

with st.sidebar:
    st.header("Google Ads outputs")
    sizes_raw = st.text_area(
        "Sizes",
        value=default_sizes,
        help="One size per line, written as WIDTHxHEIGHT.",
        height=150,
    )
    remove_background = st.toggle("AI remove background", value=False)
    background_color = "#FFFFFF"
    if remove_background:
        background_color = st.color_picker("Replacement background color", value="#FFFFFF")
    background_model = "u2netp"
    output_format = st.selectbox("Format", ["PNG", "JPEG", "WEBP"], index=0)
    jpeg_quality = st.slider("JPEG quality", min_value=60, max_value=100, value=92, disabled=output_format != "JPEG")
    do_trim = st.toggle("Auto trim product margins", value=True)

try:
    selected_sizes = parse_sizes(sizes_raw)
    st.write("Selected sizes:", ", ".join(f"{width}x{height}" for width, height in selected_sizes))
    can_process = uploaded_zip is not None
except ValueError as error:
    selected_sizes = []
    can_process = False
    st.error(str(error))

if uploaded_zip is None:
    st.info("Choose a .zip file to get started.")

if uploaded_zip is not None:
    try:
        with zipfile.ZipFile(uploaded_zip) as source:
            image_count = sum(
                1
                for item in source.infolist()
                if not item.is_dir() and Path(item.filename).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
            )
        st.success(f"Found {image_count} image file{'s' if image_count != 1 else ''} in the zip.")
    except zipfile.BadZipFile:
        st.error("That file does not look like a valid zip archive.")
        can_process = False

if st.button("Build ad assets", type="primary", disabled=not can_process):
    progress = st.progress(0, text="Preparing images...")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / uploaded_zip.name
        input_path.write_bytes(uploaded_zip.getvalue())

        def update_progress(done: int, total: int, filename: str):
            if total:
                progress.progress(done / total, text=f"Processing {done}/{total}: {filename}")

        try:
            output_zip, output_folder, report = process_zip(
                input_path,
                tmp_dir,
                selected_sizes,
                do_trim=do_trim,
                background_color=background_color,
                remove_background=remove_background,
                background_model=background_model,
                output_format=output_format,
                jpeg_quality=jpeg_quality,
                progress_callback=update_progress,
            )
        except Exception as error:
            progress.empty()
            st.exception(error)
        else:
            progress.progress(1.0, text="Done")
            st.download_button(
                "Download Google Ads assets",
                data=output_zip.read_bytes(),
                file_name="google_ads_assets.zip",
                mime="application/zip",
            )

            with st.expander("Processing report"):
                st.text("\n".join(report) if report else "No images were processed.")
