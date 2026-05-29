from pathlib import Path
from PIL import Image
import zipfile, shutil, os, re
import numpy as np

# Change these paths if running locally.
source_zip = Path("JCB New PRODUCTS.zip")
output_dir = Path("JCB_New_PRODUCTS_resized_organised")
output_zip = Path("JCB_New_PRODUCTS_resized_organised_with_code.zip")

targets = [
    (1200, 628),
    (1080, 1920),
    (1200, 1200),
    (960, 1200),
]

valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}

def safe_stem(name: str) -> str:
    stem, _ = os.path.splitext(name)
    return re.sub(r"[^\w\-. ]+", "_", stem).strip()

def detect_outer_margin_trim(img: Image.Image):
    arr = np.array(img.convert("RGB")).astype(np.float32)
    h, w = arr.shape[:2]
    row_means = arr.mean(axis=1)

    top_ref = row_means[:20].mean(axis=0)
    bottom_ref = row_means[-20:].mean(axis=0)

    top_dist = np.linalg.norm(row_means - top_ref, axis=1)
    bottom_dist = np.linalg.norm(row_means - bottom_ref, axis=1)

    top = 0
    for i in range(min(h // 3, h - 20)):
        if np.mean(top_dist[i:i+12]) > 5.0:
            top = i
            break

    bottom = h
    for i in range(h - 1, max(2 * h // 3, 12), -1):
        start = max(0, i - 11)
        if np.mean(bottom_dist[start:i+1]) > 5.0:
            bottom = i + 1
            break

    max_trim = int(h * 0.22)
    top = min(top, max_trim)
    bottom = max(bottom, h - max_trim)

    if bottom - top < h * 0.5:
        top, bottom = 0, h

    return top, bottom

def apply_inner_buffer(top: int, bottom: int, h: int):
    span = bottom - top
    buffer_px = max(18, int(min(h * 0.025, span * 0.05)))
    top2 = min(top + buffer_px, h - 1)
    bottom2 = max(bottom - buffer_px, top2 + 1)

    if bottom2 - top2 < h * 0.45:
        return top, bottom

    return top2, bottom2

def resize_fit(img: Image.Image, target_w: int, target_h: int):
    scale = min(target_w / img.width, target_h / img.height)
    new_w = max(1, round(img.width * scale))
    new_h = max(1, round(img.height * scale))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

def paste_with_feather_bg(canvas: Image.Image, fg: Image.Image, x: int, y: int, feather: int = 10):
    canvas_arr = np.array(canvas).astype(np.float32)
    bg_arr = canvas_arr.copy()
    fg_arr = np.array(fg).astype(np.float32)

    h, w = fg_arr.shape[:2]
    canvas_arr[y:y+h, x:x+w, :] = fg_arr

    if y > 0:
        f = min(feather, h)
        for i in range(f):
            alpha = (i + 1) / (f + 1)
            row_canvas = y + i
            bg_row = bg_arr[row_canvas, x:x+w, :]
            fg_row = fg_arr[i, :, :]
            canvas_arr[row_canvas, x:x+w, :] = bg_row * (1 - alpha) + fg_row * alpha

    bottom_pad = canvas_arr.shape[0] - (y + h)
    if bottom_pad > 0:
        f = min(feather, h)
        for i in range(f):
            alpha = (i + 1) / (f + 1)
            row_canvas = y + h - 1 - i
            bg_row = bg_arr[row_canvas, x:x+w, :]
            fg_row = fg_arr[h - 1 - i, :, :]
            canvas_arr[row_canvas, x:x+w, :] = bg_row * (1 - alpha) + fg_row * alpha

    return Image.fromarray(np.clip(np.round(canvas_arr), 0, 255).astype(np.uint8), mode="RGB")

_rembg_session = None

def remove_ai_background(img: Image.Image, model_name: str = "u2netp"):
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
    os.environ.setdefault("U2NET_HOME", "/tmp/u2net-cache")
    try:
        from rembg import new_session, remove
    except ImportError as error:
        raise RuntimeError(
            "AI background removal needs the rembg package. Install requirements.txt and restart the app."
        ) from error

    global _rembg_session
    if _rembg_session is None:
        _rembg_session = new_session(model_name)

    return remove(img.convert("RGBA"), session=_rembg_session)

def paste_with_alpha(canvas: Image.Image, fg: Image.Image, x: int, y: int):
    canvas_rgba = canvas.convert("RGBA")
    fg_rgba = fg.convert("RGBA")
    canvas_rgba.alpha_composite(fg_rgba, (x, y))
    return canvas_rgba.convert("RGB")

def hex_to_rgb(value: str):
    value = value.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError("Background color must be a 6-digit hex value.")

    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))

def build_output(
    trimmed: Image.Image,
    target_w: int,
    target_h: int,
    *,
    background_color: tuple[int, int, int] = (255, 255, 255),
):
    fitted = resize_fit(trimmed, target_w, target_h)
    fw, fh = fitted.size
    x = (target_w - fw) // 2
    y = (target_h - fh) // 2

    bg = Image.new("RGB", (target_w, target_h), background_color)

    if fitted.mode == "RGBA":
        out = paste_with_alpha(bg, fitted, x, y)
    else:
        out = paste_with_feather_bg(bg, fitted, x, y, feather=10)

    return (
        out,
        background_color,
    )

def process_zip(
    input_zip: Path,
    work_dir: Path,
    size_targets: list[tuple[int, int]] | tuple[tuple[int, int], ...] = targets,
    *,
    do_trim: bool = True,
    background_color: str | tuple[int, int, int] = (255, 255, 255),
    remove_background: bool = False,
    background_model: str = "u2netp",
    output_format: str = "PNG",
    jpeg_quality: int = 92,
    progress_callback=None,
):
    output_format = output_format.upper()
    extension = "jpg" if output_format == "JPEG" else output_format.lower()
    if isinstance(background_color, str):
        background_color = hex_to_rgb(background_color)

    output_folder = work_dir / "resized_organised"
    output_archive = work_dir / "resized_organised.zip"
    extract_dir = work_dir / "_extract_temp"

    if output_folder.exists():
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(input_zip, "r") as z:
        z.extractall(extract_dir)

    image_files = [
        p for p in extract_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_exts and not p.name.startswith("._")
    ]

    report = []

    for index, img_path in enumerate(sorted(image_files), start=1):
        if progress_callback:
            progress_callback(index, len(image_files), img_path.name)

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            title = safe_stem(img_path.name)
            product_folder = output_folder / title
            product_folder.mkdir(parents=True, exist_ok=True)

            if do_trim:
                top, bottom = detect_outer_margin_trim(img)
                top2, bottom2 = apply_inner_buffer(top, bottom, img.height)
                trimmed = img.crop((0, top2, img.width, bottom2))
            else:
                top, bottom = 0, img.height
                top2, bottom2 = 0, img.height
                trimmed = img

            background_removal_method = None
            if remove_background:
                if progress_callback:
                    progress_callback(index, len(image_files), f"AI removing background: {img_path.name}")
                trimmed = remove_ai_background(trimmed, model_name=background_model)
                background_removal_method = "ai"

            info_parts = [
                f"{img_path.name}",
                f"original={img.size}",
                f"outer_trim=({top},{bottom})",
                f"buffered_crop=(0,{top2},{img.width},{bottom2})",
                f"trimmed={trimmed.size}",
            ]

            for tw, th in size_targets:
                out, bg_rgb = build_output(
                    trimmed,
                    tw,
                    th,
                    background_color=background_color,
                )
                output_filename = f"{title}_{tw}x{th}.{extension}"
                save_kwargs = {"optimize": True}
                if output_format == "JPEG":
                    save_kwargs["quality"] = jpeg_quality
                out.save(product_folder / output_filename, output_format, **save_kwargs)

                info_parts.append(f"{tw}x{th}_background_color={bg_rgb}")
                if background_removal_method:
                    info_parts.append(f"{tw}x{th}_background_removal={background_removal_method}")

            report.append(" | ".join(info_parts))

    (output_folder / "_processing_report.txt").write_text("\n".join(report), encoding="utf-8")

    with zipfile.ZipFile(output_archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in output_folder.rglob("*"):
            z.write(p, p.relative_to(output_folder.parent))

    shutil.rmtree(extract_dir)

    return output_archive, output_folder, report

def main():
    if output_dir.exists():
        shutil.rmtree(output_dir)

    work_dir = Path("_jcb_cli_output")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    archive, folder, _ = process_zip(source_zip, work_dir, targets)

    shutil.copytree(folder, output_dir)
    shutil.copy2(archive, output_zip)
    shutil.rmtree(work_dir)

if __name__ == "__main__":
    main()
