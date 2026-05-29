from pathlib import Path
from PIL import Image
import zipfile, shutil, os, re
import numpy as np
from collections import deque

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

def robust_band_colour(band: np.ndarray):
    pts = band.reshape(-1, 3).astype(np.float32)

    brightness = pts.mean(axis=1)
    cutoff = np.percentile(brightness, 25)
    pts = pts[brightness >= cutoff]

    if len(pts) == 0:
        pts = band.reshape(-1, 3).astype(np.float32)

    med = np.median(pts, axis=0)
    d = np.linalg.norm(pts - med, axis=1)
    keep = pts[d <= np.percentile(d, 70)]

    if len(keep) < 50:
        keep = pts

    return np.median(keep, axis=0)

def make_gradient_background(target_w: int, target_h: int, top_rgb, bottom_rgb):
    top_rgb = np.array(top_rgb, dtype=np.float32)
    bottom_rgb = np.array(bottom_rgb, dtype=np.float32)

    y = np.linspace(0.0, 1.0, target_h, dtype=np.float32)[:, None]
    grad = top_rgb * (1.0 - y) + bottom_rgb * y
    grad = np.repeat(grad[:, None, :], target_w, axis=1)
    grad = np.clip(np.round(grad), 0, 255).astype(np.uint8)

    return Image.fromarray(grad, mode="RGB")

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

def connected_edge_mask(mask: np.ndarray):
    h, w = mask.shape
    connected = np.zeros((h, w), dtype=bool)
    queue = deque()

    for x in range(w):
        if mask[0, x]:
            queue.append((0, x))
            connected[0, x] = True
        if mask[h - 1, x] and not connected[h - 1, x]:
            queue.append((h - 1, x))
            connected[h - 1, x] = True

    for y in range(h):
        if mask[y, 0] and not connected[y, 0]:
            queue.append((y, 0))
            connected[y, 0] = True
        if mask[y, w - 1] and not connected[y, w - 1]:
            queue.append((y, w - 1))
            connected[y, w - 1] = True

    while queue:
        y, x = queue.popleft()
        for y2, x2 in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= y2 < h and 0 <= x2 < w and mask[y2, x2] and not connected[y2, x2]:
                connected[y2, x2] = True
                queue.append((y2, x2))

    return connected

def remove_edge_background(img: Image.Image, tolerance: int = 36, feather: int = 18):
    rgba = img.convert("RGBA")
    arr = np.array(rgba).astype(np.float32)
    rgb = arr[:, :, :3]
    h, w = rgb.shape[:2]

    edge_pixels = np.concatenate((rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]), axis=0)
    bg_rgb = np.median(edge_pixels, axis=0)
    distance = np.linalg.norm(rgb - bg_rgb, axis=2)
    bg_candidate = distance <= (tolerance + feather)
    connected = connected_edge_mask(bg_candidate)

    alpha = np.full((h, w), 255, dtype=np.float32)
    fade = np.clip((distance - tolerance) / max(1, feather), 0, 1) * 255
    alpha[connected] = fade[connected]

    arr[:, :, 3] = np.minimum(arr[:, :, 3], alpha)
    return Image.fromarray(np.clip(np.round(arr), 0, 255).astype(np.uint8), mode="RGBA"), tuple(int(v) for v in bg_rgb.round())

def paste_with_alpha(canvas: Image.Image, fg: Image.Image, x: int, y: int):
    canvas_rgba = canvas.convert("RGBA")
    fg_rgba = fg.convert("RGBA")
    canvas_rgba.alpha_composite(fg_rgba, (x, y))
    return canvas_rgba.convert("RGB")

def build_output(
    trimmed: Image.Image,
    target_w: int,
    target_h: int,
    *,
    background_mode: str = "gradient",
    remove_background: bool = False,
    background_tolerance: int = 36,
):
    trimmed = trimmed.convert("RGB")
    colour_sample = trimmed
    removed_bg_rgb = None
    if remove_background:
        trimmed, removed_bg_rgb = remove_edge_background(trimmed, tolerance=background_tolerance)

    fitted = resize_fit(trimmed, target_w, target_h)
    fw, fh = fitted.size
    x = (target_w - fw) // 2
    y = (target_h - fh) // 2

    sample_fitted = resize_fit(colour_sample, target_w, target_h)
    arr = np.array(sample_fitted.convert("RGB")).astype(np.uint8)
    band_h = max(6, min(24, fh // 18))

    top_band = arr[:band_h, :, :]
    bottom_band = arr[-band_h:, :, :]

    top_rgb = robust_band_colour(top_band)
    bottom_rgb = robust_band_colour(bottom_band)

    if background_mode == "white":
        bg = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    else:
        bg = make_gradient_background(target_w, target_h, top_rgb, bottom_rgb)

    if fitted.mode == "RGBA":
        out = paste_with_alpha(bg, fitted, x, y)
    else:
        out = paste_with_feather_bg(bg, fitted, x, y, feather=10)

    return (
        out,
        tuple(int(v) for v in top_rgb.round()),
        tuple(int(v) for v in bottom_rgb.round()),
        removed_bg_rgb,
    )

def process_zip(
    input_zip: Path,
    work_dir: Path,
    size_targets: list[tuple[int, int]] | tuple[tuple[int, int], ...] = targets,
    *,
    do_trim: bool = True,
    background_mode: str = "gradient",
    remove_background: bool = False,
    background_tolerance: int = 36,
    output_format: str = "PNG",
    jpeg_quality: int = 92,
    progress_callback=None,
):
    output_format = output_format.upper()
    extension = "jpg" if output_format == "JPEG" else output_format.lower()

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

            info_parts = [
                f"{img_path.name}",
                f"original={img.size}",
                f"outer_trim=({top},{bottom})",
                f"buffered_crop=(0,{top2},{img.width},{bottom2})",
                f"trimmed={trimmed.size}",
            ]

            for tw, th in size_targets:
                out, top_rgb, bottom_rgb, removed_bg_rgb = build_output(
                    trimmed,
                    tw,
                    th,
                    background_mode=background_mode,
                    remove_background=remove_background,
                    background_tolerance=background_tolerance,
                )
                output_filename = f"{title}_{tw}x{th}.{extension}"
                save_kwargs = {"optimize": True}
                if output_format == "JPEG":
                    save_kwargs["quality"] = jpeg_quality
                out.save(product_folder / output_filename, output_format, **save_kwargs)

                info_parts.append(f"{tw}x{th}_bg_top={top_rgb}")
                info_parts.append(f"{tw}x{th}_bg_bottom={bottom_rgb}")
                info_parts.append(f"{tw}x{th}_background_mode={background_mode}")
                if removed_bg_rgb:
                    info_parts.append(f"{tw}x{th}_removed_bg_sample={removed_bg_rgb}")

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
