import io, os, zipfile, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

import streamlit as st
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageChops
import fitz  # PyMuPDF

# ===== HEIC/HEIF =====
HEIF_OK = False
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_OK = True
except Exception:
    HEIF_OK = False

# ==========================
# PAGE & SIDEBAR
# ==========================
st.set_page_config(page_title="Multi-ZIP → JPG & Kompres 168–174 KB", page_icon="📦", layout="wide")
st.title("📦 Multi-ZIP / Files → JPG & Kompres 168–174 KB (auto)")
st.caption("Konversi gambar (termasuk JFIF/HEIC) & PDF ke JPG. Target otomatis: min 168 KB, max 174 KB. Video tidak diterima (tidak muncul di uploader & tidak diproses).")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    SPEED_PRESET = st.selectbox("Preset kecepatan", ["fast", "balanced"], index=0)
    MIN_SIDE_PX = st.number_input("Sisi terpendek minimum (px)", 64, 2048, 256, 32)
    SCALE_MIN = st.slider("Skala minimum saat downscale", 0.10, 0.75, 0.35, 0.05)

    # 🔎 Optimasi dokumen/tulisan (lebih natural)
    DOC_OPTIMIZE = st.checkbox("Optimasi dokumen/tulisan (auto)", True)
    DOC_STRENGTH = st.slider("Kekuatan optimasi dokumen", 0.0, 2.0, 0.8, 0.05)

    # ✅ Mode ini memprioritaskan kualitas visual untuk menghindari titik-titik/pecah
    QUALITY_SAFE_MODE = st.checkbox("Mode anti-artifact (disarankan)", True)

    UPSCALE_MAX = 1.0 if QUALITY_SAFE_MODE else st.slider("Batas upscale maksimum", 1.0, 3.0, 2.0, 0.1)
    SHARPEN_ON_RESIZE = st.checkbox("Sharpen ringan setelah resize", True)
    SHARPEN_AMOUNT = st.slider("Sharpen amount (resize)", 0.0, 2.0, 0.5 if QUALITY_SAFE_MODE else 0.8, 0.1)

    # 🎛️ Tuning Manual (opsional)
    MANUAL_TUNE = st.checkbox("Tuning manual (brightness/contrast/saturation/clarity)", False)
    if MANUAL_TUNE:
        BRIGHT_DELTA = st.slider("Brightness ± (%)", -40, 40, 0, 1)
        CONTRAST_DELTA = st.slider("Contrast ± (%)", -40, 40, 0, 1)
        SAT_DELTA = st.slider("Saturasi ± (%)", -40, 40, 0, 1)
        CLARITY_DELTA = st.slider("Clarity/Local contrast ± (%)", -40, 40, 0, 1)
        HIGHLIGHTS_COMP = st.slider("Jinakkan highlights", 0.0, 0.6, 0.15, 0.01)
        SHADOWS_LIFT = st.slider("Angkat shadows", 0.0, 0.6, 0.15, 0.01)
    else:
        # Default auto yang aman/natural untuk dokumen/tulisan
        BRIGHT_DELTA = int(8 * DOC_STRENGTH)     # ~+8% per strength
        CONTRAST_DELTA = int(12 * DOC_STRENGTH)  # ~+12% per strength
        SAT_DELTA = int(6 * DOC_STRENGTH)        # ~+6% per strength (dokumen berwarna)
        CLARITY_DELTA = int(8 * DOC_STRENGTH)    # ~+8% per strength
        HIGHLIGHTS_COMP = 0.10 * DOC_STRENGTH
        SHADOWS_LIFT = 0.10 * DOC_STRENGTH

    PDF_DPI = 180 if SPEED_PRESET == "fast" else 220  # sedikit lebih tinggi untuk hasil PDF lebih halus
    MASTER_ZIP_NAME = st.text_input("Nama master ZIP", "compressed.zip")
    st.markdown("**Target otomatis:** 168–174 KB (kualitas dijaga, tidak memaksa upscale)")

# ===== Tunables =====
MAX_QUALITY = 95
MIN_QUALITY = 45 if QUALITY_SAFE_MODE else 15  # jaga kualitas supaya tidak pecah
BG_FOR_ALPHA = (255, 255, 255)
THREADS = min(4, max(2, (os.cpu_count() or 2)))  # <= Batasin agar stabil
ZIP_COMP_ALGO = zipfile.ZIP_STORED if SPEED_PRESET == "fast" else zipfile.ZIP_DEFLATED

# ✅ Target size fixed by system
TARGET_KB = 174
MIN_KB = 168
IMG_EXT = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif", ".heic", ".heif"}
PDF_EXT = {".pdf"}
ALLOW_ZIP = True

# ===== Helper flags for JPEG encoder quality =====
# Subsampling 4:4:4 mengurangi color bleeding/"titik" pada tepi kontras tinggi.
JPEG_SUBSAMPLING = 0 if QUALITY_SAFE_MODE else 2
JPEG_OPTIMIZE = True
JPEG_PROGRESSIVE = True

# ==========================
# Helpers (quality tuned)
# ==========================
def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def maybe_sharpen(img: Image.Image, do_it=True, amount=1.0) -> Image.Image:
    if not do_it or amount <= 0:
        return img
    # Unsharp kalem saat tahap resize
    return img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(130 * amount), threshold=2))

def pre_smooth_if_downscaling(img: Image.Image, scale: float) -> Image.Image:
    """Sedikit Gaussian blur sebelum downscale untuk anti-aliasing & anti-ringing."""
    if QUALITY_SAFE_MODE and scale < 1.0:
        try:
            return img.filter(ImageFilter.GaussianBlur(radius=0.3))
        except Exception:
            return img
    return img

def to_rgb_flat(img: Image.Image, bg=BG_FOR_ALPHA) -> Image.Image:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        base = Image.new("RGB", img.size, bg)
        base.paste(img, mask=img.convert("RGBA").split()[-1])
        return base
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

def save_jpg_bytes(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    img.save(
        buf,
        format="JPEG",
        quality=int(quality),
        optimize=JPEG_OPTIMIZE,
        progressive=JPEG_PROGRESSIVE,
        subsampling=JPEG_SUBSAMPLING,
    )
    return buf.getvalue()

def try_quality_bs(img: Image.Image, target_kb: int, q_min=MIN_QUALITY, q_max=MAX_QUALITY):
    lo, hi = q_min, q_max
    best_bytes = None
    best_q = None
    while lo <= hi:
        mid = (lo + hi) // 2
        data = save_jpg_bytes(img, mid)
        if len(data) <= target_kb * 1024:
            best_bytes, best_q = data, mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best_bytes, best_q

def resize_to_scale(img: Image.Image, scale: float, do_sharpen=True, amount=1.0) -> Image.Image:
    w, h = img.size
    base = pre_smooth_if_downscaling(img, scale)
    nw, nh = max(int(w * scale), 1), max(int(h * scale), 1)
    out = base.resize((nw, nh), Image.LANCZOS)
    return maybe_sharpen(out, do_sharpen, amount)

def ensure_min_side(img: Image.Image, min_side_px: int, do_sharpen=True, amount=1.0) -> Image.Image:
    w, h = img.size
    if min(w, h) >= min_side_px:
        return img
    scale = max(min_side_px / max(min(w, h), 1), 1.0)
    return resize_to_scale(img, scale, do_sharpen, amount)

def load_image_from_bytes(name: str, raw: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(raw))
    return ImageOps.exif_transpose(im)

def gif_first_frame(im: Image.Image) -> Image.Image:
    try:
        im.seek(0)
    except Exception:
        pass
    return im.convert("RGBA") if im.mode == "P" else im

# ==========================
# 🔎 Dokumen/Tulisan Enhancer (natural)
# ==========================
def _tone_curve_lut(shadows: float, highlights: float) -> list:
    """LUT lembut untuk L channel: angkat shadow & jinakkan highlight (0..1)."""
    shadows = clamp01(shadows); highlights = clamp01(highlights)
    lut = []
    for i in range(256):
        x = i / 255.0
        y = x + shadows * (1.0 - x) * (1.0 - x)  # angkat shadow
        y = y - highlights * (x * x)             # jinakkan highlight
        lut.append(int(round(clamp01(y) * 255)))
    return lut

def _apply_lab_local_contrast(img: Image.Image, clarity_factor: float, auto_contrast_cut=1) -> Image.Image:
    """Local/global contrast via kanal L (LAB), versi natural."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    lab = img.convert("LAB")
    L, A, B = lab.split()
    L_eq = ImageOps.equalize(L)
    # Lebih kalem: alpha base 0.25 (dibatasi oleh clarity_factor)
    alpha = clamp01(0.25 * (0.5 + 0.5 * abs(clarity_factor)))
    L_mix = Image.blend(L, L_eq, alpha=alpha)
    L_ac = ImageOps.autocontrast(L_mix, cutoff=auto_contrast_cut)
    out = Image.merge("LAB", (L_ac, A, B)).convert("RGB")
    return out

def _edge_mask(img: Image.Image, edge_th: int = 20, blur: float = 0.8) -> Image.Image:
    """Mask tepi (huruf/garis) untuk batasi sharpening agar tidak berlebihan."""
    g = img.convert("L").filter(ImageFilter.FIND_EDGES)
    g = ImageOps.autocontrast(g)
    g = g.point(lambda p: 255 if p >= edge_th else 0)
    if blur > 0:
        g = g.filter(ImageFilter.GaussianBlur(radius=blur))
    return g

def doc_unsharp_smart(img: Image.Image, amount: float = 1.0,
                      radius: float = 0.9, percent_base: int = 90, threshold: int = 4) -> Image.Image:
    """
    Smart sharpen untuk dokumen: unsharp halus + blend pakai edge mask
    supaya teks tegas tanpa halo/artefak.
    """
    percent = int(percent_base * (1.0 + 0.6 * max(0.0, min(2.0, amount - 0.5))))
    shp = img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
    m = _edge_mask(img, edge_th=20, blur=0.8)
    return Image.composite(shp, img, m)

def enhance_for_text(
    img: Image.Image,
    enabled: bool,
    strength: float,
    bright_delta: int,
    contrast_delta: int,
    sat_delta: int,
    clarity_delta: int,
    highlights_comp: float,
    shadows_lift: float,
) -> Image.Image:
    """
    Pipeline dokumen 'natural':
      1) Local contrast di kanal L (LAB) (ringan)
      2) Tone curve (angkat shadow & jinakkan highlight) (ringan)
      3) Global tune kecil (brightness/contrast/sat)
      4) Smart sharpen edge-aware (kalem)
    """
    if not enabled:
        return to_rgb_flat(img)

    base = to_rgb_flat(img)

    # 1) Local contrast (lembut)
    clarity_factor = max(-0.3, min(0.3, clarity_delta / 100.0))
    out = _apply_lab_local_contrast(base, clarity_factor=clarity_factor, auto_contrast_cut=1 if QUALITY_SAFE_MODE else 0)

    # 2) Tone curve via L (sedikit dikurangi)
    lab = out.convert("LAB")
    L, A, B = lab.split()
    lut = _tone_curve_lut(shadows=shadows_lift * 0.8, highlights=highlights_comp * 0.8)
    L2 = L.point(lut * 1)
    out = Image.merge("LAB", (L2, A, B)).convert("RGB")

    # 3) Global tune (kalem)
    if bright_delta != 0:
        out = ImageEnhance.Brightness(out).enhance(1.0 + (bright_delta / 100.0) * 0.8)
    if contrast_delta != 0:
        out = ImageEnhance.Contrast(out).enhance(1.0 + (contrast_delta / 100.0) * 0.75)
    if sat_delta != 0:
        out = ImageEnhance.Color(out).enhance(1.0 + (sat_delta / 100.0) * 0.6)

    # 4) Smart sharpen edge-aware (kalem)
    out = doc_unsharp_smart(out, amount=0.5 + 0.45 * max(0.0, min(2.0, strength)), radius=0.9, percent_base=90, threshold=4)

    return out

# ==========================
# Kompresi
# ==========================
def compress_into_range(
    base_img: Image.Image,
    min_kb: int,
    max_kb: int,
    min_side_px: int,
    scale_min: float,
    upscale_max: float,
    do_sharpen: bool,
    sharpen_amount: float,
):
    base = to_rgb_flat(base_img)

    # 1) Coba tanpa resize dulu di kualitas optimal
    data, q = try_quality_bs(base, max_kb)
    if data is not None and (len(data) >= min_kb * 1024 or QUALITY_SAFE_MODE):
        return data, 1.0, q, len(data)

    # 2) Kalau perlu resize, prioritaskan resize dibanding nurunin kualitas terlalu jauh
    lo, hi = scale_min, 1.0
    best_pack = None
    max_steps = 8 if SPEED_PRESET == "fast" else 12
    for _ in range(max_steps):
        mid = (lo + hi) / 2
        candidate = resize_to_scale(base, mid, do_sharpen, sharpen_amount)
        candidate = ensure_min_side(candidate, min_side_px, do_sharpen, sharpen_amount)
        d, q2 = try_quality_bs(candidate, max_kb)
        if d is not None:
            best_pack = (d, mid, q2, len(d))
            lo = mid + (hi - mid) * 0.35
        else:
            hi = mid - (mid - lo) * 0.35
        if hi - lo < 1e-3:
            break

    if best_pack is None:
        smallest = resize_to_scale(base, scale_min, do_sharpen, sharpen_amount)
        smallest = ensure_min_side(smallest, min_side_px, do_sharpen, sharpen_amount)
        d = save_jpg_bytes(smallest, MIN_QUALITY)
        result = (d, scale_min, MIN_QUALITY, len(d))
    else:
        result = best_pack

    data, scale_used, q_used, size_b = result

    # 3) Jika masih < MIN_KB, JANGAN paksa upscale saat safe mode
    if size_b < min_kb * 1024 and not QUALITY_SAFE_MODE:
        img_now = resize_to_scale(base, scale_used, do_sharpen, sharpen_amount)
        img_now = ensure_min_side(img_now, min_side_px, do_sharpen, sharpen_amount)
        d, q2 = try_quality_bs(img_now, max_kb, max(q_used, MIN_QUALITY), MAX_QUALITY)
        if d is not None and len(d) > size_b:
            data, q_used, size_b = d, q2, len(d)
        cur_scale = scale_used
        iters = 0
        max_iters = 6 if SPEED_PRESET == "fast" else 12
        while size_b < min_kb * 1024 and cur_scale < upscale_max and iters < max_iters:
            cur_scale = min(cur_scale * 1.2, upscale_max)
            candidate = resize_to_scale(base, cur_scale, do_sharpen, sharpen_amount)
            candidate = ensure_min_side(candidate, min_side_px, do_sharpen, sharpen_amount)
            d, q3 = try_quality_bs(candidate, max_kb, MIN_QUALITY, MAX_QUALITY)
            if d is None:
                cur_scale *= 0.95
                iters += 1
                continue
            if len(d) > size_b:
                data, q_used, size_b, scale_used = d, q3, len(d), cur_scale
            iters += 1

    return data, scale_used, q_used, size_b

def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int) -> List[Image.Image]:
    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            rect = page.rect
            long_inch = max(rect.width, rect.height) / 72.0
            target_long_px = 2400 if QUALITY_SAFE_MODE else 2000
            dpi_eff = int(min(max(dpi, 72), max(72, target_long_px / max(long_inch, 1e-6))))
            mat = fitz.Matrix(dpi_eff / 72.0, dpi_eff / 72.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(ImageOps.exif_transpose(img))
    return images

def extract_zip_to_memory(zf_bytes: bytes) -> List[Tuple[Path, bytes]]:
    out = []
    with zipfile.ZipFile(io.BytesIO(zf_bytes), 'r') as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            with zf.open(info, 'r') as f:
                data = f.read()
            out.append((Path(info.filename), data))
    return out

def guess_base_name_from_zip(zipname: str) -> str:
    base = Path(zipname).stem
    return base or "output"

def process_one_file_entry(relpath: Path, raw_bytes: bytes, input_root_label: str):
    processed: List[Tuple[str, int, float, int, bool]] = []
    outputs: Dict[str, bytes] = {}
    skipped: List[Tuple[str, str]] = []
    ext = relpath.suffix.lower()
    try:
        if ext in PDF_EXT:
            pages = pdf_bytes_to_images(raw_bytes, dpi=PDF_DPI)
            for idx, pil_img in enumerate(pages, start=1):
                try:
                    if DOC_OPTIMIZE:
                        pil_img = enhance_for_text(
                            pil_img, True, DOC_STRENGTH,
                            BRIGHT_DELTA, CONTRAST_DELTA, SAT_DELTA, CLARITY_DELTA,
                            HIGHLIGHTS_COMP, SHADOWS_LIFT
                        )
                    data, scale, q, size_b = compress_into_range(
                        pil_img,
                        MIN_KB,
                        TARGET_KB,
                        MIN_SIDE_PX,
                        SCALE_MIN,
                        UPSCALE_MAX,
                        SHARPEN_ON_RESIZE,
                        SHARPEN_AMOUNT,
                    )
                    out_rel = relpath.with_suffix("").as_posix() + f"_p{idx}.jpg"
                    outputs[out_rel] = data
                    processed.append((out_rel, size_b, scale, q, MIN_KB * 1024 <= size_b <= TARGET_KB * 1024))
                except Exception as e:
                    skipped.append((f"{relpath} (page {idx})", str(e)))
        elif ext in IMG_EXT and (ext not in {".heic", ".heif"} or HEIF_OK):
            im = load_image_from_bytes(relpath.name, raw_bytes)
            if ext == ".gif":
                im = gif_first_frame(im)
            if DOC_OPTIMIZE:
                im = enhance_for_text(
                    im, True, DOC_STRENGTH,
                    BRIGHT_DELTA, CONTRAST_DELTA, SAT_DELTA, CLARITY_DELTA,
                    HIGHLIGHTS_COMP, SHADOWS_LIFT
                )
            data, scale, q, size_b = compress_into_range(
                im,
                MIN_KB,
                TARGET_KB,
                MIN_SIDE_PX,
                SCALE_MIN,
                UPSCALE_MAX,
                SHARPEN_ON_RESIZE,
                SHARPEN_AMOUNT,
            )
            out_rel = relpath.with_suffix(".jpg").as_posix()
            outputs[out_rel] = data
            processed.append((out_rel, size_b, scale, q, MIN_KB * 1024 <= size_b <= TARGET_KB * 1024))
        elif ext in {".heic", ".heif"} and not HEIF_OK:
            skipped.append((str(relpath), "Butuh pillow-heif (tidak tersedia)"))
        # else: ignore unknown
    except Exception as e:
        skipped.append((str(relpath), str(e)))

    return input_root_label, processed, skipped, outputs

# ==========================
# UI Upload & Run
# ==========================
st.subheader("1) Upload ZIP atau File Lepas")
allowed_exts_for_uploader = sorted({e.lstrip('.') for e in IMG_EXT.union(PDF_EXT)} | ({"zip"} if ALLOW_ZIP else set()))

uploaded_files = st.file_uploader(
    "Upload beberapa ZIP (berisi folder/gambar/PDF) dan/atau file lepas (gambar/PDF). Video ditolak otomatis (tidak dimuat).",
    type=allowed_exts_for_uploader,
    accept_multiple_files=True,
)

run = st.button("🚀 Proses & Buat Master ZIP", type="primary", disabled=not uploaded_files)

if run:
    if not uploaded_files:
        st.warning("Silakan upload minimal satu file.")
        st.stop()

    jobs = []
    used_labels = set()

    def unique_name(base: str, used: set) -> str:
        name = base
        idx = 2
        while name in used:
            name = f"{base}_{idx}"
            idx += 1
        used.add(name)
        return name

    zip_inputs, loose_inputs = [], []
    for f in uploaded_files:
        name, raw = f.name, f.read()
        if name.lower().endswith(".zip"):
            zip_inputs.append((name, raw))
        else:
            loose_inputs.append((name, raw))

    allowed = IMG_EXT.union(PDF_EXT)

    for zname, zbytes in zip_inputs:
        try:
            pairs = extract_zip_to_memory(zbytes)
            base_label = unique_name(guess_base_name_from_zip(zname), used_labels)
            items = [(relp, data) for (relp, data) in pairs if relp.suffix.lower() in allowed]
            if items:
                jobs.append({"label": base_label, "items": items})
        except Exception as e:
            st.error(f"Gagal membuka ZIP {zname}: {e}")

    if loose_inputs:
        ts = time.strftime("%Y%m%d_%H%M%S")
        base_label = unique_name(f"compressed_pict_{ts}", used_labels)
        items = [(Path(name), data) for (name, data) in loose_inputs if Path(name).suffix.lower() in allowed]
        if items:
            jobs.append({"label": base_label, "items": items})

    if not jobs:
        st.error("Tidak ada berkas valid (butuh gambar/PDF, atau ZIP berisi file-file tersebut).")
        st.stop()

    st.write(f"🔧 Ditemukan **{sum(len(j['items']) for j in jobs)}** berkas dari **{len(jobs)}** input.")

    summary: Dict[str, List[Tuple[str, int, float, int, bool]]] = defaultdict(list)
    skipped_all: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    master_buf = io.BytesIO()
    zip_write_lock = threading.Lock()
    with zipfile.ZipFile(master_buf, "w", compression=ZIP_COMP_ALGO) as master:
        top_folders: Dict[str, str] = {}
        for job in jobs:
            top = f"{job['label']}_compressed"
            top_folders[job['label']] = top
            master.writestr(f"{top}/", "")

        def add_to_master_zip_threadsafe(top_folder: str, rel_path: str, data: bytes):
            with zip_write_lock:
                master.writestr(f"{top_folder}/{rel_path}", data)

        def worker(label: str, relp: Path, raw: bytes):
            return process_one_file_entry(relp, raw, label)

        all_tasks = [(job["label"], relp, data) for job in jobs for (relp, data) in job["items"]]
        total, done = len(all_tasks), 0
        progress = st.progress(0.0)

        with ThreadPoolExecutor(max_workers=THREADS) as ex:
            futures = [ex.submit(worker, *t) for t in all_tasks]
            for fut in as_completed(futures):
                label, prc, skp, outs = fut.result()
                summary[label].extend(prc)
                skipped_all[label].extend(skp)
                if outs:
                    top = top_folders[label]
                    for rel_path, data in outs.items():
                        add_to_master_zip_threadsafe(top, rel_path, data)
                done += 1
                progress.progress(min(done / total, 1.0))

    master_buf.seek(0)

    # ==========================
    # Ringkasan
    # ==========================
    st.subheader("📊 Ringkasan")
    grand_ok = 0
    grand_cnt = 0
    MAX_ROWS_PER_JOB = 300

    for job in jobs:
        base = job["label"]
        items = summary[base]
        skipped = skipped_all[base]
        with st.expander(f"📦 {base} — {len(items)} file diproses, {len(skipped)} dilewati/errored"):
            ok = 0
            shown = 0
            for name, size_b, scale, q, in_range in items:
                if shown >= MAX_ROWS_PER_JOB:
                    break
                kb = size_b / 1024
                flag = "✅" if in_range else ("ℹ️" if (QUALITY_SAFE_MODE and kb < MIN_KB) else ("🔼" if kb < MIN_KB else "⚠️"))
                st.write(f"{flag} {name} → **{kb:.1f} KB** | scale≈{scale:.3f} | quality={q}")
                ok += 1 if in_range else 0
                shown += 1
            extra = len(items) - shown
            if extra > 0:
                st.caption(f"(+{extra} baris lainnya disembunyikan untuk menjaga performa UI)")

            if skipped:
                st.write("**Dilewati/Errored:**")
                for n, reason in skipped[:50]:
                    st.write(f"- {n}: {reason}")

            st.caption(f"Berhasil di rentang {MIN_KB}–{TARGET_KB} KB: **{ok}/{len(items)}**")
            grand_ok += ok
            grand_cnt += len(items)

    st.write("---")
    st.write(f"**Total file OK di rentang:** {grand_ok}/{grand_cnt}")

    st.download_button(
        "⬇️ Download Master ZIP",
        data=master_buf.getvalue(),
        file_name=MASTER_ZIP_NAME.strip() or "compressed.zip",
        mime="application/zip",
    )

    st.success("Selesai! Master ZIP siap diunduh. Mode anti-artifact aktif untuk kualitas optimal.")
