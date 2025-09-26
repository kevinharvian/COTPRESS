import io, os, zipfile, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import fitz  # PyMuPDF

# ===== HEIC/HEIF =====
HEIF_OK = False
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_OK = True
except Exception:
    HEIF_OK = False

# ===== PAGE =====
st.set_page_config(page_title="Multi-ZIP → JPG & Kompres 100–170 KB", page_icon="📦", layout="wide")
st.title("📦 Multi-ZIP / Files → JPG & Kompres 100–170 KB (auto)")
st.caption("Konversi gambar (termasuk JFIF/HEIC) & PDF ke JPG. Target ukuran otomatis: min 100 KB, max 170 KB. File video TIDAK diterima (silent reject).")

# ===== Sidebar Settings =====
with st.sidebar:
    st.header("⚙️ Pengaturan")
    SPEED_PRESET = st.selectbox("Preset kecepatan", ["fast", "balanced"], index=0)
    MIN_SIDE_PX = st.number_input("Sisi terpendek minimum (px)", 64, 2048, 256, 32)
    SCALE_MIN = st.slider("Skala minimum saat downscale", 0.10, 0.75, 0.35, 0.05)
    UPSCALE_MAX = st.slider("Batas upscale maksimum", 1.0, 3.0, 2.0, 0.1)
    SHARPEN_ON_RESIZE = st.checkbox("Sharpen ringan setelah resize", True)
    SHARPEN_AMOUNT = st.slider("Sharpen amount", 0.0, 2.0, 1.0, 0.1)
    PDF_DPI = 150 if SPEED_PRESET == "fast" else 200
    MASTER_ZIP_NAME = st.text_input("Nama master ZIP", "compressed.zip")
    st.markdown("**Target otomatis:** 100–170 KB (tidak bisa diubah)")

# ===== Tunables =====
MAX_QUALITY = 95
MIN_QUALITY = 15
BG_FOR_ALPHA = (255, 255, 255)
THREADS = min(4, max(2, (os.cpu_count() or 2)))  # <= Batasin agar stabil
ZIP_COMP_ALGO = zipfile.ZIP_STORED if SPEED_PRESET == "fast" else zipfile.ZIP_DEFLATED

# ✅ Target size fixed by system
TARGET_KB = 170
MIN_KB = 100

IMG_EXT = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif", ".heic", ".heif"}
PDF_EXT = {".pdf"}
# (Optional) daftar video jika ingin dipakai untuk validasi eksternal, tapi TIDAK digunakan di filter:
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".3gp", ".wmv", ".flv", ".mpg", ".mpeg"}

# ===== Helpers (quality tuned) =====
def maybe_sharpen(img: Image.Image, do_it=True, amount=1.0) -> Image.Image:
    if not do_it or amount <= 0:
        return img
    return img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(150*amount), threshold=2))

def to_rgb_flat(img: Image.Image, bg=BG_FOR_ALPHA) -> Image.Image:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        base = Image.new("RGB", img.size, bg)
        base.paste(img, mask=img.convert("RGBA").split()[-1])
        return base
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

# ⬇️ Balanced mode tetap pakai optimize/progressive; fast dimatikan untuk hemat CPU
def save_jpg_bytes(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    if SPEED_PRESET == "fast":
        img.save(
            buf,
            format="JPEG",
            quality=quality,
            optimize=False,
            progressive=False,
            subsampling=2,
        )
    else:
        img.save(
            buf,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=True,
            subsampling=2,
        )
    return buf.getvalue()

def try_quality_bs(img: Image.Image, target_kb: int, q_min=MIN_QUALITY, q_max=MAX_QUALITY):
    """Binary search kualitas sampai <= target_kb."""
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
    nw, nh = max(int(w*scale), 1), max(int(h*scale), 1)
    out = img.resize((nw, nh), Image.LANCZOS)
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

    # A. tanpa resize
    data, q = try_quality_bs(base, max_kb)
    if data is not None:
        result = (data, 1.0, q, len(data))
    else:
        # B. downscale cepat (few iterations)
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

    # C. kalau masih < min_kb → naikkan kualitas/detail (tanpa lewat target)
    if size_b < min_kb * 1024:
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

# ===== PDF → PIL via PyMuPDF (DPI adaptif) =====
def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int) -> List[Image.Image]:
    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            rect = page.rect
            long_inch = max(rect.width, rect.height) / 72.0  # 1pt = 1/72 inch
            target_long_px = 2000  # sasaran aman agar RAM stabil, kualitas masih bagus
            dpi_eff = int(min(max(dpi, 72), max(72, target_long_px / max(long_inch, 1e-6))))
            zoom = dpi_eff / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(ImageOps.exif_transpose(img))
    return images

# ===== ZIP/Jobs Handling =====
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

# ===== ZIP/Processing =====
def process_one_file_entry(relpath: Path, raw_bytes: bytes, input_root_label: str):
    processed = []
    outputs: Dict[str, bytes] = {}
    skipped: List[Tuple[str, str]] = []  # kept for interface parity, but will be unused for videos (silent reject)

    ext = relpath.suffix.lower()
    try:
        if ext in PDF_EXT:
            pages = pdf_bytes_to_images(raw_bytes, dpi=PDF_DPI)
            for idx, pil_img in enumerate(pages, start=1):
                try:
                    data, scale, q, size_b = compress_into_range(
                        pil_img, MIN_KB, TARGET_KB, MIN_SIDE_PX,
                        SCALE_MIN, UPSCALE_MAX, SHARPEN_ON_RESIZE, SHARPEN_AMOUNT
                    )
                    out_rel = relpath.with_suffix("").as_posix() + f"_p{idx}.jpg"
                    outputs[out_rel] = data
                    processed.append((out_rel, size_b, scale, q, MIN_KB*1024 <= size_b <= TARGET_KB*1024))
                except Exception as e:
                    skipped.append((f"{relpath} (page {idx})", str(e)))
        elif ext in IMG_EXT and (ext not in {".heic", ".heif"} or HEIF_OK):
            im = load_image_from_bytes(relpath.name, raw_bytes)
            if ext == ".gif":
                im = gif_first_frame(im)
            data, scale, q, size_b = compress_into_range(
                im, MIN_KB, TARGET_KB, MIN_SIDE_PX,
                SCALE_MIN, UPSCALE_MAX, SHARPEN_ON_RESIZE, SHARPEN_AMOUNT
            )
            out_rel = relpath.with_suffix(".jpg").as_posix()
            outputs[out_rel] = data
            processed.append((out_rel, size_b, scale, q, MIN_KB*1024 <= size_b <= TARGET_KB*1024))
        elif ext in {".heic", ".heif"} and not HEIF_OK:
            skipped.append((str(relpath), "Butuh pillow-heif (tidak tersedia)"))
        # else: unknown types ignored
    except Exception as e:
        skipped.append((str(relpath), str(e)))

    return input_root_label, processed, skipped, outputs

# ===== UI Upload & Run =====
st.subheader("1) Upload ZIP atau File Lepas")
uploaded_files = st.file_uploader(
    "Upload beberapa ZIP (berisi folder/gambar/PDF) dan/atau file lepas (gambar/PDF). Video ditolak otomatis (tidak dimuat).",
    type=None, accept_multiple_files=True
)

run = st.button("🚀 Proses & Buat Master ZIP", type="primary", disabled=not uploaded_files)

if run:
    if not uploaded_files:
        st.warning("Silakan upload minimal satu file."); st.stop()

    # Build jobs: tiap ZIP = 1 job; file lepas digabung jadi 1 job
    jobs = []
    used_labels = set()

    def unique_name(base: str, used: set) -> str:
        name = base; idx = 2
        while name in used:
            name = f"{base}_{idx}"; idx += 1
        used.add(name); return name

    # Pisahkan ZIP dan file lepas
    zip_inputs, loose_inputs = [], []
    for f in uploaded_files:
        name, raw = f.name, f.read()
        if name.lower().endswith(".zip"):
            zip_inputs.append((name, raw))
        else:
            loose_inputs.append((name, raw))

    # ZIP inputs → ambil hanya gambar/PDF dari isi ZIP (video silent reject)
    allowed = IMG_EXT.union(PDF_EXT)

    for zname, zbytes in zip_inputs:
        try:
            pairs = extract_zip_to_memory(zbytes)  # [(relpath, data)]
            base_label = unique_name(guess_base_name_from_zip(zname), used_labels)
            items = [(relp, data) for (relp, data) in pairs if relp.suffix.lower() in allowed]
            if items:
                jobs.append({"label": base_label, "items": items})
        except Exception as e:
            st.error(f"Gagal membuka ZIP {zname}: {e}")

    # Loose inputs → hanya gambar/PDF (video silent reject)
    if loose_inputs:
        ts = time.strftime("%Y%m%d_%H%M%S")
        base_label = unique_name(f"compressed_pict_{ts}", used_labels)
        items = [(Path(name), data) for (name, data) in loose_inputs if Path(name).suffix.lower() in allowed]
        if items:
            jobs.append({"label": base_label, "items": items})

    if not jobs:
        st.error("Tidak ada berkas valid (butuh gambar/PDF, atau ZIP berisi file-file tersebut)."); st.stop()

    st.write(f"🔧 Ditemukan **{sum(len(j['items']) for j in jobs)}** berkas dari **{len(jobs)}** input.")

    # ====== STREAMING WRITE KE MASTER ZIP (tidak menumpuk RAM) ======
    summary: Dict[str, List[Tuple[str, int, float, int, bool]]] = defaultdict(list)
    skipped_all: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    # Siapkan master ZIP lebih dulu
    master_buf = io.BytesIO()
    zip_write_lock = threading.Lock()

    with zipfile.ZipFile(master_buf, "w", compression=ZIP_COMP_ALGO) as master:
        # Pre-create top folders
        top_folders = {}
        for job in jobs:
            top = f"{job['label']}_compressed"
            top_folders[job['label']] = top
            master.writestr(f"{top}/", "")

        def add_to_master_zip_threadsafe(top_folder: str, rel_path: str, data: bytes):
            with zip_write_lock:
                master.writestr(f"{top_folder}/{rel_path}", data)

        # Proses multi-core, tulis hasil segera
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

                # 🔑 Tulis output langsung ke ZIP agar tidak menumpuk di RAM
                if outs:
                    top = top_folders[label]
                    for rel_path, data in outs.items():
                        add_to_master_zip_threadsafe(top, rel_path, data)

                done += 1
                progress.progress(min(done/total, 1.0))

    master_buf.seek(0)

    # Ringkasan
    st.subheader("📊 Ringkasan")
    grand_ok = 0; grand_cnt = 0

    # Kurangi spam UI: tampilkan maksimal 300 baris detail agar UI tidak lambat
    MAX_ROWS_PER_JOB = 300

    for job in jobs:
        base = job["label"]; items = summary[base]; skipped = skipped_all[base]
        with st.expander(f"📦 {base} — {len(items)} file diproses, {len(skipped)} dilewati/errored"):
            ok = 0
            shown = 0
            for name, size_b, scale, q, in_range in items:
                if shown >= MAX_ROWS_PER_JOB:
                    break
                kb = size_b/1024
                flag = "✅" if in_range else ("🔼" if kb < MIN_KB else "⚠️")
                st.write(f"{flag} `{name}` → **{kb:.1f} KB** | scale≈{scale:.3f} | quality={q}")
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
        grand_ok += ok; grand_cnt += len(items)

    st.write("---")
    st.write(f"**Total file OK di rentang:** {grand_ok}/{grand_cnt}")

    st.download_button(
        "⬇️ Download Master ZIP",
        data=master_buf.getvalue(),
        file_name=MASTER_ZIP_NAME.strip() or "compressed.zip",
        mime="application/zip",
    )
    st.success("Selesai! Master ZIP siap diunduh.")