"""
Face Detection and Recognition Web Application
Menggunakan InsightFace (RetinaFace + ArcFace) untuk akurasi tinggi.
Fallback ke LBPH jika InsightFace tidak tersedia.
"""

import os
import glob
import sqlite3
import threading
import logging
import re
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import pytesseract  # Tambahan untuk OCR

from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, flash, session
)
from werkzeug.security import generate_password_hash, check_password_hash

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== PATH CONFIG ======
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "database_wajah")
KTP_DIR = os.path.join(BASE_DIR, "data", "database_ktp") # <--- FOLDER BARU
MODEL_DIR = os.path.join(BASE_DIR, "model")
DB_PATH = os.path.join(BASE_DIR, "database.db")
MODEL_PATH = os.path.join(MODEL_DIR, "Trainer.yml")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(KTP_DIR, exist_ok=True) # <--- BUAT FOLDER OTOMATIS
os.makedirs(MODEL_DIR, exist_ok=True)

# ====== KONFIGURASI TESSERACT (OCR) ======
# Wajib diarahkan ke file exe instalasi Tesseract
path_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(path_tesseract):
    pytesseract.pytesseract.tesseract_cmd = path_tesseract
else:
    logger.warning("WARNING: Tesseract OCR tidak ditemukan di default path. Fitur scan KTP mungkin gagal.")

# ====== FLASK APP ======
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

# ====== FACE ENGINE SELECTION ======
# Try to use InsightFace first, fallback to LBPH
USE_INSIGHTFACE = os.environ.get("USE_INSIGHTFACE", "1") == "1"
FACE_ENGINE = None

try:
    if USE_INSIGHTFACE:
        import face_engine
        # face_engine.initialize() is called automatically on import
        FACE_ENGINE = "insightface"
        logger.info("Using InsightFace engine for face recognition")
except ImportError as e:
    logger.warning(f"InsightFace not available, falling back to LBPH: {e}")
    FACE_ENGINE = "lbph"
except Exception as e:
    logger.warning(f"Failed to initialize InsightFace, falling back to LBPH: {e}")
    FACE_ENGINE = "lbph"

if FACE_ENGINE is None:
    FACE_ENGINE = "lbph"

# ====== ADMIN CREDENTIALS ======
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
_default_plain = os.environ.get("ADMIN_PASSWORD_PLAIN", "Cakra@123")
ADMIN_PASSWORD_HASH = os.environ.get("ADMIN_PASSWORD_HASH", generate_password_hash(_default_plain))

def login_required(view_func):
    def wrapper(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin_login"))
        return view_func(*args, **kwargs)
    wrapper.__name__ = view_func.__name__
    return wrapper

# ====== OpenCV SETUP (for LBPH fallback) ======
def get_cascade_path(fname="haarcascade_frontalface_default.xml"):
    try:
        return os.path.join(cv2.data.haarcascades, fname)
    except Exception:
        return fname

CASCADE_FILE_MAIN = get_cascade_path("haarcascade_frontalface_default.xml")
CASCADE_FILE_ALT2 = get_cascade_path("haarcascade_frontalface_alt2.xml")

# Only check cascade if we might need LBPH fallback
detectors = []
if os.path.isfile(CASCADE_FILE_MAIN):
    det_main = cv2.CascadeClassifier(CASCADE_FILE_MAIN)
    if not det_main.empty():
        detectors.append(det_main)

if os.path.isfile(CASCADE_FILE_ALT2):
    det_alt2 = cv2.CascadeClassifier(CASCADE_FILE_ALT2)
    if not det_alt2.empty():
        detectors.append(det_alt2)

# LBPH recognizer (for fallback)
recognizer = None
if hasattr(cv2, "face"):
    recognizer = cv2.face.LBPHFaceRecognizer_create(1, 8, 8, 8)

model_loaded = False
model_lock = threading.Lock()

# Threshold & voting
LBPH_CONF_THRESHOLD = float(os.environ.get("LBPH_CONF_THRESHOLD", "120"))
VOTE_MIN_SHARE = float(os.environ.get("VOTE_MIN_SHARE", "0.35"))
MIN_VALID_FRAMES = int(os.environ.get("MIN_VALID_FRAMES", "2"))
EARLY_VOTES_REQUIRED = int(os.environ.get("EARLY_VOTES_REQUIRED", "4"))
EARLY_CONF_THRESHOLD = float(os.environ.get("EARLY_CONF_THRESHOLD", "80"))

# ====== DB ======
def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def db_init():
    with db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                nik INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                dob TEXT NOT NULL,
                address TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS queues(
                poli_name TEXT PRIMARY KEY,
                next_number INTEGER NOT NULL
            )
        """)
        c = conn.execute("SELECT COUNT(*) AS c FROM queues").fetchone()
        if c["c"] == 0:
            for poli in ["Poli Umum", "Poli Gigi", "IGD"]:
                conn.execute("INSERT INTO queues(poli_name, next_number) VALUES(?, ?)", (poli, 0))
        conn.commit()
db_init()

# ====== UTIL ======
def parse_date_flexible(dob_str: str):
    if not dob_str:
        return None
    dob_str = dob_str.strip()
    formats = [
        "%Y-%m-%d", "%d-%m-%Y",
        "%Y/%m/%d", "%d/%m/%Y",
        "%Y.%m.%d", "%d.%m.%Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(dob_str, fmt)
        except Exception:
            continue
    return None

def calculate_age(dob_str: str) -> str:
    try:
        dt = parse_date_flexible(dob_str)
        if not dt:
            return "N/A"
        today = datetime.now()
        age = today.year - dt.year - ((today.month, today.day) < (dt.month, dt.day))
        return f"{age} Tahun"
    except Exception:
        return "N/A"

def list_existing_samples(nik: int) -> int:
    return len(glob.glob(os.path.join(DATA_DIR, f"{nik}.*.jpg")))

def bytes_to_bgr(image_bytes: bytes):
    np_data = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_data, cv2.IMREAD_COLOR)

def is_blurry(gray_roi, thr: float = 80.0) -> bool:
    fm = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
    return fm < thr

def preprocess_roi(gray_roi):
    roi = cv2.resize(gray_roi, (200, 200), interpolation=cv2.INTER_CUBIC)
    roi = cv2.equalizeHist(roi)
    return roi

def detect_largest_face(gray):
    best_roi, best_rect, best_area = None, None, -1
    for det in detectors:
        faces = det.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60)
        )
        if len(faces) == 0:
            continue
        (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
        area = w * h
        if area > best_area:
            best_area = area
            best_rect = (x, y, w, h)
            best_roi = gray[y:y+h, x:x+w]
    if best_roi is None:
        return None, None
    return best_roi, best_rect

def save_face_images_from_frame(img_bgr, name: str, nik: int, idx: int) -> int:
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return 0

    crop, rect = detect_largest_face(gray)
    if crop is None:
        return 0
    if is_blurry(crop, thr=40.0):
        return 0

    preprocessed = preprocess_roi(crop)
    out_path = os.path.join(DATA_DIR, f"{nik}.{idx}.jpg")
    cv2.imwrite(out_path, preprocessed)
    return 1

def augment_img(img):
    out = img.copy()
    out = cv2.convertScaleAbs(out, alpha=1.05, beta=5)
    h, w = out.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), 3, 1.0)
    out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return out

def ensure_min_samples(nik: int, min_count: int = 20) -> int:
    pattern = os.path.join(DATA_DIR, f"{nik}.*.jpg")
    files = sorted(glob.glob(pattern), key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split(".")[1]))
    saved = len(files)
    if saved == 0:
        return 0

    next_idx = int(os.path.splitext(os.path.basename(files[-1]))[0].split(".")[1]) + 1
    added = 0
    src_imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in files if os.path.isfile(p)]
    src_imgs = [im for im in src_imgs if im is not None and im.size > 0]
    if not src_imgs:
        return 0

    i = 0
    while saved + added < min_count:
        base = src_imgs[i % len(src_imgs)]
        aug = augment_img(base)
        out_path = os.path.join(DATA_DIR, f"{nik}.{next_idx}.jpg")
        cv2.imwrite(out_path, aug)
        next_idx += 1
        added += 1
        i += 1
    return added

def get_images_and_labels():
    faces, ids = [], []
    nik_counts = {}
    for fname in os.listdir(DATA_DIR):
        if not fname.lower().endswith(".jpg"):
            continue
        fpath = os.path.join(DATA_DIR, fname)
        try:
            pil = Image.open(fpath).convert("L")
            img_np = np.array(pil, "uint8")
            parts = fname.split(".")
            if len(parts) < 3:
                continue
            nik = int(parts[0])
            faces.append(img_np)
            ids.append(nik)
            nik_counts[nik] = nik_counts.get(nik, 0) + 1
        except Exception as e:
            logger.debug(f"Skip: {fpath} - {e}")
    
    if faces:
        logger.info(f"[TRAINING] Loaded {len(faces)} images for {len(nik_counts)} unique NIKs")
    return faces, ids

def train_model_blocking():
    faces, ids = get_images_and_labels()
    if not faces:
        logger.info("[TRAINING] No training data found")
        return False, "Tidak ada data untuk training!"
    try:
        logger.info(f"[TRAINING] Starting training with {len(faces)} images...")
        recognizer.train(faces, np.array(ids))
        recognizer.save(MODEL_PATH)
        logger.info(f"[TRAINING] Model saved to {MODEL_PATH}")
        return True, "Training selesai."
    except Exception as e:
        logger.error(f"[TRAINING] Error: {e}")
        return False, f"Error training: {e}"

def load_model_if_exists():
    global model_loaded
    if os.path.isfile(MODEL_PATH):
        try:
            recognizer.read(MODEL_PATH)
            model_loaded = True
            logger.info(f"[MODEL] Successfully loaded model from {MODEL_PATH}")
            return True
        except Exception as e:
            logger.warning(f"[MODEL] Failed to load model: {e}")
            model_loaded = False
            return False
    if FACE_ENGINE == "lbph":
        logger.info(f"[MODEL] No model file found at {MODEL_PATH}")
    return False

def retrain_after_change():
    global model_loaded
    with model_lock:
        jpgs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".jpg")]
        if not jpgs:
            if os.path.isfile(MODEL_PATH):
                try:
                    os.remove(MODEL_PATH)
                except Exception as e:
                    logger.warning(f"Gagal hapus model: {e}")
            model_loaded = False
            return True, "Semua data dihapus. Model direset."
        ok, msg = train_model_blocking()
        if ok:
            try:
                recognizer.read(MODEL_PATH)
                model_loaded = True
            except Exception as e:
                model_loaded = False
                return False, f"Model tersimpan tetapi gagal dimuat: {e}"
        return ok, msg

# Load model at startup
load_model_if_exists()

# ====== ROUTES (pages tetap) ======
@app.get("/")
def index():
    return render_template("user.html", active_page="home")

@app.get("/user/register")
def user_register():
    return render_template("user.html", active_page="daftar")

@app.get("/user/recognize")
def user_recognize():
    return render_template("user.html", active_page="verif")

@app.get("/admin/login")
def admin_login():
    return render_template("admin_login.html")

@app.post("/admin/login")
def admin_login_post():
    username = request.form.get("username", "")
    password = request.form.get("password", "")
    if username == ADMIN_USERNAME and check_password_hash(ADMIN_PASSWORD_HASH, password):
        session["admin_logged_in"] = True
        session["admin_name"] = username
        return redirect(url_for("admin_dashboard"))
    flash("Username atau password salah.", "danger")
    return redirect(url_for("admin_login"))

@app.get("/admin/logout")
def admin_logout():
    session.clear()
    return redirect(url_for("admin_login"))

@app.get("/admin")
@login_required
def admin_dashboard():
    with db_connect() as conn:
        rows = conn.execute("SELECT nik, name, dob, address, created_at FROM patients ORDER BY created_at DESC").fetchall()
        queues = conn.execute("SELECT poli_name, next_number FROM queues").fetchall()
    
    data_count = len([f for f in os.listdir(DATA_DIR) if f.lower().endswith(".jpg")])
    
    engine_info = {
        'name': FACE_ENGINE.upper(),
        'model_loaded': model_loaded
    }
    
    # --- LOGIKA BARU: Cek Hardware Status (JUJUR) ---
    hardware_status = "CPU (Optimization)"
    hardware_color = "yellow"
    
    if FACE_ENGINE == "insightface":
        try:
            status = face_engine.get_engine_status()
            engine_info['model_loaded'] = status.get('insightface_available', False)
            engine_info['embeddings_count'] = status.get('total_embeddings', 0)

            app_instance = face_engine._face_app
            is_gpu_active = False

            if app_instance:
                if hasattr(app_instance, 'models') and 'detection' in app_instance.models:
                    active_providers = app_instance.models['detection'].session.get_providers()
                    if 'CUDAExecutionProvider' in active_providers:
                        is_gpu_active = True
            
            if is_gpu_active:
                hardware_status = "GPU (NVIDIA RTX)"
                hardware_color = "green"
            else:
                hardware_status = "CPU (Optimization)"
                hardware_color = "yellow"

        except Exception as e:
            logger.error(f"Error checking status: {e}")
            hardware_status = "Error Check"
            hardware_color = "red"

    elif FACE_ENGINE == "lbph":
        hardware_status = "CPU (OpenCV)"
        hardware_color = "gray"
    
    return render_template(
        "admin_dashboard.html",
        patients=rows,
        model_loaded=engine_info['model_loaded'],
        model_name=engine_info['name'],
        foto_count=data_count,
        total_patients=len(rows),
        queues=queues,
        admin_name=session.get("admin_name", "Admin"),
        face_engine=FACE_ENGINE,
        hardware_status=hardware_status, 
        hardware_color=hardware_color    
    )

# ====== API: ENGINE STATUS ======
@app.get("/api/engine/status")
def api_engine_status():
    status = {
        'engine': FACE_ENGINE,
        'model_loaded': model_loaded
    }
    if FACE_ENGINE == "insightface":
        try:
            status.update(face_engine.get_engine_status())
        except Exception as e:
            status['error'] = str(e)
    return jsonify(ok=True, status=status)

# ====== API: PATIENTS (READ) ======
@app.get("/api/patients")
def api_patients():
    with db_connect() as conn:
        rows = conn.execute("""
            SELECT nik, name, dob, address, created_at
            FROM patients
            ORDER BY created_at DESC
        """).fetchall()
    out = []
    for r in rows:
        out.append({
            "nik": r["nik"],
            "name": r["name"],
            "dob": r["dob"],
            "address": r["address"],
            "created_at": r["created_at"],
            "age": calculate_age(r["dob"])
        })
    return jsonify(ok=True, patients=out)

@app.get("/api/patient/<int:nik>")
def api_patient_detail(nik: int):
    with db_connect() as conn:
        r = conn.execute("""
            SELECT nik, name, dob, address, created_at
            FROM patients WHERE nik = ?
        """, (nik,)).fetchone()
    if not r:
        return jsonify(ok=False, msg="Pasien tidak ditemukan."), 404
    return jsonify(ok=True, patient={
        "nik": r["nik"],
        "name": r["name"],
        "dob": r["dob"],
        "address": r["address"],
        "created_at": r["created_at"],
        "age": calculate_age(r["dob"])
    })

# ====== API: OCR KTP (Final Production: Contextual Line-by-Line) ======
@app.post("/api/ocr/ktp")
def api_ocr_ktp():
    file = request.files.get("image")
    if not file:
        return jsonify(ok=False, msg="Tidak ada gambar dikirim.")

    try:
        # 1. PREPROCESSING (Sama persis dengan test_ktp.py)
        img = bytes_to_bgr(file.read())
        
        # Auto-Rotate (Jika Portrait -> Landscape)
        if img.shape[0] > img.shape[1]: 
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Upscale
        scale = 2.0 
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # CLAHE (Ratakan Kontras)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        # Thresholding (Otsu)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # Dilation Tipis (Agar huruf tidak putus)
        kernel = np.ones((2,2), np.uint8)
        gray = cv2.erode(gray, kernel, iterations=1)

        # 2. EKSTRAKSI TEKS
        text = pytesseract.image_to_string(gray, lang='ind', config='--psm 6')
        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 2]
        full_text = " ".join(lines) # Backup untuk search global jika perlu

        data = {"nik": "", "nama": "", "dob": "", "alamat": ""}

        # --- HELPER FUNCTIONS ---
        def clean_garbage(text_val):
            # Hapus simbol aneh di awal/akhir
            text_val = re.sub(r'^[^A-Z0-9]+', '', text_val.upper())
            text_val = re.sub(r'[^A-Z0-9]+$', '', text_val)
            return text_val.strip()

        def force_alpha(text_val):
            # Paksa angka jadi huruf (0->O, 1->I, 5->S, dll)
            replacements = {'0': 'O', '1': 'I', '5': 'S', '2': 'Z', '4': 'A', '8': 'B', '6': 'G', '7': 'Z', '3': 'E'}
            text_val = text_val.upper()
            for digit, char in replacements.items():
                text_val = text_val.replace(digit, char)
            # Hapus simbol aneh, sisakan huruf, spasi, titik, koma
            return re.sub(r'[^A-Z\s\.,]', '', text_val).strip()

        # 3. PARSING BARIS DEMI BARIS (LOGIKA UTAMA)
        addr_buffer = [] # Penampung alamat

        for i, line in enumerate(lines):
            line_upper = line.upper()

            # A. NIK
            if "NIK" in line_upper or re.search(r'\d{16}', line_upper):
                digits = re.sub(r'[^0-9]', '', line_upper)
                # Prioritas Jatim (35)
                match = re.search(r'(35\d{14})', digits)
                if not match: match = re.search(r'(3\d{15})', digits) # Umum
                if not match: match = re.search(r'\d{16}', digits)    # Fallback
                
                if match: data['nik'] = match.group(0)

            # B. NAMA
            if "NAMA" in line_upper:
                raw = re.sub(r'nama\s*[:.\-]*\s*', '', line_upper, flags=re.IGNORECASE)
                clean = force_alpha(raw)
                
                # Hapus Sampah Akhir (misal " Y" atau " TG")
                clean = re.sub(r'\s+[A-Z]{1,2}$', '', clean)
                
                # Hapus kata NIK jika kebawa
                if "NIK" in clean: clean = clean.split("NIK")[1]

                if len(clean) > 2:
                    data['nama'] = clean
                elif i + 1 < len(lines):
                    # Cek baris bawah
                    potential = force_alpha(lines[i+1])
                    potential = re.sub(r'\s+[A-Z]{1,2}$', '', potential)
                    if "LAHIR" not in potential: data['nama'] = potential

            # C. ALAMAT (Jalan)
            if "ALAMAT" in line_upper:
                val = re.sub(r'alamat\s*[:.\-]*\s*', '', line_upper, flags=re.IGNORECASE)
                val = clean_garbage(val)
                
                # Hapus kata pendek di awal (TI, IL)
                words = val.split(' ')
                if len(words) > 0 and len(words[0]) <= 2: val = " ".join(words[1:])
                
                # Stop jika kena RT/RW di baris yang sama
                if "RT" in val or "RW" in val: 
                    val = re.split(r'RT|RW', val)[0]

                if len(val) > 2: addr_buffer.append(val.strip())

            # D. RT/RW (Cari Eksplisit di Baris Ini)
            if "RT" in line_upper or "RW" in line_upper:
                nums = re.findall(r'\d+', line_upper)
                if len(nums) >= 2:
                    rt = nums[0] if len(nums[0]) <= 3 else nums[0][-3:]
                    rw = nums[1] if len(nums[1]) <= 3 else nums[1][-3:]
                    addr_buffer.append(f"RT/RW {rt}/{rw}")
                elif len(nums) == 1 and len(nums[0]) > 4: # Kasus 005002 nempel
                    combined = nums[0]
                    mid = len(combined) // 2
                    addr_buffer.append(f"RT/RW {combined[:mid]}/{combined[mid:]}")

            # E. KELURAHAN
            if re.search(r'(KEL|DESA)', line_upper):
                val = re.sub(r'(KEL|DESA|/DASA|ILESA)[\.\s:]*', '', line_upper)
                val = force_alpha(val)
                # Hapus sampah
                for stop in ["KEC", "JENIS", "LAKI", "AGAMA"]:
                    if stop in val: val = val.split(stop)[0]
                
                if len(val) > 2: addr_buffer.append(f"Kel. {val}")

            # F. KECAMATAN
            if "KECAMATAN" in line_upper:
                val = line_upper.replace("KECAMATAN", "").strip()
                val = force_alpha(val)
                # Fix Typo Umum (Opsional, tapi bagus buat jaga-jaga)
                val = val.replace("DUKLUIN", "DUKUN").replace("DUKUIN", "DUKUN")
                
                for stop in ["AGAMA", "KAWIN", "STATUS"]:
                    if stop in val: val = val.split(stop)[0]
                
                # Hapus sampah akhir
                val = re.sub(r'\s+[A-Z]{1,2}$', '', val)
                
                if len(val) > 2: addr_buffer.append(f"Kec. {val}")

        # 4. FINALISASI DATA
        
        # DOB dari NIK (Paling Valid)
        if data['nik'] and len(data['nik']) == 16:
            try:
                tgl = int(data['nik'][6:8])
                bln = int(data['nik'][8:10])
                thn = int(data['nik'][10:12])
                if tgl > 40: tgl -= 40
                
                curr_y = int(datetime.now().strftime("%y"))
                full_y = 2000 + thn if thn <= curr_y else 1900 + thn
                data['dob'] = f"{full_y}-{bln:02d}-{tgl:02d}"
            except: pass

        # Gabung Alamat (Hapus Duplikat)
        if addr_buffer:
            seen = set()
            final_addr = [x for x in addr_buffer if not (x in seen or seen.add(x))]
            data['alamat'] = ", ".join(final_addr)

        # Simpan Foto
        if data['nik']:
            filename = f"{data['nik']}.jpg"
            save_path = os.path.join(KTP_DIR, filename)
            cv2.imwrite(save_path, img) # Simpan hasil rotate & upscale yang bersih
            logger.info(f"Foto KTP tersimpan: {save_path}")

        return jsonify(ok=True, data=data)

    except Exception as e:
        logger.error(f"OCR Error: {e}")
        return jsonify(ok=False, msg="Gagal proses OCR.")
    
# ====== API: REGISTER ======
@app.post("/api/register")
def api_register():
    nik_str = request.form.get("nik", "").strip()
    name = (request.form.get("nama") or request.form.get("name") or "").strip()
    dob = (request.form.get("ttl") or request.form.get("dob") or "").strip()
    address = (request.form.get("alamat") or request.form.get("address") or "").strip()

    files = request.files.getlist("files[]")
    if not files:
        files = request.files.getlist("frames[]")

    if not (nik_str and name and dob and address):
        return jsonify(ok=False, msg="Semua field wajib diisi."), 400
    try:
        nik = int(nik_str)
    except ValueError:
        return jsonify(ok=False, msg="NIK harus angka."), 400
    if not files:
        return jsonify(ok=False, msg="Tidak ada gambar dari webcam."), 400

    now_iso = datetime.now().isoformat(timespec="seconds")
    with db_connect() as conn:
        conn.execute("""
            INSERT INTO patients(nik, name, dob, address, created_at)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(nik) DO UPDATE SET name=excluded.name, dob=excluded.dob, address=excluded.address
        """, (nik, name, dob, address, now_iso))
        conn.commit()

    frames = []
    for f in files:
        try:
            img = bytes_to_bgr(f.read())
            if img is not None:
                frames.append(img)
        except Exception as e:
            logger.warning(f"Failed to process frame: {e}")

    if not frames:
        with db_connect() as conn:
            conn.execute("DELETE FROM patients WHERE nik = ?", (nik,))
            conn.commit()
        return jsonify(ok=False, msg="Tidak ada frame yang valid."), 400

    if FACE_ENGINE == "insightface":
        try:
            enrolled, msg = face_engine.enroll_multiple_frames(frames, nik, min_embeddings=5)
            if enrolled > 0:
                logger.info(f"[REGISTER] InsightFace success for NIK {nik}: {enrolled} embeddings")
                return jsonify(ok=True, msg=f"Registrasi OK (InsightFace). {enrolled} embedding berhasil disimpan.")
            else:
                logger.warning(f"[REGISTER] InsightFace returned 0 enrollments for NIK {nik}: {msg}, trying LBPH fallback")
        except Exception as e:
            logger.error(f"[REGISTER] InsightFace error: {e}")

    existing = list_existing_samples(nik)
    next_idx = existing + 1
    saved_total = 0
    
    for img in frames:
        try:
            saved = save_face_images_from_frame(img, name, nik, next_idx + saved_total)
            saved_total += saved
            if saved_total >= 20:
                break
        except Exception as e:
            logger.warning(f"Failed to save frame: {e}")

    if saved_total > 0 and saved_total < 20:
        try:
            added = ensure_min_samples(nik, 20)
            saved_total += added
        except Exception as e:
            logger.warning(f"Pad samples error: {e}")

    if saved_total == 0:
        with db_connect() as conn:
            conn.execute("DELETE FROM patients WHERE nik = ?", (nik,))
            conn.commit()
        logger.warning(f"[REGISTER] LBPH failed for NIK {nik}: No valid frames")
        return jsonify(ok=False, msg="Registrasi gagal: Tidak ada frame yang lolos validasi."), 400

    logger.info(f"[REGISTER] LBPH success for NIK {nik}: {saved_total} frames")
    ok, msg = retrain_after_change()
    return jsonify(ok=True, msg=f"Registrasi OK (LBPH). {saved_total} frame disimpan. {msg}")

# ====== API: RECOGNIZE ======
@app.post("/api/recognize")
def api_recognize():
    files = request.files.getlist("files[]")
    if not files:
        files = request.files.getlist("frames[]")

    if not files:
        return jsonify(ok=False, msg="Tidak ada gambar yang dikirim."), 400

    frames = []
    for f in files:
        try:
            img = bytes_to_bgr(f.read())
            if img is not None:
                frames.append(img)
        except Exception:
            pass

    if not frames:
        return jsonify(ok=True, found=False, msg="Tidak ada frame yang valid.")

    if FACE_ENGINE == "insightface":
        try:
            result = face_engine.recognize_face_multi_frame(frames)
            if result is not None:
                nik = result['nik']
                with db_connect() as conn:
                    row = conn.execute(
                        "SELECT nik, name, dob, address FROM patients WHERE nik = ?",
                        (nik,)
                    ).fetchone()
                
                if row:
                    age = calculate_age(row["dob"])
                    confidence = result.get('confidence', int(result['similarity'] * 100))
                    
                    logger.info(f"[RECOGNIZE] InsightFace success: NIK={nik}, sim={result['similarity']:.3f}")
                    return jsonify(
                        ok=True, found=True,
                        nik=row["nik"], name=row["name"], dob=row["dob"], address=row["address"],
                        age=age, confidence=confidence,
                        engine="insightface",
                        similarity=result['similarity']
                    )
                else:
                    logger.warning(f"[RECOGNIZE] InsightFace matched NIK {nik} but not found in patients DB, trying LBPH")
            else:
                logger.info("[RECOGNIZE] InsightFace: No match found, trying LBPH fallback")
        except Exception as e:
            logger.error(f"[RECOGNIZE] InsightFace error: {e}")

    if not model_loaded or not os.path.isfile(MODEL_PATH):
        return jsonify(ok=False, msg="Model belum tersedia. Silakan register dulu."), 400

    if recognizer is None:
        return jsonify(ok=False, msg="LBPH recognizer tidak tersedia."), 500

    from collections import defaultdict, Counter
    votes = defaultdict(list)
    processed = 0
    best_nik, best_avg = None, 99999.0

    for img in frames:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi_raw, rect = detect_largest_face(gray)
            
            if roi_raw is None or is_blurry(roi_raw, 25.0):
                continue

            roi = preprocess_roi(roi_raw)
            Id_pred, conf = recognizer.predict(roi)
            votes[int(Id_pred)].append(float(conf))
            processed += 1

            for nk, cfs in votes.items():
                avg = sum(cfs) / len(cfs)
                if avg < best_avg:
                    best_avg = avg
                    best_nik = nk
            if best_nik is not None:
                share = len(votes[best_nik]) / max(1, processed)
                if share >= VOTE_MIN_SHARE and len(votes[best_nik]) >= EARLY_VOTES_REQUIRED and best_avg <= EARLY_CONF_THRESHOLD:
                    break

        except Exception as e:
            logger.warning(f"LBPH predict error: {e}")

    if processed == 0 or not votes:
        return jsonify(ok=True, found=False, msg="Tidak ada wajah terdeteksi.")

    all_preds = [(nk, c) for nk, lst in votes.items() for c in lst]
    major = Counter([nk for nk, _ in all_preds]).most_common(1)[0][0]
    confs_for_major = votes.get(major, [])
    
    if not confs_for_major:
        return jsonify(ok=True, found=False, msg="Tidak dikenali.")

    vote_share = len(confs_for_major) / processed
    median_conf = float(np.median(confs_for_major))

    if vote_share < VOTE_MIN_SHARE or len(confs_for_major) < MIN_VALID_FRAMES or median_conf >= LBPH_CONF_THRESHOLD:
        return jsonify(ok=True, found=False, msg="Tidak dikenali.")

    with db_connect() as conn:
        row = conn.execute(
            "SELECT nik, name, dob, address FROM patients WHERE nik = ?",
            (major,)
        ).fetchone()

    if not row:
        return jsonify(ok=True, found=False, msg="Tidak dikenali.")

    confidence_percent = int(max(0, min(100, 100 - median_conf)))
    age = calculate_age(row["dob"])
    
    logger.info(f"[RECOGNIZE] LBPH success: NIK={major}, conf={median_conf:.2f}")
    return jsonify(
        ok=True, found=True,
        nik=row["nik"], name=row["name"], dob=row["dob"], address=row["address"],
        age=age, confidence=confidence_percent,
        engine="lbph"
    )

# ====== API: QUEUE ======
@app.post("/api/queue/assign")
def api_queue_assign():
    data = request.json if request.is_json else {}
    poli = (data.get("poli") or "").strip()
    if poli not in ["Poli Umum", "Poli Gigi", "IGD"]:
        return jsonify(ok=False, msg="Poli tidak valid."), 400
    with db_connect() as conn:
        row = conn.execute("SELECT next_number FROM queues WHERE poli_name=?", (poli,)).fetchone()
        if not row:
            return jsonify(ok=False, msg="Poli tidak ditemukan."), 404
        last_number = row["next_number"]
        nomor = last_number + 1
        conn.execute("UPDATE queues SET next_number=? WHERE poli_name=?", (nomor, poli))
        conn.commit()
    return jsonify(ok=True, poli=poli, nomor=nomor)

@app.post("/api/queue/set")
@login_required
def api_queue_set():
    data = request.json if request.is_json else {}
    poli = (data.get("poli") or "").strip()
    nomor = data.get("nomor")
    if poli not in ["Poli Umum", "Poli Gigi", "IGD"]:
        return jsonify(ok=False, msg="Poli tidak valid."), 400
    try:
        n = int(nomor)
        if n < 0: raise ValueError
    except:
        return jsonify(ok=False, msg="Nomor harus >= 0."), 400
    with db_connect() as conn:
        conn.execute("UPDATE queues SET next_number=? WHERE poli_name=?", (n, poli))
        conn.commit()
    return jsonify(ok=True, msg=f"Nomor terakhir {poli} di-set ke {n}.")

# ====== ADMIN: RETRAIN / DELETE ======
@app.post("/admin/retrain")
@login_required
def admin_retrain():
    ok, msg = retrain_after_change()
    flash(("Retrain sukses." if ok else f"Retrain gagal: {msg}"), "success" if ok else "danger")
    return redirect(url_for("admin_dashboard"))

@app.post("/admin/patient/<int:nik>/delete")
@login_required
def admin_delete_patient(nik: int):
    with db_connect() as conn:
        conn.execute("DELETE FROM patients WHERE nik = ?", (nik,))
        conn.commit()
    
    removed = 0
    for path in glob.glob(os.path.join(DATA_DIR, f"{nik}.*.jpg")):
        try:
            os.remove(path)
            removed += 1
        except Exception as e:
            logger.warning(f"Failed to delete file {path}: {e}")
    
    if FACE_ENGINE == "insightface":
        try:
            deleted_emb = face_engine.delete_embeddings_for_nik(nik)
            logger.info(f"Deleted {deleted_emb} embeddings for NIK {nik}")
        except Exception as e:
            logger.warning(f"Failed to delete embeddings: {e}")
    
    ok, msg = retrain_after_change()
    flash(f"Hapus NIK {nik}: {removed} file dihapus. {msg}", "success" if ok else "danger")
    return redirect(url_for("admin_dashboard"))

@app.post("/admin/patient/update")
@login_required
def admin_update_patient():
    try:
        old_nik_str = request.form.get("old_nik", "").strip()
        nik_str = request.form.get("nik", "").strip()
        dob = request.form.get("dob", "").strip()
        address = request.form.get("address", "").strip()

        if not all([old_nik_str, nik_str, dob, address]):
            return jsonify(ok=False, msg="Semua field wajib diisi."), 400

        old_nik = int(old_nik_str)
        nik = int(nik_str)

        with db_connect() as conn:
            if nik != old_nik and conn.execute("SELECT 1 FROM patients WHERE nik = ?", (nik,)).fetchone():
                return jsonify(ok=False, msg=f"NIK {nik} sudah terdaftar untuk pasien lain."), 409

            conn.execute("""
                UPDATE patients SET nik=?, dob=?, address=? WHERE nik=?
            """, (nik, dob, address, old_nik))
            conn.commit()

        if nik != old_nik:
            renamed_count = 0
            pattern = os.path.join(DATA_DIR, f"{old_nik}.*.jpg")
            for old_path in glob.glob(pattern):
                fname = os.path.basename(old_path)
                parts = fname.split('.')
                if len(parts) >= 3:
                    new_fname = f"{nik}.{'.'.join(parts[1:])}"
                    new_path = os.path.join(DATA_DIR, new_fname)
                    try:
                        os.rename(old_path, new_path)
                        renamed_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to rename {old_path}: {e}")
            
            if FACE_ENGINE == "insightface":
                try:
                    updated_emb = face_engine.update_nik_in_embeddings(old_nik, nik)
                    logger.info(f"Updated {updated_emb} embeddings from NIK {old_nik} to {nik}")
                except Exception as e:
                    logger.warning(f"Failed to update embeddings: {e}")
            
            ok_retrain, msg_retrain = retrain_after_change()
            msg_rename = f"{renamed_count} file gambar di-rename. {msg_retrain}"
            if not ok_retrain:
                return jsonify(ok=False, msg=f"Data diupdate tapi retrain gagal: {msg_retrain}"), 500
        else:
            msg_rename = "NIK tidak berubah, tidak perlu retrain."

        return jsonify(ok=True, msg=f"Data pasien NIK {old_nik} berhasil diupdate. {msg_rename}")

    except ValueError:
        return jsonify(ok=False, msg="NIK harus berupa angka."), 400
    except Exception as e:
        logger.error(f"Error update patient: {e}")
        return jsonify(ok=False, msg=f"Terjadi error di server: {e}"), 500

# ====== API BARU: CHECK FACE ======
@app.post("/api/check_face")
def api_check_face():
    """
    API ringan untuk mengecek apakah ada wajah di frame.
    """
    file = request.files.get("frame")
    if not file:
        return jsonify(ok=False, found=False)
    try:
        img = bytes_to_bgr(file.read())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        roi, rect = detect_largest_face(gray)
        found = roi is not None
        return jsonify(ok=True, found=found)
    except Exception:
        return jsonify(ok=False, found=False)
    
# ====== API BARU: CEK KEBERADAAN KTP (Fast Check) ======
@app.post("/api/check_ktp_presence")
def api_check_ktp_presence():
    """
    Mendeteksi apakah ada objek menyerupai KTP (Kotak & Biru Dominan).
    Ringan & Cepat untuk auto-trigger.
    """
    file = request.files.get("frame")
    if not file:
        return jsonify(ok=False, found=False)
    
    try:
        # 1. Baca Gambar
        img = bytes_to_bgr(file.read())
        
        # 2. Resize kecil biar ngebut prosesnya
        small = cv2.resize(img, (320, 240))
        
        # 3. Deteksi Warna Biru KTP (HSV)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        
        # Rentang warna diperluas agar KTP yang agak gelap/terang tetap kena
        lower_blue = np.array([70, 30, 30])
        upper_blue = np.array([140, 255, 255])
        
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # --- FITUR DEBUG (FOTO TERSIMPAN DI FOLDER WEB-FACE) ---
        # Hapus tanda '#' di bawah ini untuk mengaktifkan simpan foto
        cv2.imwrite("debug_ktp_source.jpg", small)
        cv2.imwrite("debug_ktp_mask.jpg", mask)
        # -------------------------------------------------------

        # Bersihkan noise dikit (Morphology)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # 4. Cari Kontur di area biru
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        found = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Minimal luas area tertentu (biar ga deteksi noise kecil)
            if area > 800: 
                # Cek rasio aspek (KTP itu persegi panjang)
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                
                # Rasio KTP biasanya lebar (sekitar 1.5 - 1.6), kita kasih toleransi
                if 1.1 < aspect_ratio < 2.5:
                    found = True
                    break
        
        return jsonify(ok=True, found=found)
        
    except Exception as e:
        # Silent error biar ga menuhin log
        return jsonify(ok=False, found=False)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)