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
import subprocess
import json
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import Output  # Penting untuk fitur kotak hijau

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
MODEL_DIR = os.path.join(BASE_DIR, "model")
DB_PATH = os.path.join(BASE_DIR, "database.db")
MODEL_PATH = os.path.join(MODEL_DIR, "Trainer.yml")

# Konfigurasi Folder KTP (4 Variasi)
DATA_KTP_DIR = os.path.join(BASE_DIR, "data", "database_ktp")
DIR_RGB = os.path.join(DATA_KTP_DIR, "source_rgb")       # Foto Asli Cropped
DIR_NEG = os.path.join(DATA_KTP_DIR, "negative")         # Foto Negatif
DIR_GRAY = os.path.join(DATA_KTP_DIR, "grayscale")       # Foto Siap OCR
DIR_BOX = os.path.join(DATA_KTP_DIR, "debug_box")        # Foto dengan Kotak Hijau

# Buat semua folder jika belum ada
for d in [DATA_DIR, MODEL_DIR, DIR_RGB, DIR_NEG, DIR_GRAY, DIR_BOX]:
    os.makedirs(d, exist_ok=True)

# ====== KONFIGURASI TESSERACT (OCR) ======
path_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(path_tesseract):
    pytesseract.pytesseract.tesseract_cmd = path_tesseract
else:
    logger.warning("WARNING: Tesseract OCR tidak ditemukan. Fitur scan KTP mungkin gagal.")

# ====== FLASK APP ======
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

# ====== FACE ENGINE SELECTION ======
USE_INSIGHTFACE = os.environ.get("USE_INSIGHTFACE", "1") == "1"
FACE_ENGINE = None

try:
    if USE_INSIGHTFACE:
        import face_engine
        FACE_ENGINE = "insightface"
        logger.info("Using InsightFace engine for face recognition")
except ImportError as e:
    logger.warning(f"InsightFace fallback: {e}")
    FACE_ENGINE = "lbph"
except Exception as e:
    logger.warning(f"InsightFace init failed: {e}")
    FACE_ENGINE = "lbph"

if FACE_ENGINE is None:
    FACE_ENGINE = "lbph"

# ====== LBPH CONSTANTS ======
MIN_VALID_FRAMES = 3
LBPH_CONF_THRESHOLD = 50  # Confidence threshold untuk recognition
model_lock = threading.Lock()

# ====== ADMIN CREDENTIALS ======
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
_default_plain = os.environ.get("ADMIN_PASSWORD_PLAIN", "Cakra@123")
ADMIN_PASSWORD_HASH = os.environ.get("ADMIN_PASSWORD_HASH", generate_password_hash(_default_plain))

# ====== CLOUDFLARED TUNNEL MANAGEMENT ======
CLOUDFLARED_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cloudflared.exe")
TUNNEL_STATUS_FILE = os.path.join(BASE_DIR, "tunnel_status.json")

# Global tunnel state
tunnel_state = {
    "status": "stopped",  # stopped, starting, running, error
    "url": "",
    "process": None,
    "error": "",
    "started_at": None
}

def save_tunnel_status():
    """Save tunnel status to file"""
    status_data = {
        "status": tunnel_state["status"],
        "url": tunnel_state["url"],
        "error": tunnel_state["error"],
        "started_at": tunnel_state["started_at"],
        "last_updated": datetime.now().isoformat()
    }
    try:
        with open(TUNNEL_STATUS_FILE, "w") as f:
            json.dump(status_data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save tunnel status: {e}")

def load_tunnel_status():
    """Load tunnel status from file"""
    try:
        if os.path.exists(TUNNEL_STATUS_FILE):
            with open(TUNNEL_STATUS_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load tunnel status: {e}")
    return {
        "status": "stopped",
        "url": "",
        "error": "",
        "started_at": None,
        "last_updated": None
    }

def start_cloudflared_tunnel():
    """Start cloudflared tunnel in background"""
    global tunnel_state
    
    if not os.path.exists(CLOUDFLARED_PATH):
        tunnel_state["status"] = "error"
        tunnel_state["error"] = f"cloudflared.exe tidak ditemukan di: {CLOUDFLARED_PATH}"
        save_tunnel_status()
        return False
    
    if tunnel_state["status"] == "running" or tunnel_state["status"] == "starting":
        return True  # Already running
    
    try:
        tunnel_state["status"] = "starting"
        tunnel_state["error"] = ""
        save_tunnel_status()
        
        # Start cloudflared process
        cmd = [CLOUDFLARED_PATH, "tunnel", "--url", "http://127.0.0.1:5000"]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        tunnel_state["process"] = process
        tunnel_state["status"] = "running"
        tunnel_state["started_at"] = datetime.now().isoformat()
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_tunnel_process, args=(process,))
        monitor_thread.daemon = True
        monitor_thread.start()
        
        logger.info("Cloudflared tunnel started")
        save_tunnel_status()
        return True
        
    except Exception as e:
        tunnel_state["status"] = "error"
        tunnel_state["error"] = str(e)
        save_tunnel_status()
        logger.error(f"Failed to start cloudflared tunnel: {e}")
        return False

def stop_cloudflared_tunnel():
    """Stop cloudflared tunnel"""
    global tunnel_state
    
    if tunnel_state["process"]:
        try:
            tunnel_state["process"].terminate()
            tunnel_state["process"] = None
        except:
            pass
    
    tunnel_state["status"] = "stopped"
    tunnel_state["url"] = ""
    tunnel_state["error"] = ""
    tunnel_state["started_at"] = None
    save_tunnel_status()
    logger.info("Cloudflared tunnel stopped")

def monitor_tunnel_process(process):
    """Monitor cloudflared process and extract tunnel URL"""
    global tunnel_state
    
    url_found = False
    try:
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            
            line = line.strip()
            logger.info(f"Cloudflared: {line}")
            
            # Extract tunnel URL from output
            if "trycloudflare.com" in line and not url_found:
                # Extract URL from cloudflared output
                import re
                url_match = re.search(r'https://[^\s]+\.trycloudflare\.com', line)
                if url_match:
                    tunnel_state["url"] = url_match.group(0)
                    url_found = True
                    save_tunnel_status()
                    logger.info(f"Tunnel URL: {tunnel_state['url']}")
            
            # Check if process is still running
            if process.poll() is not None:
                break
                
    except Exception as e:
        logger.error(f"Error monitoring tunnel process: {e}")
    finally:
        if process.poll() is not None:
            # Process ended
            tunnel_state["status"] = "stopped"
            tunnel_state["url"] = ""
            save_tunnel_status()
            logger.info("Cloudflared tunnel process ended")

def login_required(view_func):
    def wrapper(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin_login"))
        return view_func(*args, **kwargs)
    wrapper.__name__ = view_func.__name__
    return wrapper

# ====== HELPER FUNCTIONS (OCR & IMAGE PROCESSING) ======

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def smart_crop_card(image):
    try:
        orig = image.copy()
        ratio = image.shape[0] / 500.0
        h = 500
        w = int(image.shape[1] / ratio)
        small = cv2.resize(image, (w, h))

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        edged = cv2.Canny(gray, 75, 200)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        screenCnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                total_area = w * h
                contour_area = cv2.contourArea(c)
                if contour_area > (total_area * 0.10): 
                    screenCnt = approx
                    break 

        if screenCnt is None:
            return image

        pts = screenCnt.reshape(4, 2) * ratio
        rect = order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    except:
        return image

def clean_name(val):
    val = re.sub(r'^[^A-Z0-9]+', '', val.upper())
    val = re.sub(r'[^A-Z]+$', '', val)
    val = re.sub(r'\s+[A-Z]{1,2}\s*\d*$', '', val)
    val = val.replace('1', 'I').replace('7', 'Z')
    if "NIK" in val: val = val.split("NIK")[1]
    return val.strip()

def scrub_address(addr_text):
    stop_words = ["LAKI", "PEREMPUAN", "GOL", "DARAH", "AGAMA", "KECAMATAN", "KWN", "KAWIN", "JENIS", "KELAMIN", "AMIN"]
    addr_upper = addr_text.upper()
    for sw in stop_words:
        if sw in addr_upper:
            addr_text = addr_text[:addr_upper.index(sw)]
            addr_upper = addr_text.upper()
    return re.sub(r'[^A-Z0-9\s\.\/]', '', addr_text).strip()

def clean_garbage(text_val):
    text_val = re.sub(r'^[^A-Z0-9]+', '', text_val.upper())
    text_val = re.sub(r'[^A-Z0-9]+$', '', text_val)
    return text_val.strip()

def force_alpha(text):
    replacements = {'0': 'O', '1': 'I', '5': 'S', '2': 'Z', '4': 'A', '8': 'B', '6': 'G', '7': 'Z', '3': 'E'}
    text = text.upper()
    for digit, char in replacements.items():
        text = text.replace(digit, char)
    return re.sub(r'[^A-Z\s\.,]', '', text).strip()

# ====== DB INIT ======
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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scan_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL, -- 'success' atau 'failed'
                ip_address TEXT,
                nik TEXT, -- untuk success: NIK user, untuk failed: '-'
                name TEXT, -- untuk success: nama user, untuk failed: '-'
                dob TEXT, -- untuk success: tanggal lahir, untuk failed: '-'
                address TEXT, -- untuk success: alamat, untuk failed: '-'
                age TEXT, -- untuk success: umur, untuk failed: '-'
                message TEXT -- detail pesan (opsional)
            )
        """)
        c = conn.execute("SELECT COUNT(*) AS c FROM queues").fetchone()
        if c["c"] == 0:
            for poli in ["Poli Umum", "Poli Gigi", "IGD"]:
                conn.execute("INSERT INTO queues(poli_name, next_number) VALUES(?, ?)", (poli, 0))
        conn.commit()
db_init()

# ====== UTIL FUNGSI LAINNYA ======
def calculate_age(dob_str: str) -> str:
    try:
        for fmt in ["%Y-%m-%d", "%d-%m-%Y"]:
            try:
                dt = datetime.strptime(dob_str, fmt)
                today = datetime.now()
                age = today.year - dt.year - ((today.month, today.day) < (dt.month, dt.day))
                return f"{age} Tahun"
            except: continue
        return "N/A"
    except: return "N/A"

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

# ====== DETECTOR FACE (FALLBACK CPU) ======
detectors = []
for fname in ["haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"]:
    path = os.path.join(cv2.data.haarcascades, fname)
    if os.path.isfile(path):
        detectors.append(cv2.CascadeClassifier(path))

def detect_largest_face(gray):
    best_roi, best_rect, best_area = None, None, -1
    for det in detectors:
        faces = det.detectMultiScale(gray, 1.1, 3, minSize=(60, 60))
        if len(faces) == 0: continue
        (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
        area = w * h
        if area > best_area:
            best_area = area
            best_rect = (x, y, w, h)
            best_roi = gray[y:y+h, x:x+w]
    return best_roi, best_rect

def save_face_images_from_frame(img_bgr, name: str, nik: int, idx: int) -> int:
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        crop, _ = detect_largest_face(gray)
        if crop is None or is_blurry(crop, 40.0): return 0
        
        preprocessed = preprocess_roi(crop)
        out_path = os.path.join(DATA_DIR, f"{nik}.{idx}.jpg")
        cv2.imwrite(out_path, preprocessed)
        return 1
    except: return 0

def ensure_min_samples(nik: int, min_count: int = 20) -> int:
    pattern = os.path.join(DATA_DIR, f"{nik}.*.jpg")
    files = sorted(glob.glob(pattern), key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split(".")[1]))
    saved = len(files)
    if saved == 0: return 0

    next_idx = int(os.path.splitext(os.path.basename(files[-1]))[0].split(".")[1]) + 1
    added = 0
    src_imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in files if os.path.isfile(p)]
    src_imgs = [im for im in src_imgs if im is not None and im.size > 0]
    
    if not src_imgs: return 0
    i = 0
    while saved + added < min_count:
        base = src_imgs[i % len(src_imgs)].copy()
        out = cv2.convertScaleAbs(base, alpha=1.05, beta=5) # simple augment
        out_path = os.path.join(DATA_DIR, f"{nik}.{next_idx}.jpg")
        cv2.imwrite(out_path, out)
        next_idx += 1
        added += 1
        i += 1
    return added

# ====== LBPH LOGIC ======
recognizer = None
if hasattr(cv2, "face"):
    recognizer = cv2.face.LBPHFaceRecognizer_create(1, 8, 8, 8)

def load_model_if_exists():
    global model_loaded
    if os.path.isfile(MODEL_PATH):
        try:
            recognizer.read(MODEL_PATH)
            model_loaded = True
            logger.info(f"Model LBPH loaded from {MODEL_PATH}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load LBPH model: {e}")
    model_loaded = False
    return False

def retrain_after_change():
    global model_loaded
    with model_lock:
        faces, ids = [], []
        for fname in os.listdir(DATA_DIR):
            if not fname.endswith(".jpg"): continue
            try:
                nik = int(fname.split(".")[0])
                img = Image.open(os.path.join(DATA_DIR, fname)).convert("L")
                faces.append(np.array(img, "uint8"))
                ids.append(nik)
            except: pass
        
        if not faces:
            if os.path.isfile(MODEL_PATH): os.remove(MODEL_PATH)
            model_loaded = False
            return True, "Data kosong, model direset."
        
        try:
            recognizer.train(faces, np.array(ids))
            recognizer.save(MODEL_PATH)
            model_loaded = True
            return True, "Training selesai."
        except Exception as e:
            model_loaded = False
            return False, str(e)

load_model_if_exists()

# ====== LOGGING FUNCTIONS ======
def log_scan_result(status: str, nik: str = None, name: str = None, dob: str = None, address: str = None, age: str = None, message: str = None):
    """
    Mencatat hasil scan/verifikasi ke database scan_logs
    status: 'success' atau 'failed'
    """
    try:
        # PERBAIKAN: Baca IP Asli dari Header Tunnel (Cloudflare/Ngrok)
        if request.headers.getlist("X-Forwarded-For"):
            ip_address = request.headers.getlist("X-Forwarded-For")[0]
        else:
            ip_address = request.remote_addr if request else None
            
        timestamp = datetime.now().isoformat(timespec="seconds")
        
        with db_connect() as conn:
            conn.execute(
                """
                INSERT INTO scan_logs
                (timestamp, status, ip_address, nik, name, dob, address, age, message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (timestamp, status, ip_address, nik, name, dob, address, age, message)
            )
            conn.commit()
        logger.info(f"Log recorded: {status} | IP: {ip_address} | NIK: {nik} | Msg: {message}")
    except Exception as e:
        logger.error(f"Failed to log scan result: {e}")

# ====== ROUTES ======
@app.get("/")
def index(): return render_template("user.html", active_page="home")

@app.get("/user/register")
def user_register(): return render_template("user.html", active_page="daftar")

@app.get("/user/recognize")
def user_recognize(): return render_template("user.html", active_page="verif")

@app.get("/admin/login")
def admin_login(): return render_template("admin_login.html")

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
    
    engine_info = {'name': FACE_ENGINE.upper(), 'model_loaded': model_loaded}
    
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
                hardware_status = "GPU (NVIDIA RTX 3050)"
                hardware_color = "green"
        except:
            hardware_status = "Error Check"
            hardware_color = "red"
    elif FACE_ENGINE == "lbph":
        hardware_status = "CPU (11th Gen Intel Core i7-11600H)"
        hardware_color = "gray"
    
    return render_template("admin_dashboard.html", patients=rows, model_loaded=engine_info['model_loaded'],
                           model_name=engine_info['name'], foto_count=data_count, total_patients=len(rows),
                           queues=queues, admin_name=session.get("admin_name", "Admin"),
                           face_engine=FACE_ENGINE, hardware_status=hardware_status, hardware_color=hardware_color)

# ====== API: STATUS & CRUD ======
@app.get("/api/engine/status")
def api_engine_status():
    status = {'engine': FACE_ENGINE, 'model_loaded': model_loaded}
    if FACE_ENGINE == "insightface":
        try: status.update(face_engine.get_engine_status())
        except Exception as e: status['error'] = str(e)
    return jsonify(ok=True, status=status)

# ====== API: CLOUDFLARED TUNNEL MANAGEMENT ======
@app.get("/api/tunnel/status")
def api_tunnel_status():
    """Get current tunnel status"""
    current_status = load_tunnel_status()
    return jsonify(ok=True, tunnel=current_status)

@app.post("/api/tunnel/start")
@login_required
def api_tunnel_start():
    """Start cloudflared tunnel"""
    if start_cloudflared_tunnel():
        return jsonify(ok=True, msg="Tunnel berhasil dimulai")
    else:
        return jsonify(ok=False, msg=tunnel_state.get("error", "Gagal memulai tunnel"))

@app.post("/api/tunnel/stop")
@login_required
def api_tunnel_stop():
    """Stop cloudflared tunnel"""
    stop_cloudflared_tunnel()
    return jsonify(ok=True, msg="Tunnel berhasil dihentikan")

@app.get("/api/patients")
def api_patients():
    with db_connect() as conn:
        rows = conn.execute("SELECT nik, name, dob, address, created_at FROM patients ORDER BY created_at DESC").fetchall()
    return jsonify(ok=True, patients=[{k: r[k] for k in r.keys()} | {"age": calculate_age(r["dob"])} for r in rows])

@app.get("/api/patient/<int:nik>")
def api_patient_detail(nik: int):
    with db_connect() as conn:
        r = conn.execute("SELECT * FROM patients WHERE nik = ?", (nik,)).fetchone()
    if not r: return jsonify(ok=False, msg="Pasien tidak ditemukan."), 404
    return jsonify(ok=True, patient={k: r[k] for k in r.keys()} | {"age": calculate_age(r["dob"])})

@app.get("/api/scan_logs")
def api_scan_logs():
    """Get scan logs with limit"""
    try:
        limit = request.args.get('limit', 100, type=int)
        with db_connect() as conn:
            rows = conn.execute(
                "SELECT timestamp, status, ip_address, nik, name, dob, address, age, message FROM scan_logs ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
        logs = [dict(r) for r in rows]
        return jsonify(ok=True, logs=logs)
    except Exception as e:
        logger.error(f"Error fetching scan logs: {e}")
        return jsonify(ok=False, msg=str(e)), 500

# ====== API: SCAN KTP (PIPELINE NEGATIVE -> GRAYSCALE) ======
@app.post("/api/ocr/ktp")
def api_ocr_ktp():
    file = request.files.get("image")
    if not file: return jsonify(ok=False, msg="Tidak ada gambar.")

    try:
        # 1. READ & CROP
        img = bytes_to_bgr(file.read())
        img_cropped = smart_crop_card(img)

        # 2. ROTATE & SCALE
        if img_cropped.shape[0] > img_cropped.shape[1]:
            img_cropped = cv2.rotate(img_cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        scale = 2.0 
        img_rgb = cv2.resize(img_cropped, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # 3. PIPELINE BARU (RGB -> NEGATIVE -> GRAYSCALE)
        img_negative = cv2.bitwise_not(img_rgb)
        img_gray = cv2.cvtColor(img_negative, cv2.COLOR_BGR2GRAY)

        # 4. ENHANCEMENT & BINARIZATION
        gray_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        kernel = np.ones((2,2), np.uint8)
        gray_final = cv2.erode(gray_binary, kernel, iterations=1)

        # 5. OCR & VISUALISASI KOTAK (DEBUG)
        d = pytesseract.image_to_data(gray_final, lang='ind', output_type=Output.DICT, config='--psm 6')
        
        img_box = img_rgb.copy() # Canvas box di atas gambar asli
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if int(d['conf'][i]) > 40 and d['text'][i].strip() != "":
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                cv2.rectangle(img_box, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 6. PARSING DATA
        full_text_raw = pytesseract.image_to_string(gray_final, lang='ind', config='--psm 6')
        logger.info("\n--- OCR RAW RESULT ---\n" + full_text_raw + "\n----------------------")

        lines = [line.strip() for line in full_text_raw.split('\n') if len(line.strip()) > 2]
        full_text_oneline = " ".join(lines)
        data = {"nik": "", "nama": "", "dob": "", "alamat": ""}

        # Logic NIK
        digits = re.sub(r'[^0-9]', '', full_text_oneline)
        match_jatim = re.search(r'35\d{14}', digits)
        match_general = re.search(r'3\d{15}', digits)
        if match_jatim: data['nik'] = match_jatim.group(0)
        elif match_general: data['nik'] = match_general.group(0)
        else:
            match_any = re.search(r'\d{16}', digits)
            if match_any: data['nik'] = match_any.group(0)

        # Logic Tanggal (Dari NIK)
        if data['nik'] and len(data['nik']) == 16:
            try:
                tgl, bln, thn = int(data['nik'][6:8]), int(data['nik'][8:10]), int(data['nik'][10:12])
                if tgl > 40: tgl -= 40
                curr_y = int(datetime.now().strftime("%y"))
                full_y = 2000 + thn if thn <= curr_y else 1900 + thn
                data['dob'] = f"{full_y}-{bln:02d}-{tgl:02d}"
            except: pass

        # Logic Nama
        for i, line in enumerate(lines):
            if "NAMA" in line.upper():
                raw = re.sub(r'nama\s*[:.\-]*\s*', '', line.lower())
                clean = clean_name(raw)
                clean = re.sub(r'[_—\-:;&]', '', clean)
                clean = re.sub(r'\bBAN\b', '', clean)
                for w in ["LAKI", "PEREMPUAN", "GOL", "DARAH"]:
                    if w in clean: clean = clean.split(w)[0]
                if len(clean) > 2: data['nama'] = clean.strip()
                elif i + 1 < len(lines):
                    pot = lines[i+1].strip().upper()
                    if "LAHIR" not in pot: data['nama'] = clean_name(pot)
                break

        # Logic Alamat
        addr_buffer = []
        for i, line in enumerate(lines):
            if "ALAMAT" in line.upper():
                val = re.sub(r'alamat\s*[:.\-]*\s*', '', line.lower()).upper()
                val = clean_garbage(val)
                if len(val.split(' ')[0]) <= 2: val = " ".join(val.split(' ')[1:])
                if "RT" in val: val = val.split("RT")[0]
                val = scrub_address(val)
                if len(val) > 2: addr_buffer.append(val.strip())
                break
        
        rtrw_match = re.search(r'(\d{3})\s*[\/|1l7]\s*(\d{3})', full_text_oneline)
        if rtrw_match: addr_buffer.append(f"RT/RW {rtrw_match.group(1)}/{rtrw_match.group(2)}")
        
        for line in lines:
            up = line.upper()
            if re.search(r'(KEL|DESA)', up):
                val = re.sub(r'(KEL|DESA|/DASA|ILESA)[\.\s:]*', '', up)
                val = scrub_address(val)
                if "KEC" in val: val = val.split("KEC")[0]
                val = force_alpha(val)
                if len(val) > 2: addr_buffer.append(f"Kel. {val}")
            if "KECAMATAN" in up:
                val = up.replace("KECAMATAN", "").strip()
                val = scrub_address(val)
                val = val.replace("DUKLUIN", "DUKUN").replace("DUKUIN", "DUKUN")
                val = force_alpha(val)
                val = re.sub(r'\s+[A-Z]{1,2}$', '', val)
                if len(val) > 2: addr_buffer.append(f"Kec. {val}")

        if addr_buffer:
            seen = set()
            data['alamat'] = ", ".join([x for x in addr_buffer if not (x in seen or seen.add(x))])

        # --- SIMPAN HASIL KE 4 FOLDER ---
        filename_base = data['nik'] if data['nik'] else "unknown_" + datetime.now().strftime("%H%M%S")
        
        cv2.imwrite(os.path.join(DIR_RGB, f"{filename_base}.jpg"), img_rgb)
        cv2.imwrite(os.path.join(DIR_NEG, f"{filename_base}.jpg"), img_negative)
        cv2.imwrite(os.path.join(DIR_GRAY, f"{filename_base}.jpg"), gray_final)
        cv2.imwrite(os.path.join(DIR_BOX, f"{filename_base}.jpg"), img_box) # Box visual

        return jsonify(ok=True, data=data)

    except Exception as e:
        logger.error(f"OCR Error: {e}")
        return jsonify(ok=False, msg="Gagal proses OCR.")

# ====== API: CHECK KTP (Fast) ======
@app.post("/api/check_ktp_presence")
def api_check_ktp_presence():
    file = request.files.get("frame")
    if not file: return jsonify(ok=False, found=False)
    try:
        img = bytes_to_bgr(file.read())
        small = cv2.resize(img, (320, 240))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([70, 30, 30]), np.array([140, 255, 255]))
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 800:
                x,y,w,h = cv2.boundingRect(cnt)
                if 1.1 < (float(w)/h) < 2.5: return jsonify(ok=True, found=True)
        return jsonify(ok=True, found=False)
    except: return jsonify(ok=False, found=False)

# ====== API: CHECK FACE (Fast) ======
@app.post("/api/check_face")
def api_check_face():
    file = request.files.get("frame")
    if not file: return jsonify(ok=False, found=False)
    try:
        img = bytes_to_bgr(file.read())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        roi, _ = detect_largest_face(gray)
        return jsonify(ok=True, found=(roi is not None))
    except: return jsonify(ok=False, found=False)

# ====== REGISTRASI ======
@app.post("/api/register")
def api_register():
    nik_str = request.form.get("nik", "").strip()
    name = (request.form.get("nama") or request.form.get("name") or "").strip()
    dob = (request.form.get("ttl") or request.form.get("dob") or "").strip()
    address = (request.form.get("alamat") or request.form.get("address") or "").strip()
    files = request.files.getlist("files[]") if request.files.getlist("files[]") else request.files.getlist("frames[]")

    if not (nik_str and name and dob and address): return jsonify(ok=False, msg="Isi semua data."), 400
    if not files: return jsonify(ok=False, msg="Tidak ada foto wajah."), 400
    
    try: nik = int(nik_str)
    except: return jsonify(ok=False, msg="NIK harus angka."), 400

    now_iso = datetime.now().isoformat(timespec="seconds")
    with db_connect() as conn:
        conn.execute("INSERT INTO patients(nik, name, dob, address, created_at) VALUES(?, ?, ?, ?, ?) ON CONFLICT(nik) DO UPDATE SET name=excluded.name, dob=excluded.dob, address=excluded.address", (nik, name, dob, address, now_iso))
        conn.commit()

    frames = [bytes_to_bgr(f.read()) for f in files]
    
    if FACE_ENGINE == "insightface":
        try:
            enrolled, msg = face_engine.enroll_multiple_frames(frames, nik, min_embeddings=5)
            if enrolled > 0:
                # Log successful InsightFace registration
                log_scan_result(
                    status="success",
                    nik=str(nik),
                    name=name,
                    dob=dob,
                    address=address,
                    age=calculate_age(dob),
                    message=f"Register OK (InsightFace). {enrolled} foto."
                )
                return jsonify(ok=True, msg=f"Register OK (InsightFace). {enrolled} foto.")
        except Exception as e:
            logger.error(f"InsightFace Error: {e}")
            # Log failed InsightFace registration
            log_scan_result(
                status="failed",
                nik=str(nik),
                name=name,
                dob=dob,
                address=address,
                age=calculate_age(dob),
                message=f"InsightFace Error: {str(e)}"
            )

    # LBPH Fallback
    next_idx = list_existing_samples(nik) + 1
    saved = 0
    for img in frames:
        saved += save_face_images_from_frame(img, name, nik, next_idx + saved)
        if saved >= 20: break
    
    if saved < 20: saved += ensure_min_samples(nik, 20)
    
    if saved == 0:
        with db_connect() as conn: conn.execute("DELETE FROM patients WHERE nik = ?", (nik,))
        # Log failed registration
        log_scan_result(
            status="failed",
            nik=str(nik),
            name=name,
            dob=dob,
            address=address,
            age=calculate_age(dob),
            message="Gagal simpan wajah (LBPH)."
        )
        return jsonify(ok=False, msg="Gagal simpan wajah (LBPH)."), 400
        
    retrain_after_change()
    # Log successful registration
    log_scan_result(
        status="success",
        nik=str(nik),
        name=name,
        dob=dob,
        address=address,
        age=calculate_age(dob),
        message=f"Register OK (LBPH). {saved} foto."
    )
    return jsonify(ok=True, msg=f"Register OK (LBPH). {saved} foto.")

# ... (kode impor dan config lainnya tetap sama)

# ====== RECOGNIZE ======
@app.post("/api/recognize")
def api_recognize():
    files = request.files.getlist("files[]") if request.files.getlist("files[]") else request.files.getlist("frames[]")
    if not files: 
        log_scan_result(status="failed", message="Tidak ada frame yang dikirim.")
        return jsonify(ok=False, msg="No frames."), 400
    
    frames = [bytes_to_bgr(f.read()) for f in files]

    # --- 1. INSIGHTFACE ENGINE ---
    if FACE_ENGINE == "insightface":
        try:
            res = face_engine.recognize_face_multi_frame(frames)
            if res:
                with db_connect() as conn:
                    r = conn.execute("SELECT * FROM patients WHERE nik=?", (res['nik'],)).fetchone()
                if r:
                    # SUKSES
                    log_scan_result(
                        status="success",
                        nik=str(r['nik']),
                        name=r['name'],
                        dob=r['dob'],
                        address=r['address'],
                        age=calculate_age(r['dob']),
                        message=f"Verifikasi berhasil (InsightFace). Confidence: {res.get('confidence')}%"
                    )
                    return jsonify(ok=True, found=True, nik=r['nik'], name=r['name'], 
                                   dob=r['dob'], address=r['address'], age=calculate_age(r['dob']),
                                   confidence=res.get('confidence'), engine="insightface")
            else:
                 # GAGAL: Wajah ada tapi tidak cocok dengan database
                 log_scan_result(status="failed", message="Wajah tidak dikenali di database (InsightFace).")
        
        except Exception as e:
            logger.error(f"Recog Error: {e}")
            log_scan_result(status="failed", message=f"System Error: {str(e)}")

    # --- 2. LBPH FALLBACK ---
    if not model_loaded: 
        log_scan_result(status="failed", message="Model belum di-load/training.")
        return jsonify(ok=False, msg="Wajah Tidak Dikenali."), 400
        
    if not recognizer: 
        return jsonify(ok=False, msg="LBPH error."), 500

    from collections import defaultdict, Counter
    votes = defaultdict(list)
    for img in frames:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        crop, _ = detect_largest_face(gray)
        if crop is None: continue
        try:
            id_pred, conf = recognizer.predict(preprocess_roi(crop))
            votes[id_pred].append(conf)
        except: pass
    
    if not votes:
        # GAGAL: Wajah tidak terdeteksi sama sekali
        log_scan_result(status="failed", message="Wajah tidak terdeteksi dalam frame.")
        return jsonify(ok=True, found=False, msg="Wajah tidak terdeteksi.")
    
    major = Counter([id for id, _ in [item for sublist in votes.items() for item in zip([sublist[0]]*len(sublist[1]), sublist[1])]]).most_common(1)[0][0]
    confs = votes.get(major, [])
    med_conf = float(np.median(confs))
    
    if len(confs) < MIN_VALID_FRAMES or med_conf >= LBPH_CONF_THRESHOLD:
        # GAGAL: Confidence jelek
        log_scan_result(status="failed", message=f"Akurasi rendah (Conf: {int(med_conf)}). Wajah tidak valid.")
        return jsonify(ok=True, found=False, msg="Tidak dikenali.")
        
    with db_connect() as conn:
        r = conn.execute("SELECT * FROM patients WHERE nik=?", (major,)).fetchone()
    
    if not r: 
        # GAGAL: ID diprediksi tapi data pasien sudah dihapus
        log_scan_result(status="failed", message=f"NIK {major} terdeteksi tapi data tidak ada di DB.")
        return jsonify(ok=True, found=False, msg="User tidak ada di DB.")
    
    age = calculate_age(r['dob'])
    
    # SUKSES LBPH
    log_scan_result(
        status="success",
        nik=str(r['nik']),
        name=r['name'],
        dob=r['dob'],
        address=r['address'],
        age=age,
        message=f"Verifikasi berhasil (LBPH). Confidence: {int(max(0, min(100, 100 - med_conf)))}%"
    )
    return jsonify(ok=True, found=True, nik=r['nik'], name=r['name'], dob=r['dob'], 
                   address=r['address'], age=age, 
                   confidence=int(max(0, min(100, 100 - med_conf))), engine="lbph")


# ====== ADMIN ACTIONS ======
@app.post("/admin/patient/<int:nik>/delete")
@login_required
def admin_delete_patient(nik: int): # RENAMED
    with db_connect() as conn: conn.execute("DELETE FROM patients WHERE nik=?", (nik,))
    for f in glob.glob(os.path.join(DATA_DIR, f"{nik}.*.jpg")): os.remove(f)
    if FACE_ENGINE == "insightface": face_engine.delete_embeddings_for_nik(nik)
    retrain_after_change()
    flash(f"Data NIK {nik} dihapus.", "success")
    return redirect(url_for("admin_dashboard"))

@app.post("/admin/retrain")
@login_required
def admin_retrain(): # RENAMED
    ok, msg = retrain_after_change()
    flash(msg, "success" if ok else "danger")
    return redirect(url_for("admin_dashboard"))

@app.post("/admin/patient/update")
@login_required
def admin_update_patient(): # RENAMED
    try:
        old, new = int(request.form.get("old_nik")), int(request.form.get("nik"))
        with db_connect() as conn:
            conn.execute("UPDATE patients SET nik=?, dob=?, address=? WHERE nik=?", 
                         (new, request.form.get("dob"), request.form.get("address"), old))
        
        if old != new:
            for f in glob.glob(os.path.join(DATA_DIR, f"{old}.*.jpg")):
                os.rename(f, os.path.join(DATA_DIR, f"{new}.{f.split('.')[-2]}.jpg"))
            if FACE_ENGINE == "insightface": face_engine.update_nik_in_embeddings(old, new)
            retrain_after_change()
        return jsonify(ok=True, msg="Update sukses.")
    except Exception as e: return jsonify(ok=False, msg=str(e))

@app.post("/api/queue/assign")
def api_queue_assign(): # RENAMED
    p = request.json.get("poli")
    if p not in ["Poli Umum", "Poli Gigi", "IGD"]: return jsonify(ok=False)
    with db_connect() as conn:
        curr = conn.execute("SELECT next_number FROM queues WHERE poli_name=?", (p,)).fetchone()[0]
        conn.execute("UPDATE queues SET next_number=? WHERE poli_name=?", (curr+1, p))
    return jsonify(ok=True, poli=p, nomor=curr+1)

@app.post("/api/queue/set")
@login_required
def api_queue_set(): # RENAMED
    p, n = request.json.get("poli"), request.json.get("nomor")
    with db_connect() as conn: conn.execute("UPDATE queues SET next_number=? WHERE poli_name=?", (n, p))
    return jsonify(ok=True)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)