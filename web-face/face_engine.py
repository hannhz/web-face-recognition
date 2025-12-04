"""
Face Detection and Recognition Engine using InsightFace (RetinaFace + ArcFace)
This module provides high-accuracy face detection and recognition.

Pipeline:
1. Detection: RetinaFace (from InsightFace) - fast and accurate
2. Alignment: 5-point landmark alignment
3. Recognition: ArcFace embedding model - 512-dim embeddings
4. Matching: Cosine similarity with threshold tuning
"""

import os

# --- JURUS PAKSA: TUNTUN PYTHON KE FOLDER CUDA ---
try:
    # Ini alamat folder CUDA 11.8 kamu yang sudah lengkap isinya
    cuda_bin_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
    
    if os.path.exists(cuda_bin_path):
        # Perintah ini memaksa Windows meload DLL dari sini
        os.add_dll_directory(cuda_bin_path)
        print(f"✅ SUKSES: Path CUDA ditambahkan paksa: {cuda_bin_path}")
    else:
        print("❌ ERROR: Folder CUDA 11.8 tidak ditemukan! Cek instalasi.")
except AttributeError:
    # os.add_dll_directory hanya ada di Python 3.8+ Windows
    os.environ['PATH'] = cuda_bin_path + ';' + os.environ['PATH']
# ------------------------------------------------
import json
import sqlite3
import threading
import logging
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('FaceEngine')

# ====== CONFIGURATION ======
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "database_wajah")
MODEL_DIR = os.path.join(BASE_DIR, "model")
EMBEDDING_DB_PATH = os.path.join(MODEL_DIR, "embeddings.db")
EMBEDDING_NPY_PATH = os.path.join(MODEL_DIR, "embeddings.npy")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Face detection/recognition thresholds
DETECTION_THRESHOLD = float(os.environ.get("DETECTION_THRESHOLD", "0.5"))  # Detection confidence
RECOGNITION_THRESHOLD = float(os.environ.get("RECOGNITION_THRESHOLD", "0.4"))  # Cosine similarity threshold (lower = stricter)
MIN_FACE_SIZE = int(os.environ.get("MIN_FACE_SIZE", "60"))  # Minimum face size in pixels
EMBEDDING_DIM = 512  # ArcFace embedding dimension

# Registration thresholds (relaxed for easier enrollment)
REGISTRATION_DETECTION_THRESHOLD = float(os.environ.get("REGISTRATION_DETECTION_THRESHOLD", "0.3"))  # Lower threshold for registration
REGISTRATION_QUALITY_THRESHOLD = float(os.environ.get("REGISTRATION_QUALITY_THRESHOLD", "0.1"))  # Lower quality threshold for registration

# Voting parameters for multi-frame recognition
VOTE_MIN_SHARE = float(os.environ.get("VOTE_MIN_SHARE", "0.35"))  # Minimum vote share
MIN_VALID_FRAMES = int(os.environ.get("MIN_VALID_FRAMES", "2"))  # Minimum valid frames
EARLY_VOTES_REQUIRED = int(os.environ.get("EARLY_VOTES_REQUIRED", "4"))  # Early stop votes
EARLY_SIM_THRESHOLD = float(os.environ.get("EARLY_SIM_THRESHOLD", "0.55"))  # Early stop similarity

# Global state
_engine_lock = threading.Lock()
_face_app = None
_embeddings_db = {}  # {nik: [embeddings]}
_embeddings_loaded = False


def _get_face_app():
    """Lazy load InsightFace app to avoid startup delay"""
    global _face_app
    if _face_app is None:
        try:
            from insightface.app import FaceAnalysis
            logger.info("Initializing InsightFace app...")
            
            # --- MODIFIKASI: Gunakan GPU (CUDA) ---
            # Prioritaskan CUDAExecutionProvider agar GPU dipakai
            _face_app = FaceAnalysis(
                name='buffalo_l',
                root=MODEL_DIR,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] 
            )
            
            # ctx_id = 0 artinya gunakan GPU pertama. (-1 untuk CPU)
            _face_app.prepare(ctx_id=0, det_size=(640, 640))
            
            # Sanity check: ensure recognition head exists
            test_attr = hasattr(_face_app, 'models') or True
            logger.info("InsightFace app initialized successfully with GPU")
        except ImportError as e:
            logger.warning(f"InsightFace not available in this Python environment: {e}")
            _face_app = None
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace: {e}")
            _face_app = None
    return _face_app

def _get_face_app():
    """Lazy load InsightFace app to avoid startup delay"""
    global _face_app
    if _face_app is None:
        try:
            from insightface.app import FaceAnalysis
            logger.info("Initializing InsightFace app...")
            
            # --- MODIFIKASI: Gunakan GPU (CUDA) ---
            # Prioritaskan CUDAExecutionProvider agar GPU dipakai
            _face_app = FaceAnalysis(
                name='buffalo_l',
                root=MODEL_DIR,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] 
            )
            
            # ctx_id = 0 artinya gunakan GPU pertama. (-1 untuk CPU)
            _face_app.prepare(ctx_id=0, det_size=(640, 640))
            
            # Sanity check: ensure recognition head exists
            test_attr = hasattr(_face_app, 'models') or True
            logger.info("InsightFace app initialized successfully with GPU")
        except ImportError as e:
            logger.warning(f"InsightFace not available in this Python environment: {e}")
            _face_app = None
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace: {e}")
            _face_app = None
    return _face_app

def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """L2 normalize embedding vector"""
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    return embedding


def _cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings"""
    return float(np.dot(emb1, emb2))


def _euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate euclidean distance between two embeddings"""
    return float(np.linalg.norm(emb1 - emb2))


# Public utility functions for testing and external use
def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """L2 normalize embedding vector (public API)"""
    return _normalize_embedding(embedding)


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings (public API)"""
    return _cosine_similarity(emb1, emb2)


# ====== EMBEDDING DATABASE ======

def init_embedding_db():
    """Initialize SQLite database for embeddings"""
    conn = sqlite3.connect(EMBEDDING_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nik INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL,
            quality_score REAL DEFAULT 0.0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS threshold_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            threshold REAL NOT NULL,
            accuracy REAL DEFAULT 0.0,
            false_positive_rate REAL DEFAULT 0.0,
            timestamp TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_nik ON embeddings(nik)")
    conn.commit()
    conn.close()
    logger.info(f"Embedding database initialized at {EMBEDDING_DB_PATH}")


def save_embedding(nik: int, embedding: np.ndarray, quality_score: float = 0.0) -> bool:
    """Save embedding to database"""
    try:
        conn = sqlite3.connect(EMBEDDING_DB_PATH)
        normalized = _normalize_embedding(embedding)
        blob = normalized.astype(np.float32).tobytes()
        conn.execute(
            "INSERT INTO embeddings (nik, embedding, created_at, quality_score) VALUES (?, ?, ?, ?)",
            (nik, blob, datetime.now().isoformat(), quality_score)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Failed to save embedding: {e}")
        return False


def load_all_embeddings() -> Dict[int, List[np.ndarray]]:
    """Load all embeddings from database into memory"""
    global _embeddings_db, _embeddings_loaded
    try:
        if not os.path.exists(EMBEDDING_DB_PATH):
            init_embedding_db()
            _embeddings_loaded = True
            return {}

        conn = sqlite3.connect(EMBEDDING_DB_PATH)
        cursor = conn.execute("SELECT nik, embedding FROM embeddings ORDER BY quality_score DESC")

        _embeddings_db = {}
        count = 0
        for row in cursor:
            nik = int(row[0])
            emb = np.frombuffer(row[1], dtype=np.float32)
            if nik not in _embeddings_db:
                _embeddings_db[nik] = []
            _embeddings_db[nik].append(emb)
            count += 1

        conn.close()
        _embeddings_loaded = True
        logger.info(f"Loaded {count} embeddings for {len(_embeddings_db)} unique NIKs")
        return _embeddings_db
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        _embeddings_loaded = True
        return {}


def delete_embeddings_for_nik(nik: int) -> int:
    """Delete all embeddings for a given NIK"""
    global _embeddings_db
    try:
        conn = sqlite3.connect(EMBEDDING_DB_PATH)
        cursor = conn.execute("DELETE FROM embeddings WHERE nik = ?", (nik,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        if nik in _embeddings_db:
            del _embeddings_db[nik]

        logger.info(f"Deleted {deleted} embeddings for NIK {nik}")
        return deleted
    except Exception as e:
        logger.error(f"Failed to delete embeddings: {e}")
        return 0


def update_nik_in_embeddings(old_nik: int, new_nik: int) -> int:
    """Update NIK in embeddings database"""
    global _embeddings_db
    try:
        conn = sqlite3.connect(EMBEDDING_DB_PATH)
        cursor = conn.execute(
            "UPDATE embeddings SET nik = ? WHERE nik = ?",
            (new_nik, old_nik)
        )
        updated = cursor.rowcount
        conn.commit()
        conn.close()

        if old_nik in _embeddings_db:
            _embeddings_db[new_nik] = _embeddings_db.pop(old_nik)

        logger.info(f"Updated {updated} embeddings from NIK {old_nik} to {new_nik}")
        return updated
    except Exception as e:
        logger.error(f"Failed to update NIK in embeddings: {e}")
        return 0


def get_embedding_count() -> int:
    """Get total number of embeddings in database"""
    try:
        if not os.path.exists(EMBEDDING_DB_PATH):
            return 0
        conn = sqlite3.connect(EMBEDDING_DB_PATH)
        cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        logger.error(f"Failed to get embedding count: {e}")
        return 0


def get_unique_nik_count() -> int:
    """Get number of unique NIKs in database"""
    try:
        if not os.path.exists(EMBEDDING_DB_PATH):
            return 0
        conn = sqlite3.connect(EMBEDDING_DB_PATH)
        cursor = conn.execute("SELECT COUNT(DISTINCT nik) FROM embeddings")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        logger.error(f"Failed to get unique NIK count: {e}")
        return 0


# ====== FACE DETECTION ======

def detect_faces(img_bgr: np.ndarray, detection_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Detect faces in image using InsightFace RetinaFace.
    Returns list of face dictionaries with bbox, landmarks, det_score, embedding.

    Args:
        img_bgr: BGR image array
        detection_threshold: Optional custom detection threshold (defaults to DETECTION_THRESHOLD)
    """
    if detection_threshold is None:
        detection_threshold = DETECTION_THRESHOLD

    app = _get_face_app()
    if app is None:
        return _detect_faces_fallback(img_bgr)

    try:
        faces = app.get(img_bgr)
        results = []
        for face in faces:
            if face.det_score < detection_threshold:
                continue

            bbox = face.bbox.astype(int).tolist()
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            # Filter small faces
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue

            results.append({
                'bbox': bbox,
                'landmarks': face.kps.tolist() if getattr(face, 'kps', None) is not None else None,
                'det_score': float(face.det_score),
                'embedding': _normalize_embedding(face.embedding) if getattr(face, 'embedding', None) is not None else None,
                'age': getattr(face, 'age', None),
                'gender': getattr(face, 'gender', None)
            })

        # Sort by face size (largest first)
        results.sort(key=lambda x: (x['bbox'][2]-x['bbox'][0]) * (x['bbox'][3]-x['bbox'][1]), reverse=True)
        return results
    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        return _detect_faces_fallback(img_bgr)


def _detect_faces_fallback(img_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """Fallback face detection using Haar Cascade when InsightFace is unavailable"""
    try:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        detector = cv2.CascadeClassifier(cascade_path)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))

        results = []
        for (x, y, w, h) in faces:
            results.append({
                'bbox': [x, y, x+w, y+h],
                'landmarks': None,
                'det_score': 0.9,  # Haar doesn't give confidence
                'embedding': None,
                'age': None,
                'gender': None
            })

        return results
    except Exception as e:
        logger.error(f"Fallback detection failed: {e}")
        return []


def detect_largest_face(img_bgr: np.ndarray, detection_threshold: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """
    Detect and return the largest face in the image.

    Args:
        img_bgr: BGR image array
        detection_threshold: Optional custom detection threshold (defaults to DETECTION_THRESHOLD)
    """
    faces = detect_faces(img_bgr, detection_threshold=detection_threshold)
    if not faces:
        return None
    return faces[0]  # Already sorted by size


# ====== IMAGE QUALITY ASSESSMENT ======

def is_blurry(img_gray: np.ndarray, threshold: float = 100.0) -> bool:
    """Check if image is blurry using Laplacian variance"""
    variance = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    return variance < threshold


def calculate_quality_score(face_dict: Dict[str, Any], img_bgr: np.ndarray) -> float:
    """Calculate quality score for a detected face (0-1 range)"""
    score = 0.0

    # Detection confidence
    det_score = face_dict.get('det_score', 0.0)
    score += det_score * 0.4

    # Face size score
    bbox = face_dict.get('bbox', [0, 0, 0, 0])
    face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    size_ratio = min(face_area / img_area, 0.5) / 0.5  # Normalize to 0-1
    score += size_ratio * 0.3

    # Sharpness score
    x1, y1, x2, y2 = bbox
    face_region = img_bgr[y1:y2, x1:x2]
    if face_region.size > 0:
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 500.0, 1.0)  # Normalize
        score += sharpness_score * 0.3

    return min(score, 1.0)


# ====== FACE ALIGNMENT ======

def align_face(img_bgr: np.ndarray, landmarks: Optional[List[List[float]]] = None) -> np.ndarray:
    """
    Align face using 5-point landmarks.
    If landmarks not provided, return original image.
    """
    if landmarks is None or len(landmarks) < 5:
        return img_bgr

    try:
        # Standard face alignment target points (112x112)
        src_pts = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

        dst_pts = np.array(landmarks[:5], dtype=np.float32)

        # Estimate transformation matrix
        M = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC)[0]

        if M is None:
            return img_bgr

        # Apply transformation
        aligned = cv2.warpAffine(img_bgr, M, (112, 112), borderMode=cv2.BORDER_REPLICATE)
        return aligned
    except Exception as e:
        logger.warning(f"Face alignment failed: {e}")
        return img_bgr


# ====== FACE RECOGNITION ======

def get_embedding(img_bgr: np.ndarray, face_dict: Optional[Dict[str, Any]] = None) -> Optional[np.ndarray]:
    """
    Get face embedding from image.
    If face_dict is provided, use its embedding if available.
    Otherwise, detect face and compute embedding.
    """
    # Use provided embedding if available
    if face_dict is not None and face_dict.get('embedding') is not None:
        return face_dict['embedding']

    # Detect face and get embedding
    app = _get_face_app()
    if app is None:
        return None

    try:
        faces = app.get(img_bgr)
        if not faces:
            return None

        # Get largest face embedding
        largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        if getattr(largest, 'embedding', None) is None:
            return None

        return _normalize_embedding(largest.embedding)
    except Exception as e:
        logger.error(f"Failed to get embedding: {e}")
        return None


def find_matching_identity(
    query_embedding: np.ndarray,
    threshold: float = None,
    top_k: int = 5
) -> List[Tuple[int, float]]:
    """
    Find matching identity from database.
    Returns list of (nik, similarity) tuples sorted by similarity.
    """
    global _embeddings_db, _embeddings_loaded

    if threshold is None:
        threshold = RECOGNITION_THRESHOLD

    if not _embeddings_loaded:
        load_all_embeddings()

    if not _embeddings_db:
        return []

    matches = []
    for nik, embeddings in _embeddings_db.items():
        # Calculate similarity with all embeddings for this NIK
        similarities = [_cosine_similarity(query_embedding, emb) for emb in embeddings]
        max_sim = max(similarities) if similarities else 0.0

        if max_sim >= threshold:
            matches.append((nik, max_sim))

    # Sort by similarity (highest first)
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:top_k]


def recognize_face_in_image(
    img_bgr: np.ndarray,
    threshold: float = None
) -> Optional[Tuple[int, float]]:
    """
    Recognize face in single image.
    Returns (nik, similarity) or None if not recognized.
    """
    face = detect_largest_face(img_bgr)
    if face is None:
        return None

    embedding = face.get('embedding')
    if embedding is None:
        embedding = get_embedding(img_bgr, face)

    if embedding is None:
        return None

    matches = find_matching_identity(embedding, threshold)
    if matches:
        return matches[0]

    return None


def recognize_face_multi_frame(
    frames: List[np.ndarray],
    threshold: float = None
) -> Optional[Dict[str, Any]]:
    """
    Recognize face across multiple frames with voting.
    Returns result dict with nik, similarity, confidence, etc.
    """
    global _embeddings_db, _embeddings_loaded

    if threshold is None:
        threshold = RECOGNITION_THRESHOLD

    if not _embeddings_loaded:
        load_all_embeddings()

    if not _embeddings_db:
        logger.info("No embeddings in database")
        return None

    from collections import defaultdict
    votes = defaultdict(list)  # nik -> list of similarities
    processed = 0

    for frame in frames:
        face = detect_largest_face(frame)
        if face is None:
            continue

        # Check quality
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        bbox = face.get('bbox', [0, 0, 0, 0])
        face_gray = gray[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if face_gray.size > 0 and is_blurry(face_gray, 50.0):
            continue

        embedding = face.get('embedding')
        if embedding is None:
            embedding = get_embedding(frame, face)
            if embedding is None:
                continue

        processed += 1

        # Find matches
        for nik, embs in _embeddings_db.items():
            similarities = [_cosine_similarity(embedding, emb) for emb in embs]
            max_sim = max(similarities) if similarities else 0.0
            if max_sim >= threshold:
                votes[nik].append(max_sim)

        # Early stop if confident
        if votes:
            best_nik = max(votes.keys(), key=lambda k: np.mean(votes[k]))
            vote_share = len(votes[best_nik]) / processed
            avg_sim = np.mean(votes[best_nik])

            if (vote_share >= VOTE_MIN_SHARE and
                len(votes[best_nik]) >= EARLY_VOTES_REQUIRED and
                avg_sim >= EARLY_SIM_THRESHOLD):
                logger.info(f"Early stop: NIK={best_nik}, sim={avg_sim:.3f}, votes={len(votes[best_nik])}")
                break

    if processed == 0 or not votes:
        logger.info(f"Recognition failed: processed={processed}, votes={len(votes)}")
        return None

    # Find winner by vote count and average similarity
    winner = None
    best_score = -1

    for nik, sims in votes.items():
        vote_count = len(sims)
        avg_sim = np.mean(sims)
        score = vote_count * avg_sim  # Combined score

        if score > best_score:
            best_score = score
            winner = {
                'nik': nik,
                'similarity': float(avg_sim),
                'vote_count': vote_count,
                'vote_share': vote_count / processed,
                'processed_frames': processed,
                'confidence': int(min(avg_sim * 100, 100))
            }

    if winner is None:
        return None

    # Validate minimum requirements
    if (winner['vote_share'] < VOTE_MIN_SHARE or
        winner['vote_count'] < MIN_VALID_FRAMES or
        winner['similarity'] < threshold):
        logger.info(f"Recognition rejected: {winner}")
        return None

    logger.info(f"Recognition success: NIK={winner['nik']}, sim={winner['similarity']:.3f}")
    return winner


# ====== REGISTRATION / ENROLLMENT ======

def enroll_face(
    img_bgr: np.ndarray,
    nik: int
) -> Tuple[bool, str, Optional[np.ndarray]]:
    """
    Enroll a single face image to database.
    Returns (success, message, embedding).

    Uses relaxed detection and quality thresholds for easier registration.
    """
    # Use relaxed detection threshold for registration
    face = detect_largest_face(img_bgr, detection_threshold=REGISTRATION_DETECTION_THRESHOLD)
    if face is None:
        return False, "No face detected", None

    # Check quality with relaxed threshold for registration
    quality = calculate_quality_score(face, img_bgr)
    if quality < REGISTRATION_QUALITY_THRESHOLD:
        return False, f"Face quality too low: {quality:.2f}", None

    # If detection embedding missing, try to compute explicitly
    embedding = face.get('embedding')
    if embedding is None:
        embedding = get_embedding(img_bgr, face)
        if embedding is None:
            return False, "Could not extract embedding (is InsightFace installed and models downloaded?)", None

    # Save embedding
    if save_embedding(nik, embedding, quality):
        # Update in-memory cache
        global _embeddings_db
        if nik not in _embeddings_db:
            _embeddings_db[nik] = []
        _embeddings_db[nik].append(embedding)

        return True, f"Enrolled with quality {quality:.2f}", embedding

    return False, "Failed to save embedding", None


def enroll_multiple_frames(
    frames: List[np.ndarray],
    nik: int,
    min_embeddings: int = 5
) -> Tuple[int, str]:
    """
    Enroll multiple frames for a single NIK.
    Returns (num_enrolled, message).
    """
    enrolled = 0

    for frame in frames:
        success, msg, _ = enroll_face(frame, nik)
        if success:
            enrolled += 1
            if enrolled >= 20:  # Max 20 embeddings per person
                break

    # Augment if needed (by duplicating best embeddings)
    global _embeddings_db
    if enrolled > 0 and enrolled < min_embeddings and nik in _embeddings_db:
        current_count = len(_embeddings_db[nik])
        needed = min_embeddings - current_count

        # Add slightly noisy versions of existing embeddings
        for i in range(needed):
            idx = i % current_count
            base_emb = _embeddings_db[nik][idx]
            noise = np.random.normal(0, 0.01, base_emb.shape)
            augmented = _normalize_embedding(base_emb + noise)
            save_embedding(nik, augmented, 0.5)
            _embeddings_db[nik].append(augmented)
            enrolled += 1

    if enrolled == 0:
        return 0, "No valid face frames to enroll"

    return enrolled, f"Successfully enrolled {enrolled} embeddings"


# ====== THRESHOLD TUNING ======

def log_threshold_performance(
    threshold: float,
    accuracy: float,
    false_positive_rate: float
):
    """Log threshold performance for analysis"""
    try:
        conn = sqlite3.connect(EMBEDDING_DB_PATH)
        conn.execute(
            "INSERT INTO threshold_history (threshold, accuracy, false_positive_rate, timestamp) VALUES (?, ?, ?, ?)",
            (threshold, accuracy, false_positive_rate, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log threshold: {e}")


def suggest_threshold() -> float:
    """
    Suggest optimal threshold based on embedding distribution.
    Analyzes intra-class and inter-class distances.
    """
    global _embeddings_db, _embeddings_loaded

    if not _embeddings_loaded:
        load_all_embeddings()

    if len(_embeddings_db) < 2:
        return RECOGNITION_THRESHOLD

    try:
        intra_sims = []  # Same person similarities
        inter_sims = []  # Different person similarities

        niks = list(_embeddings_db.keys())

        # Calculate intra-class similarities
        for nik in niks:
            embs = _embeddings_db[nik]
            for i in range(len(embs)):
                for j in range(i+1, len(embs)):
                    intra_sims.append(_cosine_similarity(embs[i], embs[j]))

        # Calculate inter-class similarities (sample)
        import random
        for _ in range(min(1000, len(niks) * len(niks))):
            nik1, nik2 = random.sample(niks, 2)
            emb1 = random.choice(_embeddings_db[nik1])
            emb2 = random.choice(_embeddings_db[nik2])
            inter_sims.append(_cosine_similarity(emb1, emb2))

        if not intra_sims or not inter_sims:
            return RECOGNITION_THRESHOLD

        # Find threshold that maximizes separation
        intra_min = np.percentile(intra_sims, 10)
        inter_max = np.percentile(inter_sims, 90)

        suggested = (intra_min + inter_max) / 2
        logger.info(f"Suggested threshold: {suggested:.3f} (intra_min={intra_min:.3f}, inter_max={inter_max:.3f})")

        return max(0.3, min(0.6, suggested))
    except Exception as e:
        logger.error(f"Threshold suggestion failed: {e}")
        return RECOGNITION_THRESHOLD


# ====== INITIALIZATION ======

_initialized = False
_init_lock = threading.Lock()

# ====== INITIALIZATION (AGGRESSIVE WARM UP) ======

_initialized = False
_init_lock = threading.Lock()

# GANTI BAGIAN PALING BAWAH (Fungsi initialize) DENGAN INI:

def initialize():
    """
    Initialize face engine at startup with REAL IMAGE WARM UP.
    Menggunakan foto asli 'warmup.jpg' untuk memancing seluruh pipeline (Deteksi + Recog)
    agar bangun sempurna.
    """
    global _initialized
    with _init_lock:
        if _initialized:
            return  # Already initialized

        # 1. Load Database
        init_embedding_db()
        load_all_embeddings()

        logger.info("🔥 WARM UP: Memanaskan Mesin AI...")
        
        # 2. Load Model Utama
        app = _get_face_app()

        if app:
            try:
                # Cari file warmup.jpg di folder yang sama dengan script ini
                warmup_path = os.path.join(BASE_DIR, "warmup.jpg")
                
                # Cek apakah user sudah menaruh fotonya?
                if os.path.exists(warmup_path):
                    logger.info(f"   ├─ Ditemukan foto pancingan: {warmup_path}")
                    
                    # Baca gambar
                    img = cv2.imread(warmup_path)
                    
                    if img is not None:
                        logger.info("   ├─ Memaksa GPU memproses foto asli 3x...")
                        
                        # KITA PAKSA DIA KERJA 3 KALI!
                        # 1x buat bangunin Deteksi
                        # 1x buat bangunin Recognition (karena wajah terdeteksi)
                        # 1x buat memantapkan cache memori
                        for i in range(3):
                            start = datetime.now()
                            app.get(img) # <--- Ini trigger FULL PIPELINE (Deteksi -> Recog)
                            logger.info(f"   │  └─ Pass {i+1} selesai: {(datetime.now() - start).total_seconds():.2f}s")
                            
                        logger.info("✅ WARM UP SUKSES! GPU sudah panas & siap tempur.")
                    else:
                        logger.warning("⚠️ File warmup.jpg ada tapi rusak/tidak terbaca.")
                else:
                    # FALLBACK: Kalau kamu lupa naruh foto, kita pakai cara dummy lama
                    logger.warning("⚠️ File 'warmup.jpg' TIDAK DITEMUKAN! Menggunakan dummy noise...")
                    logger.warning("   (Tips: Taruh foto wajah bernama 'warmup.jpg' di folder project agar scan pertama instan)")
                    
                    # Dummy fallback
                    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                    app.get(dummy) 

            except Exception as e:
                logger.warning(f"⚠️ Warmup Warning: {e}")

        _initialized = True
        logger.info("Face engine initialized completely.")


def is_available() -> bool:
    """Check if face engine is available (InsightFace or fallback)"""
    insightface_ready = _get_face_app() is not None
    # Fallback is always available (uses Haar Cascade)
    fallback_ready = True
    return insightface_ready or fallback_ready


def get_engine_status() -> Dict[str, Any]:
    """Get face engine status"""
    global _embeddings_db, _embeddings_loaded

    return {
        'insightface_available': _get_face_app() is not None,
        'embeddings_loaded': _embeddings_loaded,
        'total_embeddings': get_embedding_count(),
        'unique_niks': get_unique_nik_count(),
        'recognition_threshold': RECOGNITION_THRESHOLD,
        'detection_threshold': DETECTION_THRESHOLD
    }


# Initialize on import
if os.environ.get("FACE_ENGINE_INIT", "1") == "1":
    initialize()