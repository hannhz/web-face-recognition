# WEB-FACE - Sistem Cerdas Registrasi & Verifikasi Wajah

Aplikasi web Flask untuk registrasi pasien menggunakan **Scan E-KTP (OCR)** dan verifikasi identitas menggunakan **Pengenalan Wajah (InsightFace)**. Aplikasi ini dirancang untuk akurasi tinggi dan performa cepat dengan dukungan akselerasi GPU NVIDIA.

## 🚀 Fitur Unggulan

### 1. 📷 Smart Face Recognition

- **Engine:** InsightFace (RetinaFace + ArcFace) dengan akselerasi GPU.
- **Auto-Scan:** Mendeteksi wajah secara otomatis tanpa perlu menekan tombol.
- **Anti-Spoofing:** Menggunakan _Multi-Frame Voting_ untuk meningkatkan akurasi dan mencegah pemalsuan.
- **Face Alignment:** Normalisasi posisi wajah (5-point landmarks) untuk hasil pengenalan yang optimal.
- **Fallback:** Otomatis pindah ke mode CPU (LBPH) jika model berat gagal dimuat atau GPU tidak tersedia.

### 2. 🆔 Scan E-KTP Otomatis (OCR)

- **Auto-Capture:** Kamera otomatis memotret saat mendeteksi bentuk kartu KTP.
- **Smart Parsing:** Membaca NIK, Nama, Tanggal Lahir, dan Alamat dengan cerdas menggunakan regex kontekstual.
- **Auto-Correction:** Memperbaiki typo umum OCR (misal: `7005` menjadi `2005`, `DUKLUIN` menjadi `DUKUN`).
- **Data Extraction:** Otomatis mengisi form registrasi dari data hasil scan KTP.

---

## 📁 Struktur Direktori

```text
WEB-FACE/
├── app.py                    # Aplikasi Flask utama (Backend Logic)
├── face_engine.py            # Engine deteksi dan pengenalan wajah
├── requirements.txt          # Daftar Dependensi Python (Versi Stabil)
├── database.db               # Database SQLite (auto-generated)
├── data/
│   ├── database_wajah/       # Penyimpanan foto wajah user (Privasi)
│   └── database_ktp/         # Penyimpanan foto KTP user (Privasi)
├── model/
│   ├── embeddings.db         # Database embedding wajah (InsightFace)
│   └── buffalo_l/            # Model InsightFace (auto-download)
├── templates/
│   ├── user.html             # Interface pengguna (Kiosk)
│   ├── admin_login.html      # Halaman Login Admin
│   └── admin_dashboard.html  # Dashboard Admin
├── static/js/
│   ├── user.js               # Logic Frontend User
│   └── admin.js              # Logic Frontend Admin
├── README.md                 # Dokumentasi utama
└── README_INSIGHTFACE.md     # Dokumentasi teknis mendalam InsightFace
```

## 🛠️ Instalasi Cepat (CPU Only)

Jika Anda hanya ingin menjalankan tanpa GPU (lebih lambat untuk InsightFace), ikuti langkah ini:

# Clone repository

git clone [https://github.com/MuhammadDias/web-face-recognition.git](https://github.com/MuhammadDias/web-face-recognition.git)
cd web-face-recognition

# Buat virtual environment

```bash
python -m venv .venv
```

# Windows:

```bash
.venv\Scripts\activate
```

# Install dependencies

```bash
pip install -r requirements.txt
```

# Jalankan aplikasi

```bash
python app.py
```

## 🚀 Instalasi GPU Support (Wajib untuk Performa Tinggi)

Aplikasi ini dioptimalkan untuk CUDA 11.8. Ikuti langkah ini agar GPU NVIDIA terbaca.

## 📋 Prasyarat Hardware

Laptop/PC dengan GPU NVIDIA (RTX/GTX).

Driver NVIDIA terbaru.

## 🛠️ Langkah 1: Install CUDA Toolkit 11.8

Versi library yang digunakan di project ini membutuhkan CUDA 11.8 (Bukan 12.x).

-Download CUDA Toolkit 11.8.
-Install exe (local) pilih mode Express.

## 📦 Langkah 2: Lengkapi File DLL (cuDNN & Zlib)

CUDA Installer tidak menyertakan cuDNN dan zlibwapi. Anda harus menambahkannya manual agar Python bisa membacanya.

Download:

-cuDNN v8.x (untuk CUDA 11.x) dari NVIDIA Developer.
-zlibwapi.dll (x64) dari sumber terpercaya (misal: dll-files.com).

Copy & Paste:
-Extract cuDNN, ambil semua file di folder bin (seperti cudnn64_8.dll, dll).
-Ambil file zlibwapi.dll.

Paste semuanya ke dalam folder instalasi CUDA:
**C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin**

## 🐍 Langkah 3: Setup Python Environment

Pastikan menggunakan versi library yang tepat (sudah diatur di [requirements.txt]).

-Buat & Aktifkan Venv

```bash
python -m venv .venv
.venv\Scripts\activate
```

-Install Library (Versi dikunci agar stabil)

# Ini akan menginstall onnxruntime-gpu versi 1.16.3 yang kompatibel dengan CUDA 11.8

```bash
pip install -r requirements.txt
```

## ✅ Langkah 4: Jalankan

```bash
python app.py
```

Jika sukses, log terminal akan menampilkan: INFO:FaceEngine:InsightFace app initialized successfully with GPU

## 🔗 Akses Aplikasi

### Akses Lokal

- **User**: http://127.0.0.1:5000/
- **Admin**: http://127.0.0.1:5000/admin/login
  - Username: `admin`
  - Password: `Cakra@123`

### Akses Online dengan Cloudflared (Mengganti ngrok)

Untuk akses dari internet, gunakan **Cloudflared Quick Tunnel** (gratis dan lebih stabil dari ngrok):

#### Cara 1: Otomatis dengan Script

```bash
# Windows Batch (CMD)
start-with-cloudflared.bat

# PowerShell
.\start-with-cloudflared.ps1
```

#### Cara 2: Manual

```bash
# Terminal 1: Jalankan Flask
python app.py

# Terminal 2: Jalankan Cloudflared tunnel
..\cloudflared.exe tunnel --url http://127.0.0.1:5000
```

**Catatan**:

- `cloudflared.exe` sudah tersedia di folder parent
- Cloudflared akan memberikan URL publik seperti: `https://abc123.trycloudflare.com`
- URL ini bisa diakses dari mana saja untuk testing atau demo

## 📊 Arsitektur Pipeline

```
graph LR
A[Input Webcam] --> B[Deteksi RetinaFace]
B --> C[Alignment & Preprocess]
C --> D[Extract Embedding ArcFace]
D --> E[Normalize L2]
E --> F[Compare Cosine Similarity]
F --> G[Multi-Frame Voting]
G --> H[Output Hasil]
```

## ⚙️ Konfigurasi

| Variable                | Default | Deskripsi                     |
| ----------------------- | ------- | ----------------------------- |
| `USE_INSIGHTFACE`       | `1`     | Set ke `0` untuk gunakan LBPH |
| `RECOGNITION_THRESHOLD` | `0.4`   | Threshold similarity (0-1)    |
| `DETECTION_THRESHOLD`   | `0.5`   | Threshold deteksi wajah       |

## 📚 Dokumentasi Lengkap

Lihat **[README_INSIGHTFACE.md](README_INSIGHTFACE.md)** untuk:

- Setup detail
- Arsitektur sistem
- Tips meningkatkan akurasi
- API Reference
- Troubleshooting

## 🧪 Testing

```bash
python test_basic.py
python test_recognition_workflow.py
```

## 📝 Changelog

### v2.2.0 (Final Stable Release) - _Current_

**New Features (Fitur Baru):**

- **🆔 Smart KTP OCR:** Sistem scan E-KTP otomatis menggunakan Tesseract dengan logika cerdas:
  - **Auto-Capture:** Mendeteksi bentuk kartu KTP dan mengambil foto otomatis saat stabil.
  - **Auto-Rotate:** Otomatis memutar foto vertikal menjadi horizontal agar terbaca.
  - **Smart Parsing:** Menggunakan _Contextual Regex_ untuk membaca NIK, Nama, dan Alamat baris-per-baris (lebih akurat daripada scan global).
  - **Auto-Correction:** Memperbaiki typo umum OCR (misal: Tahun `7005` -> `2005`, Nama `I7ZAT` -> `IZZAT`, `DUKLUIN` -> `DUKUN`).
  - **Noise Filtering:** Membersihkan sampah teks di alamat (menghapus kata "LAKI", "GOL DARAH" yang bocor).
- **🌗 Dark Mode UI:** Tampilan _Light/Dark Mode_ yang bisa di-switch pada halaman User dan Admin Dashboard.
- **🖥️ Hardware Monitor:** Indikator Real-time di Dashboard Admin untuk memantau apakah sistem berjalan menggunakan **GPU (NVIDIA)** atau **CPU**.

## v2.1.0

Fix: Kompatibilitas penuh dengan CUDA 11.8 & cuDNN 8.

Fix: Menyelesaikan konflik versi NumPy 2.0 dan OpenCV.

Feat: Dashboard Admin menampilkan status real-time penggunaan Hardware (GPU/CPU).

Feat: Dukungan Dark Mode pada UI User dan Admin.

### v2.0.0

- Migrasi ke InsightFace (RetinaFace + ArcFace)
- Face alignment dengan 5-point landmarks
- SQLite embedding storage
- Multi-frame voting dengan early stop
- Auto-fallback ke LBPH

### v1.0.0 (Legacy)

- Haar Cascade + LBPH

## 📄 Lisensi

Internal / Sesuai kebutuhan proyek.
