import cv2
import pytesseract
import re
import os
import numpy as np
from datetime import datetime

# Path Tesseract
path_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(path_tesseract):
    pytesseract.pytesseract.tesseract_cmd = path_tesseract

# ==========================================
# HELPERS
# ==========================================
def clean_garbage(text):
    text = re.sub(r'^[^A-Z0-9]+', '', text.upper())
    text = re.sub(r'[^A-Z0-9]+$', '', text)
    return text.strip()

def force_alpha(text):
    replacements = {'0': 'O', '1': 'I', '5': 'S', '2': 'Z', '4': 'A', '8': 'B', '6': 'G', '7': 'Z'}
    text = text.upper()
    for digit, char in replacements.items():
        text = text.replace(digit, char)
    return re.sub(r'[^A-Z\s\.,]', '', text).strip()

# ==========================================
# ENGINE UTAMA
# ==========================================
def process_ktp(image_path):
    print(f"\n📸 Memproses: {image_path}")
    img = cv2.imread(image_path)
    
    # 1. PREPROCESSING (Tetap sama karena sudah bagus)
    if img.shape[0] > img.shape[1]: 
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    scale = 2.0 
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    kernel = np.ones((2,2), np.uint8)
    gray = cv2.erode(gray, kernel, iterations=1)

    # 2. OCR
    text = pytesseract.image_to_string(gray, lang='ind', config='--psm 6')
    lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 2]
    
    # Debug Raw Text (Biar kita tau apa yang dibaca)
    print("\n--- RAW TEXT ---")
    for l in lines: print(l)
    print("----------------\n")

    result = {"nik": None, "nama": None, "dob": None, "alamat": None}
    
    # BUFFER ALAMAT
    addr_parts = []
    
    # 3. PARSING BARIS DEMI BARIS (LEBIH AMAN)
    for i, line in enumerate(lines):
        line_upper = line.upper()

        # A. NIK (Cari baris yang mengandung NIK atau angka 16 digit)
        if "NIK" in line_upper or re.search(r'\d{16}', line_upper):
            digits = re.sub(r'[^0-9]', '', line_upper)
            # Prioritas Jatim (35)
            match = re.search(r'(35\d{14})', digits)
            if not match: match = re.search(r'(3\d{15})', digits) # Umum
            if not match: match = re.search(r'\d{16}', digits)    # Fallback
            
            if match: result['nik'] = match.group(0)

        # B. NAMA (Baris yang ada "NAMA")
        if "NAMA" in line_upper:
            raw = re.sub(r'nama\s*[:.\-]*\s*', '', line_upper, flags=re.IGNORECASE)
            clean = force_alpha(raw)
            
            # Hapus Sampah Akhir (misal " Y" atau " TG")
            # Logic: Hapus 1-2 huruf di akhir string yang didahului spasi
            clean = re.sub(r'\s+[A-Z]{1,2}$', '', clean)
            
            # Hapus kata NIK jika kebawa
            if "NIK" in clean: clean = clean.split("NIK")[1]

            if len(clean) > 2:
                result['nama'] = clean
            elif i + 1 < len(lines):
                # Cek baris bawah
                potential = force_alpha(lines[i+1])
                potential = re.sub(r'\s+[A-Z]{1,2}$', '', potential) # Bersihkan juga
                if "LAHIR" not in potential: result['nama'] = potential

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

            if len(val) > 2: addr_parts.append(val.strip())

        # D. RT/RW (Cari di baris yang EKSPLISIT ada kata RT atau RW)
        # Ini perbaikan utamanya! Biar ga ngambil NIK.
        if "RT" in line_upper or "RW" in line_upper:
            # Ambil angka di baris ini saja
            nums = re.findall(r'\d+', line_upper)
            # Biasanya ada 2 grup angka (005 dan 002)
            if len(nums) >= 2:
                # Ambil 2 angka terakhir (asumsi rt rw berurutan)
                # Pastikan panjangnya masuk akal (1-3 digit)
                rt = nums[0] if len(nums[0]) <= 3 else nums[0][-3:]
                rw = nums[1] if len(nums[1]) <= 3 else nums[1][-3:]
                addr_parts.append(f"RT/RW {rt}/{rw}")
            # Kadang kebaca 0051002 (nyambung)
            elif len(nums) == 1 and len(nums[0]) > 4:
                combined = nums[0]
                # Ambil 3 digit dan 3 digit
                mid = len(combined) // 2
                addr_parts.append(f"RT/RW {combined[:mid]}/{combined[mid:]}")

        # E. KELURAHAN
        if re.search(r'(KEL|DESA)', line_upper):
            val = re.sub(r'(KEL|DESA|/DASA|ILESA)[\.\s:]*', '', line_upper)
            val = force_alpha(val)
            # Hapus sampah JENIS/KEC
            for stop in ["KEC", "JENIS", "LAKI", "AGAMA"]:
                if stop in val: val = val.split(stop)[0]
            
            if len(val) > 2: addr_parts.append(f"Kel. {val}")

        # F. KECAMATAN
        if "KECAMATAN" in line_upper:
            val = line_upper.replace("KECAMATAN", "").strip()
            val = force_alpha(val)
            # Fix Typo Umum
            val = val.replace("DUKLUIN", "DUKUN").replace("DUKUIN", "DUKUN")
            
            for stop in ["AGAMA", "KAWIN"]:
                if stop in val: val = val.split(stop)[0]
            
            # Hapus sampah akhir 1-2 huruf
            val = re.sub(r'\s+[A-Z]{1,2}$', '', val)
            
            if len(val) > 2: addr_parts.append(f"Kec. {val}")

    # 4. FINALISASI DATA
    
    # DOB dari NIK (Paling Valid)
    if result['nik'] and len(result['nik']) == 16:
        try:
            tgl = int(result['nik'][6:8])
            bln = int(result['nik'][8:10])
            thn = int(result['nik'][10:12])
            if tgl > 40: tgl -= 40
            
            curr_y = int(datetime.now().strftime("%y"))
            full_y = 2000 + thn if thn <= curr_y else 1900 + thn
            result['dob'] = f"{full_y}-{bln:02d}-{tgl:02d}"
        except: pass

    # Gabung Alamat
    if addr_parts:
        # Hapus duplikat dan gabung
        seen = set()
        final_addr = [x for x in addr_parts if not (x in seen or seen.add(x))]
        result['alamat'] = ", ".join(final_addr)

    print("\n✅ === HASIL EKSTRAKSI ===")
    print(f"NIK    : {result['nik']}")
    print(f"NAMA   : {result['nama']}")
    print(f"DOB    : {result['dob']}")
    print(f"ALAMAT : {result['alamat']}")
    print("=========================\n")

if __name__ == "__main__":
    file_foto = "3525012708050123.jpg" # Pastikan nama file benar
    process_ktp(file_foto)