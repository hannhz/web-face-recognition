// User front-end: FINAL FIX (Waktu Verif + Countdown Reset + IP Fix Ready)
(() => {
  // --- DOM Elements ---
  const pageHome = document.getElementById('page-home');
  const pageRegistrasi = document.getElementById('page-registrasi');
  const pagePoli = document.getElementById('page-poli');
  const pagePoliGateway = document.getElementById('page-poli-gateway');

  const navHome = document.getElementById('nav-home');
  const navRegistrasi = document.getElementById('nav-registrasi');
  const navPoli = document.getElementById('nav-poli');
  const btnHomeRegistrasi = document.getElementById('btn-home-registrasi');
  const btnHomeKePoli = document.getElementById('btn-home-ke-poli');

  // Registrasi
  const formRegistrasi = document.getElementById('form-registrasi');
  const inputNik = document.getElementById('reg-nik');
  const inputNama = document.getElementById('reg-nama');
  const inputDob = document.getElementById('reg-ttl');
  const inputAlamat = document.getElementById('reg-alamat');
  const videoReg = document.getElementById('video-reg');
  const statusReg = document.getElementById('status-reg');
  const countReg = document.getElementById('count-reg');

  // Verifikasi & Auto Scan UI
  const videoVerif = document.getElementById('video-verif');
  const btnScan = document.getElementById('btn-scan');
  const btnNikFallback = document.getElementById('btn-nik-fallback');
  const verifResult = document.getElementById('verif-result');
  const verifData = document.getElementById('verif-data');
  const verifNikBox = document.getElementById('verif-nik');
  const fallbackNik = document.getElementById('fallback-nik');
  const btnCariNik = document.getElementById('btn-cari-nik');
  const statusVerif = document.getElementById('status-verif');
  const btnLanjutForm = document.getElementById('btn-lanjut-form');
  const btnDetailData = document.getElementById('btn-detail-data');

  // Switch Camera Buttons
  const btnSwitchCamReg = document.getElementById('btn-switch-cam-reg');
  const btnSwitchCamVerif = document.getElementById('btn-switch-cam-verif');

  // Auto Scan Overlay Elements
  const overlayAuto = document.getElementById('auto-scan-overlay');
  const textCountdown = document.getElementById('auto-scan-countdown');
  const circleProgress = document.getElementById('auto-scan-circle');
  const focusBox = document.getElementById('verif-focus-box');
  // Note: nextScanText element is created dynamically inside verifData now

  // Poli gateway
  const formPoliGateway = document.getElementById('form-poli-gateway');
  const gwNama = document.getElementById('gw-nama');
  const gwUmur = document.getElementById('gw-umur');
  const gwAlamat = document.getElementById('gw-alamat');
  const gwPoli = document.getElementById('gw-poli');
  const gwKeluhan = document.getElementById('gw-keluhan');

  // Modals
  const modalAlert = document.getElementById('modal-alert');
  const alertMessage = document.getElementById('alert-message');
  const btnAlertOk = document.getElementById('btn-modal-alert-ok');

  const modalLoading = document.getElementById('modal-loading');
  const loadingText = document.getElementById('loading-text');
  const progressInner = document.getElementById('progress-inner');

  const modalAntrian = document.getElementById('modal-antrian');
  const antrianPoli = document.getElementById('antrian-poli');
  const antrianNomor = document.getElementById('antrian-nomor');
  const btnAntrianTutup = document.getElementById('btn-modal-antrian-tutup');

  const modalRegisSuccess = document.getElementById('modal-regis-success');
  const btnModalRegisTutup = document.getElementById('btn-modal-regis-tutup');
  const btnModalLanjutPoli = document.getElementById('btn-modal-lanjut-poli');

  // Camera Permission Modal
  const modalCameraPermission = document.getElementById('modal-camera-permission');
  const btnCameraPermission = document.getElementById('btn-camera-permission');
  const btnCameraPermissionLater = document.getElementById('btn-camera-permission-later');

  // Camera Status Indicator
  const cameraStatus = document.getElementById('camera-status');
  const cameraStatusDot = document.getElementById('camera-status-dot');
  const cameraStatusText = document.getElementById('camera-status-text');

  // -- STATE VARIABLES --
  let streamReg = null;
  let streamVerif = null;
  let currentFacingMode = 'user';
  let activeStreamMode = null;
  let activePatient = null;
  let verificationStartTime = null; // Variable untuk hitung durasi

  // Auto Scan State
  let autoCheckInterval = null;
  let nextScanTimer = null;
  let isScanning = false;
  let faceDetectedTime = 0;
  const CHECK_INTERVAL = 400;
  const REQUIRED_TIME = 2000;
  const CIRCLE_FULL = 264;

  // --- NAVIGATION ---
  function showPage(id) {
    stopAutoCheck();
    stopNextScanCountdown();

    [pageHome, pageRegistrasi, pagePoli, pagePoliGateway].forEach((p) => p && p.classList.add('hidden'));
    document.querySelectorAll('.nav-button').forEach((b) => b.classList.remove('active'));

    if (id === 'page-home') {
      pageHome.classList.remove('hidden');
    }
    if (id === 'page-registrasi') {
      pageRegistrasi.classList.remove('hidden');
      if (navRegistrasi) navRegistrasi.classList.add('active');
    }
    if (id === 'page-poli') {
      pagePoli.classList.remove('hidden');
      if (navPoli) navPoli.classList.add('active');
      resetVerif();
    }
    if (id === 'page-poli-gateway') {
      pagePoliGateway.classList.remove('hidden');
      if (navPoli) navPoli.classList.add('active');
    }
  }

  function resetVerif() {
    if (verifResult) verifResult.classList.add('hidden');
    if (verifNikBox) verifNikBox.classList.add('hidden');
    if (statusVerif) statusVerif.textContent = 'Menunggu wajah...';
    if (verifData) verifData.innerHTML = '';
    isScanning = false;
    stopNextScanCountdown();
    resetCountdownUI();

    ensureCamera('verif').then(() => {
      if (streamVerif && !pagePoli.classList.contains('hidden')) {
        startAutoCheck();
      }
    });
  }

  // --- CAMERA LOGIC ---
  async function initWebcam(videoEl, mode = 'user') {
    if (videoEl.srcObject) {
      videoEl.srcObject.getTracks().forEach((t) => t.stop());
    }

    const constraints = {
      video: {
        facingMode: mode,
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
      audio: false,
    };

    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      videoEl.srcObject = stream;
      if (mode === 'environment') {
        videoEl.classList.add('video-back-cam');
      } else {
        videoEl.classList.remove('video-back-cam');
      }
      return stream;
    } catch (e) {
      if (mode === 'environment') {
        return initWebcam(videoEl, 'user');
      }
      showAlert('Gagal akses kamera: ' + e.message);
      return null;
    }
  }

  async function ensureCamera(type) {
    const status = localStorage.getItem('cameraPermissionStatus');
    if (status !== 'granted') {
      showCameraPermissionModal();
      return;
    }
    if (type === 'verif' && streamVerif && streamVerif.active && activeStreamMode === currentFacingMode) return;
    if (type === 'reg' && streamReg && streamReg.active && activeStreamMode === currentFacingMode) return;

    if (type === 'reg') streamReg = await initWebcam(videoReg, currentFacingMode);
    if (type === 'verif') streamVerif = await initWebcam(videoVerif, currentFacingMode);

    activeStreamMode = currentFacingMode;
    updateCameraStatus('granted');
  }

  async function switchCamera() {
    currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
    activeStreamMode = null;
    if (!pageRegistrasi.classList.contains('hidden')) await ensureCamera('reg');
    else if (!pagePoli.classList.contains('hidden')) {
      stopAutoCheck();
      await ensureCamera('verif');
      startAutoCheck();
    }
  }

  if (btnSwitchCamReg)
    btnSwitchCamReg.addEventListener('click', (e) => {
      e.preventDefault();
      switchCamera();
    });
  if (btnSwitchCamVerif)
    btnSwitchCamVerif.addEventListener('click', (e) => {
      e.preventDefault();
      switchCamera();
    });

  // --- AUTO SCAN LOGIC ---
  function startAutoCheck() {
    if (autoCheckInterval) clearInterval(autoCheckInterval);
    faceDetectedTime = 0;

    autoCheckInterval = setInterval(async () => {
      if (pagePoli.classList.contains('hidden')) return;
      if (!streamVerif || videoVerif.paused || videoVerif.ended) return;
      if (isScanning) return;
      if (!modalLoading.classList.contains('hidden')) return;
      if (!verifResult.classList.contains('hidden')) return;

      const frameBlob = await captureSingleFrame(videoVerif, 240, 0.5);
      if (!frameBlob) return;

      const fd = new FormData();
      fd.append('frame', frameBlob, 'check.jpg');

      try {
        const r = await fetch('/api/check_face', { method: 'POST', body: fd });
        const d = await r.json();

        if (d.ok && d.found) {
          faceDetectedTime += CHECK_INTERVAL;
          statusVerif.textContent = `Wajah terdeteksi... ${Math.ceil((REQUIRED_TIME - faceDetectedTime) / 1000)}s`;
          updateCountdownUI(faceDetectedTime, REQUIRED_TIME);
          if (faceDetectedTime >= REQUIRED_TIME) triggerAutoScan();
        } else {
          faceDetectedTime = 0;
          statusVerif.textContent = 'Menunggu wajah...';
          resetCountdownUI();
        }
      } catch (err) {}
    }, CHECK_INTERVAL);
  }

  function stopAutoCheck() {
    if (autoCheckInterval) {
      clearInterval(autoCheckInterval);
      autoCheckInterval = null;
    }
    resetCountdownUI();
  }

  function updateCountdownUI(current, total) {
    if (overlayAuto) overlayAuto.classList.remove('hidden');
    if (focusBox) {
      focusBox.classList.remove('border-red-600');
      focusBox.classList.add('border-primary-500');
    }
    const remaining = Math.ceil((total - current) / 1000);
    if (textCountdown) textCountdown.textContent = remaining > 0 ? remaining : 'Scan';
    if (circleProgress) {
      const percentage = Math.min(current / total, 1);
      const offset = CIRCLE_FULL - percentage * CIRCLE_FULL;
      circleProgress.style.strokeDashoffset = offset;
    }
  }

  function resetCountdownUI() {
    if (overlayAuto) overlayAuto.classList.add('hidden');
    if (focusBox) {
      focusBox.classList.remove('border-red-600');
      focusBox.classList.remove('border-primary-500');
    }
    if (textCountdown) textCountdown.textContent = '3';
    if (circleProgress) circleProgress.style.strokeDashoffset = CIRCLE_FULL;
  }

  async function triggerAutoScan() {
    isScanning = true;
    stopAutoCheck();
    btnScan.click();
  }

  // --- NEXT SCAN COUNTDOWN (RESTORED) ---
  function startNextScanCountdown() {
    stopNextScanCountdown();
    let seconds = 10;

    // Helper untuk update text di UI yang baru di-generate
    const updateText = () => {
      const el = document.getElementById('next-scan-text-dynamic');
      if (el) el.textContent = `Verifikasi selanjutnya dalam ${seconds} detik...`;
    };

    updateText();
    nextScanTimer = setInterval(() => {
      seconds--;
      updateText();
      if (seconds <= 0) {
        stopNextScanCountdown();
        resetVerif();
      }
    }, 1000);
  }

  function stopNextScanCountdown() {
    if (nextScanTimer) {
      clearInterval(nextScanTimer);
      nextScanTimer = null;
    }
  }

  // --- IMAGE CAPTURE ---
  function captureSingleFrame(videoEl, width = 320, quality = 0.5) {
    return new Promise((resolve) => {
      if (!videoEl.videoWidth) return resolve(null);
      const canvas = document.createElement('canvas');
      const scale = width / videoEl.videoWidth;
      canvas.width = width;
      canvas.height = videoEl.videoHeight * scale;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(resolve, 'image/jpeg', quality);
    });
  }

  function captureFrames(videoEl, total = 3, gap = 150, counterEl = null, label = 'Frame', quality = 0.7) {
    return new Promise((resolve) => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      const frames = [];
      let taken = 0;
      const targetWidth = 400;
      const scale = targetWidth / videoEl.videoWidth;
      canvas.width = targetWidth;
      canvas.height = videoEl.videoHeight * scale;

      const grab = () => {
        if (!videoEl.videoWidth) return requestAnimationFrame(grab);
        ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(
          (b) => {
            frames.push(b);
            taken++;
            if (counterEl) counterEl.textContent = taken;
            updateProgress(taken, total, label);
            if (taken >= total) resolve(frames);
            else setTimeout(grab, gap);
          },
          'image/jpeg',
          quality
        );
      };
      grab();
    });
  }

  // --- HELPER UI ---
  function showAlert(msg) {
    if (alertMessage) alertMessage.textContent = msg;
    if (modalAlert) modalAlert.classList.remove('hidden');
  }
  function showLoading(text) {
    if (loadingText) loadingText.textContent = text;
    if (progressInner) progressInner.style.width = '0%';
    if (modalLoading) modalLoading.classList.remove('hidden');
  }
  function hideLoading() {
    if (modalLoading) modalLoading.classList.add('hidden');
  }
  function updateProgress(c, t, label) {
    const pct = Math.round((c / t) * 100);
    if (progressInner) progressInner.style.width = pct + '%';
    if (loadingText) loadingText.textContent = `${label} ${c}/${t} (${pct}%)`;
  }
  function computeAge(dob) {
    if (!dob) return '-';
    const d = new Date(dob);
    const today = new Date();
    let age = today.getFullYear() - d.getFullYear();
    const m = today.getMonth() - d.getMonth();
    if (m < 0 || (m === 0 && today.getDate() < d.getDate())) age--;
    return `${age} Tahun`;
  }

  // --- EVENT LISTENERS ---

  // Registrasi
  formRegistrasi.addEventListener('submit', async (e) => {
    e.preventDefault();
    const nikVal = inputNik.value.trim();

    const status = localStorage.getItem('cameraPermissionStatus');
    if (status !== 'granted') {
      showCameraPermissionModal();
      return;
    }

    await ensureCamera('reg');
    if (!streamReg) {
      statusReg.textContent = 'Gagal kamera';
      return;
    }

    statusReg.textContent = 'Mengambil foto...';
    countReg.textContent = '0';
    showLoading('Registrasi: mengambil foto...');

    const frames = await captureFrames(videoReg, 15, 100, countReg, 'Foto', 0.8);
    updateProgress(15, 15, 'Mengirim');

    const fd = new FormData();
    fd.append('nik', nikVal);
    fd.append('name', inputNama.value.trim());
    fd.append('dob', inputDob.value);
    fd.append('address', inputAlamat.value.trim());
    frames.forEach((b, i) => fd.append('frames[]', b, `frame_${i}.jpg`));

    try {
      const r = await fetch('/api/register', { method: 'POST', body: fd });
      const d = await r.json();
      hideLoading();
      if (!d.ok) {
        showAlert(d.msg || 'Gagal');
        statusReg.textContent = 'Gagal';
        return;
      }
      activePatient = { nik: nikVal, name: inputNama.value.trim(), address: inputAlamat.value.trim(), dob: inputDob.value };
      formRegistrasi.reset();
      countReg.textContent = '0';
      if (modalRegisSuccess) modalRegisSuccess.classList.remove('hidden');
    } catch (err) {
      hideLoading();
      showAlert('Error: ' + err.message);
    }
  });

  if (btnModalRegisTutup)
    btnModalRegisTutup.addEventListener('click', () => {
      if (modalRegisSuccess) modalRegisSuccess.classList.add('hidden');
      showPage('page-home');
    });

  // --- VERIFIKASI (RESTORED DETAILS) ---
  btnScan.addEventListener('click', async () => {
    isScanning = true;
    resetCountdownUI();
    stopNextScanCountdown();

    const status = localStorage.getItem('cameraPermissionStatus');
    if (status !== 'granted') {
      showCameraPermissionModal();
      isScanning = false;
      return;
    }

    await ensureCamera('verif');
    if (!streamVerif) {
      isScanning = false;
      return;
    }

    // HITUNG DURASI
    verificationStartTime = Date.now();
    statusVerif.textContent = 'Memverifikasi...';
    showLoading('Verifikasi: mengambil foto...');

    const frames = await captureFrames(videoVerif, 3, 100, null, 'Verifikasi', 0.7);
    updateProgress(3, 3, 'Memproses');

    const fd = new FormData();
    frames.forEach((b, i) => fd.append('frames[]', b, `scan_${i}.jpg`));

    try {
      const r = await fetch('/api/recognize', { method: 'POST', body: fd });
      const d = await r.json();
      hideLoading();

      if (!d.ok || !d.found) {
        statusVerif.textContent = 'Gagal/Tidak Dikenali';
        showAlert(d.msg || 'Wajah tidak dikenali.');
        activePatient = null;
        isScanning = false;
        setTimeout(startAutoCheck, 2000);
        return;
      }

      // HITUNG DURASI SELESAI
      const elapsed = ((Date.now() - verificationStartTime) / 1000).toFixed(1);

      statusVerif.textContent = 'Berhasil';
      activePatient = { nik: d.nik, name: d.name, address: d.address, dob: d.dob, age: d.age };

      // RESTORED UI: Lengkap dengan Waktu Verifikasi & Timer Reset
      verifData.innerHTML = `
        <div class="space-y-2">
            <p><strong>NIK:</strong> <span class="font-mono">${d.nik}</span></p>
            <p><strong>Nama:</strong> ${d.name}</p>
            <p><strong>Tanggal Lahir:</strong> ${d.dob || '-'}</p>
            <p><strong>Umur:</strong> ${d.age}</p>
            <p><strong>Alamat:</strong> ${d.address}</p>
            <p><strong>Waktu Verifikasi:</strong> ${elapsed} detik</p>


            <div class="bg-blue-50 dark:bg-[#1e293b] text-blue-800 dark:text-blue-300 text-sm px-4 py-2 rounded-lg mt-2 flex items-center gap-2 animate-pulse border dark:border-border">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                <span id="next-scan-text-dynamic" class="font-medium">Verifikasi selanjutnya dalam 10 detik...</span>
            </div>
        </div>
      `;
      verifResult.classList.remove('hidden');
      startNextScanCountdown(); // Timer jalan lagi
    } catch (err) {
      hideLoading();
      showAlert('Error: ' + err.message);
      isScanning = false;
    }
  });

  // Navigation Event Listeners
  if (navHome) navHome.addEventListener('click', () => showPage('page-home'));
  if (navRegistrasi) navRegistrasi.addEventListener('click', () => showPage('page-registrasi'));
  if (navPoli) navPoli.addEventListener('click', () => showPage('page-poli'));
  if (btnHomeRegistrasi) btnHomeRegistrasi.addEventListener('click', () => showPage('page-registrasi'));
  if (btnHomeKePoli) btnHomeKePoli.addEventListener('click', () => showPage('page-poli'));

  // Modal Closers
  if (btnAlertOk) btnAlertOk.addEventListener('click', () => modalAlert.classList.add('hidden'));
  if (btnAntrianTutup)
    btnAntrianTutup.addEventListener('click', () => {
      modalAntrian.classList.add('hidden');
      showPage('page-home');
    });

  if (btnDetailData) btnDetailData.addEventListener('click', stopNextScanCountdown);
  if (btnLanjutForm)
    btnLanjutForm.addEventListener('click', () => {
      stopNextScanCountdown();
      if (activePatient) {
        gwNama.textContent = activePatient.name;
        gwUmur.textContent = activePatient.age || computeAge(activePatient.dob);
        gwAlamat.textContent = activePatient.address;
        showPage('page-poli-gateway');
      }
    });
  if (btnModalLanjutPoli)
    btnModalLanjutPoli.addEventListener('click', () => {
      if (modalRegisSuccess) modalRegisSuccess.classList.add('hidden');
      if (activePatient) {
        gwNama.textContent = activePatient.name;
        gwUmur.textContent = computeAge(activePatient.dob);
        gwAlamat.textContent = activePatient.address;
        showPage('page-poli-gateway');
      }
    });

  // Poli Submit
  if (formPoliGateway)
    formPoliGateway.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (!activePatient) return;
      showLoading('Mengambil antrian...');
      try {
        const r = await fetch('/api/queue/assign', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ poli: gwPoli.value }),
        });
        const d = await r.json();
        hideLoading();
        if (d.ok) {
          antrianPoli.textContent = d.poli;
          antrianNomor.textContent = d.nomor;
          modalAntrian.classList.remove('hidden');
          formPoliGateway.reset();
          activePatient = null;
        } else {
          showAlert('Gagal ambil nomor');
        }
      } catch (err) {
        hideLoading();
        showAlert('Error: ' + err.message);
      }
    });

  // --- PERMISSION LOGIC ---
  function showCameraPermissionModal() {
    if (modalCameraPermission) modalCameraPermission.classList.remove('hidden');
  }

  function hideCameraPermissionModal() {
    if (modalCameraPermission) modalCameraPermission.classList.add('hidden');
  }

  function updateCameraStatus(s) {
    if (!cameraStatus || !cameraStatusDot || !cameraStatusText) return;
    if (s === 'granted') {
      cameraStatusDot.className = 'w-2 h-2 rounded-full bg-green-500';
      cameraStatusText.textContent = 'Camera: Siap';
      cameraStatus.className = 'flex items-center gap-2 mb-8 px-3 py-1 rounded-full text-xs font-medium bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 w-fit mx-auto';
    } else {
      cameraStatusDot.className = 'w-2 h-2 rounded-full bg-red-500';
      cameraStatusText.textContent = 'Camera: Belum Izin';
      cameraStatus.className = 'flex items-center gap-2 mb-8 px-3 py-1 rounded-full text-xs font-medium bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 w-fit mx-auto';
    }
  }

  if (btnCameraPermission)
    btnCameraPermission.addEventListener('click', async () => {
      try {
        const s = await navigator.mediaDevices.getUserMedia({ video: true });
        s.getTracks().forEach((t) => t.stop());
        localStorage.setItem('cameraPermissionStatus', 'granted');
        hideCameraPermissionModal();
        showAlert('Kamera diizinkan! Silakan lanjutkan.');
        updateCameraStatus('granted');
      } catch (e) {
        localStorage.setItem('cameraPermissionStatus', 'denied');
        hideCameraPermissionModal();
        showAlert('Akses ditolak. Fitur scan wajah tidak bisa digunakan.');
        updateCameraStatus('denied');
      }
    });

  if (btnCameraPermissionLater)
    btnCameraPermissionLater.addEventListener('click', () => {
      hideCameraPermissionModal();
    });

  if (localStorage.getItem('cameraPermissionStatus') === 'granted') {
    updateCameraStatus('granted');
  } else {
    updateCameraStatus('denied');
    setTimeout(showCameraPermissionModal, 1500);
  }

  showPage('page-home');
})();
