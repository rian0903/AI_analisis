// ==========================================================================
// Modern JavaScript Controller for Content Moderation App Container
// ==========================================================================

const CLASS_CONFIG = {
  Safe: {
    title: "SAFE CONTENT",
    desc: "Konten aman dan wajar tanpa indikasi kekerasan atau pornografi.",
    emoji: "✅",
    theme: "safe",
  },
  Violence: {
    title: "VIOLENCE DETECTED",
    desc: "Terdeteksi visual kekerasan fisik, senjata, atau darah.",
    emoji: "⚠️",
    theme: "violence",
  },
  NSFW: {
    title: "NSFW / ADULT CONTENT",
    desc: "Terdeteksi konten eksplisit dewasa, ketelanjangan, atau sensual.",
    emoji: "🔞",
    theme: "nsfw",
  },
};

let currentImageData = null; // { type: 'file'|'base64', data: '...', name: '...' }
let currentGradcamData = { original: null, heatmap: null, overlay: null };
let activeGradcamView = 'overlay';

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initUploadHandlers();
  initSampleButtons();
  initGradcamTabs();
  initBatchHandlers();
  initModelSelector();

  // Wait for pywebview ready event
  window.addEventListener('pywebviewready', () => {
    showToast('🚀 Desktop App Container Siap!', 'success');
    loadModelsList();
    loadModelInfo();
  });
});

// --------------------------------------------------------------------------
// Navigation Tabs
// --------------------------------------------------------------------------
function initTabs() {
  const tabs = document.querySelectorAll('.nav-tab');
  const panes = document.querySelectorAll('.tab-pane');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      panes.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const targetId = tab.getAttribute('data-tab');
      document.getElementById(targetId).classList.add('active');
    });
  });
}

// --------------------------------------------------------------------------
// Upload & File Handling
// --------------------------------------------------------------------------
function initUploadHandlers() {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const btnBrowse = document.getElementById('btnBrowseFile');
  const btnRemove = document.getElementById('btnRemoveImage');
  const btnAnalyze = document.getElementById('btnAnalyze');

  btnBrowse.addEventListener('click', async (e) => {
    e.stopPropagation();
    if (window.pywebview && window.pywebview.api && window.pywebview.api.open_file_dialog) {
      try {
        const filePath = await window.pywebview.api.open_file_dialog();
        if (filePath) {
          handleSelectedFilePath(filePath);
        }
      } catch (err) {
        fileInput.click();
      }
    } else {
      fileInput.click();
    }
  });

  dropzone.addEventListener('click', () => {
    if (!currentImageData) {
      btnBrowse.click();
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  });

  // Drag & Drop
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  btnRemove.addEventListener('click', (e) => {
    e.stopPropagation();
    clearSelectedImage();
  });

  btnAnalyze.addEventListener('click', () => {
    runAnalysis();
  });
}

function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    showToast('Harap pilih file gambar valid (JPG, PNG, WEBP).', 'warn');
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    const base64Data = e.target.result;
    displayPreview(base64Data);
    currentImageData = {
      type: 'base64',
      data: base64Data,
      name: file.name
    };
    document.getElementById('btnAnalyze').disabled = false;
  };
  reader.readAsDataURL(file);
}

async function handleSelectedFilePath(filePath) {
  if (!filePath) return;
  const fileName = filePath.split(/[\\/]/).pop();
  try {
    let previewSrc = filePath;
    if (window.pywebview && window.pywebview.api.get_file_base64) {
      previewSrc = await window.pywebview.api.get_file_base64(filePath);
    }
    displayPreview(previewSrc);
    currentImageData = {
      type: 'path',
      data: filePath,
      name: fileName
    };
    document.getElementById('btnAnalyze').disabled = false;
  } catch (err) {
    showToast('Gagal memuat pratinjau citra: ' + err, 'error');
  }
}

function displayPreview(src) {
  const dropzoneContent = document.querySelector('.dropzone-content');
  const previewContainer = document.getElementById('imagePreviewContainer');
  const imgPreview = document.getElementById('imagePreview');

  dropzoneContent.classList.add('hidden');
  previewContainer.classList.remove('hidden');
  imgPreview.src = src;
}

function clearSelectedImage() {
  currentImageData = null;
  document.getElementById('fileInput').value = '';
  document.querySelector('.dropzone-content').classList.remove('hidden');
  document.getElementById('imagePreviewContainer').classList.add('hidden');
  document.getElementById('btnAnalyze').disabled = true;

  document.getElementById('emptyResultPlaceholder').classList.remove('hidden');
  document.getElementById('resultContent').classList.add('hidden');
}

// --------------------------------------------------------------------------
// Quick Samples
// --------------------------------------------------------------------------
function initSampleButtons() {
  const chips = document.querySelectorAll('.chip');
  chips.forEach(chip => {
    chip.addEventListener('click', async () => {
      const sampleType = chip.getAttribute('data-sample');
      if (window.pywebview && window.pywebview.api && window.pywebview.api.get_sample_image) {
        showToast(`Memuat contoh citra ${sampleType}...`, 'info');
        const res = await window.pywebview.api.get_sample_image(sampleType);
        if (res && res.base64) {
          displayPreview(res.base64);
          currentImageData = {
            type: 'base64',
            data: res.base64,
            name: res.name || `sample_${sampleType}.jpg`
          };
          document.getElementById('btnAnalyze').disabled = false;
          runAnalysis();
        } else {
          showToast('Contoh gambar tidak ditemukan pada dataset lokal.', 'warn');
        }
      } else {
        showToast(`Fitur sampel siap saat container Python aktif.`, 'info');
      }
    });
  });
}

// --------------------------------------------------------------------------
// Single Image Inference
// --------------------------------------------------------------------------
async function runAnalysis() {
  if (!currentImageData) return;

  const btnAnalyze = document.getElementById('btnAnalyze');
  const spinner = document.getElementById('analyzeSpinner');
  btnAnalyze.disabled = true;
  spinner.classList.remove('hidden');

  try {
    let result = null;
    if (window.pywebview && window.pywebview.api) {
      if (currentImageData.type === 'path') {
        result = await window.pywebview.api.predict_file(currentImageData.data);
      } else {
        result = await window.pywebview.api.predict_base64(currentImageData.data, currentImageData.name);
      }
    } else {
      // Demo mock fallback if opened without pywebview
      await new Promise(r => setTimeout(r, 600));
      result = getMockResult();
    }

    if (result && !result.error) {
      renderResult(result);
    } else {
      showToast('Gagal melakukan analisis: ' + (result ? result.error : 'Unknown error'), 'error');
    }
  } catch (err) {
    showToast('Terjadi kesalahan analisis: ' + err, 'error');
  } finally {
    btnAnalyze.disabled = false;
    spinner.classList.add('hidden');
  }
}

function renderResult(res) {
  document.getElementById('emptyResultPlaceholder').classList.add('hidden');
  document.getElementById('resultContent').classList.remove('hidden');

  const predClass = res.prediction;
  const confPct = (res.confidence * 100).toFixed(1);
  const cfg = CLASS_CONFIG[predClass] || CLASS_CONFIG.Safe;

  // Banner
  const banner = document.getElementById('resultBanner');
  banner.className = `result-banner ${cfg.theme}`;
  document.getElementById('bannerEmoji').textContent = cfg.emoji;
  document.getElementById('bannerClass').textContent = cfg.title;
  document.getElementById('bannerDesc').textContent = cfg.desc;
  document.getElementById('bannerConfidence').textContent = `${confPct}%`;

  // Inference Time
  document.getElementById('inferenceTimer').textContent = `Inference: ${res.inference_time_ms || 0} ms`;

  // Indicators
  const human = res.human_analysis || {};
  document.getElementById('indHuman').textContent = human.person_detected
    ? `Ya (${human.person_count || 1} orang)`
    : 'Tidak Terdeteksi';
  document.getElementById('indPose').textContent = human.pose_detected ? 'Terdeteksi' : 'None / Fallback';
  document.getElementById('indPipeline').textContent = res.pipeline || 'Global + Human + Pose';

  // Low confidence warning
  const lowConfWarn = document.getElementById('lowConfidenceWarning');
  if (res.low_confidence) {
    lowConfWarn.classList.remove('hidden');
  } else {
    lowConfWarn.classList.add('hidden');
  }

  // Probability Bars
  const probs = res.probabilities || { Safe: 0, Violence: 0, NSFW: 0 };
  const pSafe = ((probs.Safe || 0) * 100).toFixed(1);
  const pVio = ((probs.Violence || 0) * 100).toFixed(1);
  const pNsfw = ((probs.NSFW || 0) * 100).toFixed(1);

  document.getElementById('valSafe').textContent = `${pSafe}%`;
  document.getElementById('barSafe').style.width = `${pSafe}%`;

  document.getElementById('valViolence').textContent = `${pVio}%`;
  document.getElementById('barViolence').style.width = `${pVio}%`;

  document.getElementById('valNsfw').textContent = `${pNsfw}%`;
  document.getElementById('barNsfw').style.width = `${pNsfw}%`;

  // Explanation
  document.getElementById('explanationBox').textContent =
    res.explanation_summary || 'Model memproses fitur global, ekstraksi visual orang, dan relasi pose untuk menentukan klasifikasi.';

  // Grad-CAM Visualizer
  loadGradCAM(res);
}

// --------------------------------------------------------------------------
// Grad-CAM Controller
// --------------------------------------------------------------------------
function initGradcamTabs() {
  const tabs = document.querySelectorAll('.g-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      activeGradcamView = tab.getAttribute('data-view');
      updateGradcamDisplay();
    });
  });
}

async function loadGradCAM(res) {
  const imgElem = document.getElementById('gradcamImage');
  const spinner = document.getElementById('gradcamLoading');

  spinner.classList.remove('hidden');
  imgElem.classList.add('hidden');

  try {
    if (window.pywebview && window.pywebview.api && window.pywebview.api.compute_gradcam) {
      const gradcamRes = await window.pywebview.api.compute_gradcam(
        currentImageData.type === 'path' ? currentImageData.data : currentImageData.data,
        res.prediction
      );
      if (gradcamRes) {
        currentGradcamData = {
          original: gradcamRes.original,
          heatmap: gradcamRes.heatmap,
          overlay: gradcamRes.overlay,
        };
      }
    } else {
      // Fallback preview
      currentGradcamData = {
        original: currentImageData ? currentImageData.data : '',
        heatmap: currentImageData ? currentImageData.data : '',
        overlay: currentImageData ? currentImageData.data : '',
      };
    }
  } catch (err) {
    console.error('Grad-CAM calculation error:', err);
  } finally {
    spinner.classList.add('hidden');
    imgElem.classList.remove('hidden');
    updateGradcamDisplay();
  }
}

function updateGradcamDisplay() {
  const imgElem = document.getElementById('gradcamImage');
  const caption = document.getElementById('gradcamCaption');

  if (activeGradcamView === 'overlay' && currentGradcamData.overlay) {
    imgElem.src = currentGradcamData.overlay;
    caption.textContent = 'Overlay Heatmap: Wilayah merah-kuning menunjukkan fokus perhatian konvolusi tertinggi.';
  } else if (activeGradcamView === 'heatmap' && currentGradcamData.heatmap) {
    imgElem.src = currentGradcamData.heatmap;
    caption.textContent = 'Raw Heatmap: Peta aktivasi gradien murni (Jet Colormap) sebelum digabungkan.';
  } else if (activeGradcamView === 'original' && currentGradcamData.original) {
    imgElem.src = currentGradcamData.original;
    caption.textContent = 'Citra Asli: Format masukan ukuran standar 224x224.';
  }
}

// --------------------------------------------------------------------------
// Batch Testing
// --------------------------------------------------------------------------
function initBatchHandlers() {
  const btnSelect = document.getElementById('btnSelectFolder');
  const btnStart = document.getElementById('btnStartBatch');

  let selectedFolder = null;

  btnSelect.addEventListener('click', async () => {
    if (window.pywebview && window.pywebview.api && window.pywebview.api.open_folder_dialog) {
      const folder = await window.pywebview.api.open_folder_dialog();
      if (folder) {
        selectedFolder = folder;
        document.getElementById('selectedFolderPath').textContent = folder;
        btnStart.disabled = false;
        showToast('Folder dipilih: ' + folder, 'info');
      }
    } else {
      showToast('Buka di Container Desktop untuk memilih folder lokal.', 'warn');
    }
  });

  btnStart.addEventListener('click', async () => {
    if (!selectedFolder) return;
    btnStart.disabled = true;
    btnSelect.disabled = true;

    const progressBox = document.getElementById('batchProgressBox');
    const progressBar = document.getElementById('batchProgressBar');
    const progressText = document.getElementById('batchProgressText');
    const progressPct = document.getElementById('batchProgressPct');
    const statsGrid = document.getElementById('batchStats');
    const tbody = document.getElementById('batchTableBody');

    progressBox.classList.remove('hidden');
    statsGrid.classList.remove('hidden');
    tbody.innerHTML = '<tr><td colspan="8" class="text-center">Sedang memproses citra dalam folder...</td></tr>';

    try {
      if (window.pywebview && window.pywebview.api && window.pywebview.api.predict_batch) {
        const results = await window.pywebview.api.predict_batch(selectedFolder);
        renderBatchResults(results);
      } else {
        showToast('Batch testing memerlukan koneksi backend Python.', 'warn');
      }
    } catch (err) {
      showToast('Terjadi kesalahan batch: ' + err, 'error');
    } finally {
      btnStart.disabled = false;
      btnSelect.disabled = false;
    }
  });
}

function renderBatchResults(results) {
  const tbody = document.getElementById('batchTableBody');
  tbody.innerHTML = '';

  let countSafe = 0;
  let countVio = 0;
  let countNsfw = 0;
  let countLowConf = 0;

  results.forEach((r, idx) => {
    if (r.prediction === 'Safe') countSafe++;
    else if (r.prediction === 'Violence') countVio++;
    else if (r.prediction === 'NSFW') countNsfw++;

    if (r.low_confidence) countLowConf++;

    const row = document.createElement('tr');
    const confPct = (r.confidence * 100).toFixed(1);
    const badgeColor = r.prediction === 'Safe' ? '#10b981' : (r.prediction === 'Violence' ? '#f59e0b' : '#ef4444');

    row.innerHTML = `
      <td>${idx + 1}</td>
      <td style="font-family:var(--font-mono); font-size:0.78rem;">${r.file_path.split(/[\\/]/).pop()}</td>
      <td><strong style="color:${badgeColor}">${r.prediction}</strong></td>
      <td style="font-family:var(--font-mono);">${confPct}%</td>
      <td>${r.human_analysis && r.human_analysis.person_detected ? 'Ya (' + r.human_analysis.person_count + ')' : 'Tidak'}</td>
      <td>${r.human_analysis && r.human_analysis.pose_detected ? '✅' : '—'}</td>
      <td><span class="badge-pill">${r.pipeline || 'Multi'}</span></td>
      <td>${r.inference_time_ms || 0}</td>
    `;
    tbody.appendChild(row);
  });

  document.getElementById('statTotal').textContent = results.length;
  document.getElementById('statSafe').textContent = countSafe;
  document.getElementById('statViolence').textContent = countVio;
  document.getElementById('statNsfw').textContent = countNsfw;
  document.getElementById('statLowConf').textContent = countLowConf;

  document.getElementById('batchProgressBar').style.width = '100%';
  document.getElementById('batchProgressText').textContent = `Selesai: ${results.length} citra diproses.`;
  document.getElementById('batchProgressPct').textContent = '100%';
  showToast(`Evaluasi batch selesai (${results.length} citra)`, 'success');
}

// --------------------------------------------------------------------------
// Model Selector & Info
// --------------------------------------------------------------------------
function initModelSelector() {
  const selector = document.getElementById('modelSelector');
  const btnReload = document.getElementById('btnReloadModel');

  selector.addEventListener('change', async (e) => {
    const selectedModel = e.target.value;
    if (window.pywebview && window.pywebview.api && window.pywebview.api.load_model) {
      showToast(`Memuat model: ${selectedModel}...`, 'info');
      const res = await window.pywebview.api.load_model(selectedModel);
      if (res && res.success) {
        showToast(`Model ${selectedModel} berhasil dimuat!`, 'success');
        loadModelInfo();
      } else {
        showToast('Gagal memuat model: ' + (res ? res.error : ''), 'error');
      }
    }
  });

  btnReload.addEventListener('click', () => {
    loadModelsList();
    loadModelInfo();
    showToast('Daftar model diperbarui.', 'info');
  });
}

async function loadModelsList() {
  if (window.pywebview && window.pywebview.api && window.pywebview.api.get_available_models) {
    try {
      const models = await window.pywebview.api.get_available_models();
      const selector = document.getElementById('modelSelector');
      selector.innerHTML = '';
      models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.filename;
        opt.textContent = m.filename + (m.is_active ? ' (Aktif)' : '');
        if (m.is_active) opt.selected = true;
        selector.appendChild(opt);
      });
    } catch (err) {
      console.error('Error loading models:', err);
    }
  }
}

async function loadModelInfo() {
  if (window.pywebview && window.pywebview.api && window.pywebview.api.get_model_info) {
    try {
      const info = await window.pywebview.api.get_model_info();
      if (info) {
        document.getElementById('infoModelPath').textContent = info.model_path || '-';
        document.getElementById('infoInputShapes').textContent = info.input_shapes ? info.input_shapes.join(', ') : '-';
        document.getElementById('infoTotalParams').textContent = info.total_params ? Number(info.total_params).toLocaleString() : '-';
      }
    } catch (err) {
      console.error('Error loading model info:', err);
    }
  }
}

// --------------------------------------------------------------------------
// Toast Notifications
// --------------------------------------------------------------------------
let toastTimeout = null;
function showToast(message, type = 'info') {
  const toast = document.getElementById('toast');
  const toastIcon = document.getElementById('toastIcon');
  const toastMsg = document.getElementById('toastMsg');

  clearTimeout(toastTimeout);

  const icons = { info: 'ℹ️', success: '✅', warn: '⚠️', error: '❌' };
  toastIcon.textContent = icons[type] || 'ℹ️';
  toastMsg.textContent = message;

  toast.classList.remove('hidden');

  toastTimeout = setTimeout(() => {
    toast.classList.add('hidden');
  }, 3500);
}

// Fallback Mock for testing
function getMockResult() {
  return {
    prediction: 'Safe',
    confidence: 0.945,
    probabilities: { Safe: 0.945, Violence: 0.035, NSFW: 0.020 },
    human_analysis: { person_detected: true, person_count: 1, pose_detected: true },
    pipeline: 'global+human+pose',
    low_confidence: false,
    inference_time_ms: 45.2,
    explanation_summary: 'Gambar diklasifikasikan sebagai Safe karena tidak ditemukan indikasi kekerasan ataupun ketelanjangan.',
  };
}
