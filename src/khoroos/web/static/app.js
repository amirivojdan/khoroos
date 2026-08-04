/* Khoroos web UI.
 *
 * Colour policy, applied throughout: the 15 behaviour classes exceed the safe
 * categorical ceiling, so identity colour is carried by the six *behaviour groups*
 * (comfort, locomotion, inactive, ingestive, foraging, other) and the class name is
 * carried by labels, tooltips and tables. "Uncertain" is a neutral grey — it marks
 * absent information, so it is never given a series colour.
 */
'use strict';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  config: null,
  selection: null,        // { kind: 'upload', file, name, sizeMb }
  preset: 'balanced',
  jobId: null,
  result: null,
  eventSource: null,
  focusTrack: null,       // track id the user clicked, or null for the whole flock
  setupStage: 'file',     // current step within the opening-screen setup flow
  overlay: null,          // { tracks: Map, predictions: Map } — see buildOverlayIndex
  overlayFrame: null,     // pending requestAnimationFrame handle
};

const STAGES = [
  ['probe', 'Reading video'],
  ['detect', 'Detecting birds'],
  ['track', 'Following individuals'],
  ['clips', 'Selecting stable clips'],
  ['classify', 'Recognising actions'],
  ['metrics', 'Computing indicators'],
];

const GROUP_SLOT = {
  comfort: 1, locomotion: 2, inactive: 3, ingestive: 4, foraging: 5, other: 6,
};

const GROUP_LABEL = {
  comfort: 'Comfort', locomotion: 'Locomotion', inactive: 'Inactive',
  ingestive: 'Feeding & drinking', foraging: 'Foraging', other: 'Other',
};

// Tracking is identity, not behaviour: a bird keeps one trail colour while its action
// box changes colour as classifications change.
const TRACK_COLOURS = [
  '#00c2e8', '#ffd166', '#ef476f', '#06d6a0', '#a78bfa',
  '#f97316', '#84cc16', '#f472b6', '#38bdf8', '#facc15',
];
const TRACK_TRAIL_SECONDS = 2.0;
const TRACK_TRAIL_POINTS = 14;

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function groupOf(label) {
  if (!state.config || label === 'uncertain') return null;
  for (const [group, members] of Object.entries(state.config.behaviour_groups)) {
    if (members.includes(label)) return group;
  }
  return null;
}

function colourFor(label) {
  const group = groupOf(label);
  if (!group) return cssVar('--series-none');
  return cssVar(`--series-${GROUP_SLOT[group]}`);
}

function trackColour(trackId) {
  return TRACK_COLOURS[Math.abs(Number(trackId) || 0) % TRACK_COLOURS.length];
}

/**
 * Ink that stays legible on a filled swatch.
 *
 * The six group colours span a wide lightness range — white reads on the blues but
 * disappears on the amber — so the label picks its ink from the fill's luminance.
 */
function contrastText(colour) {
  const hex = colour.replace('#', '');
  const full = hex.length === 3 ? hex.split('').map((c) => c + c).join('') : hex;
  if (full.length !== 6) return '#ffffff';
  const channel = (offset) => {
    const value = parseInt(full.slice(offset, offset + 2), 16) / 255;
    return value <= 0.03928 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  };
  const luminance = 0.2126 * channel(0) + 0.7152 * channel(2) + 0.0722 * channel(4);
  return luminance > 0.4 ? '#101010' : '#ffffff';
}

function prettyLabel(label) {
  return label.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase());
}

function fmtPct(x) { return `${(x * 100).toFixed(1)}%`; }
function fmtTime(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${String(sec).padStart(2, '0')}`;
}
function fmtDuration(s) {
  if (s < 60) return `${s.toFixed(0)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m`;
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, options);
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch { /* not json */ }
    throw new Error(detail);
  }
  return response.json();
}

// ---------------------------------------------------------------------------
// Screens
// ---------------------------------------------------------------------------

function showScreen(name) {
  document.body.dataset.screen = name;
  for (const screen of ['select', 'running', 'results']) {
    $(`#screen-${screen}`).hidden = screen !== name;
  }
  $('#new-analysis').hidden = name === 'select';
  window.scrollTo({ top: 0, behavior: 'instant' });
}

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem('khoroos-theme', theme); } catch { /* private mode */ }
  if (state.result) renderResults(state.result);
}

// Restore the last choice, or honour an explicit ?theme= override.
(function initTheme() {
  const requested = new URLSearchParams(window.location.search).get('theme');
  let stored = null;
  try { stored = localStorage.getItem('khoroos-theme'); } catch { /* private mode */ }
  const theme = requested || stored;
  if (theme === 'light' || theme === 'dark') document.documentElement.dataset.theme = theme;
}());

$('#theme-toggle').addEventListener('click', () => {
  const root = document.documentElement;
  const dark = root.dataset.theme
    ? root.dataset.theme === 'dark'
    : window.matchMedia('(prefers-color-scheme: dark)').matches;
  applyTheme(dark ? 'light' : 'dark');
});

// ---------------------------------------------------------------------------
// Select screen — presets
// ---------------------------------------------------------------------------

const PRESET_COPY = {
  fast: ['Fast', 'Every 4th frame, 2-second steps. Best for a first look at long footage.'],
  balanced: ['Balanced', 'Every 2nd frame, 1-second steps. The recommended default.'],
  thorough: ['Thorough', 'Every frame, half-second steps. Slowest, most temporal detail.'],
};

function renderPresets() {
  const container = $('#preset-options');
  container.innerHTML = '';
  for (const name of Object.keys(state.config.presets)) {
    const [title, description] = PRESET_COPY[name] || [name, ''];
    const option = document.createElement('label');
    option.className = 'preset-option' + (name === state.preset ? ' is-active' : '');
    option.innerHTML = `
      <input type="radio" name="preset" value="${name}" ${name === state.preset ? 'checked' : ''}>
      <span>
        <span class="po-name">${title}</span><br>
        <span class="po-desc">${description}</span>
      </span>`;
    option.addEventListener('change', () => {
      state.preset = name;
      renderPresets();
    });
    container.appendChild(option);
  }
  renderBatchDefaults();
}

//: Batch-size inputs, and the parameter each one overrides.
const BATCH_INPUTS = {
  '#opt-det-batch': 'detection_batch_size',
  '#opt-act-batch': 'action_batch_size',
};

/**
 * Show the selected preset's batch sizes as placeholders.
 *
 * Unlike the confidence fields, these differ per preset, so the inputs stay empty and
 * carry the preset's value as a placeholder. Switching detail level then updates what
 * the field says it will do, and an untouched field never sends a stale number.
 */
function renderBatchDefaults() {
  const preset = (state.config.presets || {})[state.preset] || {};
  for (const [selector, field] of Object.entries(BATCH_INPUTS)) {
    const input = $(selector);
    if (input) input.placeholder = preset[field] !== undefined ? preset[field] : 'default';
  }
}

//: Copy for the alert thresholds, keyed by the field names WelfareThresholds defines.
//: A field with no entry here still renders — the server owns the list, not this file.
const THRESHOLD_COPY = {
  min_comfort_share: ['Minimum comfort behaviour', 'Preening, dust bathing, wing flapping, stretching.'],
  min_locomotion_share: ['Minimum locomotion', 'Walking and running. Low values can indicate leg problems.'],
  max_inactive_share: ['Maximum inactivity', 'Resting and standing.'],
  min_ingestive_share: ['Minimum feeding and drinking', ''],
  min_observation_seconds: ['Minimum observation (bird-seconds)', 'Indicators below this are shown but never alerted on.'],
  min_class_f1_for_alert: ['Minimum class reliability (F1)', 'Behaviours the model predicts less reliably do not raise alerts.'],
};

function renderThresholds() {
  const container = $('#threshold-options');
  container.innerHTML = '';
  for (const [name, value] of Object.entries(state.config.thresholds || {})) {
    const [title, description] = THRESHOLD_COPY[name] || [prettyLabel(name), ''];
    const isShare = name.endsWith('_share') || name.endsWith('_f1_for_alert');

    const row = document.createElement('div');
    row.className = 'field-row';
    row.innerHTML = `
      <label for="thr-${name}">${title}</label>
      <input id="thr-${name}" data-threshold="${name}" type="number"
             min="0" ${isShare ? 'max="1" step="0.05"' : 'step="10"'} value="${value}">
      ${description ? `<span class="hint">${description}</span>` : ''}`;
    container.appendChild(row);
  }
}

/** Collect only the thresholds the user actually moved, so defaults stay server-side. */
function thresholdOverrides() {
  const defaults = state.config.thresholds || {};
  const overrides = {};
  for (const input of document.querySelectorAll('[data-threshold]')) {
    const name = input.dataset.threshold;
    if (input.value === '' || Number(input.value) === defaults[name]) continue;
    overrides[name] = Number(input.value);
  }
  return overrides;
}

// ---------------------------------------------------------------------------
// Select screen — two-stage setup
// ---------------------------------------------------------------------------

function showSelectLanding() {
  document.body.dataset.selectView = 'landing';
  $('#select-landing').hidden = false;
  $('#select-setup').hidden = true;
}

function showSelectSetup() {
  document.body.dataset.selectView = 'setup';
  $('#select-landing').hidden = true;
  $('#select-setup').hidden = false;
  showSelectStage(state.setupStage);
}

function showSelectStage(stage) {
  state.setupStage = stage;
  const choosingFile = stage === 'file';
  $('#select-stage-file').hidden = !choosingFile;
  $('#select-stage-options').hidden = choosingFile;

  const videoStep = $('#step-video');
  const optionsStep = $('#step-options');
  videoStep.classList.toggle('is-active', choosingFile);
  videoStep.classList.toggle('is-complete', !choosingFile);
  optionsStep.classList.toggle('is-active', !choosingFile);

  if (choosingFile) {
    videoStep.setAttribute('aria-current', 'step');
    optionsStep.removeAttribute('aria-current');
  } else {
    videoStep.removeAttribute('aria-current');
    optionsStep.setAttribute('aria-current', 'step');
  }
}

$('#getting-started').addEventListener('click', showSelectSetup);
$('#setup-back').addEventListener('click', showSelectLanding);

function setSelection(selection) {
  state.selection = selection;
  $('#selection').hidden = false;
  $('#selection-name').textContent = selection.name;
  $('#selection-size').textContent = selection.sizeMb ? `${selection.sizeMb} MB` : '';
  $('#start-btn').disabled = false;
  showSelectStage('options');
  requestAnimationFrame(() => $('#select-stage-options').focus({ preventScroll: true }));
}

// Upload
const dropzone = $('#dropzone');
$('#file-input').addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (file) {
    setSelection({
      kind: 'upload', file, name: file.name, sizeMb: (file.size / 1e6).toFixed(1),
    });
  }
});
['dragover', 'dragenter'].forEach((type) => dropzone.addEventListener(type, (e) => {
  e.preventDefault(); dropzone.classList.add('is-over');
}));
['dragleave', 'drop'].forEach((type) => dropzone.addEventListener(type, (e) => {
  e.preventDefault(); dropzone.classList.remove('is-over');
}));
dropzone.addEventListener('drop', (event) => {
  const file = event.dataTransfer.files[0];
  if (file) {
    setSelection({
      kind: 'upload', file, name: file.name, sizeMb: (file.size / 1e6).toFixed(1),
    });
  }
});

$('#change-video').addEventListener('click', () => {
  $('#file-input').value = '';
  $('#start-error').hidden = true;
  showSelectStage('file');
});

// ---------------------------------------------------------------------------
// Starting a job
// ---------------------------------------------------------------------------

$('#start-btn').addEventListener('click', async () => {
  if (!state.selection) return;
  const button = $('#start-btn');
  button.disabled = true;
  button.textContent = 'Starting…';
  $('#start-error').hidden = true;

  const form = new FormData();
  form.append('preset', state.preset);
  form.append('file', state.selection.file);
  const maxSeconds = $('#opt-max-seconds').value;
  if (maxSeconds) form.append('max_seconds', maxSeconds);
  form.append('min_confidence', $('#opt-min-conf').value);
  form.append('detection_confidence', $('#opt-det-conf').value);
  // Only sent when the user actually set one, so the preset's own value stands otherwise.
  for (const [selector, field] of Object.entries(BATCH_INPUTS)) {
    const value = $(selector).value.trim();
    if (value) form.append(field, value);
  }
  form.append('render_overlay', $('#opt-overlay').checked ? 'true' : 'false');
  const thresholds = thresholdOverrides();
  if (Object.keys(thresholds).length) form.append('thresholds', JSON.stringify(thresholds));

  try {
    const job = await api('/jobs', { method: 'POST', body: form });
    state.jobId = job.job_id;
    startWatching(job);
  } catch (error) {
    $('#start-error').textContent = error.message;
    $('#start-error').hidden = false;
  } finally {
    button.disabled = false;
    button.textContent = 'Start analysis';
  }
});

$('#new-analysis').addEventListener('click', () => {
  if (state.eventSource) state.eventSource.close();
  stopOverlayLoop();
  $('#player').pause();
  state.jobId = null; state.result = null; state.focusTrack = null; state.overlay = null;
  state.selection = null;
  $('#file-input').value = '';
  $('#selection').hidden = true;
  $('#start-btn').disabled = true;
  showSelectStage('file');
  showSelectLanding();
  showScreen('select');
  loadRecentJobs();
});

$('#cancel-btn').addEventListener('click', async () => {
  if (!state.jobId) return;
  try { await api(`/jobs/${state.jobId}/cancel`, { method: 'POST' }); } catch { /* already done */ }
});

// ---------------------------------------------------------------------------
// Running screen
// ---------------------------------------------------------------------------

function startWatching(job) {
  showScreen('running');
  $('#running-file').textContent = job.filename;
  $('#running-title').textContent = 'Analysing';
  renderStages('probe');

  if (state.eventSource) state.eventSource.close();
  const source = new EventSource(`/api/jobs/${job.job_id}/events`);
  state.eventSource = source;

  source.onmessage = (event) => {
    const status = JSON.parse(event.data);
    updateProgress(status);
    if (['completed', 'failed', 'cancelled'].includes(status.state)) {
      source.close();
      state.eventSource = null;
      if (status.state === 'completed') loadResult(job.job_id);
      else if (status.state === 'cancelled') showScreen('select');
      else {
        $('#running-title').textContent = 'Analysis failed';
        const message = $('#running-message');
        message.innerHTML = '';
        const notice = document.createElement('span');
        notice.className = 'notice notice-error';
        notice.textContent = status.error || status.message;
        message.appendChild(notice);
      }
    }
  };
  source.onerror = () => {
    // The stream ends when the job terminates; poll once to find out which.
    source.close();
    state.eventSource = null;
    api(`/jobs/${job.job_id}`).then((status) => {
      if (status.state === 'completed') loadResult(job.job_id);
    }).catch(() => { /* job vanished */ });
  };
}

function updateProgress(status) {
  const pct = Math.round(status.progress * 100);
  $('#progress-fill').style.width = `${pct}%`;
  $('#progress-pct').textContent = `${pct}%`;
  $('.progress-track').setAttribute('aria-valuenow', pct);
  $('#running-message').textContent = status.message || '';

  const eta = status.detail && status.detail.eta_s;
  $('#progress-eta').textContent = eta
    ? `about ${fmtDuration(eta)} remaining`
    : (status.state === 'running' ? 'estimating…' : '');

  renderStages(status.stage);
}

function renderStages(activeStage) {
  const activeIndex = STAGES.findIndex(([key]) => key === activeStage);
  $('#stage-list').innerHTML = STAGES.map(([key, label], index) => {
    let cls = 'stage';
    if (activeIndex >= 0 && index < activeIndex) cls += ' is-done';
    if (key === activeStage) cls += ' is-active';
    return `<li class="${cls}"><span class="dot"></span>${label}</li>`;
  }).join('');
}

async function loadResult(jobId) {
  try {
    const result = await api(`/jobs/${jobId}/result`);
    state.result = result;
    state.jobId = jobId;
    // Reveal the section *before* drawing: canvases measure clientWidth, which is 0
    // while the parent is hidden, and would render blank at the wrong size.
    showScreen('results');
    renderResults(result);
  } catch (error) {
    $('#running-title').textContent = 'Could not load results';
    $('#running-message').textContent = error.message;
  }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

function renderResults(result) {
  const metrics = result.metrics || {};
  renderKpis(result, metrics);
  setupPlayer(result);
  renderLegend();
  drawStackTimeline();
  drawBirdTimeline();
  renderBudget(metrics);
  renderPopulation(metrics);
  renderSpatial(metrics, result);
  renderBouts(metrics);
  renderExports();
  renderParams(result);
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function renderKpis(result, metrics) {
  const indicators = metrics.indicators || {};
  const budget = metrics.time_budget || {};
  const population = metrics.population || {};

  const tiles = [
    {
      label: 'Birds tracked',
      value: (result.tracks || []).length,
      note: `${population.mean ?? 0} on average in frame`,
    },
    {
      label: 'Comfort behaviour',
      value: fmtPct(indicators.comfort_index || 0),
      note: 'share of observed bird-time',
    },
    {
      label: 'Locomotion',
      value: fmtPct(indicators.locomotion_score || 0),
      note: 'share of observed bird-time',
    },
    {
      label: 'Inactive',
      value: fmtPct(indicators.inactivity_ratio || 0),
      note: 'share of observed bird-time',
    },
    {
      label: 'Feeding & drinking',
      value: fmtPct(indicators.ingestive_share || 0),
      note: 'share of observed bird-time',
    },
    {
      label: 'Observed',
      value: fmtDuration(budget.total_bird_seconds || 0),
      note: `of bird-time · ${fmtPct(budget.uncertain_share || 0)} uncertain`,
    },
  ];

  $('#kpi-row').innerHTML = tiles.map((tile) => {
    return `<div class="kpi">
      <div class="kpi-label">${tile.label}</div>
      <div class="kpi-value">${tile.value}</div>
      <div class="kpi-note">${tile.note}</div>
    </div>`;
  }).join('');
}

// ---------------------------------------------------------------------------
// Player
// ---------------------------------------------------------------------------

function setupPlayer(result) {
  const player = $('#player');
  // renderResults() re-runs on a theme change; reloading the source then would restart
  // playback and lose the annotated choice, so the video is only (re)pointed at a job
  // when the job itself changes.
  if (player._jobId !== state.jobId) {
    player._jobId = state.jobId;
    $('#overlay-switch').checked = false;
    player.src = `/api/jobs/${state.jobId}/video`;
  }
  buildOverlayIndex(result);

  api(`/jobs/${state.jobId}`).then((status) => {
    const wrap = $('#overlay-switch-wrap');
    wrap.hidden = !status.has_overlay;
    if (status.has_overlay) {
      $('#overlay-switch').onchange = (event) => {
        const time = player.currentTime;
        const resume = !player.paused && !player.ended;
        stopOverlayLoop();
        player.src = `/api/jobs/${state.jobId}/video?annotated=${event.target.checked}`;
        player.addEventListener('loadedmetadata', () => {
          player.currentTime = Math.min(time, player.duration || time);
          if (resume) player.play().catch(() => { /* browser may require user input */ });
          drawPlayerOverlay();
        }, { once: true });
      };
    }
  }).catch(() => { /* status unavailable */ });

  // renderResults() runs again on every theme change, so listeners are attached once.
  if (player._wired) {
    drawPlayerOverlay();
    return;
  }
  player._wired = true;

  player.addEventListener('timeupdate', () => {
    drawStackTimeline();
    drawBirdTimeline();
    drawPlayerOverlay();
  });
  // `timeupdate` fires only ~4×/s — too coarse for boxes that must sit on a moving
  // bird, so playback drives its own frame loop and the discrete events cover the
  // states where no loop is running.
  player.addEventListener('play', startOverlayLoop);
  ['pause', 'seeked', 'loadedmetadata', 'loadeddata', 'ended'].forEach((event) => {
    player.addEventListener(event, drawPlayerOverlay);
  });
  $('#tracks-switch').addEventListener('change', drawPlayerOverlay);

  // The SVG charts are drawn at a measured pixel width, so unlike the canvases they do
  // not simply re-flow — they have to be rebuilt when the column width changes. Guarded
  // on the width actually changing because mobile browsers fire `resize` every time the
  // URL bar retracts, and rebuilding four charts on a scroll gesture is visible jank.
  let lastWidth = window.innerWidth;
  window.addEventListener('resize', debounce(() => {
    drawStackTimeline();
    drawBirdTimeline();
    drawPlayerOverlay();
    if (window.innerWidth !== lastWidth) {
      lastWidth = window.innerWidth;
      redrawCharts();
    }
  }, 120));
  if ('ResizeObserver' in window) {
    player._overlayObserver = new ResizeObserver(drawPlayerOverlay);
    player._overlayObserver.observe(player);
  }
}

/** Re-measure and rebuild the four SVG charts. Cheap enough to run on a resize. */
function redrawCharts() {
  if (!state.result) return;
  const metrics = state.result.metrics || {};
  // A container swapped out for its table view has no box to measure; leave it alone
  // rather than redrawing it against a fallback width it would then keep.
  if ($('#chart-budget').clientWidth) renderBudget(metrics);
  renderPopulation(metrics);
  renderSpatial(metrics, state.result);
  renderBouts(metrics);
}

/** Pixel width to draw an SVG chart at.
 *
 * The chart fills its column rather than sitting at a fixed size, so it stays sharp
 * instead of being scaled by `max-width: 100%`. The floor stops the fixed label
 * gutters from swallowing the plot on a very narrow phone — below it the CSS scales
 * the chart down, which is still better than an unreadably short bar.
 */
function chartWidth(container, fallback = 460) {
  return Math.max(container.clientWidth || fallback, 260);
}

function debounce(fn, ms) {
  let handle;
  return (...args) => { clearTimeout(handle); handle = setTimeout(() => fn(...args), ms); };
}

function seekTo(seconds) {
  const player = $('#player');
  if (Number.isFinite(seconds)) player.currentTime = Math.max(0, seconds);
}

function setFocusTrack(trackId) {
  state.focusTrack = trackId;
  const chip = $('#focus-chip');
  if (trackId === null) {
    chip.hidden = true;
  } else {
    chip.hidden = false;
    chip.innerHTML = `Following bird #${trackId} <button type="button" aria-label="Clear">×</button>`;
    chip.querySelector('button').onclick = () => setFocusTrack(null);
  }
  drawBirdTimeline();
  drawPlayerOverlay();
  renderBudget(state.result.metrics);
}

// ---------------------------------------------------------------------------
// Track overlay on the video
//
// The result already carries every track's box history and every classified window,
// so the boxes are drawn live on a canvas over the player rather than requiring the
// pre-rendered annotated file. Nothing is re-encoded and seeking stays instant.
// ---------------------------------------------------------------------------

/** Index tracks and predictions for time lookup during playback. */
function buildOverlayIndex(result) {
  const tracks = new Map();
  for (const track of result.tracks || []) {
    const rows = track.boxes || [];
    if (!rows.length) continue;
    // Each row is [t, x1, y1, x2, y2] at a frame the detector actually ran on.
    tracks.set(track.id, {
      times: Float64Array.from(rows, (row) => row[0]),
      rows,
      start: track.start_s,
      end: track.end_s,
    });
  }

  // Uncertain windows are excluded from the overlay entirely. An uncertain result is the
  // absence of a finding, and painting one on the footage invites it to be read as a
  // behaviour that was observed. It stays in the timelines and the budget, where it is
  // reported as uncertain time rather than shown against an individual bird.
  const predictions = new Map();
  for (const prediction of result.predictions || []) {
    if (prediction.uncertain) continue;
    if (!predictions.has(prediction.track_id)) predictions.set(prediction.track_id, []);
    predictions.get(prediction.track_id).push(prediction);
  }
  for (const list of predictions.values()) list.sort((a, b) => a.start_s - b.start_s);

  state.overlay = { tracks, predictions };
}

/** Index of the last entry at or before `t`, or -1. */
function lastAtOrBefore(times, t) {
  let low = 0;
  let high = times.length - 1;
  let found = -1;
  while (low <= high) {
    const mid = (low + high) >> 1;
    if (times[mid] <= t) { found = mid; low = mid + 1; } else { high = mid - 1; }
  }
  return found;
}

/**
 * Box for one track at time `t`, in source-video pixels, or null when absent.
 *
 * The detector runs on a stride, so observations are sparse; boxes between them are
 * linearly interpolated, and nothing is extrapolated beyond the track's own span. This
 * matches what render_overlay() does server-side, so the two views agree.
 */
function boxAt(entry, t) {
  if (t < entry.start || t > entry.end) return null;
  const index = lastAtOrBefore(entry.times, t);
  if (index < 0) return null;

  const a = entry.rows[index];
  if (index === entry.rows.length - 1) return [a[1], a[2], a[3], a[4]];

  const b = entry.rows[index + 1];
  const span = b[0] - a[0];
  if (span <= 0) return [a[1], a[2], a[3], a[4]];

  const k = (t - a[0]) / span;
  return [
    a[1] + (b[1] - a[1]) * k,
    a[2] + (b[2] - a[2]) * k,
    a[3] + (b[3] - a[3]) * k,
    a[4] + (b[4] - a[4]) * k,
  ];
}

/** The classified window covering `t` for one track, or null between windows. */
function predictionAt(trackId, t) {
  const list = state.overlay.predictions.get(trackId);
  if (!list) return null;
  let low = 0;
  let high = list.length - 1;
  let found = null;
  while (low <= high) {
    const mid = (low + high) >> 1;
    if (list[mid].start_s <= t) { found = list[mid]; low = mid + 1; } else { high = mid - 1; }
  }
  return found && t < found.end_s ? found : null;
}

/**
 * Where the video actually sits inside its element.
 *
 * A <video> letterboxes its content, so the element box is not the picture. Boxes are
 * in source pixels and have to be mapped through the same contain-fit.
 */
function videoFit(player, canvas) {
  const sourceWidth = player.videoWidth || (state.result && state.result.video.width);
  const sourceHeight = player.videoHeight || (state.result && state.result.video.height);
  // The video is the only layout authority. Reading dimensions back from a canvas whose
  // backing store we mutate can create a resize feedback loop and stretch the page.
  const boxWidth = player.clientWidth;
  const boxHeight = player.clientHeight;
  if (!sourceWidth || !sourceHeight || !boxWidth || !boxHeight) return null;

  const stageRect = player.parentElement.getBoundingClientRect();
  const playerRect = player.getBoundingClientRect();
  canvas.style.left = `${playerRect.left - stageRect.left}px`;
  canvas.style.top = `${playerRect.top - stageRect.top}px`;
  canvas.style.width = `${boxWidth}px`;
  canvas.style.height = `${boxHeight}px`;

  const scale = Math.min(boxWidth / sourceWidth, boxHeight / sourceHeight);
  return {
    scale,
    offsetX: (boxWidth - sourceWidth * scale) / 2,
    offsetY: (boxHeight - sourceHeight * scale) / 2,
    width: boxWidth,
    height: boxHeight,
  };
}

function startOverlayLoop() {
  if (state.overlayFrame !== null) return;
  const step = () => {
    drawPlayerOverlay();
    const player = $('#player');
    state.overlayFrame = (player && !player.paused && !player.ended)
      ? requestAnimationFrame(step)
      : null;
  };
  state.overlayFrame = requestAnimationFrame(step);
}

function stopOverlayLoop() {
  if (state.overlayFrame !== null) cancelAnimationFrame(state.overlayFrame);
  state.overlayFrame = null;
}

function drawTrackTrail(ctx, entry, trackId, t, fit, opacity) {
  const colour = trackColour(trackId);
  for (let index = TRACK_TRAIL_POINTS - 1; index >= 0; index -= 1) {
    const age = (index / (TRACK_TRAIL_POINTS - 1)) * TRACK_TRAIL_SECONDS;
    const box = boxAt(entry, t - age);
    if (!box) continue;

    const freshness = 1 - index / TRACK_TRAIL_POINTS;
    const size = Math.round(3 + freshness * 6);
    const x = Math.round(fit.offsetX + ((box[0] + box[2]) / 2) * fit.scale);
    const y = Math.round(fit.offsetY + ((box[1] + box[3]) / 2) * fit.scale);

    // Square trail pixels echo the interface artwork; the dark under-pixel keeps pale
    // colours visible over litter and white plumage.
    ctx.globalAlpha = opacity * freshness * 0.42;
    ctx.fillStyle = '#000000';
    ctx.fillRect(x - size / 2 - 1, y - size / 2 - 1, size + 2, size + 2);

    ctx.globalAlpha = opacity * (0.12 + freshness * 0.88);
    ctx.fillStyle = colour;
    ctx.fillRect(x - size / 2, y - size / 2, size, size);
  }
}

function drawActionBox(ctx, box, trackId, prediction, fit, focused, opacity) {
  const colour = colourFor(prediction.label);
  const x = fit.offsetX + box[0] * fit.scale;
  const y = fit.offsetY + box[1] * fit.scale;
  const width = Math.max(1, (box[2] - box[0]) * fit.scale);
  const height = Math.max(1, (box[3] - box[1]) * fit.scale);

  // Dark under-stroke gives the action outline contrast on both bright and dark footage.
  ctx.globalAlpha = opacity * 0.65;
  ctx.strokeStyle = '#000000';
  ctx.lineWidth = focused ? 6 : 5;
  ctx.strokeRect(x, y, width, height);
  ctx.globalAlpha = opacity;
  ctx.strokeStyle = colour;
  ctx.lineWidth = focused ? 3 : 2;
  ctx.strokeRect(x, y, width, height);

  // Confidence is visible both numerically and as a short bar along the box edge.
  const confidence = Math.max(0, Math.min(1, Number(prediction.confidence) || 0));
  ctx.fillStyle = colour;
  ctx.fillRect(x, y + height - 3, width * confidence, 3);

  const text = `#${trackId}  ${prettyLabel(prediction.label)}  ${Math.round(confidence * 100)}%`;
  const textWidth = ctx.measureText(text).width;
  const labelX = Math.max(0, Math.min(x, fit.width - textWidth - 10));
  const labelY = y > 19 ? y - 5 : Math.min(y + height + 16, fit.height - 3);
  ctx.fillStyle = colour;
  ctx.fillRect(labelX, labelY - 13, textWidth + 10, 17);
  ctx.fillStyle = contrastText(colour);
  ctx.fillText(text, labelX + 5, labelY);
}

function drawPlayerOverlay() {
  const canvas = $('#player-overlay');
  const player = $('#player');
  if (!canvas || !player || !state.overlay || !state.result) return;

  const fit = videoFit(player, canvas);
  if (!fit) return;

  const ratio = window.devicePixelRatio || 1;
  const backingWidth = Math.round(fit.width * ratio);
  const backingHeight = Math.round(fit.height * ratio);
  // Changing width/height clears the canvas and can trigger layout work. Only resize
  // when the displayed video actually changed size or moved between pixel densities.
  if (canvas.width !== backingWidth || canvas.height !== backingHeight) {
    canvas.width = backingWidth;
    canvas.height = backingHeight;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, fit.width, fit.height);

  if (!$('#tracks-switch').checked) return;

  const t = player.currentTime;
  if (!Number.isFinite(t)) return;

  ctx.font = '700 11px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.textBaseline = 'alphabetic';

  for (const [trackId, entry] of state.overlay.tracks) {
    const box = boxAt(entry, t);
    if (!box) continue;

    const focused = state.focusTrack === trackId;
    const dimmed = state.focusTrack !== null && !focused;

    const opacity = dimmed ? 0.22 : 1;
    drawTrackTrail(ctx, entry, trackId, t, fit, opacity);

    // The downloaded annotated video already burns action boxes into its frames. Keep
    // the live trails, but do not draw a duplicate action box over that version.
    const prediction = predictionAt(trackId, t);
    if (prediction && !$('#overlay-switch').checked) {
      drawActionBox(ctx, box, trackId, prediction, fit, focused, opacity);
    }
    ctx.globalAlpha = 1;
  }
  ctx.globalAlpha = 1;
}

// ---------------------------------------------------------------------------
// Timelines (canvas — thousands of segments render poorly as DOM)
// ---------------------------------------------------------------------------

function prepareCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.height / (canvas._ratio || 1);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.height = `${height}px`;
  canvas._ratio = ratio;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

/** Stacked composition of behaviour groups over video time. */
function drawStackTimeline() {
  const canvas = $('#timeline-stack');
  if (!canvas.clientWidth || !state.result) return;
  const timeline = state.result.metrics.timeline;
  const duration = state.result.video.duration_s;
  const { ctx, width, height } = prepareCanvas(canvas);

  const padBottom = 18;
  const plotHeight = height - padBottom;
  ctx.fillStyle = cssVar('--surface-2');
  ctx.fillRect(0, 0, width, plotHeight);

  if (timeline && timeline.bins.length) {
    const binWidth = (timeline.bin_seconds / duration) * width;
    timeline.bins.forEach((bin) => {
      const total = bin.seconds.reduce((a, b) => a + b, 0);
      if (total <= 0) return;
      const x = (bin.t / duration) * width;

      // Aggregate the 15 classes into the six coloured groups.
      const byGroup = new Map();
      timeline.classes.forEach((label, index) => {
        const value = bin.seconds[index];
        if (value <= 0) return;
        const key = groupOf(label) || 'uncertain';
        byGroup.set(key, (byGroup.get(key) || 0) + value);
      });

      let y = plotHeight;
      const order = [...Object.keys(GROUP_SLOT), 'uncertain'];
      order.forEach((group) => {
        const value = byGroup.get(group);
        if (!value) return;
        const segment = (value / total) * plotHeight;
        ctx.fillStyle = group === 'uncertain'
          ? cssVar('--series-none')
          : cssVar(`--series-${GROUP_SLOT[group]}`);
        // 2px surface gap between stacked segments, never a stroke.
        ctx.fillRect(x, y - segment, Math.max(1, binWidth - 1), Math.max(0, segment - 2));
        y -= segment;
      });
    });
  } else {
    ctx.fillStyle = cssVar('--text-muted');
    ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText('No classified behaviour to show', 10, plotHeight / 2);
  }

  drawTimeAxis(ctx, width, plotHeight, height, duration);
  drawPlayhead(ctx, width, plotHeight, duration);
}

/** One row per tracked bird, coloured by behaviour group. */
function drawBirdTimeline() {
  const canvas = $('#timeline-birds');
  if (!canvas.clientWidth || !state.result) return;
  const duration = state.result.video.duration_s;
  const predictions = state.result.predictions || [];

  const byTrack = new Map();
  predictions.forEach((p) => {
    if (!byTrack.has(p.track_id)) byTrack.set(p.track_id, []);
    byTrack.get(p.track_id).push(p);
  });
  const trackIds = Array.from(byTrack.keys()).sort((a, b) => a - b);

  // Each row is a tap target — "click a row to follow that bird" — so a finger gets a
  // taller row than a mouse pointer does.
  const rowHeight = window.matchMedia('(pointer: coarse)').matches ? 24 : 14;
  const padBottom = 18;
  const desired = Math.max(60, trackIds.length * rowHeight) + padBottom;
  canvas.height = desired * (canvas._ratio || 1);
  const { ctx, width, height } = prepareCanvas(canvas);
  const plotHeight = height - padBottom;

  canvas._rows = [];

  if (!trackIds.length) {
    ctx.fillStyle = cssVar('--text-muted');
    ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText('No individual birds were classified in this video', 10, 26);
    return;
  }

  trackIds.forEach((trackId, index) => {
    const y = index * rowHeight;
    const dimmed = state.focusTrack !== null && state.focusTrack !== trackId;
    canvas._rows.push({ trackId, y, height: rowHeight });

    ctx.fillStyle = cssVar('--surface-2');
    ctx.fillRect(0, y + 1, width, rowHeight - 3);

    ctx.globalAlpha = dimmed ? 0.25 : 1;
    byTrack.get(trackId).forEach((p) => {
      const x1 = (p.start_s / duration) * width;
      const x2 = (p.end_s / duration) * width;
      ctx.fillStyle = colourFor(p.label);
      ctx.fillRect(x1, y + 1, Math.max(1.5, x2 - x1 - 1), rowHeight - 3);
    });
    ctx.globalAlpha = 1;

    if (state.focusTrack === trackId) {
      ctx.strokeStyle = cssVar('--text-primary');
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, y + 0.5, width - 1, rowHeight - 2);
    }
  });

  drawTimeAxis(ctx, width, plotHeight, height, duration);
  drawPlayhead(ctx, width, plotHeight, duration);
}

function drawTimeAxis(ctx, width, plotHeight, height, duration) {
  const targetTicks = Math.max(2, Math.floor(width / 90));
  const step = niceStep(duration / targetTicks);
  ctx.strokeStyle = cssVar('--border');
  ctx.fillStyle = cssVar('--text-muted');
  ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.lineWidth = 1;
  for (let t = 0; t <= duration; t += step) {
    const x = (t / duration) * width;
    ctx.beginPath();
    ctx.moveTo(Math.round(x) + 0.5, plotHeight);
    ctx.lineTo(Math.round(x) + 0.5, plotHeight + 4);
    ctx.stroke();
    ctx.fillText(fmtTime(t), Math.min(x + 3, width - 30), height - 5);
  }
}

function drawPlayhead(ctx, width, plotHeight, duration) {
  const player = $('#player');
  if (!player || !Number.isFinite(player.currentTime) || !duration) return;
  const x = (player.currentTime / duration) * width;
  ctx.strokeStyle = cssVar('--text-primary');
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x, 0);
  ctx.lineTo(x, plotHeight);
  ctx.stroke();
}

function niceStep(raw) {
  const candidates = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600];
  return candidates.find((c) => c >= raw) || 3600;
}

// Timeline interaction: click to seek, click a bird row to follow it.
['#timeline-stack', '#timeline-birds'].forEach((selector) => {
  const canvas = $(selector);
  canvas.addEventListener('click', (event) => {
    if (!state.result) return;
    const rect = canvas.getBoundingClientRect();
    const fraction = (event.clientX - rect.left) / rect.width;
    seekTo(fraction * state.result.video.duration_s);

    if (selector === '#timeline-birds' && canvas._rows) {
      const y = event.clientY - rect.top;
      const row = canvas._rows.find((r) => y >= r.y && y < r.y + r.height);
      if (row) setFocusTrack(state.focusTrack === row.trackId ? null : row.trackId);
    }
  });

  canvas.addEventListener('mousemove', (event) => {
    if (!state.result) return;
    const rect = canvas.getBoundingClientRect();
    const time = ((event.clientX - rect.left) / rect.width) * state.result.video.duration_s;

    if (selector === '#timeline-birds' && canvas._rows) {
      const y = event.clientY - rect.top;
      const row = canvas._rows.find((r) => y >= r.y && y < r.y + r.height);
      if (row) {
        const hit = (state.result.predictions || []).find(
          (p) => p.track_id === row.trackId && time >= p.start_s && time < p.end_s,
        );
        if (hit) {
          showTooltip(event, `Bird #${hit.track_id}`, [
            ['Behaviour', prettyLabel(hit.label)],
            ['Confidence', hit.confidence.toFixed(2)],
            ['Time', `${fmtTime(hit.start_s)}–${fmtTime(hit.end_s)}`],
          ]);
          return;
        }
      }
    }
    hideTooltip();
  });
  canvas.addEventListener('mouseleave', hideTooltip);
});

function showTooltip(event, title, rows) {
  const tooltip = $('#timeline-tooltip');
  tooltip.innerHTML = `<div class="tt-title">${escapeHtml(title)}</div>` +
    rows.map(([k, v]) => `<div class="tt-row"><span>${k}</span><strong>${escapeHtml(String(v))}</strong></div>`).join('');
  tooltip.hidden = false;
  const offset = 14;
  const rect = tooltip.getBoundingClientRect();
  let left = event.clientX + offset;
  if (left + rect.width > window.innerWidth) left = event.clientX - rect.width - offset;
  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${Math.min(event.clientY + offset, window.innerHeight - rect.height - 8)}px`;
}

function hideTooltip() { $('#timeline-tooltip').hidden = true; }

function renderLegend() {
  const items = Object.entries(GROUP_LABEL).map(([group, label]) => ({
    label, colour: cssVar(`--series-${GROUP_SLOT[group]}`),
  }));
  items.push({ label: 'Uncertain', colour: cssVar('--series-none') });
  $('#legend-groups').innerHTML = items.map((item) =>
    `<span class="legend-item">
       <span class="legend-swatch" style="background:${item.colour}"></span>${item.label}
     </span>`).join('');
}

// ---------------------------------------------------------------------------
// Charts (hand-rolled SVG — no external libraries, works offline)
// ---------------------------------------------------------------------------

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
  return el;
}

function emptyChart(container, message) {
  container.innerHTML = `<p class="hint" style="padding:22px 0">${message}</p>`;
}

/** Horizontal bars: behaviour share. Colour carries the group, length the magnitude. */
function renderBudget(metrics) {
  const container = $('#chart-budget');
  const budget = metrics.time_budget || {};
  let entries = Object.entries(budget.by_class || {});

  // When a bird is focused, show that bird's budget instead of the flock's.
  if (state.focusTrack !== null) {
    const bird = (metrics.per_bird || {})[state.focusTrack];
    if (bird) {
      entries = Object.entries(bird.by_class).map(([k, v]) => [k, { share: v, seconds: 0, n_windows: 0 }]);
    }
  }
  entries = entries.filter(([, v]) => v.share > 0).sort((a, b) => b[1].share - a[1].share);

  if (!entries.length) {
    emptyChart(container, 'No behaviours were classified.');
    $('#table-budget').innerHTML = '';
    return;
  }

  const rowHeight = 26;
  const barHeight = 16;           // thin marks; well under the 24px cap
  const width = chartWidth(container);
  // Narrow columns give the gutters back to the bars: the longest behaviour name
  // ("Litter scratching") still fits the compact gutter at the smaller type size.
  const compact = width < 380;
  const fontSize = compact ? 11 : 12;
  const labelWidth = compact ? 108 : 132;
  const valueWidth = compact ? 46 : 54;
  const plotWidth = width - labelWidth - valueWidth;
  const height = entries.length * rowHeight + 8;
  const max = Math.max(...entries.map(([, v]) => v.share));

  const svg = svgEl('svg', { width, height, viewBox: `0 0 ${width} ${height}`, role: 'img' });
  svg.setAttribute('aria-label', 'Share of observed bird-time per behaviour');

  entries.forEach(([label, values], index) => {
    const y = index * rowHeight + 4;
    const barWidth = Math.max(2, (values.share / max) * plotWidth);
    const colour = colourFor(label);

    const text = svgEl('text', {
      x: labelWidth - 10, y: y + barHeight / 2 + 4, 'text-anchor': 'end',
      fill: cssVar('--text-secondary'), 'font-size': fontSize,
    });
    text.textContent = prettyLabel(label);
    svg.appendChild(text);

    // 4px rounded data-end, square at the baseline.
    const bar = svgEl('path', {
      d: roundedBarPath(labelWidth, y, barWidth, barHeight, 4),
      fill: colour,
    });
    bar.appendChild(svgEl('title')).textContent =
      `${prettyLabel(label)} — ${fmtPct(values.share)}` +
      (values.seconds ? ` (${values.seconds.toFixed(0)} bird-seconds, ${values.n_windows} clips)` : '');
    svg.appendChild(bar);

    const value = svgEl('text', {
      x: labelWidth + barWidth + 8, y: y + barHeight / 2 + 4,
      fill: cssVar('--text-primary'), 'font-size': fontSize,
      'font-family': 'ui-monospace, monospace',
    });
    value.textContent = fmtPct(values.share);
    svg.appendChild(value);
  });

  container.innerHTML = '';
  container.appendChild(svg);
  buildBudgetTable(entries);
}

/** Bar with rounded corners on the data-end only, square against the baseline. */
function roundedBarPath(x, y, width, height, radius) {
  const r = Math.min(radius, width);
  return `M${x},${y} H${x + width - r} Q${x + width},${y} ${x + width},${y + r} ` +
         `V${y + height - r} Q${x + width},${y + height} ${x + width - r},${y + height} H${x} Z`;
}

function buildBudgetTable(entries) {
  const rows = entries.map(([label, values]) => `
    <tr>
      <td><span class="swatch-cell">
        <span class="legend-swatch" style="background:${colourFor(label)}"></span>
        ${prettyLabel(label)}</span></td>
      <td>${GROUP_LABEL[groupOf(label)] || '—'}</td>
      <td class="num">${fmtPct(values.share)}</td>
      <td class="num">${values.seconds ? values.seconds.toFixed(0) : '—'}</td>
      <td class="num">${values.n_windows || '—'}</td>
    </tr>`).join('');
  $('#table-budget').innerHTML = `<table>
    <thead><tr><th>Behaviour</th><th>Group</th><th class="num">Share</th>
    <th class="num">Bird-seconds</th><th class="num">Clips</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

$('[data-toggle-table="budget"]').addEventListener('click', (event) => {
  const table = $('#table-budget');
  const chart = $('#chart-budget');
  const showTable = table.hidden;
  table.hidden = !showTable;
  chart.hidden = showTable;
  event.target.textContent = showTable ? 'Chart' : 'Table';
  // The chart could not be measured while it was hidden, so any resize that happened in
  // the meantime left it at a stale width. Redraw it now that it has a box again.
  if (!showTable && state.result) renderBudget(state.result.metrics || {});
});

/** Single-series line: birds detected per frame. No legend — the title names it. */
function renderPopulation(metrics) {
  const container = $('#chart-population');
  const series = (metrics.population || {}).series || [];
  if (series.length < 2) {
    emptyChart(container, 'Not enough frames to plot a count.');
    return;
  }

  const width = chartWidth(container);
  const height = 220;
  const pad = { top: 12, right: 14, bottom: 26, left: 38 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const maxTime = series[series.length - 1][0] || 1;
  const maxCount = Math.max(...series.map((d) => d[1]), 1);
  const yMax = niceCeil(maxCount);

  const x = (t) => pad.left + (t / maxTime) * plotW;
  const y = (v) => pad.top + plotH - (v / yMax) * plotH;

  const svg = svgEl('svg', { width, height, viewBox: `0 0 ${width} ${height}`, role: 'img' });
  svg.setAttribute('aria-label', 'Number of birds detected over time');

  // Hairline, solid, recessive gridlines.
  for (let i = 0; i <= 4; i++) {
    const value = (yMax / 4) * i;
    const gy = y(value);
    svg.appendChild(svgEl('line', {
      x1: pad.left, x2: width - pad.right, y1: gy, y2: gy,
      stroke: cssVar('--border'), 'stroke-width': 1,
    }));
    const label = svgEl('text', {
      x: pad.left - 7, y: gy + 4, 'text-anchor': 'end',
      fill: cssVar('--text-muted'), 'font-size': 11,
    });
    label.textContent = Math.round(value);
    svg.appendChild(label);
  }

  // Fewer time labels on a narrow column, so they never collide or run off the edge.
  const xTicks = width < 340 ? 2 : 4;
  for (let i = 0; i <= xTicks; i++) {
    const t = (maxTime / xTicks) * i;
    // The first and last labels are anchored inward; centring them would hang half the
    // text outside the viewBox, where it is clipped.
    const anchor = i === 0 ? 'start' : (i === xTicks ? 'end' : 'middle');
    const label = svgEl('text', {
      x: x(t), y: height - 8, 'text-anchor': anchor,
      fill: cssVar('--text-muted'), 'font-size': 11,
    });
    label.textContent = fmtTime(t);
    svg.appendChild(label);
  }

  const points = series.map((d) => `${x(d[0]).toFixed(1)},${y(d[1]).toFixed(1)}`).join(' ');
  // Area wash at ~10% opacity beneath the line.
  svg.appendChild(svgEl('polygon', {
    points: `${pad.left},${y(0)} ${points} ${x(maxTime)},${y(0)}`,
    fill: cssVar('--series-1'), opacity: 0.1,
  }));
  svg.appendChild(svgEl('polyline', {
    points, fill: 'none', stroke: cssVar('--series-1'),
    'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round',
  }));

  // Direct-label the mean rather than every point.
  const mean = (metrics.population || {}).mean || 0;
  const meanY = y(mean);
  svg.appendChild(svgEl('line', {
    x1: pad.left, x2: width - pad.right, y1: meanY, y2: meanY,
    stroke: cssVar('--text-muted'), 'stroke-width': 1, opacity: 0.7,
  }));
  // The label sits over the series, so it carries a surface backing to stay readable.
  const meanText = `mean ${mean}`;
  const labelWidth = meanText.length * 6 + 8;
  svg.appendChild(svgEl('rect', {
    x: width - pad.right - labelWidth, y: meanY - 16,
    width: labelWidth, height: 14, rx: 3, fill: cssVar('--surface-1'), opacity: 0.9,
  }));
  const meanLabel = svgEl('text', {
    x: width - pad.right - 4, y: meanY - 5, 'text-anchor': 'end',
    fill: cssVar('--text-secondary'), 'font-size': 11,
  });
  meanLabel.textContent = meanText;
  svg.appendChild(meanLabel);

  container.innerHTML = '';
  container.appendChild(svg);
}

function niceCeil(value) {
  const magnitude = 10 ** Math.floor(Math.log10(Math.max(value, 1)));
  return Math.ceil(value / magnitude) * magnitude;
}

/** Heatmap: continuous magnitude, so a single-hue sequential ramp with a scale legend. */
function renderSpatial(metrics, result) {
  const container = $('#chart-spatial');
  const spatial = metrics.spatial;
  if (!spatial || !spatial.counts) {
    emptyChart(container, 'No spatial data available.');
    return;
  }
  const counts = spatial.counts;
  const max = Math.max(...counts.flat());
  if (max === 0) {
    emptyChart(container, 'No observations to map.');
    return;
  }

  const aspect = result.video.height / result.video.width;
  const width = chartWidth(container, 420);
  // The map is locked to the camera's aspect ratio, so filling a wide column would make
  // it taller than the charts beside it. Past the cap it stops growing and centres.
  let plotW = width - 8;
  let plotH = Math.round(plotW * aspect);
  if (plotH > 320) {
    plotH = 320;
    plotW = Math.round(plotH / aspect);
  }
  const plotX = Math.round((width - plotW) / 2);
  // Cells are rectangular, matching the camera's aspect ratio — square cells would
  // misrepresent where in the house each observation happened.
  const cellW = plotW / spatial.grid;
  const cellH = plotH / spatial.grid;
  const height = plotH + 42;

  const ramp = ['--seq-100', '--seq-200', '--seq-300', '--seq-400', '--seq-500', '--seq-600', '--seq-700']
    .map(cssVar);

  const svg = svgEl('svg', { width, height, viewBox: `0 0 ${width} ${height}`, role: 'img' });
  svg.setAttribute('aria-label', 'Spatial distribution of observed birds');

  counts.forEach((row, r) => {
    row.forEach((count, c) => {
      const intensity = count / max;
      const fill = count === 0 ? cssVar('--surface-2') : ramp[Math.min(ramp.length - 1, Math.floor(intensity * ramp.length))];
      const rect = svgEl('rect', {
        x: c * cellW + plotX, y: r * cellH,
        width: Math.max(1, cellW - 2), height: Math.max(1, cellH - 2),
        fill, rx: 2,
      });
      const dominant = spatial.dominant_behaviour[`${r},${c}`];
      rect.appendChild(svgEl('title')).textContent =
        `${count} observation${count === 1 ? '' : 's'}` +
        (dominant ? ` · mostly ${prettyLabel(dominant)}` : '');
      svg.appendChild(rect);
    });
  });

  // Scale legend — mandatory for a sequential ramp. The swatch pitch is derived from
  // the available width so the ramp shrinks with the column instead of running past it.
  const legendY = plotH + 14;
  const pitch = Math.min(26, (plotW - 4) / ramp.length);
  ramp.forEach((colour, index) => {
    svg.appendChild(svgEl('rect', {
      x: plotX + index * pitch, y: legendY, width: Math.max(6, pitch - 2), height: 8,
      fill: colour, rx: 2,
    }));
  });
  const low = svgEl('text', { x: plotX, y: legendY + 24, fill: cssVar('--text-muted'), 'font-size': 11 });
  low.textContent = 'fewer birds';
  svg.appendChild(low);
  // Anchored to the right edge: left-anchoring it past the ramp pushes it off a narrow
  // chart, where the ramp already occupies most of the width.
  const high = svgEl('text', {
    x: plotX + plotW, y: legendY + 24, 'text-anchor': 'end',
    fill: cssVar('--text-muted'), 'font-size': 11,
  });
  high.textContent = `more (max ${max})`;
  svg.appendChild(high);

  container.innerHTML = '';
  container.appendChild(svg);
}

/** Bout counts per behaviour — magnitude bars, coloured by group for continuity. */
function renderBouts(metrics) {
  const container = $('#chart-bouts');
  const bouts = metrics.bouts || {};
  const entries = Object.entries(bouts)
    .filter(([, v]) => v.n_bouts > 0)
    .sort((a, b) => b[1].n_bouts - a[1].n_bouts)
    .slice(0, 10);

  if (!entries.length) {
    emptyChart(container, 'No behaviour bouts were recorded.');
    return;
  }

  const rowHeight = 26;
  const barHeight = 16;
  const width = chartWidth(container);
  const compact = width < 380;
  const fontSize = compact ? 11 : 12;
  const labelWidth = compact ? 108 : 132;
  const valueWidth = compact ? 74 : 96;
  const plotWidth = width - labelWidth - valueWidth;
  const height = entries.length * rowHeight + 8;
  const max = Math.max(...entries.map(([, v]) => v.n_bouts));

  const svg = svgEl('svg', { width, height, viewBox: `0 0 ${width} ${height}`, role: 'img' });
  svg.setAttribute('aria-label', 'Number of behaviour bouts');

  entries.forEach(([label, values], index) => {
    const y = index * rowHeight + 4;
    const barWidth = Math.max(2, (values.n_bouts / max) * plotWidth);

    const name = svgEl('text', {
      x: labelWidth - 10, y: y + barHeight / 2 + 4, 'text-anchor': 'end',
      fill: cssVar('--text-secondary'), 'font-size': fontSize,
    });
    name.textContent = prettyLabel(label);
    svg.appendChild(name);

    const bar = svgEl('path', {
      d: roundedBarPath(labelWidth, y, barWidth, barHeight, 4),
      fill: colourFor(label),
    });
    bar.appendChild(svgEl('title')).textContent =
      `${values.n_bouts} bouts · mean ${values.mean_duration_s}s · ${values.total_s}s total`;
    svg.appendChild(bar);

    const value = svgEl('text', {
      x: labelWidth + barWidth + 8, y: y + barHeight / 2 + 4,
      fill: cssVar('--text-primary'), 'font-size': fontSize,
      'font-family': 'ui-monospace, monospace',
    });
    value.textContent = `${values.n_bouts} × ${values.mean_duration_s}s`;
    svg.appendChild(value);
  });

  container.innerHTML = '';
  container.appendChild(svg);
}

function renderExports() {
  const artifacts = [
    ['predictions', 'Predictions (CSV)'],
    ['time_budget', 'Time budget (CSV)'],
    ['per_bird', 'Per-bird budget (CSV)'],
    ['metrics', 'Metrics (JSON)'],
    ['result', 'Full result (JSON)'],
  ];
  $('#export-links').innerHTML = artifacts.map(([key, label]) =>
    `<a class="btn btn-small" href="/api/jobs/${state.jobId}/export/${key}" download>${label}</a>`,
  ).join('');
}

function renderParams(result) {
  const rows = Object.entries(result.params || {})
    .map(([key, value]) => `<tr><td>${key.replace(/_/g, ' ')}</td>
      <td class="num">${value === null ? '—' : value}</td></tr>`).join('');
  $('#params-dump').innerHTML = `<table><tbody>${rows}</tbody></table>`;
}

// ---------------------------------------------------------------------------
// Recent jobs
// ---------------------------------------------------------------------------

async function loadRecentJobs() {
  try {
    const { jobs } = await api('/jobs');
    const completed = jobs.filter((job) => job.state === 'completed');
    $('#recent-jobs').hidden = completed.length === 0;
    $('#recent-list').innerHTML = completed.slice(0, 8).map((job) => `
      <div class="recent-item">
        <span class="name">${escapeHtml(job.filename)}</span>
        <span class="muted">${fmtDuration(job.runtime_s)}</span>
        <button class="btn btn-small" data-open-job="${job.job_id}">Open</button>
      </div>`).join('');
    $$('[data-open-job]').forEach((button) => {
      button.addEventListener('click', () => loadResult(button.dataset.openJob).then(() => {
        showScreen('results');
      }));
    });
  } catch { /* listing is optional */ }
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

async function boot() {
  try {
    state.config = await api('/config');
  } catch (error) {
    $('#model-warning').hidden = false;
    $('#model-warning').textContent = `Could not reach the Khoroos server: ${error.message}`;
    return;
  }

  state.preset = state.config.default_preset || 'balanced';
  renderPresets();
  renderThresholds();

  $('#device-badge').hidden = false;
  $('#device-badge').textContent = state.config.device;

  const missing = Object.entries(state.config.models)
    .filter(([, info]) => !info.available)
    .map(([kind]) => kind);
  if (missing.length) {
    $('#model-warning').hidden = false;
    $('#model-warning').innerHTML =
      `⚠ <span>The <strong>${missing.join('</strong> and <strong>')}</strong> model
       ${missing.length > 1 ? 'checkpoints are' : 'checkpoint is'} not present locally and will be
       downloaded on the first analysis. This may take several minutes.</span>`;
  }

  loadRecentJobs();

  // Deep link: /?job=<id> opens an existing analysis directly, so a result can be
  // bookmarked or passed to a colleague on the same machine.
  const requested = new URLSearchParams(window.location.search).get('job');
  if (requested) {
    try {
      const status = await api(`/jobs/${requested}`);
      if (status.state === 'completed') await loadResult(requested);
      else startWatching(status);
    } catch (error) {
      $('#start-error').hidden = false;
      $('#start-error').textContent = `Could not open analysis ${requested}: ${error.message}`;
    }
  }
}

boot();
