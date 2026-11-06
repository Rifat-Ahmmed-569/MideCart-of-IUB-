const $ = id => document.getElementById(id);
const KEYMAP = {ArrowUp:'up', ArrowDown:'down', ArrowLeft:'left', ArrowRight:'right',
                w:'up', s:'down', a:'left', d:'right', W:'up', S:'down', A:'left', D:'right'};
const held = new Set();
let st = {mode: 'stopped', stop_token: 0};
let sentIdle = true;

function post(path, body) {
  return fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'},
                      body: JSON.stringify(body || {})});
}

/* ---------- driving ---------- */
function sendControl() {
  if (st.mode !== 'manual') return;
  const f = (held.has('up') ? 1 : 0) - (held.has('down') ? 1 : 0);
  const t = (held.has('right') ? 1 : 0) - (held.has('left') ? 1 : 0);
  sentIdle = held.size === 0;
  post('/api/control', {forward: f, turn: t, keys: [...held]}).catch(() => {});
}
function setKey(key, down) {
  if (down === held.has(key)) return;
  down ? held.add(key) : held.delete(key);
  document.querySelectorAll('.pad button').forEach(b => b.classList.toggle('on', held.has(b.dataset.key)));
  sendControl();                       // instant, no waiting for the timer
}
function releaseAll() { held.clear(); document.querySelectorAll('.pad .on').forEach(b => b.classList.remove('on')); sendControl(); }

// keep the lease alive while a key is held
setInterval(() => { if (held.size) sendControl(); }, 80);

window.addEventListener('keydown', e => {
  if (e.code === 'Space' || e.code === 'Escape') { e.preventDefault(); estop(); return; }
  const k = KEYMAP[e.key];
  if (!k) return;
  e.preventDefault();                  // arrows must not scroll the page or move the slider
  if (st.mode === 'manual') setKey(k, true);
});
window.addEventListener('keyup', e => { const k = KEYMAP[e.key]; if (k) { e.preventDefault(); setKey(k, false); } });
window.addEventListener('blur', releaseAll);
document.addEventListener('visibilitychange', () => { if (document.hidden) releaseAll(); });

document.querySelectorAll('.pad button').forEach(b => {
  const k = b.dataset.key;
  b.addEventListener('pointerdown', e => { e.preventDefault(); setKey(k, true); });
  ['pointerup', 'pointerleave', 'pointercancel'].forEach(ev => b.addEventListener(ev, () => setKey(k, false)));
});

/* ---------- modes / stop / speed ---------- */
async function estop() {
  held.clear();
  const r = await post('/api/stop');
  st = {...st, ...(await r.json())};
  render();
}
async function setMode(mode) {
  await refresh();                     // need the current stop_token
  const r = await post('/api/mode', {mode, stop_token: st.stop_token});
  if (r.ok) st.mode = mode;
  render();
}
$('stop').onclick = estop;
$('btnAuto').onclick = () => setMode('auto');
$('btnManual').onclick = () => setMode('manual');

let speedTimer;
$('speed').addEventListener('input', e => {
  $('speedVal').textContent = e.target.value;
  clearTimeout(speedTimer);
  speedTimer = setTimeout(() => post('/api/speed', {speed: +e.target.value}), 100);
});
$('speed').addEventListener('change', e => e.target.blur());   // so arrow keys drive, not the slider

/* ---------- status ---------- */
function render() {
  const manual = st.mode === 'manual', auto = st.mode === 'auto';
  $('btnAuto').classList.toggle('active', auto);
  $('btnManual').classList.toggle('active', manual);
  $('manualPanel').classList.toggle('disabled', !manual);

  let cls = 'stopped', title = 'STOPPED', text = 'Choose Autonomous or Manual to start.';
  if (st.serial && st.serial !== 'connected' && st.mode !== 'stopped') {
    title = 'WAITING FOR ARDUINO'; text = 'Serial: ' + st.serial;
  } else if (auto) {
    if (st.vision === 'STOP' || st.vision === 'PERSON_DETECTED') {
      cls = 'paused'; title = 'AUTO - PAUSED'; text = 'Person detected close to the robot. Resumes when they move away.';
    } else { cls = 'auto'; title = 'AUTO - FOLLOWING LINE'; text = 'Human stop is active.'; }
  } else if (manual) {
    cls = 'manual'; title = 'MANUAL'; text = 'Drive with the arrow keys. Human detection is off.';
  }
  $('banner').className = 'banner ' + cls;
  $('bannerTitle').textContent = title;
  $('bannerText').textContent = text;

  $('people').textContent = manual ? 'off' : st.people ?? 0;
  $('nearest').textContent = manual || st.nearest_distance == null ? '-' : (st.nearest_distance * 3.28084).toFixed(1) + ' ft';
  $('fps').textContent = st.camera_fps ?? 0;
  $('motors').textContent = (st.motor_left ?? 0) + ' / ' + (st.motor_right ?? 0);
  $('serial').textContent = st.serial ?? '-';
}

let firstSpeed = true;
async function refresh() {
  try {
    const r = await fetch('/api/status', {cache: 'no-store'});
    const s = await r.json();
    if (st.mode === 'manual' && s.mode !== 'manual') releaseAll();
    st = s;
    if (firstSpeed) { $('speed').value = s.speed; $('speedVal').textContent = s.speed; firstSpeed = false; }
    $('dot').classList.add('ok'); $('connText').textContent = 'Connected';
    render();
  } catch {
    $('dot').classList.remove('ok'); $('connText').textContent = 'Disconnected';
  }
}
setInterval(refresh, 300);
refresh();
