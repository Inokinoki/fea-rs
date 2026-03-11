import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';

const app = document.getElementById('app');
const statusEl = document.getElementById('status');
const scaleEl = document.getElementById('scale');
const scaleValEl = document.getElementById('scaleVal');
const showUndeformedEl = document.getElementById('showUndeformed');
const showDeformedEl = document.getElementById('showDeformed');
const showStressLegendEl = document.getElementById('showStressLegend');
const stressLegendEl = document.getElementById('stressLegend');
const animateEl = document.getElementById('animate');
const animSpeedEl = document.getElementById('animSpeed');
const animSpeedValEl = document.getElementById('animSpeedVal');

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x0b0f14, 1.0);
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x0b0f14, 5, 60);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 500);
camera.position.set(3, 2, 6);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dir = new THREE.DirectionalLight(0xffffff, 0.55);
dir.position.set(3, 6, 4);
scene.add(dir);

const grid = new THREE.GridHelper(20, 20, 0x334155, 0x111827);
grid.material.opacity = 0.25;
grid.material.transparent = true;
scene.add(grid);

const root = new THREE.Group();
scene.add(root);

// Modal animation state
let modalData = null;
let animating = false;
let animTime = 0;
let animSpeed = 1.0;
let modeShape = null;
let animId = null;

function stressToColor(s, smin, smax) {
  // Diverging colormap: blue (compression) -> white -> red (tension)
  if (!Number.isFinite(s)) return new THREE.Color(0x93c5fd);
  if (smax <= smin) return new THREE.Color(0x93c5fd);

  const mid = (smax + smin) / 2;
  const range = (smax - smin) / 2;
  const t = range > 0 ? (s - mid) / range : 0;

  const c_compression = new THREE.Color(0x2563eb); // blue
  const c_neutral = new THREE.Color(0xe5e7eb);     // light gray
  const c_tension = new THREE.Color(0xef4444);     // red

  if (t < 0) {
    return c_compression.clone().lerp(c_neutral, -t);
  } else {
    return c_neutral.clone().lerp(c_tension, t);
  }
}

function frequencyToColor(f, fmin, fmax) {
  // Rainbow colormap for frequency visualization
  if (!Number.isFinite(f) || fmax <= fmin) return new THREE.Color(0x93c5fd);
  const t = (f - fmin) / (fmax - fmin);

  // HSV rainbow: red -> orange -> yellow -> green -> cyan -> blue -> violet
  const hue = (1.0 - t) * 0.67; // blue to red
  return new THREE.Color().setHSL(hue, 0.8, 0.5);
}

function createStressLegendTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 32;
  const ctx = canvas.getContext('2d');

  const gradient = ctx.createLinearGradient(0, 0, 256, 0);
  gradient.addColorStop(0, '#2563eb');   // compression
  gradient.addColorStop(0.5, '#e5e7eb'); // neutral
  gradient.addColorStop(1, '#ef4444');   // tension

  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 256, 32);

  // Add labels
  ctx.fillStyle = '#000';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('-', 5, 22);
  ctx.textAlign = 'center';
  ctx.fillText('0', 128, 22);
  ctx.textAlign = 'right';
  ctx.fillText('+', 251, 22);

  return canvas;
}

function computeBounds(points) {
  const box = new THREE.Box3();
  for (const p of points) box.expandByPoint(new THREE.Vector3(p[0], p[1], p[2]));
  return box;
}

function makeLine(a, b, color, linewidth = 1) {
  const geom = new THREE.BufferGeometry();
  geom.setFromPoints([a, b]);
  const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95 });
  const line = new THREE.Line(geom, mat);
  return line;
}

let undeformedGroup = null;
let deformedGroup = null;
let modeGroup = null;

function buildScene(data, scale) {
  root.clear();
  undeformedGroup = new THREE.Group();
  deformedGroup = new THREE.Group();
  modeGroup = new THREE.Group();

  // Handle both simple and enhanced JSON formats
  const points = data.points || data.undeformed?.points || [];
  const cells = data.cells || data.undeformed?.cells || [];
  const disp = data.point_data?.displacement || data.undeformed?.point_data?.displacement || [];
  const stress = data.cell_data?.axial_stress || [];
  const metadata = data.metadata || {};
  const modes = data.modes || [];

  // Compute stress range
  const finiteStress = stress.filter(Number.isFinite);
  const smin = finiteStress.length > 0 ? Math.min(...finiteStress) : 0;
  const smax = finiteStress.length > 0 ? Math.max(...finiteStress) : 0;

  const p0 = (i) => new THREE.Vector3(points[i][0], points[i][1], points[i][2]);
  const p1 = (i) => {
    const d = disp[i] || [0, 0, 0];
    return new THREE.Vector3(
      points[i][0] + scale * d[0],
      points[i][1] + scale * d[1],
      points[i][2] + scale * d[2]
    );
  };

  // Draw undeformed geometry
  for (let ci = 0; ci < cells.length; ci++) {
    const [i, j] = cells[ci];
    undeformedGroup.add(makeLine(p0(i), p0(j), 0x64748b));
  }

  // Draw deformed geometry with stress coloring
  for (let ci = 0; ci < cells.length; ci++) {
    const [i, j] = cells[ci];
    const c = stressToColor(stress[ci], smin, smax);
    deformedGroup.add(makeLine(p1(i), p1(j), c));
  }

  // Add nodes (deformed)
  const box = computeBounds(points);
  const size = box.getSize(new THREE.Vector3()).length();
  const r = Math.max(0.01, size * 0.015);
  const sphereGeom = new THREE.SphereGeometry(r, 14, 14);
  const sphereMat = new THREE.MeshStandardMaterial({ color: 0xe5e7eb, roughness: 0.65, metalness: 0.0 });

  for (let i = 0; i < points.length; i++) {
    const m = new THREE.Mesh(sphereGeom, sphereMat);
    m.position.copy(p1(i));
    deformedGroup.add(m);
  }

  // Add undeformed nodes (ghosted)
  for (let i = 0; i < points.length; i++) {
    const m = new THREE.Mesh(sphereGeom, sphereMat.clone());
    m.material.opacity = 0.3;
    m.material.transparent = true;
    m.position.copy(p0(i));
    undeformedGroup.add(m);
  }

  // Build mode shapes if available
  if (modes && modes.length > 0) {
    buildModeShapes(modes, cells, points, r, sphereGeom, sphereMat);
  }

  root.add(undeformedGroup);
  root.add(deformedGroup);
  root.add(modeGroup);

  // Frame camera
  const worldBox = new THREE.Box3().setFromObject(root);
  const center = worldBox.getCenter(new THREE.Vector3());
  const radius = worldBox.getSize(new THREE.Vector3()).length() * 0.5;
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(radius * 0.9, radius * 0.6, radius * 1.2));
  camera.near = Math.max(0.001, radius / 200);
  camera.far = Math.max(50, radius * 10);
  camera.updateProjectionMatrix();
  controls.update();

  // Update status with metadata if available
  if (metadata.num_nodes || metadata.num_elements) {
    statusEl.innerHTML = `Nodes: <code>${metadata.num_nodes || points.length}</code> | Elements: <code>${metadata.num_elements || cells.length}</code> | Max disp: <code>${(metadata.max_displacement || 0).toExponential(2)}</code> m${modes.length > 0 ? ` | Modes: <code>${modes.length}</code>` : ''}`;
  }
}

function buildModeShapes(modes, cells, points, nodeRadius, sphereGeom, sphereMat) {
  if (!modes || modes.length === 0) return;

  const fmin = Math.min(...modes.map(m => m.frequency));
  const fmax = Math.max(...modes.map(m => m.frequency));

  // Create a line for each mode shape
  modes.forEach((mode, modeIdx) => {
    const modeGroupInner = new THREE.Group();
    const modeShape = mode.shape || [];
    const freq = mode.frequency;
    const color = frequencyToColor(freq, fmin, fmax);

    // Draw mode shape at offset position
    const yOffset = -2 - modeIdx * 0.5;

    for (let ci = 0; ci < cells.length; ci++) {
      const [i, j] = cells[ci];
      const pi = new THREE.Vector3(points[i][0], points[i][1] + yOffset, points[i][2]);
      const pj = new THREE.Vector3(points[j][0], points[j][1] + yOffset, points[j][2]);
      modeGroupInner.add(makeLine(pi, pj, color));
    }

    // Add frequency label (as a point with color)
    const labelMat = sphereMat.clone();
    labelMat.color = color;
    const labelSphere = new THREE.Mesh(sphereGeom, labelMat);
    labelSphere.position.set(0, yOffset, 0);
    modeGroupInner.add(labelSphere);

    modeGroupInner.visible = false;
    modeGroupInner.userData = { modeIndex: modeIdx, frequency: freq };
    modeGroup.add(modeGroupInner);
  });
}

function setVisibility() {
  if (undeformedGroup) undeformedGroup.visible = !!showUndeformedEl.checked;
  if (deformedGroup) deformedGroup.visible = !!showDeformedEl.checked;
  stressLegendEl.style.display = showStressLegendEl.checked ? 'block' : 'none';

  // Show mode shapes only if animation is off or no mode selected
  if (modeGroup) {
    modeGroup.children.forEach((g, i) => {
      g.visible = !animating && window.showModeCheckbox && window.showModeCheckbox[i]?.checked;
    });
  }
}

let currentData = null;

async function loadModel() {
  statusEl.textContent = 'Loading model.json...';
  try {
    const res = await fetch('./model.json', { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    currentData = data;

    const scale = Number(scaleEl.value);
    buildScene(data, scale);
    setVisibility();

    // Build mode checkboxes if modes exist
    if (data.modes && data.modes.length > 0) {
      buildModeControls(data.modes);
    }

    // Rebuild deformed view on scale change
    scaleEl.addEventListener('input', () => {
      const s = Number(scaleEl.value);
      scaleValEl.textContent = `${s}×`;
      if (currentData) buildScene(currentData, s);
      setVisibility();
    });

    showUndeformedEl.addEventListener('change', setVisibility);
    showDeformedEl.addEventListener('change', setVisibility);
    showStressLegendEl.addEventListener('change', () => {
      setVisibility();
      if (currentData && showStressLegendEl.checked) {
        buildScene(currentData, Number(scaleEl.value));
      }
    });
  } catch (e) {
    statusEl.textContent = `Failed to load model.json (${e}). Serve via: python3 -m http.server`;
    throw e;
  }
}

function buildModeControls(modes) {
  // Create mode selector in HUD
  let modesDiv = document.getElementById('modesContainer');
  if (!modesDiv) {
    modesDiv = document.createElement('div');
    modesDiv.id = 'modesContainer';
    modesDiv.style.cssText = 'margin-top: 8px; border-top: 1px solid rgba(148,163,184,0.22); padding-top: 8px;';
    document.querySelector('#hud .row:last-of-type').after(modesDiv);
  }

  modesDiv.innerHTML = '<div style="font-size:12px;margin-bottom:4px;">Mode shapes (click to visualize):</div>';

  window.showModeCheckbox = [];
  modes.forEach((mode, i) => {
    const row = document.createElement('div');
    row.className = 'row';
    row.style.margin = '4px 0';

    const label = document.createElement('label');
    label.style.cssText = 'display: flex; align-items: center; gap: 6px; font-size: 11px;';

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.onchange = () => {
      if (modeGroup) {
        modeGroup.children[i].visible = checkbox.checked;
      }
    };
    window.showModeCheckbox.push(checkbox);

    label.appendChild(checkbox);
    label.appendChild(document.createTextNode(`Mode ${i + 1}: ${(mode.frequency / (2 * Math.PI)).toFixed(2)} Hz`));
    row.appendChild(label);
    modesDiv.appendChild(row);
  });

  // Add animate button
  const animRow = document.createElement('div');
  animRow.className = 'row';
  animRow.innerHTML = `
    <label><input id="animate" type="checkbox" /> Animate mode</label>
    <input id="animSpeed" type="range" min="0.1" max="5" step="0.1" value="1" style="width:100px;" />
    <span id="animSpeedVal" class="pill">1.0×</span>
  `;
  modesDiv.appendChild(animRow);

  // Hook up animation controls
  setTimeout(() => {
    window.animateCheckbox = document.getElementById('animate');
    const speedSlider = document.getElementById('animSpeed');

    window.animateCheckbox?.addEventListener('change', (e) => {
      animating = e.target.checked;
      if (animating) {
        startAnimation();
      } else {
        stopAnimation();
      }
    });

    speedSlider?.addEventListener('input', (e) => {
      animSpeed = parseFloat(e.target.value);
      document.getElementById('animSpeedVal').textContent = `${animSpeed.toFixed(1)}×`;
    });
  }, 0);
}

function startAnimation() {
  if (!currentData || !currentData.modes || currentData.modes.length === 0) return;

  // Find first visible mode
  const activeModeIdx = modeGroup?.children.findIndex(g => g.visible) ?? 0;
  if (activeModeIdx < 0 || activeModeIdx >= currentData.modes.length) return;

  modeShape = currentData.modes[activeModeIdx].shape;
  if (!modeShape) return;

  animTime = 0;
  animateMode();
}

function animateMode() {
  if (!animating || !modeShape || !currentData) return;

  animTime += animSpeed * 0.05;
  const scale = Math.sin(animTime) * 0.3; // Oscillate between -0.3 and 0.3

  // Update deformed shape with modal displacement
  const points = currentData.points || currentData.undeformed?.points || [];
  const cells = currentData.cells || currentData.undeformed?.cells || [];

  // Clear and rebuild deformed group with animated shape
  if (deformedGroup) {
    while(deformedGroup.children.length > 0) {
      deformedGroup.remove(deformedGroup.children[0]);
    }

    const sphereGeom = new THREE.SphereGeometry(0.02, 14, 14);
    const sphereMat = new THREE.MeshStandardMaterial({ color: 0x4ade80, roughness: 0.65, metalness: 0.0 });

    // Draw mode shape
    for (let ci = 0; ci < cells.length; ci++) {
      const [i, j] = cells[ci];
      const pi = new THREE.Vector3(
        points[i][0] + scale * (modeShape[i*3] || 0),
        points[i][1] + scale * (modeShape[i*3+1] || 0),
        points[i][2] + scale * (modeShape[i*3+2] || 0)
      );
      const pj = new THREE.Vector3(
        points[j][0] + scale * (modeShape[j*3] || 0),
        points[j][1] + scale * (modeShape[j*3+1] || 0),
        points[j][2] + scale * (modeShape[j*3+2] || 0)
      );
      deformedGroup.add(makeLine(pi, pj, 0x4ade80));
    }

    // Add nodes
    for (let i = 0; i < points.length; i++) {
      const m = new THREE.Mesh(sphereGeom, sphereMat);
      m.position.set(
        points[i][0] + scale * (modeShape[i*3] || 0),
        points[i][1] + scale * (modeShape[i*3+1] || 0),
        points[i][2] + scale * (modeShape[i*3+2] || 0)
      );
      deformedGroup.add(m);
    }
  }

  animId = requestAnimationFrame(animateMode);
}

function stopAnimation() {
  if (animId) {
    cancelAnimationFrame(animId);
    animId = null;
  }
  // Restore static deformed view
  if (currentData) {
    buildScene(currentData, Number(scaleEl.value));
    setVisibility();
  }
}

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

window.addEventListener('resize', onResize);

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

scaleValEl.textContent = `${scaleEl.value}×`;
loadModel().finally(() => animate());
