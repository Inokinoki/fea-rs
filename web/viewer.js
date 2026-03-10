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
let stressLegendMesh = null;

function buildScene(data, scale) {
  root.clear();
  undeformedGroup = new THREE.Group();
  deformedGroup = new THREE.Group();

  // Handle both simple and enhanced JSON formats
  const points = data.points || data.undeformed?.points || [];
  const cells = data.cells || data.undeformed?.cells || [];
  const disp = data.point_data?.displacement || data.undeformed?.point_data?.displacement || [];
  const stress = data.cell_data?.axial_stress || data.cell_data?.axial_stress || [];
  const metadata = data.metadata || {};

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

  // Add undeformed nodes
  for (let i = 0; i < points.length; i++) {
    const m = new THREE.Mesh(sphereGeom, sphereMat.clone());
    m.material.opacity = 0.4;
    m.material.transparent = true;
    m.position.copy(p0(i));
    undeformedGroup.add(m);
  }

  root.add(undeformedGroup);
  root.add(deformedGroup);

  // Update stress legend
  if (stressLegendMesh) {
    stressLegendEl.removeChild(stressLegendMesh);
  }
  if (showStressLegendEl.checked && finiteStress.length > 0) {
    const legendCanvas = createStressLegendTexture();
    const legendImg = document.createElement('img');
    legendImg.src = legendCanvas.toDataURL();
    legendImg.style.cssText = 'width: 200px; height: 25px; margin-top: 4px;';
    stressLegendEl.innerHTML = '';
    stressLegendEl.appendChild(legendImg);

    const statsDiv = document.createElement('div');
    statsDiv.style.cssText = 'font-size: 11px; margin-top: 4px; opacity: 0.8;';
    statsDiv.innerHTML = `Stress: ${smin.toFixed(2)} to ${smax.toFixed(2)} Pa`;
    stressLegendEl.appendChild(statsDiv);
  }

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
    statusEl.innerHTML = `Nodes: <code>${metadata.num_nodes || points.length}</code> | Elements: <code>${metadata.num_elements || cells.length}</code> | Max disp: <code>${(metadata.max_displacement || 0).toExponential(2)}</code> m`;
  }
}

function setVisibility() {
  if (undeformedGroup) undeformedGroup.visible = !!showUndeformedEl.checked;
  if (deformedGroup) deformedGroup.visible = !!showDeformedEl.checked;
  stressLegendEl.style.display = showStressLegendEl.checked ? 'block' : 'none';
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
