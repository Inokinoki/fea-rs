import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';

const app = document.getElementById('app');
const statusEl = document.getElementById('status');
const scaleEl = document.getElementById('scale');
const scaleValEl = document.getElementById('scaleVal');
const showUndeformedEl = document.getElementById('showUndeformed');
const showDeformedEl = document.getElementById('showDeformed');

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
  // Simple blue -> gray -> red mapping around mid.
  if (!Number.isFinite(s) || smax <= smin) return new THREE.Color(0x93c5fd);
  const t = Math.min(1, Math.max(0, (s - smin) / (smax - smin)));
  const c1 = new THREE.Color(0x2563eb); // blue
  const c2 = new THREE.Color(0xe5e7eb); // light gray
  const c3 = new THREE.Color(0xef4444); // red
  if (t < 0.5) return c1.clone().lerp(c2, t * 2);
  return c2.clone().lerp(c3, (t - 0.5) * 2);
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
  // Note: linewidth is ignored in most WebGL implementations.
  return line;
}

let undeformedGroup = null;
let deformedGroup = null;

function buildScene(data, scale) {
  root.clear();
  undeformedGroup = new THREE.Group();
  deformedGroup = new THREE.Group();

  const pts = data.points;
  const disp = data.point_data?.displacement || [];
  const stress = data.cell_data?.axial_stress || [];
  const smin = Math.min(...stress.filter(Number.isFinite));
  const smax = Math.max(...stress.filter(Number.isFinite));

  const p0 = (i) => new THREE.Vector3(pts[i][0], pts[i][1], pts[i][2]);
  const p1 = (i) => {
    const d = disp[i] || [0, 0, 0];
    return new THREE.Vector3(pts[i][0] + scale * d[0], pts[i][1] + scale * d[1], pts[i][2] + scale * d[2]);
  };

  for (let ci = 0; ci < data.cells.length; ci++) {
    const [i, j] = data.cells[ci];
    undeformedGroup.add(makeLine(p0(i), p0(j), 0x94a3b8));
    const c = stressToColor(stress[ci], smin, smax);
    deformedGroup.add(makeLine(p1(i), p1(j), c));
  }

  // Add small spheres for nodes (deformed).
  const box = computeBounds(pts);
  const size = box.getSize(new THREE.Vector3()).length();
  const r = Math.max(0.01, size * 0.01);
  const sphereGeom = new THREE.SphereGeometry(r, 14, 14);
  const sphereMat = new THREE.MeshStandardMaterial({ color: 0xe5e7eb, roughness: 0.65, metalness: 0.0 });
  for (let i = 0; i < pts.length; i++) {
    const m = new THREE.Mesh(sphereGeom, sphereMat);
    m.position.copy(p1(i));
    deformedGroup.add(m);
  }

  root.add(undeformedGroup);
  root.add(deformedGroup);

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
}

function setVisibility() {
  if (undeformedGroup) undeformedGroup.visible = !!showUndeformedEl.checked;
  if (deformedGroup) deformedGroup.visible = !!showDeformedEl.checked;
}

async function loadModel() {
  statusEl.textContent = 'Loading model.json...';
  try {
    const res = await fetch('./model.json', { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    statusEl.textContent = `Loaded ${data.points.length} points, ${data.cells.length} cells.`;
    const scale = Number(scaleEl.value);
    buildScene(data, scale);
    setVisibility();

    // Rebuild deformed view on scale change (simple + robust for small meshes).
    scaleEl.addEventListener('input', () => {
      const s = Number(scaleEl.value);
      scaleValEl.textContent = `${s}\u00d7`;
      buildScene(data, s);
      setVisibility();
    });
    showUndeformedEl.addEventListener('change', setVisibility);
    showDeformedEl.addEventListener('change', setVisibility);
  } catch (e) {
    statusEl.textContent = `Failed to load model.json (${e}). If opened from disk, serve via python http.server.`;
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

scaleValEl.textContent = `${scaleEl.value}\u00d7`;
loadModel().finally(() => animate());

