import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { writeSTL } from "./lib/meshio.js";

// --- Geometry worker ---
// All mold computation runs in worker.js, in this browser tab. Nothing is uploaded anywhere.
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
let nextRequestId = 0;
const pending = new Map();

worker.onmessage = (e) => {
  const { id, progress, result } = e.data;
  const req = pending.get(id);
  if (!req) return;
  if (progress !== undefined) {
    req.onProgress?.(progress);
    return;
  }
  pending.delete(id);
  req.resolve(result);
};

worker.onerror = (e) => {
  const msg = e.message || "The geometry engine failed to start (check your internet connection and reload).";
  for (const req of pending.values()) req.resolve({ error: msg });
  pending.clear();
};

function callWorker(type, payload, { transfer = [], onProgress } = {}) {
  const id = ++nextRequestId;
  return new Promise((resolve) => {
    pending.set(id, { resolve, onProgress });
    worker.postMessage({ id, type, payload }, transfer);
  });
}

callWorker("init"); // start downloading the geometry engine right away

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a1a);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 5000);
camera.position.set(180, 140, 180);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);

window.__camera = camera;
window.__controls = controls;
window.__scene = scene;

scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(100, 200, 100);
scene.add(dirLight);

// Axis helper: X=red, Y=green, Z=blue, matching the spec's coordinate convention.
scene.add(new THREE.AxesHelper(150));
scene.add(new THREE.GridHelper(300, 30, 0x444444, 0x2a2a2a));

let propellerMesh = null;
let markerGroup = new THREE.Group();
scene.add(markerGroup);

// Backend convention: X=length, Y=flat-face normal, Z=vertical (parting-plane normal).
// three.js convention: Y is "up". Remap [x,y,z] -> [x,z,y] purely for display so the
// spec's Z axis renders vertically; underlying analysis data is untouched.
function toThree([x, y, z]) {
  return [x, z, y];
}

// Builds a three.js geometry from a worker mesh ({vertices: Float32Array, faces: Uint32Array}),
// applying the same [x,y,z] -> [x,z,y] display remap as toThree.
function toGeometry(meshData) {
  const src = meshData.vertices;
  const remapped = new Float32Array(src.length);
  for (let i = 0; i < src.length; i += 3) {
    remapped[i] = src[i];
    remapped[i + 1] = src[i + 2];
    remapped[i + 2] = src[i + 1];
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(remapped, 3));
  geometry.setIndex(new THREE.BufferAttribute(meshData.faces, 1));
  geometry.computeVertexNormals();
  return geometry;
}

function buildMesh(meshData) {
  if (propellerMesh) {
    scene.remove(propellerMesh);
    propellerMesh.geometry.dispose();
  }
  const geometry = toGeometry(meshData);

  const material = new THREE.MeshStandardMaterial({
    color: 0xffaaaa,
    side: THREE.DoubleSide,
    flatShading: false,
    transparent: true,
    opacity: 0.6,
  });
  propellerMesh = new THREE.Mesh(geometry, material);
  scene.add(propellerMesh);
}

function buildMarkers(analysis) {
  while (markerGroup.children.length) {
    markerGroup.remove(markerGroup.children[0]);
  }

  const hubMat = new THREE.MeshBasicMaterial({ color: 0xff3333 });
  const hubSphere = new THREE.Mesh(new THREE.SphereGeometry(3, 16, 16), hubMat);
  hubSphere.position.set(...toThree(analysis.hub_center));
  markerGroup.add(hubSphere);

  const extMat = new THREE.MeshBasicMaterial({ color: 0xffdd33 });
  for (const ext of analysis.extremities) {
    const s = new THREE.Mesh(new THREE.SphereGeometry(2.5, 16, 16), extMat);
    s.position.set(...toThree(ext.point));
    markerGroup.add(s);
  }

  for (const thin of analysis.thin_regions) {
    const isThin = thin.thickness_mm < 1.0;
    const mat = new THREE.MeshBasicMaterial({ color: isThin ? 0xff8800 : 0x33ccff });
    const s = new THREE.Mesh(new THREE.SphereGeometry(1.5, 12, 12), mat);
    s.position.set(...toThree(thin.location));
    s.position.z += 4;
    markerGroup.add(s);
  }
}

function spinnerHTML(text) {
  return `<span class="spinner"></span>${text}`;
}

function renderInfo(analysis) {
  const lines = [];
  lines.push(`Hub center: [${analysis.hub_center.map((v) => v.toFixed(1)).join(", ")}]`);
  lines.push(`Hub radius est: ${analysis.hub_radius_estimate.toFixed(2)} mm`);
  lines.push(`Blade length (X span): ${analysis.blade_length.toFixed(1)} mm`);
  lines.push(`<strong>Extremities</strong>`);
  lines.push(
    "<ul>" +
      analysis.extremities
        .map((e) => `<li>${e.side}: [${e.point.map((v) => v.toFixed(1)).join(", ")}]</li>`)
        .join("") +
      "</ul>"
  );
  lines.push(`<strong>Thin regions</strong> (orange = below 1mm threshold)`);
  lines.push(
    "<ul>" +
      analysis.thin_regions
        .map((t) => `<li>${t.side}: ${t.thickness_mm.toFixed(2)} mm</li>`)
        .join("") +
      "</ul>"
  );
  if (analysis.warnings.length) {
    lines.push(`<div class="warn"><strong>Warnings</strong><ul>${analysis.warnings.map((w) => `<li>${w}</li>`).join("")}</ul></div>`);
  }
  document.getElementById("info").innerHTML = lines.join("");
}

// --- Mold (Phase 2): cavity / base / gate, driven by sliders ---

const moldGroup = new THREE.Group();
scene.add(moldGroup);
window.__moldGroup = moldGroup;

function buildMoldMesh(meshData, color, opacity) {
  const geometry = toGeometry(meshData);
  const material = new THREE.MeshStandardMaterial({
    color,
    side: THREE.DoubleSide,
    transparent: opacity < 1,
    opacity,
  });
  return new THREE.Mesh(geometry, material);
}

const DEFAULTS = {
  voxelPitch: 0.2,
  wallThickness: 6,
  baseThickness: 15,
  baseExtension: 5,
  gateRunExtra: 2,
  gateDiameter: 3,
  riserDiameter: 3,
  riserTaperAngle: 5,
  partingPitch: 1,
};

const sliderIds = ["voxelPitch", "wallThickness", "baseThickness", "baseExtension", "gateRunExtra", "gateDiameter", "riserDiameter", "riserTaperAngle", "partingPitch"];

function readSliders() {
  const v = {};
  for (const id of sliderIds) v[id] = parseFloat(document.getElementById(id).value);
  return v;
}

function syncSliderLabels() {
  const units = { riserTaperAngle: "°" };
  for (const id of sliderIds) {
    const unit = units[id] ?? " mm";
    document.getElementById(id + "Val").textContent = document.getElementById(id).value + unit;
  }
}

// --- Instant block preview ---
// The real mold takes 20-30 s to compute, but the outer block is just a box sized from
// the cavity bounds and the sliders (same formulas as generate_mold_halves in mold.py).
// Draw it immediately on every slider tick so changes are visible while the real mesh catches up.
const previewGroup = new THREE.Group();
previewGroup.visible = false;
scene.add(previewGroup);
let cavityBounds = null;

function previewBox(xmin, xmax, ymin, ymax, zmin, zmax, color) {
  // Backend [x,y,z] -> three.js [x,z,y], same remap as toThree.
  const geo = new THREE.BoxGeometry(xmax - xmin, zmax - zmin, ymax - ymin);
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(geo),
    new THREE.LineBasicMaterial({ color })
  );
  edges.position.set((xmin + xmax) / 2, (zmin + zmax) / 2, (ymin + ymax) / 2);
  geo.dispose();
  return edges;
}

function updatePreview() {
  if (!cavityBounds) return;
  const v = readSliders();
  const [[cx0, cy0, cz0], [cx1, cy1, cz1]] = cavityBounds;
  const margin = v.wallThickness + v.baseExtension;
  const xmin = cx0 - margin, xmax = cx1 + margin;
  const ymin = cy0 - margin, ymax = cy1 + margin + v.gateRunExtra;
  const hTop = Math.max(v.baseThickness, cz1 + v.wallThickness);
  const hBot = Math.max(v.baseThickness, -cz0 + v.wallThickness);

  while (previewGroup.children.length) {
    const child = previewGroup.children.pop();
    child.geometry.dispose();
    child.material.dispose();
  }
  previewGroup.add(previewBox(xmin, xmax, ymin, ymax, 0, hTop, 0xffffff));
  previewGroup.add(previewBox(xmin, xmax, ymin, ymax, -hBot, 0, 0xffffff));
  previewGroup.visible = true;
}

// The server can't cancel a mold job once it starts, so never stack them: if sliders move
// while one is running, remember that and run exactly one more with the latest values.
let moldInFlight = false;
let moldDirty = false;

async function fetchMold() {
  if (moldInFlight) {
    moldDirty = true;
    return;
  }
  moldInFlight = true;
  try {
    await fetchMoldOnce();
  } finally {
    moldInFlight = false;
    if (moldDirty) {
      moldDirty = false;
      fetchMold();
    } else if (!moldDebounceTimer) {
      previewGroup.visible = false; // the real mesh now matches the sliders
    }
  }
}

let moldFetchToken = 0;

async function fetchMoldOnce() {
  const v = readSliders();
  const pitchChanged = parseFloat(document.getElementById("voxelPitch").value) !== window.__lastVoxelPitch;
  document.getElementById("moldStatus").innerHTML = spinnerHTML(
    pitchChanged ? "Re-voxelizing cavity (may take a moment)…" : "Computing mold…"
  );
  const token = ++moldFetchToken;

  const data = await callWorker("mold", {
    params: {
      voxel_pitch: v.voxelPitch,
      wall_thickness: v.wallThickness,
      base_thickness: v.baseThickness,
      base_extension: v.baseExtension,
      gate_run_extra: v.gateRunExtra,
      gate_diameter: v.gateDiameter,
      riser_diameter: v.riserDiameter,
      riser_taper_angle: v.riserTaperAngle,
      parting_pitch: v.partingPitch,
    },
  }, {
    onProgress: (text) => {
      if (token === moldFetchToken) document.getElementById("moldStatus").innerHTML = spinnerHTML(text);
    },
  });
  if (token !== moldFetchToken) return; // a newer request superseded this one

  if (data.error) {
    document.getElementById("moldStatus").innerHTML = `<span class="warn">Error: ${data.error}</span>`;
    return;
  }
  lastMold = data;

  while (moldGroup.children.length) {
    const child = moldGroup.children.pop();
    child.geometry.dispose();
    moldGroup.remove(child);
  }

  moldGroup.add(buildMoldMesh(data.mold_half_top, 0x999999, 0.35));
  moldGroup.add(buildMoldMesh(data.mold_half_bottom, 0x999999, 0.35));
  moldGroup.add(buildMoldMesh(data.cavity_top, 0x33ccff, 0.5));
  moldGroup.add(buildMoldMesh(data.cavity_bottom, 0x33ccff, 0.5));
  moldGroup.add(buildMoldMesh(data.gate, 0x2266ff, 1.0));
  for (const r of (data.risers ?? [])) {
    moldGroup.add(buildMoldMesh(r, 0xff8833, 0.85));
  }
  if (data.parting_surface) {
    moldGroup.add(buildMoldMesh(data.parting_surface, 0xffdd00, 0.7));
  }

  if (data.cavity_bounds) cavityBounds = data.cavity_bounds;
  window.__lastVoxelPitch = parseFloat(document.getElementById("voxelPitch").value);
  document.getElementById("gateLengthVal").textContent = data.gate_length.toFixed(1) + " mm";
  document.getElementById("moldStatus").textContent = "Mold up to date.";
  document.getElementById("moldWarnings").innerHTML = data.warnings.length
    ? "<ul>" + data.warnings.map((w) => `<li>${w}</li>`).join("") + "</ul>"
    : "";
}

let moldDebounceTimer = null;
function scheduleMoldUpdate() {
  syncSliderLabels();
  updatePreview();
  clearTimeout(moldDebounceTimer);
  moldDebounceTimer = setTimeout(() => {
    moldDebounceTimer = null;
    fetchMold();
  }, 200);
}

for (const id of sliderIds) {
  document.getElementById(id).addEventListener("input", scheduleMoldUpdate);
}

document.getElementById("resetBtn").addEventListener("click", () => {
  for (const id of sliderIds) document.getElementById(id).value = DEFAULTS[id];
  scheduleMoldUpdate();
});

// --- Propeller load/upload ---

let lastMold = null; // most recent successful mold, used for STL export

function loadAnalysis() {
  document.getElementById("info").textContent = "Upload a propeller (STEP/STL) to get started.";
}

document.getElementById("fileInput").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const input = e.target;
  input.disabled = true;
  const info = document.getElementById("info");
  info.innerHTML = spinnerHTML(`Analyzing ${file.name}…`);

  let data;
  try {
    const buffer = await file.arrayBuffer();
    data = await callWorker("load", { name: file.name, buffer }, {
      transfer: [buffer],
      onProgress: (text) => { info.innerHTML = spinnerHTML(text); },
    });
  } catch (err) {
    data = { error: err.message };
  }
  input.disabled = false;

  if (data.error) {
    info.innerHTML = `<span class="warn">Error: ${data.error}</span>`;
    return;
  }
  lastMold = null;
  buildMesh(data.mesh);
  buildMarkers(data.analysis);
  renderInfo(data.analysis);
  cavityBounds = data.cavity_bounds;
  updatePreview();
  fetchMold();
});

// --- STL export ---

async function downloadSTL(half) {
  const status = document.getElementById("exportStatus");
  if (!lastMold) {
    status.textContent = "Export error: No mold generated yet.";
    return;
  }
  status.textContent = "Preparing STL…";
  try {
    const m = half === "top" ? lastMold.mold_half_top : lastMold.mold_half_bottom;
    const blob = new Blob([writeSTL(m.vertices, m.faces)], { type: "application/octet-stream" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `mold_half_${half}.stl`;
    a.click();
    URL.revokeObjectURL(url);
    status.textContent = `mold_half_${half}.stl downloaded.`;
  } catch (e) {
    status.textContent = "Export failed: " + e.message;
  }
}

document.getElementById("exportTop").addEventListener("click", () => downloadSTL("top"));
document.getElementById("exportBottom").addEventListener("click", () => downloadSTL("bottom"));

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

loadAnalysis();
animate();
