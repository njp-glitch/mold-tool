// Runs all the geometry off the main thread so the 3D view stays responsive.
// Replaces the old FastAPI backend: same three operations (load, mold), same state,
// but each browser tab has its own copy, so there is nothing to share or host.

import Module from "https://cdn.jsdelivr.net/npm/manifold-3d@3.5.4/manifold.js";
import { loadMesh, bounds } from "./lib/meshio.js";
import { autoOrient, analyzePropeller } from "./lib/geometry.js";
import { solidifyCavity } from "./lib/voxel.js";
import { generateMoldHalves, makeCavity, makeOriginal } from "./lib/mold.js";

const OCCT_BASE = "https://cdn.jsdelivr.net/npm/occt-import-js@0.0.23/dist/";
const DEFAULT_VOXEL_PITCH = 0.2;

const wasmReady = Module().then((m) => (m.setup(), m));

const state = { original: null, analysis: null, cavity: null, voxelPitch: DEFAULT_VOXEL_PITCH };

function setCavity(wasm, pitch, progress) {
  const solid = solidifyCavity(state.original.mesh, pitch, progress);
  const cavity = makeCavity(wasm, solid);
  if (state.cavity) state.cavity.manifold.delete();
  state.cavity = cavity;
  state.voxelPitch = pitch;
}

/** Collects every typed-array buffer in a result so it can be transferred instead of copied. */
function buffersOf(obj, out = []) {
  if (!obj || typeof obj !== "object") return out;
  if (ArrayBuffer.isView(obj)) { out.push(obj.buffer); return out; }
  for (const v of Array.isArray(obj) ? obj : Object.values(obj)) buffersOf(v, out);
  return out;
}

const meshOut = (m) => ({ vertices: Float32Array.from(m.vertices), faces: Uint32Array.from(m.faces) });

const handlers = {
  async init() {
    await wasmReady;
    return { ok: true };
  },

  async load({ name, buffer }, progress) {
    const wasm = await wasmReady;
    progress(name.match(/\.(step|stp)$/i) ? "Reading STEP file (first time loads the CAD engine)…" : "Reading STL…");
    const raw = await loadMesh(name, buffer, OCCT_BASE);
    progress("Orienting and analyzing…");
    const oriented = autoOrient(raw);
    const analysis = analyzePropeller(oriented);
    state.original = makeOriginal(oriented);
    state.analysis = analysis;
    try {
      setCavity(wasm, DEFAULT_VOXEL_PITCH, progress);
    } catch (exc) {
      return { error: `Could not build cavity solid: ${exc.message || exc}` };
    }
    return { mesh: meshOut(oriented), analysis, cavity_bounds: state.cavity.bounds };
  },

  async mold({ params }, progress) {
    const wasm = await wasmReady;
    if (!state.original) return { error: "No propeller loaded yet." };
    const { voxel_pitch, ...moldParams } = params;
    // Re-voxelize only when the pitch changes (expensive).
    if (voxel_pitch !== state.voxelPitch) {
      progress("Re-voxelizing cavity…");
      setCavity(wasm, voxel_pitch, progress);
    }
    const extremities = state.analysis ? state.analysis.extremities.map((e) => e.point) : [];
    try {
      const result = generateMoldHalves(wasm, state.cavity, moldParams, extremities, state.original, progress);
      return { ...result, cavity_bounds: state.cavity.bounds };
    } catch (exc) {
      return { error: `Mold generation failed: ${exc.message || exc}` };
    }
  },
};

self.onmessage = async (e) => {
  const { id, type, payload } = e.data;
  const progress = (text) => self.postMessage({ id, progress: text });
  try {
    const result = await handlers[type](payload, progress);
    self.postMessage({ id, result }, [...new Set(buffersOf(result))]);
  } catch (exc) {
    console.error(exc);
    let msg = exc && exc.message ? exc.message : String(exc);
    if (exc instanceof RangeError || /memory/i.test(msg)) {
      msg = `Ran out of memory (${msg}). Try a larger voxel pitch.`;
    }
    self.postMessage({ id, result: { error: msg } });
  }
};
