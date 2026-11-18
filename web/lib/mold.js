// Port of backend/mold.py: mold cavity, base, gate, riser, and parting surface generation.
//
// Coordinate convention (post autoOrient):
//   X = blade length, Y = in-plane remaining axis, Z = flat-face normal / parting-plane normal.
// Top mold half = Z >= parting surface; bottom = Z <= parting surface.
//
// `wasm` is the initialized manifold-3d module (Manifold, Mesh, ...).

import { bounds, isWatertight } from "./meshio.js";
import { ZRayCaster, castAlongY } from "./raycast.js";
import { gaussianFilter, maximumFilter, minimumFilter, nearestFeatureIndex } from "./ndimage.js";

// ---------------------------------------------------------------------------
// Manifold helpers
// ---------------------------------------------------------------------------

export function toManifold(wasm, mesh) {
  const m = new wasm.Mesh({
    numProp: 3,
    vertProperties: Float32Array.from(mesh.vertices),
    triVerts: Uint32Array.from(mesh.faces),
  });
  const man = new wasm.Manifold(m);
  const st = man.status();
  if (st !== "NoError") {
    man.delete();
    throw new Error(`Geometry engine rejected mesh (${st}).`);
  }
  return man;
}

export function fromManifold(man) {
  const m = man.getMesh();
  const np = m.numProp;
  const n = m.vertProperties.length / np;
  const vertices = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    vertices[i * 3] = m.vertProperties[i * np];
    vertices[i * 3 + 1] = m.vertProperties[i * np + 1];
    vertices[i * 3 + 2] = m.vertProperties[i * np + 2];
  }
  return { vertices, faces: Uint32Array.from(m.triVerts) };
}

// ---------------------------------------------------------------------------
// Silhouette-following parting surface
// ---------------------------------------------------------------------------

const PARTING_SUB = 5;
const PARTING_SIGMA = 1.0;
const PARTING_PASSES = 12;
const PARTING_BLEND_PASSES = 30;
const PARTING_BLEND_SIGMA = 1.0;

/** np.arange(start, stop, step) */
function arange(start, stop, step) {
  const n = Math.max(0, Math.ceil((stop - start) / step));
  return Float64Array.from({ length: n }, (_, i) => start + i * step);
}

/** np.linspace(a, b, n) */
function linspace(a, b, n) {
  if (n === 1) return Float64Array.of(a);
  return Float64Array.from({ length: n }, (_, i) => (i === n - 1 ? b : a + ((b - a) * i) / (n - 1)));
}

/**
 * parting_z(x, y): the Z of the blade's mid-surface at each XY grid node, kept
 * inside the blade and blended into flat plateaus outside it.
 * Returns { xs, ys, Z, inside } with Z[i * ny + j] at (xs[i], ys[j]).
 */
export function computePartingHeightmap(caster, meshBounds, pitch, xRange, yRange) {
  const [xmin, xmax] = xRange;
  const [ymin, ymax] = yRange;
  const zTop = meshBounds[1][2] + 10.0;

  const xs = arange(xmin, xmax + pitch, pitch);
  const ys = arange(ymin, ymax + pitch, pitch);
  const nx = xs.length, ny = ys.length, n = nx * ny;

  let Z = new Float64Array(n);
  const mask = new Float64Array(n);
  const hz = [], hn = [];
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      hz.length = 0; hn.length = 0;
      caster.cast(xs[i], ys[j], (z, f) => { hz.push(z); hn.push(caster.nz[f]); });
      if (!hz.length) continue;
      const order = hz.map((_, k) => k).sort((p, q) => hz[q] - hz[p]); // descending Z
      const zd = order.map((k) => hz[k]);
      const nd = order.map((k) => hn[k]);
      const mid = (zd[0] + zd[zd.length - 1]) / 2;
      let partingZ = mid;
      // A twisted/folded section can cross the surface more than once; keep the
      // up->down transition nearest the overall midpoint (the true mid-plane).
      let best = Infinity;
      for (let m = 0; m < zd.length - 1; m++) {
        if (nd[m] >= -0.05 && nd[m + 1] <= 0.05) {
          const c = (zd[m] + zd[m + 1]) / 2;
          if (Math.abs(c - mid) < best) { best = Math.abs(c - mid); partingZ = c; }
        }
      }
      Z[i * ny + j] = partingZ;
      mask[i * ny + j] = 1;
    }
  }

  // Bilinear patches between nodes must stay inside the blade too, so build each
  // node's allowed band from the tightest surface bounds over every cell it touches:
  // sample on a sub-grid, then max-filter the lower surface and min-filter the upper.
  const sub = PARTING_SUB;
  const snx = (nx - 1) * sub + 1, sny = (ny - 1) * sub + 1;
  const sx = linspace(xs[0], xs[nx - 1], snx);
  const sy = linspace(ys[0], ys[ny - 1], sny);
  const sLo = new Float64Array(snx * sny).fill(-Infinity);
  const sHi = new Float64Array(snx * sny).fill(Infinity);
  for (let i = 0; i < snx; i++) {
    for (let j = 0; j < sny; j++) {
      let lo = Infinity, hi = -Infinity;
      const hits = caster.cast(sx[i], sy[j], (z) => { if (z < lo) lo = z; if (z > hi) hi = z; });
      if (hits) { sLo[i * sny + j] = lo; sHi[i * sny + j] = hi; }
    }
  }
  const win = 2 * sub + 1;
  const loF = maximumFilter(sLo, snx, sny, win);
  const hiF = minimumFilter(sHi, snx, sny, win);

  const inside = new Uint8Array(n);
  const bandLo = new Float64Array(n), bandHi = new Float64Array(n);
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const k = i * ny + j;
      const zl = loF[i * sub * sny + j * sub], zh = hiF[i * sub * sny + j * sub];
      if (Number.isFinite(zl) && Number.isFinite(zh)) {
        inside[k] = 1;
        // Middle 50% of the local thickness, so both halves get material even where thin.
        const margin = 0.25 * Math.max(zh - zl, 0);
        let bl = zl + margin, bh = zh - margin;
        // Too-coarse grid across a hub shoulder/steep twist: no value satisfies every
        // neighbouring cell. Pin to the midpoint of the neighbourhood bounds.
        if (bl > bh) bl = bh = 0.5 * (zl + zh);
        bandLo[k] = bl; bandHi[k] = bh;
      }
    }
  }

  const smooth = (arr) => {
    const am = arr.map((v, k) => v * mask[k]);
    const num = gaussianFilter(am, nx, ny, PARTING_SIGMA);
    const den = gaussianFilter(mask, nx, ny, PARTING_SIGMA);
    return num.map((v, k) => (den[k] > 0.05 ? v / den[k] : 0));
  };
  const clamp = (arr) => {
    const out = Float64Array.from(arr);
    for (let k = 0; k < n; k++) {
      if (inside[k]) out[k] = Math.min(Math.max(arr[k], bandLo[k]), bandHi[k]);
    }
    return out;
  };

  for (let p = 0; p < PARTING_PASSES; p++) Z = clamp(smooth(clamp(Z)));
  Z = clamp(Z);

  // Outside the footprint, hold the height of the nearest point on the part
  // (printable plateaus), then soften the seams where plateaus meet.
  if (inside.some((v) => v)) {
    const near = nearestFeatureIndex(inside, nx, ny);
    for (let k = 0; k < n; k++) if (!inside[k]) Z[k] = Z[near[k]];
    const held = Float64Array.from(Z);
    for (let p = 0; p < PARTING_BLEND_PASSES; p++) {
      Z = gaussianFilter(Z, nx, ny, PARTING_BLEND_SIGMA, "nearest");
      for (let k = 0; k < n; k++) if (inside[k]) Z[k] = held[k];
    }
  }
  return { xs, ys, Z, inside };
}

/** np.searchsorted(a, v) (side="left") */
function searchsorted(a, v) {
  let lo = 0, hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] < v) lo = mid + 1; else hi = mid;
  }
  return lo;
}

/**
 * Pins the parting height to a single value across the gate's corridor (so the
 * straight gate is split evenly along its whole run), then blends it in.
 * Returns { Z, gateZ }.
 */
export function flattenGateCorridor(xs, ys, Z, inside, gateX, gateHalfWidth, entryY, blockYMax) {
  const nx = xs.length, ny = ys.length;
  const iy = Math.min(Math.max(searchsorted(ys, entryY), 0), ny - 1);
  const ix = Math.min(Math.max(searchsorted(xs, gateX), 0), nx - 1);
  const gateZ = Z[ix * ny + iy];

  const corridor = new Uint8Array(nx * ny);
  let any = false;
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const k = i * ny + j;
      if (Math.abs(xs[i] - gateX) <= gateHalfWidth && ys[j] >= entryY && ys[j] <= blockYMax && !inside[k]) {
        corridor[k] = 1;
        any = true;
      }
    }
  }
  if (!any) return { Z, gateZ };

  Z = Float64Array.from(Z);
  for (let k = 0; k < Z.length; k++) if (corridor[k]) Z[k] = gateZ;
  const heldZ = Float64Array.from(Z);
  for (let p = 0; p < PARTING_BLEND_PASSES; p++) {
    Z = gaussianFilter(Z, nx, ny, PARTING_BLEND_SIGMA, "nearest");
    for (let k = 0; k < Z.length; k++) if (inside[k] || corridor[k]) Z[k] = heldZ[k];
  }
  return { Z, gateZ };
}

/** Closed solid of everything below the parting surface, down to zFloor. Used as a boolean cutter. */
function buildLowerCutter(xs, ys, Z, zFloor) {
  const nx = xs.length, ny = ys.length, n = nx * ny;
  const vertices = new Float64Array(n * 2 * 3);
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const k = i * ny + j;
      vertices.set([xs[i], ys[j], Z[k]], k * 3);
      vertices.set([xs[i], ys[j], zFloor], (n + k) * 3);
    }
  }
  const ti = (i, j) => i * ny + j;
  const bi = (i, j) => n + i * ny + j;
  const f = [];
  for (let i = 0; i < nx - 1; i++) {
    for (let j = 0; j < ny - 1; j++) {
      let a = ti(i, j), b = ti(i + 1, j), c = ti(i + 1, j + 1), d = ti(i, j + 1);
      f.push(a, b, c, a, c, d); // top, +Z
      a = bi(i, j); b = bi(i + 1, j); c = bi(i + 1, j + 1); d = bi(i, j + 1);
      f.push(a, c, b, a, d, c); // bottom, -Z
    }
  }
  for (let j = 0; j < ny - 1; j++) {
    f.push(ti(0, j), bi(0, j + 1), bi(0, j), ti(0, j), ti(0, j + 1), bi(0, j + 1)); // -X
    f.push(ti(nx - 1, j), bi(nx - 1, j), bi(nx - 1, j + 1), ti(nx - 1, j), bi(nx - 1, j + 1), ti(nx - 1, j + 1)); // +X
  }
  for (let i = 0; i < nx - 1; i++) {
    f.push(ti(i, 0), bi(i, 0), bi(i + 1, 0), ti(i, 0), bi(i + 1, 0), ti(i + 1, 0)); // -Y
    f.push(ti(i, ny - 1), bi(i + 1, ny - 1), bi(i, ny - 1), ti(i, ny - 1), ti(i + 1, ny - 1), bi(i + 1, ny - 1)); // +Y
  }
  return { vertices, faces: Uint32Array.from(f) };
}

/** Display mesh of just the parting surface. */
function buildPartingSurfaceDisplay(xs, ys, Z) {
  const nx = xs.length, ny = ys.length;
  const vertices = new Float32Array(nx * ny * 3);
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) vertices.set([xs[i], ys[j], Z[i * ny + j]], (i * ny + j) * 3);
  const f = [];
  for (let i = 0; i < nx - 1; i++) {
    for (let j = 0; j < ny - 1; j++) {
      const a = i * ny + j, b = (i + 1) * ny + j, c = (i + 1) * ny + j + 1, d = i * ny + j + 1;
      f.push(a, b, c, a, c, d);
    }
  }
  return { vertices, faces: Uint32Array.from(f) };
}

// ---------------------------------------------------------------------------
// Gate and risers
// ---------------------------------------------------------------------------

function gateEntryY(cavityMesh, x, blockYMax) {
  const ys = castAlongY(cavityMesh, x, 0.0).filter((y) => y <= blockYMax + 10.0);
  if (!ys.length) return { entryY: 0.0, warn: "Gate: no cavity found along +Y parting line; defaulting to hub centre." };
  return { entryY: Math.max(...ys), warn: null };
}

/** Straight Y-axis gate from the cavity out to the +Y mold face, centred at height z. */
function gateChannel(wasm, entryY, diameter, blockYMax, x = 0.0, z = 0.0, overlap = 1.5) {
  // entryY is the cavity's +Y surface; step back into the cavity by the overlap so
  // the runner actually connects (stepping outward would leave a plug of mold).
  const innerY = entryY - overlap;
  const length = Math.max(blockYMax - innerY, overlap);
  const channel = wasm.Manifold.cylinder(length, diameter / 2, diameter / 2, 32, true)
    .rotate([-90, 0, 0])
    .translate([x, (blockYMax + innerY) / 2, z]);
  return { channel, length };
}

/** X-axis vent pocket from the mold side wall to a blade tip (tapered for demolding). */
function riserPocket(wasm, ext, moldXMin, moldXMax, diameter, taperDeg, overlap = 1.5) {
  const [x, y, z] = ext;
  const r = diameter / 2;
  const taperR = taperDeg > 0 ? Math.tan((taperDeg * Math.PI) / 180) : 0;
  const depth = x >= 0 ? moldXMax - x + overlap : x - moldXMin + overlap;
  if (depth <= 0) return { pocket: null, isTop: z >= 0 };
  const cone = wasm.Manifold.cylinder(depth, r, r + taperR * depth, 32, false);
  const pocket = x >= 0
    ? cone.rotate([0, 90, 0]).translate([x - overlap, y, z])
    : cone.rotate([0, -90, 0]).translate([x + overlap, y, z]);
  cone.delete();
  return { pocket, isTop: z >= 0 };
}

// ---------------------------------------------------------------------------
// Main mold generation
// ---------------------------------------------------------------------------

export const DEFAULT_PARAMS = {
  wall_thickness: 6.0,
  base_thickness: 15.0,
  base_extension: 5.0,
  gate_run_extra: 2.0,
  gate_diameter: 3.0,
  riser_diameter: 3.0,
  riser_taper_angle: 5.0,
  parting_pitch: 1.0,
};

/** Python's f"{v:.1f}" / round(v, 1): ties go to the even digit (16.25 -> "16.2"). */
function py1(v) {
  const s = v * 10, f = Math.floor(s), d = s - f;
  const r = d > 0.5 + 1e-9 ? f + 1 : d < 0.5 - 1e-9 ? f : f % 2 === 0 ? f : f + 1;
  return (r / 10).toFixed(1);
}
const r1 = py1;

/**
 * Builds mold halves + gate around a precomputed cavity.
 *   cavity: { mesh, manifold, bounds }   (from solidifyCavity + toManifold)
 *   original: { mesh, caster, bounds }   (the oriented propeller, for the parting surface)
 * Returns plain meshes (Float32Array/Uint32Array) ready for display/export.
 */
export function generateMoldHalves(wasm, cavity, params, extremities, original, onProgress = () => {}) {
  const P = { ...DEFAULT_PARAMS, ...params };
  const { Manifold } = wasm;
  const trash = [];
  const keep = (m) => (trash.push(m), m);
  const warnings = [];

  try {
    const cb = cavity.bounds;
    const margin = P.wall_thickness + P.base_extension;
    const bx0 = cb[0][0] - margin, bx1 = cb[1][0] + margin;
    const by0 = cb[0][1] - margin;
    // +Y is where the gate exits, so it alone gets the extra run length.
    const by1 = cb[1][1] + margin + P.gate_run_extra;

    const hTop = Math.max(P.base_thickness, cb[1][2] + P.wall_thickness);
    const hBot = Math.max(P.base_thickness, -cb[0][2] + P.wall_thickness);
    if (hTop > P.base_thickness) warnings.push(`Top base increased to ${py1(hTop)} mm to fully enclose cavity.`);
    if (hBot > P.base_thickness) warnings.push(`Bottom base increased to ${py1(hBot)} mm to fully enclose cavity.`);

    const makeBlock = (z0, z1) =>
      keep(Manifold.cube([bx1 - bx0, by1 - by0, z1 - z0], true).translate([(bx0 + bx1) / 2, (by0 + by1) / 2, (z0 + z1) / 2]));

    // --- Gate entry point ---
    const gateX = 0.0;
    const { entryY, warn } = gateEntryY(cavity.mesh, gateX, by1);
    if (warn) warnings.push(warn);
    let gateZ = 0.0;

    // --- Risers ---
    const risersTop = [], risersBottom = [], riserDisplay = [];
    for (const ext of extremities || []) {
      const { pocket, isTop } = riserPocket(wasm, ext, bx0, bx1, P.riser_diameter, P.riser_taper_angle);
      if (!pocket) {
        warnings.push(`Riser at [${ext.map(r1).join(", ")}] skipped: outside mold bounds.`);
        continue;
      }
      keep(pocket);
      riserDisplay.push(pocket);
      (isTop ? risersTop : risersBottom).push(pocket);
    }

    // --- Parting surface split ---
    let partingSurface = null;
    let useCurved = !!original;
    let cavityTop, cavityBottom, blockTop, blockBottom;

    if (useCurved) {
      try {
        onProgress("Computing parting surface…");
        const hm = computePartingHeightmap(original.caster, original.bounds, P.parting_pitch, [bx0, bx1], [by0, by1]);
        const flat = flattenGateCorridor(hm.xs, hm.ys, hm.Z, hm.inside, gateX, P.gate_diameter / 2 + 1.0, entryY, by1);
        gateZ = flat.gateZ;

        onProgress("Splitting mold halves…");
        const zFloor = -(hBot + 10.0);
        const cutter = keep(toManifold(wasm, buildLowerCutter(hm.xs, hm.ys, flat.Z, zFloor)));
        const fullBlock = makeBlock(-hBot, hTop);

        cavityBottom = keep(cavity.manifold.intersect(cutter));
        cavityTop = keep(cavity.manifold.subtract(cutter));
        blockBottom = keep(fullBlock.intersect(cutter));
        blockTop = keep(fullBlock.subtract(cutter));

        partingSurface = buildPartingSurfaceDisplay(hm.xs, hm.ys, flat.Z);
        warnings.push(`Silhouette parting surface computed at ${py1(P.parting_pitch)} mm pitch.`);
      } catch (exc) {
        warnings.push(`Silhouette parting failed (${exc.message || exc}); falling back to flat Z=0 parting.`);
        useCurved = false;
        gateZ = 0.0;
      }
    }

    if (!useCurved) {
      cavityTop = keep(cavity.manifold.trimByPlane([0, 0, 1], 0));
      cavityBottom = keep(cavity.manifold.trimByPlane([0, 0, -1], 0));
      blockTop = makeBlock(0, hTop);
      blockBottom = makeBlock(-hBot, 0);
    }

    // --- Gate ---
    const { channel: gate, length: gateLength } = gateChannel(wasm, entryY, P.gate_diameter, by1, gateX, gateZ);
    keep(gate);

    // --- Boolean subtraction ---
    onProgress("Cutting cavity, gate and risers…");
    const voidTop = keep(Manifold.union([cavityTop, gate, ...risersTop]));
    const voidBottom = keep(Manifold.union([cavityBottom, gate, ...risersBottom]));
    const moldTop = keep(blockTop.subtract(voidTop));
    const moldBottom = keep(blockBottom.subtract(voidBottom));

    const out = {
      mold_half_top: fromManifold(moldTop),
      mold_half_bottom: fromManifold(moldBottom),
      cavity_top: fromManifold(cavityTop),
      cavity_bottom: fromManifold(cavityBottom),
      gate: fromManifold(gate),
      gate_length: gateLength,
      risers: riserDisplay.map(fromManifold),
      parting_surface: partingSurface,
      warnings,
    };
    if (moldTop.isEmpty() || !isWatertight(out.mold_half_top.faces)) warnings.push("Top mold half is not watertight; may need repair.");
    if (moldBottom.isEmpty() || !isWatertight(out.mold_half_bottom.faces)) warnings.push("Bottom mold half is not watertight; may need repair.");
    return out;
  } finally {
    for (const m of trash) m.delete();
  }
}

/** Cavity bundle for generateMoldHalves from a solidified mesh. */
export function makeCavity(wasm, solidMesh) {
  return { mesh: solidMesh, manifold: toManifold(wasm, solidMesh), bounds: bounds(solidMesh.vertices) };
}

/** Original-mesh bundle for the parting surface (ray index built once per upload). */
export function makeOriginal(mesh) {
  return { mesh, caster: new ZRayCaster(mesh), bounds: bounds(mesh.vertices) };
}
