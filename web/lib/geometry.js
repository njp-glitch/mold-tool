// Port of backend/geometry.py: auto-orient and analyze propeller geometry.
//
// Coordinate convention after orientation:
//   X = blade length axis
//   Y = in-plane remaining axis
//   Z = flat-face normal / parting-plane normal
// Hub center is placed at the origin.

import { bounds, isWatertight } from "./meshio.js";

/** Eigen-decomposition of a symmetric 3x3 matrix (Jacobi). Returns {values, vectors} with vectors as columns. */
function eigSym3(A) {
  const a = A.map((r) => r.slice());
  const V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 50; sweep++) {
    const off = Math.abs(a[0][1]) + Math.abs(a[0][2]) + Math.abs(a[1][2]);
    if (off < 1e-15) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 3; k++) {
        const akp = a[k][p], akq = a[k][q];
        a[k][p] = c * akp - s * akq;
        a[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[p][k], aqk = a[q][k];
        a[p][k] = c * apk - s * aqk;
        a[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = V[k][p], vkq = V[k][q];
        V[k][p] = c * vkp - s * vkq;
        V[k][q] = s * vkp + c * vkq;
      }
    }
  }
  return { values: [a[0][0], a[1][1], a[2][2]], vectors: V };
}

const cross = (u, v) => [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
const norm = (u) => Math.hypot(u[0], u[1], u[2]);

/** Returns {centroid, axes} where axes[0] = longest extent direction, axes[2] = thinnest. */
function pcaAxes(vertices) {
  const n = vertices.length / 3;
  const c = [0, 0, 0];
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) c[k] += vertices[i * 3 + k];
  for (let k = 0; k < 3; k++) c[k] /= n;
  const cov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < n; i++) {
    const d = [vertices[i * 3] - c[0], vertices[i * 3 + 1] - c[1], vertices[i * 3 + 2] - c[2]];
    for (let r = 0; r < 3; r++) for (let s = 0; s < 3; s++) cov[r][s] += d[r] * d[s];
  }
  for (let r = 0; r < 3; r++) for (let s = 0; s < 3; s++) cov[r][s] /= Math.max(n - 1, 1);
  const { values, vectors } = eigSym3(cov);
  const order = [0, 1, 2].sort((p, q) => values[q] - values[p]);
  const axes = order.map((col) => {
    const v = [vectors[0][col], vectors[1][col], vectors[2][col]];
    // Eigenvector sign is arbitrary (numpy's comes from LAPACK internals). Pin it
    // so the result is deterministic: largest-magnitude component positive.
    let big = 0;
    for (let k = 1; k < 3; k++) if (Math.abs(v[k]) > Math.abs(v[big])) big = k;
    return v[big] < 0 ? v.map((x) => -x) : v;
  });
  return { centroid: c, axes };
}

/** Volume-weighted center of mass (trimesh's center_mass); falls back to vertex mean for open meshes. */
export function centerMass(mesh) {
  const { vertices: V, faces: F } = mesh;
  let vol = 0;
  const acc = [0, 0, 0];
  for (let t = 0; t < F.length; t += 3) {
    const a = F[t] * 3, b = F[t + 1] * 3, c = F[t + 2] * 3;
    const A = [V[a], V[a + 1], V[a + 2]], B = [V[b], V[b + 1], V[b + 2]], C = [V[c], V[c + 1], V[c + 2]];
    const cr = cross(B, C);
    const v6 = A[0] * cr[0] + A[1] * cr[1] + A[2] * cr[2];
    vol += v6;
    for (let k = 0; k < 3; k++) acc[k] += v6 * (A[k] + B[k] + C[k]);
  }
  if (Math.abs(vol) > 1e-12) return acc.map((x) => x / (4 * vol));
  const n = V.length / 3, m = [0, 0, 0];
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) m[k] += V[i * 3 + k] / n;
  return m;
}

/**
 * Returns a rotated/translated copy of mesh so that:
 *   X = major (length) axis, Y = remaining in-plane axis, Z = minor (flat-face normal) axis.
 * Hub center of mass is placed at the origin.
 */
export function autoOrient(mesh) {
  const { centroid, axes } = pcaAxes(mesh.vertices);
  const x = axes[0];
  let z = axes[2];
  let y = cross(z, x);
  const yl = norm(y);
  y = y.map((v) => v / yl);
  z = cross(x, y); // re-orthogonalize

  const src = mesh.vertices;
  const out = new Float64Array(src.length);
  for (let i = 0; i < src.length; i += 3) {
    const d = [src[i] - centroid[0], src[i + 1] - centroid[1], src[i + 2] - centroid[2]];
    out[i] = d[0] * x[0] + d[1] * x[1] + d[2] * x[2];
    out[i + 1] = d[0] * y[0] + d[1] * y[1] + d[2] * y[2];
    out[i + 2] = d[0] * z[0] + d[1] * z[1] + d[2] * z[2];
  }

  // Most of the blade should sit on the +X side; otherwise spin 180 degrees about Z.
  let xmin = Infinity, xmax = -Infinity;
  for (let i = 0; i < out.length; i += 3) {
    xmin = Math.min(xmin, out[i]);
    xmax = Math.max(xmax, out[i]);
  }
  if (xmax < -xmin) {
    for (let i = 0; i < out.length; i += 3) {
      out[i] = -out[i];
      out[i + 1] = -out[i + 1];
    }
  }

  const oriented = { vertices: out, faces: mesh.faces };
  const hub = centerMass(oriented);
  hub[0] = 0; // length axis already centered by symmetry assumption
  for (let i = 0; i < out.length; i += 3) {
    out[i + 1] -= hub[1];
    out[i + 2] -= hub[2];
  }
  return oriented;
}

function median(arr) {
  const s = Float64Array.from(arr).sort();
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

/** Z extent of vertices within `window` mm of point along X (cheap thickness proxy). */
function localThicknessAt(vertices, point, window = 6.0) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < vertices.length; i += 3) {
    if (Math.abs(vertices[i] - point[0]) <= window) {
      lo = Math.min(lo, vertices[i + 2]);
      hi = Math.max(hi, vertices[i + 2]);
    }
  }
  return lo === Infinity ? NaN : hi - lo;
}

/** Assumes `mesh` has already been auto-oriented. */
export function analyzePropeller(mesh, thinThresholdMm = 1.0) {
  const V = mesh.vertices;
  const warnings = [];
  const [bmin, bmax] = bounds(V);

  let iMin = 0, iMax = 0;
  for (let i = 0; i < V.length; i += 3) {
    if (V[i] < V[iMin * 3]) iMin = i / 3;
    if (V[i] > V[iMax * 3]) iMax = i / 3;
  }
  const pt = (i) => [V[i * 3], V[i * 3 + 1], V[i * 3 + 2]];
  const extremities = [
    { point: pt(iMax), side: "+X" },
    { point: pt(iMin), side: "-X" },
  ];

  const bladeLength = bmax[0] - bmin[0];

  const win = Math.max(0.1 * bladeLength, 5.0);
  const radii = [];
  for (let i = 0; i < V.length; i += 3) {
    if (Math.abs(V[i]) <= win) radii.push(Math.hypot(V[i + 1], V[i + 2]));
  }
  let hubRadius = NaN;
  if (radii.length) hubRadius = median(radii);
  else warnings.push("Could not estimate hub radius: no vertices found near origin.");

  const thinRegions = [];
  for (const ext of extremities) {
    const thickness = localThicknessAt(V, ext.point);
    if (!Number.isNaN(thickness)) {
      thinRegions.push({ location: ext.point, thickness_mm: thickness, side: ext.side });
      if (thickness < thinThresholdMm) {
        warnings.push(
          `Thin edge detected at ${ext.side} extremity: ~${thickness.toFixed(2)}mm (< ${thinThresholdMm.toFixed(1)}mm threshold).`
        );
      }
    }
  }

  if (!isWatertight(mesh.faces)) {
    warnings.push("Mesh is not watertight; cavity offset/boolean steps in later phases may need repair first.");
  }

  return {
    hub_center: [0, 0, 0],
    hub_radius_estimate: hubRadius,
    blade_length: bladeLength,
    extremities,
    thin_regions: thinRegions,
    bounding_box: { min: bmin, max: bmax },
    warnings,
  };
}
