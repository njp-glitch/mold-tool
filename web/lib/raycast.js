// Axis-aligned ray casting (stands in for trimesh's mesh.ray.intersects_location).
// Every ray the mold code fires runs along a coordinate axis, so a ray/triangle
// test reduces to a 2D point-in-triangle check in the other two coordinates.

/** Spatial index over a mesh projected onto the XY plane, for rays fired along -Z. */
export class ZRayCaster {
  constructor(mesh) {
    const V = mesh.vertices, F = mesh.faces;
    this.V = V;
    this.F = F;
    const nTri = F.length / 3;

    let xmin = Infinity, ymin = Infinity, xmax = -Infinity, ymax = -Infinity;
    for (let i = 0; i < V.length; i += 3) {
      xmin = Math.min(xmin, V[i]); xmax = Math.max(xmax, V[i]);
      ymin = Math.min(ymin, V[i + 1]); ymax = Math.max(ymax, V[i + 1]);
    }
    const span = Math.max(xmax - xmin, ymax - ymin, 1e-9);
    // Aim for roughly a few triangles per cell.
    const target = Math.max(16, Math.min(2048, Math.round(Math.sqrt(nTri) * 2)));
    const cell = span / target;
    const gx = Math.max(1, Math.ceil((xmax - xmin) / cell) + 1);
    const gy = Math.max(1, Math.ceil((ymax - ymin) / cell) + 1);
    Object.assign(this, { xmin, ymin, cell, gx, gy });

    // Face normal Z (unit) per face, for the parting-surface transition test.
    this.nz = new Float64Array(nTri);
    const counts = new Uint32Array(gx * gy + 1);
    const ranges = new Int32Array(nTri * 4);
    for (let t = 0; t < nTri; t++) {
      const a = F[t * 3] * 3, b = F[t * 3 + 1] * 3, c = F[t * 3 + 2] * 3;
      const ux = V[b] - V[a], uy = V[b + 1] - V[a + 1], uz = V[b + 2] - V[a + 2];
      const vx = V[c] - V[a], vy = V[c + 1] - V[a + 1], vz = V[c + 2] - V[a + 2];
      const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nzr = ux * vy - uy * vx;
      const len = Math.hypot(nx, ny, nzr);
      this.nz[t] = len > 0 ? nzr / len : 0;
      const i0 = Math.floor((Math.min(V[a], V[b], V[c]) - xmin) / cell);
      const i1 = Math.floor((Math.max(V[a], V[b], V[c]) - xmin) / cell);
      const j0 = Math.floor((Math.min(V[a + 1], V[b + 1], V[c + 1]) - ymin) / cell);
      const j1 = Math.floor((Math.max(V[a + 1], V[b + 1], V[c + 1]) - ymin) / cell);
      ranges.set([i0, i1, j0, j1], t * 4);
      for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) counts[i * gy + j + 1]++;
    }
    for (let k = 1; k < counts.length; k++) counts[k] += counts[k - 1];
    const fill = counts.slice(0, gx * gy);
    const items = new Uint32Array(counts[gx * gy]);
    for (let t = 0; t < nTri; t++) {
      const [i0, i1, j0, j1] = ranges.subarray(t * 4, t * 4 + 4);
      for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) items[fill[i * gy + j]++] = t;
    }
    this.start = counts;
    this.items = items;
    this.eps = span * 1e-12;
  }

  /**
   * Fires a ray straight down through (px, py). Calls onHit(z, faceIndex) for every
   * triangle crossed (all hits, like multiple_hits=True). Returns the hit count.
   */
  cast(px, py, onHit) {
    const i = Math.floor((px - this.xmin) / this.cell);
    const j = Math.floor((py - this.ymin) / this.cell);
    if (i < 0 || j < 0 || i >= this.gx || j >= this.gy) return 0;
    const { V, F, items, eps } = this;
    const c0 = this.start[i * this.gy + j], c1 = this.start[i * this.gy + j + 1];
    let hits = 0;
    for (let k = c0; k < c1; k++) {
      const t = items[k];
      const a = F[t * 3] * 3, b = F[t * 3 + 1] * 3, c = F[t * 3 + 2] * 3;
      const ax = V[a], ay = V[a + 1], bx = V[b], by = V[b + 1], cx = V[c], cy = V[c + 1];
      const area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
      if (Math.abs(area) < 1e-14) continue; // edge-on to the ray
      const w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / area;
      const w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / area;
      const w2 = 1 - w0 - w1;
      const tol = eps / Math.abs(area) + 1e-12;
      if (w0 < -tol || w1 < -tol || w2 < -tol) continue;
      onHit(w0 * V[a + 2] + w1 * V[b + 2] + w2 * V[c + 2], t);
      hits++;
    }
    return hits;
  }
}

/**
 * Brute-force single ray along -Y from (x, *, z): returns the Y of every surface crossing.
 * Used once per mold for the gate entry point, so no index needed.
 */
export function castAlongY(mesh, x, z) {
  const V = mesh.vertices, F = mesh.faces;
  const out = [];
  for (let t = 0; t < F.length; t += 3) {
    const a = F[t] * 3, b = F[t + 1] * 3, c = F[t + 2] * 3;
    // 2D coordinates are (x, z).
    const ax = V[a], az = V[a + 2], bx = V[b], bz = V[b + 2], cx = V[c], cz = V[c + 2];
    if (x < Math.min(ax, bx, cx) || x > Math.max(ax, bx, cx) || z < Math.min(az, bz, cz) || z > Math.max(az, bz, cz)) continue;
    const area = (bx - ax) * (cz - az) - (bz - az) * (cx - ax);
    if (Math.abs(area) < 1e-14) continue;
    const w0 = ((bx - x) * (cz - z) - (bz - z) * (cx - x)) / area;
    const w1 = ((cx - x) * (az - z) - (cz - z) * (ax - x)) / area;
    const w2 = 1 - w0 - w1;
    if (w0 < -1e-12 || w1 < -1e-12 || w2 < -1e-12) continue;
    out.push(w0 * V[a + 1] + w1 * V[b + 1] + w2 * V[c + 1]);
  }
  return out;
}
