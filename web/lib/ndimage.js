// Ports of the scipy.ndimage functions the parting surface uses, for 2D grids.
// Arrays are Float64Array in row-major [nx][ny] order: a[i * ny + j], matching numpy's Z[i, j].

function mapIndex(k, n, mode) {
  if (mode === "nearest") return k < 0 ? 0 : k >= n ? n - 1 : k;
  // scipy "reflect": (d c b a | a b c d | d c b a)
  if (n === 1) return 0;
  const period = 2 * n;
  k = ((k % period) + period) % period;
  return k < n ? k : period - 1 - k;
}

function gaussianKernel(sigma, truncate = 4.0) {
  const r = Math.floor(truncate * sigma + 0.5);
  const w = [];
  let sum = 0;
  for (let x = -r; x <= r; x++) {
    const v = Math.exp(-0.5 * (x * x) / (sigma * sigma));
    w.push(v);
    sum += v;
  }
  return { r, w: w.map((v) => v / sum) };
}

/** scipy.ndimage.gaussian_filter for a 2D array (separable, default truncate=4). */
export function gaussianFilter(a, nx, ny, sigma, mode = "reflect") {
  const { r, w } = gaussianKernel(sigma);
  const tmp = new Float64Array(a.length);
  const out = new Float64Array(a.length);
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      let s = 0;
      for (let k = -r; k <= r; k++) s += w[k + r] * a[mapIndex(i + k, nx, mode) * ny + j];
      tmp[i * ny + j] = s;
    }
  }
  for (let i = 0; i < nx; i++) {
    const row = i * ny;
    for (let j = 0; j < ny; j++) {
      let s = 0;
      for (let k = -r; k <= r; k++) s += w[k + r] * tmp[row + mapIndex(j + k, ny, mode)];
      out[row + j] = s;
    }
  }
  return out;
}

function rankFilter(a, nx, ny, size, pick) {
  const h = Math.floor(size / 2);
  const tmp = new Float64Array(a.length);
  const out = new Float64Array(a.length);
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      let m = a[mapIndex(i - h, nx, "nearest") * ny + j];
      for (let k = -h + 1; k <= size - 1 - h; k++) m = pick(m, a[mapIndex(i + k, nx, "nearest") * ny + j]);
      tmp[i * ny + j] = m;
    }
  }
  for (let i = 0; i < nx; i++) {
    const row = i * ny;
    for (let j = 0; j < ny; j++) {
      let m = tmp[row + mapIndex(j - h, ny, "nearest")];
      for (let k = -h + 1; k <= size - 1 - h; k++) m = pick(m, tmp[row + mapIndex(j + k, ny, "nearest")]);
      out[row + j] = m;
    }
  }
  return out;
}

/** scipy.ndimage.maximum_filter(size=size, mode="nearest") */
export const maximumFilter = (a, nx, ny, size) => rankFilter(a, nx, ny, size, Math.max);
/** scipy.ndimage.minimum_filter(size=size, mode="nearest") */
export const minimumFilter = (a, nx, ny, size) => rankFilter(a, nx, ny, size, Math.min);

/**
 * For each cell, the flat index of the nearest cell where `feature` is true
 * (Euclidean). Equivalent to distance_transform_edt(~feature, return_indices=True).
 * Exact two-pass lower-envelope algorithm (Felzenszwalb & Huttenlocher).
 */
export function nearestFeatureIndex(feature, nx, ny) {
  const INF = 1e20;
  // Pass 1: along j within each row i.
  const f1 = new Float64Array(nx * ny);
  const nearJ = new Int32Array(nx * ny);
  for (let i = 0; i < nx; i++) {
    const row = i * ny;
    let last = -1;
    for (let j = 0; j < ny; j++) {
      if (feature[row + j]) last = j;
      nearJ[row + j] = last;
    }
    let next = -1;
    for (let j = ny - 1; j >= 0; j--) {
      if (feature[row + j]) next = j;
      const l = nearJ[row + j];
      let best = -1;
      if (l >= 0) best = l;
      if (next >= 0 && (best < 0 || next - j < j - best)) best = next;
      nearJ[row + j] = best;
      f1[row + j] = best < 0 ? INF : (j - best) * (j - best);
    }
  }
  // Pass 2: along i within each column j, lower envelope of parabolas.
  const out = new Int32Array(nx * ny).fill(-1);
  const v = new Int32Array(nx);
  const z = new Float64Array(nx + 1);
  for (let j = 0; j < ny; j++) {
    const f = (q) => f1[q * ny + j];
    let k = -1;
    for (let q = 0; q < nx; q++) {
      if (f(q) >= INF) continue;
      if (k < 0) {
        k = 0; v[0] = q; z[0] = -Infinity; z[1] = Infinity;
        continue;
      }
      let s;
      while (true) {
        const p = v[k];
        s = ((f(q) + q * q) - (f(p) + p * p)) / (2 * q - 2 * p);
        if (s <= z[k] && k > 0) k--;
        else break;
      }
      if (s <= z[k] && k === 0) {
        v[0] = q; z[0] = -Infinity; z[1] = Infinity;
        continue;
      }
      k++;
      v[k] = q; z[k] = s; z[k + 1] = Infinity;
    }
    if (k < 0) continue;
    let m = 0;
    for (let q = 0; q < nx; q++) {
      while (z[m + 1] < q) m++;
      const p = v[m];
      out[q * ny + j] = p * ny + nearJ[p * ny + j];
    }
  }
  return out;
}
