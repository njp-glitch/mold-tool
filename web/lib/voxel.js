// Port of solidify_cavity(): voxelize the surface, fill the interior, and extract
// a clean watertight surface at level 0.5.
//
// Python used trimesh voxelize(method="subdivide") + fill() + skimage marching cubes.
// Here: dense surface sampling (same "< pitch/2 spacing" rule as trimesh's subdivide),
// a border flood fill (same result as scipy binary_fill_holes), and marching
// tetrahedra on a Freudenthal split of each cube. Marching tetrahedra has no
// ambiguous cases, so the output is always a closed 2-manifold, which the
// boolean engine requires.

const PAD = 2;

/** Returns { vertices: Float64Array, faces: Uint32Array } of the solidified cavity. */
export function solidifyCavity(mesh, pitch = 0.4, onProgress = () => {}) {
  const V = mesh.vertices, F = mesh.faces;

  // --- Grid extents (voxel index i sits at world i * pitch, like trimesh) ---
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < V.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k], V[i + k]);
      hi[k] = Math.max(hi[k], V[i + k]);
    }
  }
  const imin = lo.map((v) => Math.round(v / pitch) - 1);
  const imax = hi.map((v) => Math.round(v / pitch) + 1);
  const N = imin.map((m, k) => imax[k] - m + 1 + 2 * PAD);
  const [NX, NY, NZ] = N;
  const total = NX * NY * NZ;
  if (total > 900e6) {
    throw new Error(`Voxel grid would be ${(total / 1e6).toFixed(0)}M cells; raise the voxel pitch.`);
  }
  const grid = new Uint8Array(total); // 0 = empty, 1 = surface/solid, 2 = outside
  const at = (x, y, z) => (z * NY + y) * NX + x;

  // --- 1. Mark every voxel the surface passes through ---
  onProgress("Voxelizing surface…");
  const h = pitch / 2;
  const mark = (x, y, z) => {
    const gx = Math.round(x / pitch) - imin[0] + PAD;
    const gy = Math.round(y / pitch) - imin[1] + PAD;
    const gz = Math.round(z / pitch) - imin[2] + PAD;
    grid[at(gx, gy, gz)] = 1;
  };
  for (let t = 0; t < F.length; t += 3) {
    const a = F[t] * 3, b = F[t + 1] * 3, c = F[t + 2] * 3;
    const ax = V[a], ay = V[a + 1], az = V[a + 2];
    const ux = V[b] - ax, uy = V[b + 1] - ay, uz = V[b + 2] - az;
    const vx = V[c] - ax, vy = V[c + 1] - ay, vz = V[c + 2] - az;
    const nu = Math.max(1, Math.ceil(Math.hypot(ux, uy, uz) / h));
    const nv = Math.max(1, Math.ceil(Math.hypot(vx, vy, vz) / h));
    for (let i = 0; i <= nu; i++) {
      const s = i / nu;
      const jmax = Math.floor((1 - s) * nv + 1e-9);
      for (let j = 0; j <= jmax; j++) {
        const r = j / nv;
        mark(ax + s * ux + r * vx, ay + s * uy + r * vy, az + s * uz + r * vz);
      }
    }
    // Lattice rows stop short of the far edge; sample it directly.
    const wx = V[c] - V[b], wy = V[c + 1] - V[b + 1], wz = V[c + 2] - V[b + 2];
    const nw = Math.max(1, Math.ceil(Math.hypot(wx, wy, wz) / h));
    for (let i = 0; i <= nw; i++) {
      const s = i / nw;
      mark(V[b] + s * wx, V[b + 1] + s * wy, V[b + 2] + s * wz);
    }
  }

  // --- 2. Fill: flood the outside from the border (6-connected); the rest is solid ---
  onProgress("Filling cavity…");
  let cap = 1 << 20;
  let queue = new Int32Array(cap);
  let head = 0, tail = 0, size = 0;
  const push = (idx) => {
    if (size === cap) {
      const bigger = new Int32Array(cap * 2);
      for (let k = 0; k < size; k++) bigger[k] = queue[(head + k) % cap];
      queue = bigger; head = 0; tail = size; cap *= 2;
    }
    queue[tail] = idx; tail = (tail + 1) % cap; size++;
  };
  // Padding guarantees the corner voxel is outside and the outside region is connected.
  grid[0] = 2;
  push(0);
  const sx = 1, sy = NX, sz = NX * NY;
  while (size > 0) {
    const idx = queue[head]; head = (head + 1) % cap; size--;
    const x = idx % NX, y = Math.floor(idx / NX) % NY, z = Math.floor(idx / sz);
    if (x > 0 && grid[idx - sx] === 0) { grid[idx - sx] = 2; push(idx - sx); }
    if (x < NX - 1 && grid[idx + sx] === 0) { grid[idx + sx] = 2; push(idx + sx); }
    if (y > 0 && grid[idx - sy] === 0) { grid[idx - sy] = 2; push(idx - sy); }
    if (y < NY - 1 && grid[idx + sy] === 0) { grid[idx + sy] = 2; push(idx + sy); }
    if (z > 0 && grid[idx - sz] === 0) { grid[idx - sz] = 2; push(idx - sz); }
    if (z < NZ - 1 && grid[idx + sz] === 0) { grid[idx + sz] = 2; push(idx + sz); }
  }
  queue = null;

  // --- 3. Marching tetrahedra at level 0.5 ---
  onProgress("Building cavity surface…");
  return marchingTets(grid, NX, NY, NZ, (g, k) => (g - PAD + imin[k]) * pitch);
}

// Freudenthal split of a cube into 6 tetrahedra along the 0->7 diagonal.
// Corner c has bits (x=1, y=2, z=4). This split is translation-invariant, so
// neighbouring cubes agree on every shared face diagonal (no cracks).
const TETS = [
  [0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7],
];

function marchingTets(grid, NX, NY, NZ, toWorld) {
  const inside = (v) => v !== 2;
  const planeSize = NX * NY;
  // Edge vertex ids keyed by (lower grid point, direction bits 1..7). Lower
  // endpoint is in plane z (cur) or z+1 (nxt); only two planes are live at once.
  let cur = new Int32Array(planeSize * 8).fill(-1);
  let nxt = new Int32Array(planeSize * 8).fill(-1);

  let vCap = 1 << 18, vCount = 0;
  let verts = new Float64Array(vCap * 3);
  let tCap = 1 << 18, tCount = 0;
  let tris = new Uint32Array(tCap * 3);

  const pos = new Float64Array(3);
  function edgeVertex(x, y, z, cLo, cHi) {
    // cLo is a bit-subset of cHi (always true for edges of Freudenthal tets).
    const lx = x + (cLo & 1), ly = y + ((cLo >> 1) & 1), lz = z + ((cLo >> 2) & 1);
    const dir = cHi & ~cLo;
    const buf = lz === z ? cur : nxt;
    const key = (ly * NX + lx) * 8 + dir;
    let id = buf[key];
    if (id >= 0) return id;
    if (vCount === vCap) {
      vCap *= 2;
      const b = new Float64Array(vCap * 3); b.set(verts); verts = b;
    }
    id = vCount++;
    verts[id * 3] = toWorld(lx + 0.5 * (dir & 1), 0);
    verts[id * 3 + 1] = toWorld(ly + 0.5 * ((dir >> 1) & 1), 1);
    verts[id * 3 + 2] = toWorld(lz + 0.5 * ((dir >> 2) & 1), 2);
    buf[key] = id;
    return id;
  }

  // Corner grid coordinates (in half-units would be exact; integers are fine for orientation).
  const cx = (c) => c & 1, cy = (c) => (c >> 1) & 1, cz = (c) => (c >> 2) & 1;

  function emit(x, y, z, ids, inC, outC) {
    let [a, b, c] = ids;
    // Orient so the normal points from the inside corner toward the outside corner.
    const A = a * 3, B = b * 3, C = c * 3;
    const ux = verts[B] - verts[A], uy = verts[B + 1] - verts[A + 1], uz = verts[B + 2] - verts[A + 2];
    const vx = verts[C] - verts[A], vy = verts[C + 1] - verts[A + 1], vz = verts[C + 2] - verts[A + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const dx = cx(outC) - cx(inC), dy = cy(outC) - cy(inC), dz = cz(outC) - cz(inC);
    if (nx * dx + ny * dy + nz * dz < 0) { const t = b; b = c; c = t; }
    if (tCount === tCap) {
      tCap *= 2;
      const nb = new Uint32Array(tCap * 3); nb.set(tris); tris = nb;
    }
    tris[tCount * 3] = a; tris[tCount * 3 + 1] = b; tris[tCount * 3 + 2] = c;
    tCount++;
  }

  const E = (x, y, z, p, q) => (p < q ? edgeVertex(x, y, z, p, q) : edgeVertex(x, y, z, q, p));
  const val = new Array(8);

  for (let z = 0; z < NZ - 1; z++) {
    for (let y = 0; y < NY - 1; y++) {
      for (let x = 0; x < NX - 1; x++) {
        const base = (z * NY + y) * NX + x;
        let mask = 0;
        for (let c = 0; c < 8; c++) {
          const idx = base + (c & 1) + ((c >> 1) & 1) * NX + ((c >> 2) & 1) * planeSize;
          val[c] = inside(grid[idx]);
          if (val[c]) mask |= 1 << c;
        }
        if (mask === 0 || mask === 255) continue;
        for (const tet of TETS) {
          const ins = [], outs = [];
          for (const c of tet) (val[c] ? ins : outs).push(c);
          if (ins.length === 0 || ins.length === 4) continue;
          if (ins.length === 1 || ins.length === 3) {
            const lone = ins.length === 1 ? ins[0] : outs[0];
            const others = ins.length === 1 ? outs : ins;
            const ids = others.map((o) => E(x, y, z, lone, o));
            const inC = ins.length === 1 ? lone : others[0];
            const outC = ins.length === 1 ? others[0] : lone;
            emit(x, y, z, ids, inC, outC);
          } else {
            const [i0, i1] = ins, [o0, o1] = outs;
            const e00 = E(x, y, z, i0, o0), e01 = E(x, y, z, i0, o1);
            const e11 = E(x, y, z, i1, o1), e10 = E(x, y, z, i1, o0);
            // Quad e00-e01-e11-e10 is planar-ordered around the tet; split in two.
            emit(x, y, z, [e00, e01, e11], i0, o0);
            emit(x, y, z, [e00, e11, e10], i0, o0);
          }
        }
      }
    }
    // Advance one plane: z+1 becomes current.
    const t = cur; cur = nxt; nxt = t; nxt.fill(-1);
  }

  return { vertices: verts.slice(0, vCount * 3), faces: tris.slice(0, tCount * 3) };
}
