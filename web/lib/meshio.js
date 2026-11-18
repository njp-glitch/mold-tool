// Mesh file I/O: STL read/write, STEP read (via occt-import-js), vertex welding.
//
// Internal mesh format used everywhere in lib/: { vertices: Float64Array (n*3), faces: Uint32Array (m*3) }.

/** Parses a binary or ASCII STL into a flat triangle soup (9 floats per triangle). */
export function parseSTL(buffer) {
  const view = new DataView(buffer);
  if (buffer.byteLength >= 84) {
    const nTri = view.getUint32(80, true);
    if (84 + nTri * 50 === buffer.byteLength) {
      const out = new Float64Array(nTri * 9);
      let off = 84;
      for (let t = 0; t < nTri; t++) {
        off += 12; // skip stored normal
        for (let k = 0; k < 9; k++) {
          out[t * 9 + k] = view.getFloat32(off, true);
          off += 4;
        }
        off += 2; // attribute byte count
      }
      return out;
    }
  }
  const text = new TextDecoder().decode(buffer);
  const nums = [];
  const re = /vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)/g;
  let m;
  while ((m = re.exec(text))) nums.push(+m[1], +m[2], +m[3]);
  if (nums.length === 0 || nums.length % 9 !== 0) {
    throw new Error("Could not parse the file as an STL (no triangles found).");
  }
  return Float64Array.from(nums);
}

/** Welds identical vertices of a triangle soup into an indexed mesh (like trimesh's process=True). */
export function weld(soup) {
  const map = new Map();
  const verts = [];
  const faces = new Uint32Array(soup.length / 3);
  for (let v = 0; v < soup.length / 3; v++) {
    const x = soup[v * 3], y = soup[v * 3 + 1], z = soup[v * 3 + 2];
    // trimesh merges at ~1e-8; round to that so float noise doesn't split shared corners.
    const key = `${Math.round(x * 1e8)},${Math.round(y * 1e8)},${Math.round(z * 1e8)}`;
    let idx = map.get(key);
    if (idx === undefined) {
      idx = verts.length / 3;
      map.set(key, idx);
      verts.push(x, y, z);
    }
    faces[v] = idx;
  }
  return { vertices: Float64Array.from(verts), faces };
}

let _occtPromise = null;

/** Lazily loads occt-import-js (OpenCascade compiled to WebAssembly). Works in a worker. */
async function loadOcct(baseUrl) {
  if (!_occtPromise) {
    _occtPromise = (async () => {
      let factory;
      if (typeof process !== "undefined" && process.versions?.node) {
        const { createRequire } = await import("node:module");
        factory = createRequire(import.meta.url)("occt-import-js");
      } else {
        // The package ships as a classic script that defines a global; module
        // workers can't importScripts, so evaluate it and grab that global.
        const src = await (await fetch(baseUrl + "occt-import-js.js")).text();
        factory = new Function(src + "\nreturn occtimportjs;")();
      }
      return factory(baseUrl ? { locateFile: (f) => baseUrl + f } : {});
    })();
  }
  return _occtPromise;
}

/** Tessellates a STEP file into an indexed mesh in millimetres. */
export async function parseSTEP(buffer, occtBaseUrl) {
  const occt = await loadOcct(occtBaseUrl);
  // Same tessellation tolerances the Python version used through cascadio
  // (0.01 mm linear, 0.5 rad angular).
  const res = occt.ReadStepFile(new Uint8Array(buffer), {
    linearUnit: "millimeter",
    linearDeflectionType: "absolute_value",
    linearDeflection: 0.01,
    angularDeflection: 0.5,
  });
  if (!res.success || !res.meshes.length) throw new Error("Could not read the STEP file.");
  const soup = [];
  for (const m of res.meshes) {
    const pos = m.attributes.position.array;
    const idx = m.index.array;
    for (let i = 0; i < idx.length; i++) {
      soup.push(pos[idx[i] * 3], pos[idx[i] * 3 + 1], pos[idx[i] * 3 + 2]);
    }
  }
  return weld(Float64Array.from(soup));
}

/** Loads an STL or STEP file (by extension) into an indexed mesh. */
export async function loadMesh(name, buffer, occtBaseUrl) {
  const ext = name.split(".").pop().toLowerCase();
  if (ext === "step" || ext === "stp") return parseSTEP(buffer, occtBaseUrl);
  if (ext === "stl") return weld(parseSTL(buffer));
  throw new Error(`Unsupported file type ".${ext}". Use STL or STEP.`);
}

/** Writes a binary STL from an indexed mesh (any numeric array types). */
export function writeSTL(vertices, faces) {
  const nTri = faces.length / 3;
  const buf = new ArrayBuffer(84 + nTri * 50);
  const view = new DataView(buf);
  let off = 84;
  view.setUint32(80, nTri, true);
  for (let t = 0; t < nTri; t++) {
    const a = faces[t * 3] * 3, b = faces[t * 3 + 1] * 3, c = faces[t * 3 + 2] * 3;
    const ux = vertices[b] - vertices[a], uy = vertices[b + 1] - vertices[a + 1], uz = vertices[b + 2] - vertices[a + 2];
    const vx = vertices[c] - vertices[a], vy = vertices[c + 1] - vertices[a + 1], vz = vertices[c + 2] - vertices[a + 2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const len = Math.hypot(nx, ny, nz) || 1;
    view.setFloat32(off, nx / len, true); view.setFloat32(off + 4, ny / len, true); view.setFloat32(off + 8, nz / len, true);
    off += 12;
    for (const i of [a, b, c]) {
      view.setFloat32(off, vertices[i], true); view.setFloat32(off + 4, vertices[i + 1], true); view.setFloat32(off + 8, vertices[i + 2], true);
      off += 12;
    }
    off += 2;
  }
  return buf;
}

/** Axis-aligned bounds as [[minx,miny,minz],[maxx,maxy,maxz]]. */
export function bounds(vertices) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < vertices.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      const v = vertices[i + k];
      if (v < lo[k]) lo[k] = v;
      if (v > hi[k]) hi[k] = v;
    }
  }
  return [lo, hi];
}

/** Every edge shared by exactly two faces (trimesh's is_watertight). */
export function isWatertight(faces) {
  const counts = new Map();
  for (let t = 0; t < faces.length; t += 3) {
    for (let e = 0; e < 3; e++) {
      const a = faces[t + e], b = faces[t + ((e + 1) % 3)];
      const key = a < b ? a * 4294967296 + b : b * 4294967296 + a;
      counts.set(key, (counts.get(key) || 0) + 1);
    }
  }
  for (const c of counts.values()) if (c !== 2) return false;
  return true;
}
