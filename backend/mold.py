"""Mold cavity, base, gate, riser, and parting surface generation.

Coordinate convention (post auto_orient):
  X = blade length, Y = in-plane remaining axis, Z = flat-face normal / parting-plane normal.
Parting plane is Z = 0 (flat) or Z = parting_z(x,y) (Phase 5 silhouette-following).
Top mold half = Z >= parting surface; bottom = Z <= parting surface.
"""
from dataclasses import dataclass, field

import numpy as np
import trimesh
from skimage import measure


# ---------------------------------------------------------------------------
# Cavity solidification
# ---------------------------------------------------------------------------

def solidify_cavity(mesh: trimesh.Trimesh, pitch: float = 0.4) -> trimesh.Trimesh:
    """Returns a clean watertight solid matching `mesh` via voxelize + marching cubes.
    Self-heals minor non-watertightness from STEP tessellation seams."""
    pad = 2
    # max_iter raised: STLs with long skinny triangles need >10 subdivision rounds.
    voxelized = mesh.voxelized(pitch, method="subdivide", max_iter=25).fill()
    matrix = np.pad(voxelized.matrix, pad, mode="constant", constant_values=False)
    verts, faces, _, _ = measure.marching_cubes(matrix.astype(np.float32), level=0.5)
    origin_offset = np.array([pad, pad, pad])
    verts_world = trimesh.transform_points(verts - origin_offset, voxelized.transform)
    result = trimesh.Trimesh(vertices=verts_world, faces=faces)
    result.fix_normals()
    return result


def slice_half(mesh: trimesh.Trimesh, keep_positive_z: bool) -> trimesh.Trimesh:
    """Slices a closed mesh at Z=0, capping the cut face."""
    normal = [0, 0, 1] if keep_positive_z else [0, 0, -1]
    return mesh.slice_plane(plane_origin=[0, 0, 0], plane_normal=normal, cap=True)


# ---------------------------------------------------------------------------
# Phase 5: silhouette-following parting surface
# ---------------------------------------------------------------------------

# Sub-samples per grid cell edge when measuring the blade's surface extents.
# Higher = tighter containment bands near a knife edge, at more raycasting.
_PARTING_SUB = 5
# Smoothing width (in grid cells) for the parting surface between clamps.
_PARTING_SIGMA = 1.0
# Smooth/clamp relaxation passes.
_PARTING_PASSES = 12
# Relaxation of the region outside the part, which blends the plateaus into one
# another where they meet. More passes = a wider, gentler transition.
_PARTING_BLEND_PASSES = 30
_PARTING_BLEND_SIGMA = 1.0

def _surface_extents(mesh: trimesh.Trimesh, px: np.ndarray, py: np.ndarray, z_start: float) -> tuple:
    """Lowest and highest surface Z above/below each (px, py), NaN where the ray misses."""
    n = len(px)
    origins = np.column_stack([px, py, np.full(n, z_start)])
    dirs = np.tile([0.0, 0.0, -1.0], (n, 1))
    locs, ray_ids, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=True)

    z_lo = np.full(n, np.nan)
    z_hi = np.full(n, np.nan)
    if len(ray_ids):
        order = np.argsort(ray_ids, kind="stable")
        rid = ray_ids[order]
        zz = locs[order][:, 2]
        uniq, starts = np.unique(rid, return_index=True)
        ends = np.append(starts[1:], len(rid))
        for k, ri in enumerate(uniq):
            seg = zz[starts[k]:ends[k]]
            z_lo[ri] = seg.min()
            z_hi[ri] = seg.max()
    return z_lo, z_hi


def compute_parting_heightmap(
    mesh: trimesh.Trimesh,
    pitch: float = 2.0,
    x_range: tuple = None,
    y_range: tuple = None,
) -> tuple:
    """
    Computes parting_z(x, y) — the Z of the surface silhouette at each XY grid point.

    Fires downward rays (+Z → -Z) and finds the Z where face normals transition
    from upward to downward.  Returns (xs, ys, Z) where Z[i, j] = parting Z
    at (xs[i], ys[j]).  Grid points outside the mesh footprint default to Z=0.
    """
    from scipy.ndimage import (
        distance_transform_edt,
        gaussian_filter,
        maximum_filter,
        minimum_filter,
    )

    bounds = mesh.bounds
    xmin = x_range[0] if x_range else bounds[0, 0]
    xmax = x_range[1] if x_range else bounds[1, 0]
    ymin = y_range[0] if y_range else bounds[0, 1]
    ymax = y_range[1] if y_range else bounds[1, 1]
    z_top = bounds[1, 2] + 10.0

    xs = np.arange(xmin, xmax + pitch, pitch)
    ys = np.arange(ymin, ymax + pitch, pitch)
    nx, ny = len(xs), len(ys)
    n = nx * ny

    pts_x = np.repeat(xs, ny)
    pts_y = np.tile(ys, nx)
    origins = np.column_stack([pts_x, pts_y, np.full(n, z_top)])
    dirs = np.tile([0.0, 0.0, -1.0], (n, 1))

    locs, ray_ids, face_ids = mesh.ray.intersects_location(origins, dirs, multiple_hits=True)

    Z_flat = np.zeros(n)
    hit_flat = np.zeros(n, dtype=bool)
    lo_flat = np.zeros(n)
    hi_flat = np.zeros(n)
    face_nz = mesh.face_normals[:, 2]

    if len(ray_ids) > 0:
        order = np.argsort(ray_ids, kind="stable")
        ray_ids_s = ray_ids[order]
        locs_s = locs[order]
        nz_s = face_nz[face_ids[order]]

        unique_rays, starts = np.unique(ray_ids_s, return_index=True)
        ends = np.append(starts[1:], len(ray_ids_s))

        for k in range(len(unique_rays)):
            ri = unique_rays[k]
            sl = slice(starts[k], ends[k])
            z_hits = locs_s[sl, 2]
            nz_hits = nz_s[sl]

            desc = np.argsort(z_hits)[::-1]
            z_d = z_hits[desc]
            nz_d = nz_hits[desc]

            # Default: midpoint of all hits
            overall_mid = float((z_d[0] + z_d[-1]) / 2)
            parting_z = overall_mid

            # A twisted/folded section can make one ray cross the surface more
            # than once, so collect every +Z -> -Z transition descending in Z
            # and keep the one nearest the overall midpoint (the blade's true
            # mid-plane there) instead of just the first one encountered —
            # otherwise a spurious crossing near the top of a fold gets picked,
            # leaving a sliver of material outside either mold half.
            candidates = [
                float((z_d[m] + z_d[m + 1]) / 2)
                for m in range(len(z_d) - 1)
                if nz_d[m] >= -0.05 and nz_d[m + 1] <= 0.05
            ]
            if candidates:
                parting_z = min(candidates, key=lambda c: abs(c - overall_mid))

            Z_flat[ri] = parting_z
            hit_flat[ri] = True
            lo_flat[ri] = float(z_d[-1])
            hi_flat[ri] = float(z_d[0])

    Z = Z_flat.reshape(nx, ny)
    mask = hit_flat.reshape(nx, ny).astype(np.float64)

    # The cutter interpolates Z bilinearly between nodes, so pinning the nodes
    # inside the blade is not enough — a patch can still bow outside it between
    # them. A bilinear patch stays within any band that contains all four of its
    # corners, so build each node's band from the tightest bounds over every
    # cell it touches: sample the true surface on a sub-grid, then max-filter
    # the lower surface and min-filter the upper over a one-cell neighborhood.
    sub = _PARTING_SUB
    sx = np.linspace(xs[0], xs[-1], (nx - 1) * sub + 1)
    sy = np.linspace(ys[0], ys[-1], (ny - 1) * sub + 1)
    SX, SY = np.meshgrid(sx, sy, indexing="ij")
    s_lo, s_hi = _surface_extents(mesh, SX.ravel(), SY.ravel(), z_top)
    s_lo = s_lo.reshape(SX.shape)
    s_hi = s_hi.reshape(SX.shape)

    win = 2 * sub + 1
    lo_f = maximum_filter(np.nan_to_num(s_lo, nan=-np.inf), size=win, mode="nearest")
    hi_f = minimum_filter(np.nan_to_num(s_hi, nan=np.inf), size=win, mode="nearest")
    z_lo = lo_f[::sub, ::sub]
    z_hi = hi_f[::sub, ::sub]

    inside = np.isfinite(z_lo) & np.isfinite(z_hi)

    # Keep the parting surface inside the blade, in the middle 50% of its local
    # thickness, so each half forms one whole face and both get material even
    # where the blade is thin.
    band_lo = np.where(inside, z_lo, -np.inf)
    band_hi = np.where(inside, z_hi, np.inf)
    thickness = np.where(inside, z_hi - z_lo, 0.0)
    margin = 0.25 * np.maximum(thickness, 0.0)
    band_lo = band_lo + margin
    band_hi = band_hi - margin

    # Across a hub shoulder or a steep twist the neighborhood can demand a lower
    # bound above its own upper bound, so no value satisfies every cell touching
    # the node — the grid is simply too coarse for that feature. Pin to the
    # midpoint of the neighborhood bounds: it splits the difference across those
    # cells, which beats honouring any single one of them.
    eroded_mid = 0.5 * (np.where(inside, z_lo, 0.0) + np.where(inside, z_hi, 0.0))
    degenerate = inside & (band_lo > band_hi)
    band_lo = np.where(degenerate, eroded_mid, band_lo)
    band_hi = np.where(degenerate, eroded_mid, band_hi)

    def smooth(arr):
        # Mask-normalized blur: smooths within the mesh footprint without being
        # dragged toward the Z=0 default just outside it (e.g. at a tapered
        # blade tip), which otherwise clips the surface short of the real peak.
        num = gaussian_filter(arr * mask, sigma=_PARTING_SIGMA)
        den = gaussian_filter(mask, sigma=_PARTING_SIGMA)
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0.05)

    def clamp(arr):
        out = arr.copy()
        out[inside] = np.clip(arr[inside], band_lo[inside], band_hi[inside])
        return out

    # Alternate smoothing and clamping so the surface relaxes toward something
    # manufacturable while still tracking a sharply twisted blade. Smoothing
    # alone cuts through the blade on a steep twist (leaving a face outside its
    # own mold half); clamping alone is faithful but kinked. Clamp last so the
    # containment guarantee holds exactly.
    for _ in range(_PARTING_PASSES):
        Z = clamp(smooth(clamp(Z)))
    Z = clamp(Z)

    # Outside the footprint, hold the height of the nearest point on the part
    # instead of sloping back to Z=0. Returning to zero makes the surface rise
    # over the blade and fall away again — an overhang that 3D-prints badly.
    # Holding it flat turns each side into a printable plateau that meets the
    # blade edge. Inside values are untouched, so containment still holds.
    if inside.any():
        _, (ni, nj) = distance_transform_edt(~inside, return_indices=True)
        Z = np.where(inside, Z, Z[ni, nj])

        # Where two plateaus at different heights meet, that nearest-point rule
        # leaves a jagged Voronoi seam. Relax the outside region with the part
        # itself held fixed: diffusion does nothing to an already-flat plateau
        # but rounds the seam into a gradual ramp. Inside values never move, so
        # containment is unaffected.
        held = Z.copy()
        for _ in range(_PARTING_BLEND_PASSES):
            Z = gaussian_filter(Z, sigma=_PARTING_BLEND_SIGMA, mode="nearest")
            Z[inside] = held[inside]
    return xs, ys, Z, inside


def flatten_gate_corridor(
    xs: np.ndarray,
    ys: np.ndarray,
    Z: np.ndarray,
    inside: np.ndarray,
    gate_x: float,
    gate_half_width: float,
    entry_y: float,
    block_y_max: float,
) -> tuple:
    """Pins the parting height to a single value across the gate's corridor,
    then blends that into the surrounding surface.

    The gate is a straight cylinder, but the silhouette-following surface
    generally isn't flat or constant along the gate's run outward — so a fixed
    cylinder centered on it gets bisected unevenly, or misses one half of the
    cut entirely if the surface drifts by more than the gate's radius. Holding
    the corridor at the height where the gate leaves the cavity keeps the
    split even along the whole channel; a straight gate can't do better than
    a single flat value anyway. `inside` (from compute_parting_heightmap) is
    never overwritten, so this can't reopen the blade-containment guarantee —
    only the truly-outside part of the corridor gets flattened.

    Returns (Z, gate_z) where gate_z is the height to build the gate at.
    """
    from scipy.ndimage import gaussian_filter

    iy = int(np.clip(np.searchsorted(ys, entry_y), 0, len(ys) - 1))
    ix = int(np.clip(np.searchsorted(xs, gate_x), 0, len(xs) - 1))
    gate_z = float(Z[ix, iy])

    X, Y = np.meshgrid(xs, ys, indexing="ij")
    corridor = (
        (np.abs(X - gate_x) <= gate_half_width)
        & (Y >= entry_y)
        & (Y <= block_y_max)
        & ~inside
    )
    if not corridor.any():
        return Z, gate_z

    Z = Z.copy()
    Z[corridor] = gate_z
    held = inside | corridor
    held_Z = Z.copy()
    for _ in range(_PARTING_BLEND_PASSES):
        Z = gaussian_filter(Z, sigma=_PARTING_BLEND_SIGMA, mode="nearest")
        Z[held] = held_Z[held]
    return Z, gate_z


def _build_lower_cutter(
    xs: np.ndarray, ys: np.ndarray, Z: np.ndarray, z_floor: float
) -> trimesh.Trimesh:
    """
    Closed mesh volume representing everything BELOW the parting surface Z[i,j]
    at grid point (xs[i], ys[j]).  Used as a boolean cutter.

    Top face  = height-map surface (outward normal +Z)
    Bottom face = flat at z_floor  (outward normal -Z)
    Four side walls connecting them.
    """
    nx, ny = len(xs), len(ys)
    n = nx * ny

    pts_x = np.repeat(xs, ny)
    pts_y = np.tile(ys, nx)

    top_verts = np.column_stack([pts_x, pts_y, Z.ravel()])
    bot_verts = np.column_stack([pts_x, pts_y, np.full(n, z_floor)])
    all_verts = np.vstack([top_verts, bot_verts])

    def ti(i, j): return i * ny + j        # top vertex
    def bi(i, j): return n + i * ny + j    # bottom vertex

    faces = []

    # Top surface — upward normal (+Z)
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = ti(i, j), ti(i + 1, j), ti(i + 1, j + 1), ti(i, j + 1)
            faces += [[a, b, c], [a, c, d]]

    # Bottom surface — downward normal (-Z), reverse winding
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = bi(i, j), bi(i + 1, j), bi(i + 1, j + 1), bi(i, j + 1)
            faces += [[a, c, b], [a, d, c]]

    # -X wall (i=0), outward normal -X
    for j in range(ny - 1):
        faces += [
            [ti(0, j), bi(0, j + 1), bi(0, j)],
            [ti(0, j), ti(0, j + 1), bi(0, j + 1)],
        ]

    # +X wall (i=nx-1), outward normal +X
    for j in range(ny - 1):
        faces += [
            [ti(nx - 1, j), bi(nx - 1, j), bi(nx - 1, j + 1)],
            [ti(nx - 1, j), bi(nx - 1, j + 1), ti(nx - 1, j + 1)],
        ]

    # -Y wall (j=0), outward normal -Y
    for i in range(nx - 1):
        faces += [
            [ti(i, 0), bi(i, 0), bi(i + 1, 0)],
            [ti(i, 0), bi(i + 1, 0), ti(i + 1, 0)],
        ]

    # +Y wall (j=ny-1), outward normal +Y
    for i in range(nx - 1):
        faces += [
            [ti(i, ny - 1), bi(i + 1, ny - 1), bi(i, ny - 1)],
            [ti(i, ny - 1), ti(i + 1, ny - 1), bi(i + 1, ny - 1)],
        ]

    mesh = trimesh.Trimesh(vertices=all_verts, faces=np.array(faces), process=True)
    mesh.fix_normals()
    return mesh


def _build_parting_surface_display(xs: np.ndarray, ys: np.ndarray, Z: np.ndarray) -> trimesh.Trimesh:
    """Thin display mesh of just the parting surface (top face of the height map)."""
    nx, ny = len(xs), len(ys)
    pts_x = np.repeat(xs, ny)
    pts_y = np.tile(ys, nx)
    verts = np.column_stack([pts_x, pts_y, Z.ravel()])

    def idx(i, j): return i * ny + j

    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            faces += [[a, b, c], [a, c, d]]

    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces))
    mesh.fix_normals()
    return mesh


# ---------------------------------------------------------------------------
# Gate channel
# ---------------------------------------------------------------------------

def _gate_entry_y(cavity_solid: trimesh.Trimesh, x: float, block_y_max: float) -> tuple:
    """Y where the cavity's +Y surface sits at hub position x, plus any warning."""
    ray_origin = np.array([[x, block_y_max + 10.0, 0.0]])
    ray_direction = np.array([[0.0, -1.0, 0.0]])
    locations, _, _ = cavity_solid.ray.intersects_location(ray_origin, ray_direction)
    if len(locations) == 0:
        return 0.0, "Gate: no cavity found along +Y parting line; defaulting to hub centre."
    return float(locations[:, 1].max()), None


def _gate_channel(
    entry_y: float,
    diameter: float,
    block_y_max: float,
    x: float = 0.0,
    z: float = 0.0,
    overlap: float = 1.5,
):
    """Straight Y-axis gate from hub outward to +Y mold face, centered at height z
    (the local parting surface there) so each half gets an even semicircular groove
    along the whole channel rather than a lopsided one.
    """
    warnings = []

    # entry_y is the cavity's +Y surface and +Y points away from the part, so
    # the channel's inner end has to step back by the overlap to bury itself in
    # the cavity. Adding it instead leaves a plug of solid mold between runner
    # and cavity, and the part has no way to fill.
    channel_inner_y = entry_y - overlap
    channel_length = max(block_y_max - channel_inner_y, overlap)

    channel = trimesh.creation.cylinder(radius=diameter / 2.0, height=channel_length, sections=32)
    rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    channel.apply_transform(rot)
    channel.apply_translation([x, (block_y_max + channel_inner_y) / 2.0, z])
    return channel, channel_length, warnings


# ---------------------------------------------------------------------------
# Riser pockets
# ---------------------------------------------------------------------------

def _make_frustum(r_bottom: float, r_top: float, height: float, sections: int = 32) -> trimesh.Trimesh:
    """Truncated cone: r_bottom at z=0, r_top at z=height, axis +Z."""
    n = sections
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    bottom_ring = np.c_[r_bottom * cos_a, r_bottom * sin_a, np.zeros(n)]
    top_ring = np.c_[r_top * cos_a, r_top * sin_a, np.full(n, height)]
    bc = np.array([[0.0, 0.0, 0.0]])
    tc = np.array([[0.0, 0.0, height]])
    verts = np.vstack([bottom_ring, top_ring, bc, tc])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, n + i, n + j], [i, n + j, j],
                  [2 * n, j, i], [2 * n + 1, n + i, n + j]]
    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    mesh.fix_normals()
    return mesh


def _riser_pocket(
    extremity_xyz,
    mold_x_min: float,
    mold_x_max: float,
    diameter: float,
    taper_angle_deg: float,
    overlap: float = 1.5,
):
    """X-axis vent pocket from mold side wall to blade tip. Returns (mesh, is_top_half)."""
    x, y, z = extremity_xyz
    r = diameter / 2.0
    taper_r = np.tan(np.radians(taper_angle_deg)) if taper_angle_deg > 0 else 0.0

    if x >= 0:
        depth = (mold_x_max - x) + overlap
        if depth <= 0:
            return None, z >= 0
        pocket = _make_frustum(r, r + taper_r * depth, depth)
        rot = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        pocket.apply_transform(rot)
        pocket.apply_translation([x - overlap, y, z])
    else:
        depth = (x - mold_x_min) + overlap
        if depth <= 0:
            return None, z >= 0
        pocket = _make_frustum(r, r + taper_r * depth, depth)
        rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0])
        pocket.apply_transform(rot)
        pocket.apply_translation([x + overlap, y, z])

    return pocket, z >= 0


# ---------------------------------------------------------------------------
# Parameters and result
# ---------------------------------------------------------------------------

@dataclass
class MoldParams:
    wall_thickness: float = 6.0
    base_thickness: float = 15.0
    base_extension: float = 5.0
    gate_run_extra: float = 2.0
    gate_diameter: float = 3.0
    riser_diameter: float = 3.0
    riser_taper_angle: float = 5.0
    parting_pitch: float = 1.0


@dataclass
class MoldResult:
    mold_half_top: trimesh.Trimesh
    mold_half_bottom: trimesh.Trimesh
    cavity_top: trimesh.Trimesh
    cavity_bottom: trimesh.Trimesh
    gate: trimesh.Trimesh
    gate_length: float
    risers: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    parting_surface: trimesh.Trimesh = None  # Phase 5 display mesh


# ---------------------------------------------------------------------------
# Main mold generation
# ---------------------------------------------------------------------------

def generate_mold_halves(
    cavity_solid: trimesh.Trimesh,
    params: MoldParams,
    extremities: list = None,
    original_mesh: trimesh.Trimesh = None,
) -> MoldResult:
    """Builds mold halves + gate around a precomputed `cavity_solid`.

    When original_mesh is provided, the cavity and mold blocks are split
    along a curved surface that follows the propeller silhouette instead of
    the flat Z=0 plane. Falls back to the flat plane if that computation fails.
    """
    warnings_list = []

    # --- Block sizing (same regardless of parting mode) ---
    cavity_bounds = cavity_solid.bounds
    margin = params.wall_thickness + params.base_extension
    block_xmin = cavity_bounds[0, 0] - margin
    block_xmax = cavity_bounds[1, 0] + margin
    block_ymin = cavity_bounds[0, 1] - margin
    # +Y is where the gate exits, so it alone gets extra run length beyond the
    # margin the other three sides share — more distance for material to flow
    # before it reaches the cavity.
    block_ymax = cavity_bounds[1, 1] + margin + params.gate_run_extra

    cavity_top_z = cavity_bounds[1, 2]
    cavity_bot_z = cavity_bounds[0, 2]
    half_h_top = max(params.base_thickness, cavity_top_z + params.wall_thickness)
    half_h_bot = max(params.base_thickness, -cavity_bot_z + params.wall_thickness)

    if half_h_top > params.base_thickness:
        warnings_list.append(
            f"Top base increased to {half_h_top:.1f} mm to fully enclose cavity."
        )
    if half_h_bot > params.base_thickness:
        warnings_list.append(
            f"Bottom base increased to {half_h_bot:.1f} mm to fully enclose cavity."
        )

    def make_block(z_min, z_max):
        extents = [block_xmax - block_xmin, block_ymax - block_ymin, z_max - z_min]
        box = trimesh.creation.box(extents=extents)
        box.apply_translation([
            (block_xmin + block_xmax) / 2,
            (block_ymin + block_ymax) / 2,
            (z_min + z_max) / 2,
        ])
        return box

    # --- Gate entry point (built once the parting height there is known) ---
    gate_x = 0.0
    entry_y, entry_warn = _gate_entry_y(cavity_solid, x=gate_x, block_y_max=block_ymax)
    if entry_warn:
        warnings_list.append(entry_warn)
    gate_z = 0.0

    # --- Risers ---
    risers_top, risers_bottom, riser_display = [], [], []
    for ext in (extremities or []):
        pocket, is_top = _riser_pocket(
            ext, block_xmin, block_xmax,
            params.riser_diameter, params.riser_taper_angle,
        )
        if pocket is None:
            warnings_list.append(
                f"Riser at {[round(v, 1) for v in ext]} skipped: outside mold bounds."
            )
            continue
        riser_display.append(pocket)
        (risers_top if is_top else risers_bottom).append(pocket)

    # --- Parting surface split ---
    parting_surface_mesh = None
    use_curved = original_mesh is not None

    if use_curved:
        try:
            xs, ys, parting_Z, inside = compute_parting_heightmap(
                original_mesh,
                pitch=params.parting_pitch,
                x_range=(block_xmin, block_xmax),
                y_range=(block_ymin, block_ymax),
            )

            # The gate is a straight cylinder; a wavy parting surface bisects it
            # unevenly (or misses one half's cut entirely) if left as-is. Flatten
            # its corridor to the height where it leaves the cavity so the split
            # stays even along the whole channel, and build the gate at that
            # same height instead of the old fixed Z=0 assumption.
            parting_Z, gate_z = flatten_gate_corridor(
                xs, ys, parting_Z, inside,
                gate_x=gate_x,
                gate_half_width=params.gate_diameter / 2.0 + 1.0,
                entry_y=entry_y,
                block_y_max=block_ymax,
            )

            z_floor = -(half_h_bot + 10.0)
            lower_cutter = _build_lower_cutter(xs, ys, parting_Z, z_floor)

            full_block = make_block(-(half_h_bot), half_h_top)

            cavity_bottom = trimesh.boolean.intersection(
                [cavity_solid, lower_cutter], engine="manifold"
            )
            cavity_top = trimesh.boolean.difference(
                [cavity_solid, lower_cutter], engine="manifold"
            )
            block_bottom = trimesh.boolean.intersection(
                [full_block, lower_cutter], engine="manifold"
            )
            block_top = trimesh.boolean.difference(
                [full_block, lower_cutter], engine="manifold"
            )

            parting_surface_mesh = _build_parting_surface_display(xs, ys, parting_Z)
            warnings_list.append(
                f"Silhouette parting surface computed at {params.parting_pitch:.1f} mm pitch."
            )
        except Exception as exc:
            warnings_list.append(
                f"Silhouette parting failed ({exc}); falling back to flat Z=0 parting."
            )
            use_curved = False
            gate_z = 0.0

    if not use_curved:
        cavity_top = slice_half(cavity_solid, keep_positive_z=True)
        cavity_bottom = slice_half(cavity_solid, keep_positive_z=False)
        block_top = make_block(0, half_h_top)
        block_bottom = make_block(-half_h_bot, 0)

    # --- Gate (built at the parting height determined above) ---
    gate, gate_length, gate_warns = _gate_channel(
        entry_y, params.gate_diameter, block_y_max=block_ymax, x=gate_x, z=gate_z
    )
    warnings_list.extend(gate_warns)

    # --- Boolean subtraction ---
    void_top = trimesh.boolean.union([cavity_top, gate] + risers_top, engine="manifold")
    void_bottom = trimesh.boolean.union([cavity_bottom, gate] + risers_bottom, engine="manifold")

    mold_half_top = trimesh.boolean.difference([block_top, void_top], engine="manifold")
    mold_half_bottom = trimesh.boolean.difference([block_bottom, void_bottom], engine="manifold")

    if not mold_half_top.is_watertight:
        warnings_list.append("Top mold half is not watertight; may need repair.")
    if not mold_half_bottom.is_watertight:
        warnings_list.append("Bottom mold half is not watertight; may need repair.")

    return MoldResult(
        mold_half_top=mold_half_top,
        mold_half_bottom=mold_half_bottom,
        cavity_top=cavity_top,
        cavity_bottom=cavity_bottom,
        gate=gate,
        gate_length=gate_length,
        risers=riser_display,
        warnings=warnings_list,
        parting_surface=parting_surface_mesh,
    )
