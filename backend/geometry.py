"""Phase 1: load, auto-orient, and analyze propeller geometry.

Coordinate convention after orientation:
  X = blade length axis
  Y = flat blade face normal
  Z = vertical / parting-plane normal
Hub center is placed at the origin.
"""
import os
import tempfile
from dataclasses import dataclass, field

import numpy as np
import trimesh


@dataclass
class PropellerAnalysis:
    hub_center: np.ndarray
    hub_radius_estimate: float
    blade_length: float
    extremities: list  # [{"point": [x,y,z], "side": "+X"/"-X"}]
    thin_regions: list  # [{"location": [x,y,z], "thickness_mm": float, "side": "+X"/"-X"}]
    bounding_box: dict
    warnings: list = field(default_factory=list)


def load_mesh(path: str) -> trimesh.Trimesh:
    """Loads an STL or STEP file into a single Trimesh.

    STEP files are tessellated to GLB via `cascadio` (OCCT-backed) first,
    since trimesh has no native STEP/BREP reader.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".step", ".stp"):
        import cascadio

        glb_path = tempfile.mktemp(suffix=".glb")
        try:
            cascadio.step_to_glb(path, glb_path)
            mesh = load_mesh(glb_path)
        finally:
            if os.path.exists(glb_path):
                os.remove(glb_path)
        # glTF (cascadio's intermediate format) mandates meters; this
        # pipeline works in mm, so rescale regardless of the STEP file's
        # original unit annotation.
        mesh.apply_scale(1000.0)
        return mesh

    loaded = trimesh.load(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(
            [g for g in loaded.geometry.values()]
        )
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Could not parse '{path}' into a single mesh (got {type(loaded)}).")
    return loaded


def _pca_axes(mesh: trimesh.Trimesh):
    """Returns (centroid, axes) where axes columns are eigenvectors of the
    vertex covariance, sorted by descending variance: axes[:,0] = longest
    extent direction, axes[:,2] = thinnest (flattest) direction.
    """
    verts = mesh.vertices
    centroid = verts.mean(axis=0)
    centered = verts - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order
    order = np.argsort(eigvals)[::-1]  # descending: most variance first
    axes = eigvecs[:, order]
    return centroid, axes


def auto_orient(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Rotates/translates a copy of mesh so that:
      new X = major (length) axis   — blade span, left-right in display
      new Y = remaining in-plane axis
      new Z = minor (flat-face normal) axis — parting-plane normal, up in display
    Putting the flat-face normal on Z means the mold splits along the flat side of
    the blade (the physically correct orientation for injection moulding).
    Hub center of mass is placed at the origin.
    """
    mesh = mesh.copy()
    centroid, axes = _pca_axes(mesh)

    x_axis = axes[:, 0]
    z_axis = axes[:, 2]  # thinnest direction = flat face normal → parting plane
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)  # re-orthogonalize

    rotation = np.eye(4)
    rotation[:3, 0] = x_axis
    rotation[:3, 1] = y_axis
    rotation[:3, 2] = z_axis
    # We want to map mesh axes -> world axes, so apply the inverse (transpose, since orthonormal)
    rot3 = rotation[:3, :3].T
    transform = np.eye(4)
    transform[:3, :3] = rot3
    transform[:3, 3] = -rot3 @ centroid

    mesh.apply_transform(transform)

    # Convention: most of the blade mass should sit at +X half and the
    # flat face should point toward +Y. Flip axes if the synthetic/real
    # part came in backwards.
    if mesh.vertices[:, 0].max() < -mesh.vertices[:, 0].min():
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1]))

    # Re-center hub at origin using center of mass as the hub estimate.
    hub_center = mesh.center_mass.copy()
    hub_center[0] = 0.0  # length axis already centered by symmetry assumption
    mesh.apply_translation(-hub_center)

    return mesh


def _local_thickness_at(mesh: trimesh.Trimesh, point: np.ndarray, window: float = 6.0) -> float:
    """Approximate local thickness (extent along Y) of vertices within
    `window` mm of `point` along X. Cheap stand-in for a full ray-cast
    thickness analysis; adequate for flagging thin tip regions.
    """
    verts = mesh.vertices
    mask = np.abs(verts[:, 0] - point[0]) <= window
    if not np.any(mask):
        return float("nan")
    local = verts[mask]
    return float(local[:, 2].max() - local[:, 2].min())


def analyze_propeller(mesh: trimesh.Trimesh, thin_threshold_mm: float = 1.0) -> PropellerAnalysis:
    """Assumes `mesh` has already been auto-oriented (see auto_orient)."""
    warnings = []
    verts = mesh.vertices
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)

    x_min_idx = np.argmin(verts[:, 0])
    x_max_idx = np.argmax(verts[:, 0])
    extremities = [
        {"point": verts[x_max_idx].tolist(), "side": "+X"},
        {"point": verts[x_min_idx].tolist(), "side": "-X"},
    ]

    blade_length = float(bbox_max[0] - bbox_min[0])

    # Hub radius estimate: distance from the X axis (centerline) for
    # vertices near the hub, i.e. within a small X window of origin.
    near_hub = verts[np.abs(verts[:, 0]) <= max(0.1 * blade_length, 5.0)]
    if len(near_hub) > 0:
        hub_radius_estimate = float(np.median(np.hypot(near_hub[:, 1], near_hub[:, 2])))
    else:
        hub_radius_estimate = float("nan")
        warnings.append("Could not estimate hub radius: no vertices found near origin.")

    thin_regions = []
    for ext in extremities:
        pt = np.array(ext["point"])
        thickness = _local_thickness_at(mesh, pt)
        if not np.isnan(thickness):
            entry = {"location": ext["point"], "thickness_mm": thickness, "side": ext["side"]}
            thin_regions.append(entry)
            if thickness < thin_threshold_mm:
                warnings.append(
                    f"Thin edge detected at {ext['side']} extremity: "
                    f"~{thickness:.2f}mm (< {thin_threshold_mm}mm threshold)."
                )

    if not mesh.is_watertight:
        warnings.append("Mesh is not watertight; cavity offset/boolean steps in later phases may need repair first.")

    return PropellerAnalysis(
        hub_center=[0.0, 0.0, 0.0],
        hub_radius_estimate=hub_radius_estimate,
        blade_length=blade_length,
        extremities=extremities,
        thin_regions=thin_regions,
        bounding_box={"min": bbox_min.tolist(), "max": bbox_max.tolist()},
        warnings=warnings,
    )
