"""Generates a synthetic 2-blade propeller mesh for development/testing.

Built directly in the canonical frame (hub at origin, length along X,
flat blade face normal along Y, vertical axis Z), then rotated and
translated to an arbitrary pose. This lets Phase 1's auto-orientation
logic be verified by checking that it recovers the canonical frame.
"""
import numpy as np
import trimesh


def _tapered_blade(hub_radius, tip_x, sign):
    """A single blade as a stack of thin tapered boxes along X.

    Thickness (Y) tapers from ~2mm at the hub to ~0.4mm at the tip so
    Phase 1's thin-edge detection has something real to find.
    Width (Z) tapers from ~14mm at the hub to ~4mm at the tip.
    """
    segments = 24
    xs = np.linspace(hub_radius, tip_x, segments + 1)
    boxes = []
    for i in range(segments):
        x0, x1 = xs[i], xs[i + 1]
        t = (x0 - hub_radius) / (tip_x - hub_radius)
        thickness = (2.0 * (1 - t) + 0.4 * t)
        width = (14.0 * (1 - t) + 4.0 * t)
        seg_len = x1 - x0
        box = trimesh.creation.box(extents=[seg_len, thickness, width])
        box.apply_translation([sign * (x0 + seg_len / 2.0), 0, 0])
        boxes.append(box)
    return trimesh.util.concatenate(boxes)


def make_synthetic_propeller(blade_length=120.0, hub_radius=10.0, hub_height=18.0,
                              seed_pose=True):
    """Returns a trimesh.Trimesh in an arbitrary (non-canonical) pose."""
    hub = trimesh.creation.cylinder(radius=hub_radius, height=hub_height, sections=32)

    blade_pos = _tapered_blade(hub_radius, blade_length, sign=1)
    blade_neg = _tapered_blade(hub_radius, blade_length, sign=-1)

    mesh = trimesh.util.concatenate([hub, blade_pos, blade_neg])

    try:
        mesh = trimesh.boolean.union([hub, blade_pos, blade_neg], engine="manifold")
    except Exception:
        # Boolean engine unavailable; fall back to a simple concatenation.
        # Non-manifold but fine for visualizing/testing orientation logic.
        mesh.merge_vertices()

    if seed_pose:
        # Rotate to an arbitrary pose so auto-orientation has real work to do:
        # canonical X (length) -> some skew direction, canonical Y (flat face) -> another.
        rot = trimesh.transformations.euler_matrix(0.7, 1.1, 0.3, axes="sxyz")
        mesh.apply_transform(rot)
        mesh.apply_translation([37.0, -52.0, 81.0])

    return mesh


if __name__ == "__main__":
    m = make_synthetic_propeller()
    m.export("synthetic_propeller.stl")
    print(f"Exported synthetic_propeller.stl: {len(m.vertices)} verts, {len(m.faces)} faces, watertight={m.is_watertight}")
