import os
import tempfile
import threading
import time
from dataclasses import asdict

import numpy as np
from fastapi import FastAPI, Header, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from geometry import analyze_propeller, auto_orient, load_mesh
from mold import MoldParams, generate_mold_halves, solidify_cavity

app = FastAPI(title="Injection Mold Design Tool")

_DEFAULT_VOXEL_PITCH = 0.2

# Per-user state, keyed by the X-Session-Id header the frontend sends (one ID per browser tab).
# Without this, two students using the hosted tool at once would overwrite each other's mesh.
_SESSION_IDLE_SECONDS = 2 * 60 * 60  # forget a session after 2 h of no requests
_MAX_SESSIONS = 30                   # hard cap so abandoned tabs can't eat all the RAM
_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()


def _new_state():
    return {"mesh": None, "analysis": None, "cavity_solid": None, "mold_result": None,
            "voxel_pitch": _DEFAULT_VOXEL_PITCH, "last_seen": time.time()}


def _get_state(session_id: str | None) -> dict:
    sid = (session_id or "default")[:64]
    now = time.time()
    with _sessions_lock:
        for key in [k for k, s in _sessions.items() if now - s["last_seen"] > _SESSION_IDLE_SECONDS]:
            del _sessions[key]
        state = _sessions.get(sid)
        if state is None:
            if len(_sessions) >= _MAX_SESSIONS:
                oldest = min(_sessions, key=lambda k: _sessions[k]["last_seen"])
                del _sessions[oldest]
            state = _sessions[sid] = _new_state()
        state["last_seen"] = now
        return state


def _set_mesh(state, mesh, analysis, voxel_pitch: float = _DEFAULT_VOXEL_PITCH):
    """Sets mesh, analysis, and cavity_solid together so they can never drift out of sync."""
    cavity_solid = solidify_cavity(mesh, pitch=voxel_pitch)
    state["mesh"] = mesh
    state["analysis"] = analysis
    state["cavity_solid"] = cavity_solid
    state["voxel_pitch"] = voxel_pitch
    state["mold_result"] = None


def _mesh_to_json(mesh):
    return {
        "vertices": mesh.vertices.tolist(),
        "faces": mesh.faces.tolist(),
    }


def _cavity_bounds(state):
    solid = state["cavity_solid"]
    return solid.bounds.tolist() if solid is not None else None


@app.get("/api/analyze")
def get_analysis(x_session_id: str | None = Header(default=None)):
    state = _get_state(x_session_id)
    mesh = state["mesh"]
    analysis = state["analysis"]
    if mesh is None:
        return {"mesh": None, "analysis": None}
    return {
        "mesh": _mesh_to_json(mesh),
        "analysis": asdict(analysis),
        "cavity_bounds": _cavity_bounds(state),
    }


@app.post("/api/upload")
def upload_propeller(file: UploadFile, x_session_id: str | None = Header(default=None)):
    # Plain def (not async) so the heavy load/voxelize work runs in FastAPI's threadpool
    # instead of blocking the event loop, which would freeze every other student's requests.
    state = _get_state(x_session_id)
    suffix = "." + file.filename.split(".")[-1].lower()
    data = file.file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        raw = load_mesh(tmp_path)
    except Exception as exc:
        return {"error": str(exc)}
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    oriented = auto_orient(raw)
    analysis = analyze_propeller(oriented)
    try:
        _set_mesh(state, oriented, analysis)
    except Exception as exc:
        return {"error": f"Could not build cavity solid: {exc}"}

    return {
        "mesh": _mesh_to_json(oriented),
        "analysis": asdict(analysis),
        "cavity_bounds": _cavity_bounds(state),
    }


class MoldParamsIn(BaseModel):
    wall_thickness: float = 6.0
    base_thickness: float = 15.0
    base_extension: float = 5.0
    gate_run_extra: float = 2.0
    gate_diameter: float = 3.0
    riser_diameter: float = 3.0
    riser_taper_angle: float = 5.0
    parting_pitch: float = 1.0
    voxel_pitch: float = _DEFAULT_VOXEL_PITCH


@app.post("/api/mold")
def get_mold(params: MoldParamsIn, x_session_id: str | None = Header(default=None)):
    state = _get_state(x_session_id)
    mesh = state["mesh"]
    analysis = state["analysis"]
    if mesh is None:
        return {"error": "No propeller loaded yet."}

    # Recompute cavity solid only when voxel pitch changes (expensive op).
    if params.voxel_pitch != state["voxel_pitch"]:
        state["cavity_solid"] = solidify_cavity(mesh, pitch=params.voxel_pitch)
        state["voxel_pitch"] = params.voxel_pitch

    cavity_solid = state["cavity_solid"]

    mold_kwargs = params.model_dump()
    mold_kwargs.pop("voxel_pitch")
    mold_params = MoldParams(**mold_kwargs)
    extremities = [e["point"] for e in analysis.extremities] if analysis else []
    try:
        result = generate_mold_halves(
            cavity_solid, mold_params,
            extremities=extremities,
            original_mesh=mesh,
        )
    except Exception as exc:
        return {"error": f"Mold generation failed: {exc}"}

    state["mold_result"] = result

    return {
        "mold_half_top": _mesh_to_json(result.mold_half_top),
        "mold_half_bottom": _mesh_to_json(result.mold_half_bottom),
        "cavity_top": _mesh_to_json(result.cavity_top),
        "cavity_bottom": _mesh_to_json(result.cavity_bottom),
        "gate": _mesh_to_json(result.gate),
        "gate_length": result.gate_length,
        "cavity_bounds": _cavity_bounds(state),
        "risers": [_mesh_to_json(r) for r in result.risers],
        "warnings": result.warnings,
        "parting_surface": _mesh_to_json(result.parting_surface) if result.parting_surface else None,
    }


@app.get("/api/export/{half}")
def export_stl(half: str, x_session_id: str | None = Header(default=None)):
    result = _get_state(x_session_id)["mold_result"]
    if result is None:
        return Response(content=b'{"error":"No mold generated yet."}', media_type="application/json", status_code=400)

    if half == "top":
        mesh = result.mold_half_top
    elif half == "bottom":
        mesh = result.mold_half_bottom
    else:
        return Response(content=b'{"error":"Unknown half."}', media_type="application/json", status_code=400)

    stl_bytes = mesh.export(file_type="stl")
    filename = f"mold_half_{half}.stl"
    return Response(
        content=stl_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
