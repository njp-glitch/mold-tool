# Mold Design Tool

Design a two-part injection mold for a propeller, right in your browser.

**Use it here:** https://njp-glitch.github.io/mold-tool/

1. Upload a propeller as STL or STEP.
2. Adjust wall thickness, base, gate, risers, and parting surface with the sliders.
3. Download the top and bottom mold halves as STL files, ready to 3D print.

Everything runs on your own computer, inside the browser tab. Nothing is uploaded, there's no account, and it's free. Very fine voxel pitches can take a minute or two on slower laptops.

## How it works

The site in `web/` is plain HTML and JavaScript. The heavy geometry runs in a Web Worker:

- **Mesh booleans:** [Manifold](https://github.com/elalish/manifold) compiled to WebAssembly
- **STEP import:** [occt-import-js](https://github.com/kovacsv/occt-import-js) (OpenCascade compiled to WebAssembly)
- **Voxel cavity, silhouette parting surface, gate and risers:** ported from the original Python version (`web/lib/`)

GitHub Pages republishes the site automatically whenever `web/` changes on `main`.

## Original Python version

`backend/` + `frontend/` hold the original FastAPI version that the browser version was ported from. To run it locally on Windows, double-click `launch.bat` (it needs the Python virtual environment in `backend/venv`).
