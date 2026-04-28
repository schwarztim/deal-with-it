# Deal With It 😎

A fully client-side "Deal With It" meme GIF generator. Upload any photo and get pixel sunglasses dropping onto the face — no server required.

![demo](https://img.shields.io/badge/try_it-live-e94560?style=for-the-badge)

## Features

- 🧠 **Auto face detection** — MediaPipe WASM finds eyes, measures face width, detects head tilt & yaw
- 🕶️ **Proper scaling** — glasses size to the face automatically (no manual tweaking needed)
- 🔄 **Head orientation** — glasses rotate with head tilt and mirror for turned faces
- 🎬 **Smooth animation** — ease-in drop with subtle bounce on landing
- 📦 **Zero backend** — everything runs in-browser (MediaPipe WASM + Canvas + gif.js)
- 🎨 **Custom glasses** — upload your own PNG or use the bundled pixel sunglasses

## Usage

Just open `index.html` in a browser. That's it.

Or host it anywhere static files are served:

```bash
# Local
python3 -m http.server 8000
# Then open http://localhost:8000/index.html
```

## How It Works

1. **MediaPipe FaceLandmarker** (468+ landmarks, iris tracking) detects face geometry
2. Glasses scale to `max(face_width × 1.45, eye_span × 2.8)` for meme-sized coverage
3. Head tilt (roll) rotates the glasses; ear asymmetry estimates yaw for mirroring
4. Flood-fill from image edges removes white background while preserving lens reflections
5. **gif.js** web workers encode the animated GIF entirely client-side

## Tech Stack

- [MediaPipe Tasks Vision](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) (WASM, from CDN)
- [gif.js](https://jnordberg.github.io/gif.js/) (web worker GIF encoder)
- Vanilla JS + Canvas API
- Single HTML file, no build step

## License

MIT
