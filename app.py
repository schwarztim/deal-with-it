#!/usr/bin/env python3
"""Deal With It 😎 — Web App"""

import os
import uuid
import tempfile
from pathlib import Path

from flask import Flask, render_template, request, send_file, jsonify
from engine import generate_gif

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB upload limit

# Store the bundled glasses PNG path (if provided)
GLASSES_DIR = Path(__file__).parent / "static" / "glasses"
GLASSES_DIR.mkdir(parents=True, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    if "photo" not in request.files:
        return jsonify({"error": "No photo uploaded"}), 400

    photo = request.files["photo"]
    if photo.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Optional custom glasses
    glasses_file = request.files.get("glasses")
    face_scale = float(request.form.get("face_scale", 1.3))
    speed = int(request.form.get("speed", 50))

    with tempfile.TemporaryDirectory() as tmp:
        photo_path = os.path.join(tmp, "photo" + Path(photo.filename).suffix)
        photo.save(photo_path)

        glasses_path = None
        if glasses_file and glasses_file.filename:
            glasses_path = os.path.join(tmp, "glasses.png")
            glasses_file.save(glasses_path)

        # Use bundled default glasses if no custom ones uploaded
        default_glasses = GLASSES_DIR / "default.png"
        if glasses_path is None and default_glasses.exists():
            glasses_path = str(default_glasses)

        try:
            gif_bytes, info = generate_gif(
                photo_path,
                glasses_path=glasses_path,
                face_scale=face_scale,
                speed_ms=speed,
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Return the GIF
    from io import BytesIO
    buf = BytesIO(gif_bytes)
    buf.seek(0)
    return send_file(buf, mimetype="image/gif", download_name="deal_with_it.gif")


@app.route("/api/detect", methods=["POST"])
def detect():
    """Debug endpoint — returns detected face metrics without generating GIF."""
    if "photo" not in request.files:
        return jsonify({"error": "No photo uploaded"}), 400

    photo = request.files["photo"]
    with tempfile.TemporaryDirectory() as tmp:
        photo_path = os.path.join(tmp, "photo" + Path(photo.filename).suffix)
        photo.save(photo_path)

        from engine import detect_face
        face = detect_face(photo_path)

        if face is None:
            return jsonify({"detected": False})

        return jsonify({
            "detected": True,
            "eye_midpoint": face.eye_midpoint,
            "eye_left": face.eye_left,
            "eye_right": face.eye_right,
            "eye_span": face.eye_span,
            "face_width": face.face_width,
            "angle": round(face.angle, 2),
        })


if __name__ == "__main__":
    app.run(debug=True, port=5678)
