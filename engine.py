#!/usr/bin/env python3
"""
Deal With It 😎 — Core engine with MediaPipe face mesh detection.

Pixel-accurate eye landmark detection for reliable, no-tweaking-needed
glasses placement on any face photo.
"""

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter

import mediapipe as mp
import numpy as np


# MediaPipe face mesh landmark indices for eyes
# https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
LEFT_EYE_CENTER = 468    # iris center (refined)
RIGHT_EYE_CENTER = 473   # iris center (refined)
# Fallback: use eye corner midpoints if iris refinement unavailable
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
# Face width landmarks (temple-to-temple approximation)
LEFT_TEMPLE = 234
RIGHT_TEMPLE = 454
# Nose tip for yaw estimation (fallback)
NOSE_TIP = 1
# Ear tragion landmarks — primary yaw signal
LEFT_EAR = 234     # left tragion (same as temple in MediaPipe mesh)
RIGHT_EAR = 454    # right tragion
# Cheek/jaw contour landmarks for ear-area visibility estimation
# These are further out toward the ears than the temples
LEFT_CHEEK_OUTER = 127    # left cheek near ear
RIGHT_CHEEK_OUTER = 356   # right cheek near ear
LEFT_JAW = 172             # left jaw near ear
RIGHT_JAW = 397            # right jaw near ear


@dataclass
class FaceMetrics:
    """Detected face geometry for glasses placement."""
    eye_midpoint: Tuple[int, int]
    eye_left: Tuple[int, int]
    eye_right: Tuple[int, int]
    eye_span: int
    face_width: int
    angle: float   # head tilt in degrees (roll)
    yaw: float     # head turn left/right: negative=looking left, positive=looking right


def detect_face(image_path: str) -> Optional[FaceMetrics]:
    """Detect face landmarks using MediaPipe FaceLandmarker (tasks API)."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    import urllib.request

    # Download model if not cached
    model_path = Path(__file__).parent / "face_landmarker.task"
    if not model_path.exists():
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        print("⬇️  Downloading face landmarker model...")
        urllib.request.urlretrieve(url, str(model_path))

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    mp_image = mp.Image.create_from_file(str(image_path))
    result = landmarker.detect(mp_image)
    landmarker.close()

    if not result.face_landmarks:
        return None

    lm = result.face_landmarks[0]
    w = mp_image.width
    h = mp_image.height

    def px(idx):
        return (int(lm[idx].x * w), int(lm[idx].y * h))

    # Eye centers — use iris landmarks if available (indices 468-477)
    try:
        left_eye = px(LEFT_EYE_CENTER)
        right_eye = px(RIGHT_EYE_CENTER)
    except (IndexError, AttributeError):
        li, lo = px(LEFT_EYE_INNER), px(LEFT_EYE_OUTER)
        ri, ro = px(RIGHT_EYE_INNER), px(RIGHT_EYE_OUTER)
        left_eye = ((li[0] + lo[0]) // 2, (li[1] + lo[1]) // 2)
        right_eye = ((ri[0] + ro[0]) // 2, (ri[1] + ro[1]) // 2)

    mid_x = (left_eye[0] + right_eye[0]) // 2
    mid_y = (left_eye[1] + right_eye[1]) // 2
    eye_span = int(math.hypot(
        right_eye[0] - left_eye[0],
        right_eye[1] - left_eye[1]
    ))
    angle = math.degrees(math.atan2(
        right_eye[1] - left_eye[1],
        right_eye[0] - left_eye[0]
    ))

    lt = px(LEFT_TEMPLE)
    rt = px(RIGHT_TEMPLE)
    face_width = int(math.hypot(rt[0] - lt[0], rt[1] - lt[1]))

    nose = px(NOSE_TIP)

    # --- Yaw estimation (head left/right turn) ---
    # Primary signal: ear asymmetry.  When the head turns, one ear moves
    # toward the face centre (occluded side) while the other moves outward
    # (visible side).  We measure the horizontal distance from each eye to
    # the corresponding ear-area landmark — the ratio tells us which side
    # is "compressed" (turned away) vs "expanded" (turned toward camera).
    left_cheek = px(LEFT_CHEEK_OUTER)
    right_cheek = px(RIGHT_CHEEK_OUTER)
    left_jaw = px(LEFT_JAW)
    right_jaw = px(RIGHT_JAW)

    # Average the cheek + jaw landmarks for a more stable "ear area" point
    left_ear_x = (left_cheek[0] + left_jaw[0]) / 2
    right_ear_x = (right_cheek[0] + right_jaw[0]) / 2

    # Distance from each eye to its ear side
    dist_left = abs(left_eye[0] - left_ear_x)    # left eye → left ear area
    dist_right = abs(right_eye[0] - right_ear_x)  # right eye → right ear area

    if dist_left + dist_right > 0:
        # Asymmetry ratio: 0 = symmetric, positive = right ear more exposed,
        # negative = left ear more exposed
        ear_asymmetry = (dist_right - dist_left) / (dist_right + dist_left)
        # Scale to approximate degrees (±1 asymmetry ≈ ±45° head turn)
        yaw_from_ears = ear_asymmetry * 45
    else:
        yaw_from_ears = 0.0

    # Fallback/secondary signal: nose offset from eye midpoint
    nose_offset = nose[0] - mid_x
    yaw_from_nose = (nose_offset / max(face_width, 1)) * 90

    # Blend: ears are more reliable when visible, nose is the fallback.
    # Weight ears 70%, nose 30% when both are available.
    yaw = yaw_from_ears * 0.7 + yaw_from_nose * 0.3

    return FaceMetrics(
        eye_midpoint=(mid_x, mid_y),
        eye_left=left_eye,
        eye_right=right_eye,
        eye_span=eye_span,
        face_width=face_width,
        angle=angle,
        yaw=yaw,
    )


def create_pixel_glasses(width: int) -> Image.Image:
    """Generate built-in pixel 'deal with it' sunglasses."""
    pattern = [
        "                             ",
        " ██████████ █ ██████████████ ",
        "█░█░██░█░██ █ █░█░██░█░█████",
        "██░██░██░███ █ ██░██░██░█████",
        "█░█░██░█░██ █ █░█░██░█░█████",
        " ██████████ █ ██████████████ ",
        "  ████████  █  ████████████  ",
        "   ██████   █   ██████████   ",
        "                             ",
    ]
    pat_w = len(pattern[0])
    pat_h = len(pattern)
    pixel_size = max(1, width // pat_w)
    img = Image.new("RGBA", (pat_w * pixel_size, pat_h * pixel_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for y, row in enumerate(pattern):
        for x, ch in enumerate(row):
            if ch in ('█', '░'):
                fill = (0, 0, 0, 255) if ch == '█' else (255, 255, 255, 255)
                draw.rectangle(
                    [x * pixel_size, y * pixel_size,
                     (x + 1) * pixel_size - 1, (y + 1) * pixel_size - 1],
                    fill=fill,
                )
    return img


def load_glasses(glasses_path: str, target_width: int) -> Image.Image:
    """Load, scale, and clean a glasses PNG — preserving interior lens detail."""
    glasses = Image.open(glasses_path).convert("RGBA")
    scale = target_width / glasses.width
    glasses = glasses.resize(
        (target_width, max(1, int(glasses.height * scale))),
        Image.LANCZOS,
    )

    # --- Flood-fill background removal from edges ---
    # Only removes white pixels connected to the image border,
    # preserving white reflections/details inside the lenses.
    w, h = glasses.size
    pixels = glasses.load()
    visited = set()
    queue = deque()

    # Seed from all border pixels
    for x in range(w):
        queue.append((x, 0))
        queue.append((x, h - 1))
    for y in range(h):
        queue.append((0, y))
        queue.append((w - 1, y))

    while queue:
        x, y = queue.popleft()
        if (x, y) in visited or x < 0 or y < 0 or x >= w or y >= h:
            continue
        visited.add((x, y))
        r, g, b, a = pixels[x, y]
        # If this pixel is white-ish or near-white, make it transparent
        if r > 180 and g > 180 and b > 180:
            pixels[x, y] = (r, g, b, 0)
            # Spread to neighbors
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited:
                    queue.append((nx, ny))

    # Clean up any thin semi-transparent fringe at the edges of the opaque region
    # by eroding the alpha channel slightly
    alpha = glasses.split()[3]
    # Threshold: anything below 128 alpha → fully transparent
    alpha = alpha.point(lambda p: 255 if p > 128 else 0)
    glasses.putalpha(alpha)

    bbox = glasses.getbbox()
    if bbox:
        glasses = glasses.crop(bbox)
    return glasses


def generate_gif(
    photo_path: str,
    glasses_path: Optional[str] = None,
    eyes_xy: Optional[Tuple[int, int]] = None,
    glasses_width: Optional[int] = None,
    face_scale: float = 1.3,
    num_frames: int = 20,
    pause_frames: int = 12,
    speed_ms: int = 50,
    output: Optional[str] = None,
) -> Tuple[bytes, dict]:
    """
    Generate a 'deal with it' GIF.

    Returns (gif_bytes, info_dict).
    If `output` is set, also writes to that path.
    """
    photo = Image.open(photo_path).convert("RGBA")
    pw, ph = photo.size
    info = {"image_size": (pw, ph)}

    angle = 0.0
    yaw = 0.0
    face = detect_face(photo_path)

    if face:
        info["face"] = {
            "eye_midpoint": face.eye_midpoint,
            "eye_left": face.eye_left,
            "eye_right": face.eye_right,
            "eye_span": face.eye_span,
            "face_width": face.face_width,
            "angle": round(face.angle, 1),
            "yaw": round(face.yaw, 1),
        }
        if eyes_xy is None:
            eyes_xy = face.eye_midpoint
            angle = face.angle
            yaw = face.yaw
        if glasses_width is None:
            # "Deal with it" glasses are intentionally oversized — they extend
            # well past the temples.  Use the wider of two estimates:
            #   1) temple-to-temple * face_scale  (default 1.0 passthrough)
            #   2) inter-pupil distance * 2.8     (anatomical: glasses ≈ 2.8× IPD)
            # Then apply a meme-sized floor of 1.45× face_width so they always
            # look boldly oversized without manual slider tweaking.
            from_face = face.face_width * face_scale
            from_ipd  = face.eye_span * 2.8
            base = max(from_face, from_ipd)
            meme_floor = face.face_width * 1.45
            glasses_width = int(max(base, meme_floor))
    else:
        info["face"] = None

    # Fallbacks
    if glasses_width is None:
        glasses_width = int(pw * 0.4)
    if eyes_xy is None:
        eyes_xy = (pw // 2, int(ph * 0.38))

    info["glasses_width"] = glasses_width
    info["landing"] = eyes_xy

    # Load glasses
    if glasses_path:
        glasses = load_glasses(glasses_path, glasses_width)
    else:
        glasses = create_pixel_glasses(glasses_width)

    # Mirror glasses based on head orientation.
    # The source glasses PNG has arms extending to the right.
    # Flip horizontally when the face is turned left (yaw < -5°)
    # so the arms track the "far" side of the face for correct perspective.
    if yaw < -5:
        glasses = glasses.transpose(Image.FLIP_LEFT_RIGHT)

    # Rotate to match head tilt
    if abs(angle) > 0.5:
        glasses = glasses.rotate(-angle, expand=True, resample=Image.BICUBIC)

    gw, gh = glasses.size
    land_x = eyes_xy[0] - gw // 2
    land_y = eyes_xy[1] - gh // 2
    start_y = -gh

    # Generate frames
    frames = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        if t < 0.85:
            t_ease = (t / 0.85) ** 2
        else:
            bounce_t = (t - 0.85) / 0.15
            t_ease = 1.0 + 0.03 * math.sin(bounce_t * math.pi)

        cur_y = int(start_y + (land_y - start_y) * t_ease)
        frame = photo.copy()
        frame.paste(glasses, (land_x, cur_y), glasses)
        rgb = Image.new("RGB", frame.size, (255, 255, 255))
        rgb.paste(frame, mask=frame.split()[3])
        frames.append(rgb)

    for _ in range(pause_frames):
        frames.append(frames[-1].copy())

    # Save to bytes
    from io import BytesIO
    buf = BytesIO()
    frames[0].save(
        buf, format="GIF", save_all=True, append_images=frames[1:],
        duration=speed_ms, loop=0, optimize=True,
    )
    gif_bytes = buf.getvalue()
    info["frames"] = len(frames)
    info["size_kb"] = len(gif_bytes) // 1024

    if output:
        Path(output).write_bytes(gif_bytes)

    return gif_bytes, info


# --- CLI entrypoint ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deal With It 😎 GIF Generator")
    parser.add_argument("photo", help="Path to face photo")
    parser.add_argument("--glasses", help="Path to glasses PNG")
    parser.add_argument("--eyes", help="Override eye midpoint X,Y")
    parser.add_argument("--width", type=int, help="Override glasses width")
    parser.add_argument("--face-scale", type=float, default=1.0,
                        help="Glasses width as fraction of face width (default: 1.0)")
    parser.add_argument("--output", default="deal_with_it.gif")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--pause", type=int, default=12)
    parser.add_argument("--speed", type=int, default=50)
    args = parser.parse_args()

    eyes = None
    if args.eyes:
        parts = args.eyes.split(",")
        eyes = (int(parts[0]), int(parts[1]))

    _, info = generate_gif(
        args.photo,
        glasses_path=args.glasses,
        eyes_xy=eyes,
        glasses_width=args.width,
        face_scale=args.face_scale,
        num_frames=args.frames,
        pause_frames=args.pause,
        speed_ms=args.speed,
        output=args.output,
    )

    if info["face"]:
        f = info["face"]
        print(f"👤 Face width: {f['face_width']}px")
        print(f"👀 Eyes: L{f['eye_left']} R{f['eye_right']}  tilt={f['angle']}°  yaw={f['yaw']}°")
    else:
        print("⚠️  No face detected — used fallback positioning")
    print(f"🕶️  Glasses: {info['glasses_width']}px wide → {info['landing']}")
    print(f"😎 Saved: {args.output}  ({info['frames']} frames, {info['size_kb']}KB)")
