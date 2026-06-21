"""
Marketing content / filmmaker agent — Dr. Theiss Naturwaren GmbH

Produces short-form vertical reel videos.

Pipeline:
1. gemini-2.5-flash       -> script + HWG-safe reel plan
2. veo-3.1-lite-generate-preview -> real AI video clips per scene
3. gemini-2.5-flash-image -> fallback scene photo if Veo fails
4. moviepy + Pillow       -> overlays, safe zones, MP4 assembly
5. gemini-2.5-flash-lite  -> caption package
"""

import os
import json
import io
import base64
import textwrap
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ── Setup ───────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_KEY = os.getenv("gemini_api_key") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing Gemini API key. Add gemini_api_key=... or GEMINI_API_KEY=... to your .env")

client = genai.Client(api_key=API_KEY)

MODEL_FLASH = "gemini-2.5-flash"
MODEL_LITE = "gemini-2.5-flash-lite"
MODEL_IMAGE = "gemini-2.5-flash-image"

# Real video generation model
MODEL_VIDEO = "veo-3.1-lite-generate-preview"
# MODEL_VIDEO = "veo-3.1-fast-generate-preview"
# MODEL_VIDEO = "veo-3.1-generate-preview"


# ── Safe zones for 1080×1920 TikTok/Instagram ───────────────────────────────
SAFE_ZONES = {
    "width": 1080,
    "height": 1920,
    "top": 140,
    "bottom": 1320,
    "left": 40,
    "right": 900,
}


PRODUCTS = [
    {"sku": "ALK-FB-01", "name": "Fuß Butter", "line": "Feet", "price": 7.71, "season": "Autumn–Winter", "segment": "45+ dry-skin, women", "hero": True},
    {"sku": "ALK-FB-02", "name": "Sole Fußbad", "line": "Feet", "price": 6.49, "season": "Winter", "segment": "Wellness, 50+", "hero": True},
    {"sku": "ALK-FB-03", "name": "Hornhaut Reduziercreme", "line": "Feet", "price": 6.99, "season": "Spring", "segment": "Women 30–60", "hero": False},
    {"sku": "ALK-FB-04", "name": "Hornhaut Entferner Maske", "line": "Feet", "price": 8.49, "season": "Spring–Summer", "segment": "Women 25–45", "hero": True},
    {"sku": "ALK-FB-05", "name": "10% Urea Fußcreme", "line": "Feet", "price": 7.25, "season": "All year", "segment": "Diabetic / very dry skin", "hero": False},
    {"sku": "ALK-FB-06", "name": "Fußpflege Deospray", "line": "Feet", "price": 6.10, "season": "Summer", "segment": "Active / men 20–45", "hero": False},
    {"sku": "ALK-LG-01", "name": "5 in 1 Beinlotion", "line": "Legs", "price": 9.95, "season": "Summer", "segment": "Women 35–65", "hero": True},
    {"sku": "ALK-LG-02", "name": "Bein Frische Gel", "line": "Legs", "price": 8.20, "season": "Summer", "segment": "Travel / standing jobs", "hero": False},
    {"sku": "ALK-LG-03", "name": "Besenreiser Pflegebalsam", "line": "Legs", "price": 11.49, "season": "Spring–Summer", "segment": "Women 40–65", "hero": False},
    {"sku": "ALK-MG-01", "name": "Mobil Gel", "line": "Muscles/Joints", "price": 5.83, "season": "Autumn–Winter", "segment": "Active 30+, 55+ joints", "hero": True},
    {"sku": "ALK-MG-02", "name": "Mobil Einreibung Extra Stark", "line": "Muscles/Joints", "price": 8.90, "season": "Winter / sport", "segment": "Sport, 25–55", "hero": False},
    {"sku": "ALK-MG-03", "name": "Mobil Eisspray akut", "line": "Muscles/Joints", "price": 9.40, "season": "Sport season", "segment": "Athletes, teams", "hero": True},
    {"sku": "ALK-MG-04", "name": "Franzbranntwein", "line": "Muscles/Joints", "price": 6.75, "season": "All year", "segment": "Traditional 55+", "hero": False},
    {"sku": "ALK-MG-05", "name": "Wärmendes Intensiv Gel", "line": "Muscles/Joints", "price": 8.30, "season": "Winter", "segment": "45+ tension/back", "hero": False},
    {"sku": "ALK-CB-01", "name": "Ur Bonbons", "line": "Cough drops", "price": 2.49, "season": "Cold season", "segment": "Mass-market", "hero": False},
]


CONTENT_ANGLES = {
    "ritual_asmr": "Ritual / ASMR foot bath — slow, sensory, satisfying",
    "post_workout": "15-sec post-workout recovery — fast, energetic",
    "heavy_legs": "'Heavy legs after a long shift' — relatable, empathetic hook",
    "origin_story": "Alpine ingredient origin — Allgäu plantation → bottle",
    "before_after": "Before / after transformation reveal",
    "sport_recovery": "Acute sport recovery — athletes & teams",
}


C_GREEN = (48, 140, 36)
C_YELLOW = (245, 197, 24)
C_WHITE = (250, 254, 250)
C_MUTED = (200, 220, 195)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()

def _resize_video_cover(clip, W=1080, H=1920):
    """Resize/crop any Veo output to exact 1080x1920."""
    cw, ch = clip.size
    scale = max(W / cw, H / ch)
    new_w, new_h = int(cw * scale), int(ch * scale)

    if hasattr(clip, "resized"):
        clip = clip.resized((new_w, new_h))
    else:
        clip = clip.resize((new_w, new_h))

    x1 = max(0, (new_w - W) // 2)
    y1 = max(0, (new_h - H) // 2)

    if hasattr(clip, "cropped"):
        clip = clip.cropped(x1=x1, y1=y1, width=W, height=H)
    else:
        clip = clip.crop(x1=x1, y1=y1, width=W, height=H)

    return clip


def _cover_crop(img: Image.Image, W: int, H: int) -> Image.Image:
    iw, ih = img.size
    scale = max(W / iw, H / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    left = (nw - W) // 2
    top = (nh - H) // 2
    return img.crop((left, top, left + W, top + H))


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
        if text.startswith("json"):
            text = text[4:].strip()
    return text


def _nearest_veo_duration(seconds: int) -> int:
    """Veo 3.1 supports 4, 6, or 8 seconds."""
    allowed = [4, 6, 8]
    return min(allowed, key=lambda x: abs(x - int(seconds)))


def _set_duration(clip, duration):
    """MoviePy v1/v2 compatibility."""
    if hasattr(clip, "with_duration"):
        return clip.with_duration(duration)
    return clip.set_duration(duration)


def _set_position(clip, pos):
    if hasattr(clip, "with_position"):
        return clip.with_position(pos)
    return clip.set_position(pos)


def _set_opacity(clip, opacity):
    if hasattr(clip, "with_opacity"):
        return clip.with_opacity(opacity)
    return clip.set_opacity(opacity)


# ── Step 1: Generate script ─────────────────────────────────────────────────
def generate_script(product: dict, angle: str, platform: str, language: str) -> dict:
    sz = SAFE_ZONES
    angle_desc = CONTENT_ANGLES.get(angle, angle)
    lang_note = "German, Du-form, warm" if language == "de" else "English"

    prompt = f"""
You are a creative director at Allgäuer Latschenkiefer / Dr. Theiss Naturwaren GmbH.

Create a studio-quality short-form vertical reel script.

Product: {product['name']} ({product['sku']}) — {product['line']}
Target segment: {product.get('segment', 'general audience')}
Peak season: {product.get('season', 'all year')}
Angle: {angle_desc}
Platform: {platform}
Language: {lang_note}
Frame: 1080×1920 px, 9:16 vertical
Total duration: around 16–24 seconds

Important:
- Create the number of scenes needed for the story.
Usually:
- ASMR ritual: 4–6 scenes
- Before/after: 4–5 scenes
- Post-workout recovery: 3–4 scenes
- Ingredient origin story: 5–7 scenes
- Each scene duration must be one of: 4, 6, or 8 seconds.
- Do not put text inside image/video prompts. Text will be composited later.
- Use cinematic camera language suitable for Veo.
Total duration should remain between 15 and 25 seconds.

SAFE ZONE RULES:
All text/logos must stay inside:
top > {sz['top']}px
bottom < {sz['bottom']}px
left > {sz['left']}px
right < {sz['right']}px

All y_position values must be integers between 360 and 980.
Do not place main text near the top, bottom, or right-side UI area.
Keep the main message in the center-left safe band:
x range: 60–840
y range: 360–980

CTA must be around y=1120 and never below 1220.

BRAND / HWG GUARDRAILS:
- Cosmetics, not drugs.
- Never say: treats, heals, cures, medical.
- Use: soothes, refreshes, cares for, revitalises, relieves tiredness.
- Sensory and experiential claims only.
- Brand feeling: Alpine heritage, natural, trusted since 1973, Made in Germany.

Return ONLY valid JSON:
{{
  "title": "short reel title",
  "product": "{product['name']}",
  "sku": "{product['sku']}",
  "platform": "{platform}",
  "angle": "{angle}",
  "duration_sec": 18,
  "hook": "opening 2-second hook",
  "scenes": [
    {{
      "scene_num": 1,
      "duration_sec": 6,
      "visual": "short camera description",
      "video_prompt": "detailed Veo video prompt, no text in video, cinematic movement, no logos unless product packaging is natural",
      "image_prompt": "fallback still image prompt, no text in image",
      "audio": "voiceover or sound direction",
      "overlay": {{
        "headline": "max 6 words",
        "subtext": "max 10 words or empty",
        "y_position": 700,
        "style": "bold-white|yellow-pop|soft-italic"
      }},
      "safe_note": "one-line safe-zone note"
    }}
  ],
  "cta": "max 8 words",
  "cta_y": 1120,
  "caption": "ready-to-post caption with 2-3 emojis, max 150 chars",
  "hashtags": ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6"],
  "music": "music vibe and BPM suggestion",
  "hwg_check": "one-line compliance confirmation"
}}
"""

    response = client.models.generate_content(
        model=MODEL_FLASH,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.8),
    )

    script = json.loads(_strip_json_fence(response.text))

    # Normalize scene durations for Veo
    for scene in script.get("scenes", []):
        scene["duration_sec"] = _nearest_veo_duration(scene.get("duration_sec", 6))

    script["duration_sec"] = sum(s.get("duration_sec", 6) for s in script.get("scenes", []))
    return script


# ── Step 2A: Real Veo video generation ──────────────────────────────────────
def generate_scene_video(
    video_prompt: str,
    product_name: str,
    output_path: str,
    duration_sec: int = 6,
    progress_cb=None,
) -> str | None:
    """
    Generates a real vertical video clip with Veo.
    Returns output_path if successful, otherwise None.
    """

    #duration_sec = _nearest_veo_duration(duration_sec)

    full_prompt = f"""
Vertical 9:16 cinematic product reel video.
Brand mood: German Alpine natural care, premium but warm, trusted since 1973.
Product: {product_name}.

Scene:
{video_prompt}

Important:
- No readable text inside the video.
- No subtitles.
- No watermark.
- Keep center-left area visually clean for later text overlays.
- Avoid medical claims.
- Natural, sensory, warm, realistic commercial style.
"""

    try:
        if progress_cb:
            progress_cb(f"Calling Veo for {duration_sec}s clip…")

        operation = client.models.generate_videos(
            model=MODEL_VIDEO,
            prompt=full_prompt,
            config=types.GenerateVideosConfig(
                aspect_ratio="9:16",
                duration_seconds=duration_sec,
                resolution="720p",
            ),
        )

        while not operation.done:
            if progress_cb:
                progress_cb("Waiting for Veo clip…")
            time.sleep(10)
            operation = client.operations.get(operation)
            print("Operation:", operation)

        generated_video = operation.response.generated_videos[0]
        client.files.download(file=generated_video.video)
        generated_video.video.save(output_path)

        return output_path

    except Exception as e:
        if progress_cb:
            progress_cb(f"Veo failed, using image fallback. Error: {e}")
        return None


# ── Step 2B: Fallback image generation ──────────────────────────────────────
def generate_scene_image(image_prompt: str, product_name: str) -> Image.Image | None:
    full_prompt = (
        f"Studio-quality product photography for a TikTok/Instagram reel. "
        f"Brand: Allgäuer Latschenkiefer, German Alpine natural care, founded 1973. "
        f"Product: {product_name}. "
        f"{image_prompt} "
        f"Photorealistic, warm soft lighting, natural Alpine textures, shallow depth of field, "
        f"portrait composition. No text, no watermarks, no overlays."
    )

    try:
        response = client.models.generate_content(
            model=MODEL_IMAGE,
            contents=full_prompt,
            config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )

        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")

    except Exception:
        return None

    return None


# ── Step 2C: Image fallback frame ───────────────────────────────────────────
def create_scene_frame(script: dict, scene: dict, bg_image: Image.Image | None = None) -> Image.Image:
    W, H = 1080, 1920
    sz = SAFE_ZONES

    if bg_image is not None:
        img = _cover_crop(bg_image, W, H)
    else:
        arr = np.zeros((H, W, 3), dtype=np.uint8)
        for y in range(H):
            t = y / (H - 1)
            arr[y, :] = [int(6 + t * 14), int(8 + t * 30), int(14 + t * 10)]
        img = Image.fromarray(arr)

    img = img.convert("RGBA")

    dark = Image.new("RGBA", (W, H), (0, 0, 0, 75))
    img = Image.alpha_composite(img, dark)

    return img.convert("RGB")


# ── Step 2D: Transparent text overlay for video/image ───────────────────────
def create_text_overlay(script: dict, scene: dict) -> Image.Image:
    W, H = 1080, 1920
    sz = SAFE_ZONES

    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Top subtle gradient
    for y in range(320):
        a = int(170 * (1 - y / 320))
        draw.line([(0, y), (W, y)], fill=(0, 0, 0, a))

    # Bottom subtle gradient
    fade_start = sz["bottom"] - 80
    for y in range(fade_start, H):
        a = int(180 * ((y - fade_start) / (H - fade_start)))
        draw.line([(0, y), (W, y)], fill=(0, 0, 0, a))

    # Brand
    f_brand = _font(38)
    draw.text(
        (sz["left"] + 24, sz["top"] + 28),
        "Allgäuer Latschenkiefer",
        fill=C_MUTED + (255,),
        font=f_brand,
    )

    bar_y = sz["top"] + 82
    draw.line(
        [(sz["left"] + 24, bar_y), (sz["left"] + 420, bar_y)],
        fill=C_GREEN + (255,),
        width=3,
    )

    # Headline
    ov_data = scene.get("overlay", {})
    headline = (ov_data.get("headline") or "").strip()

    if headline:
        f_hl = _font(76)
        y_pos = int(ov_data.get("y_position") or 750)
        y_pos = max(360, min(y_pos, 980))

        color = C_YELLOW if "yellow" in ov_data.get("style", "") else C_WHITE
        h_lines = textwrap.wrap(headline, width=13)[:2]
        line_h = 105

        pill_h = len(h_lines) * line_h + 30
        draw.rounded_rectangle(
            [sz["left"] + 10, y_pos - 18, sz["right"] - 10, y_pos + pill_h],
            radius=28,
            fill=(0, 0, 0, 135),
        )

        for i, part in enumerate(h_lines):
            bbox = draw.textbbox((0, 0), part, font=f_hl)
            tw = bbox[2] - bbox[0]
            tx = max(sz["left"] + 24, (W - tw) // 2)
            tx = min(tx, sz["right"] - tw - 24)
            ty = y_pos + i * line_h

            draw.text((tx + 5, ty + 5), part, fill=(0, 0, 0, 220), font=f_hl)
            draw.text((tx, ty), part, fill=color + (255,), font=f_hl)

        subtext = (ov_data.get("subtext") or "").strip()
        if subtext:
            f_sub = _font(42)
            sub_y = min(y_pos + len(h_lines) * line_h + 10, sz["bottom"] - 150)
            bbox = draw.textbbox((0, 0), subtext, font=f_sub)
            tw = bbox[2] - bbox[0]
            tx = max(sz["left"] + 24, (W - tw) // 2)
            tx = min(tx, sz["right"] - tw - 24)
            draw.text((tx + 3, sub_y + 3), subtext, fill=(0, 0, 0, 220), font=f_sub)
            draw.text((tx, sub_y), subtext, fill=C_MUTED + (255,), font=f_sub)

    # CTA
    cta = (script.get("cta") or "").strip()
    if cta:
        cta_y = int(script.get("cta_y", 1240))
        cta_y = max(1080, min(cta_y, 1220))

        f_cta = _font(46)
        bbox = draw.textbbox((0, 0), cta, font=f_cta)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad_x, pad_y = 30, 16

        tx = max(sz["left"] + 20, (W - tw) // 2)
        tx = min(tx, sz["right"] - tw - 20)

        draw.rounded_rectangle(
            [tx - pad_x, cta_y - pad_y, tx + tw + pad_x, cta_y + th + pad_y],
            radius=22,
            fill=C_GREEN + (245,),
        )
        draw.text((tx, cta_y), cta, fill=C_WHITE + (255,), font=f_cta)

    return img


# ── Step 2E: Assemble final MP4 ─────────────────────────────────────────────
def generate_video(script: dict, output_dir: str, progress_cb=None, use_veo: bool = True) -> str:
    """
    Generates one Veo clip per scene.
    If Veo fails, falls back to image slideshow for that scene.
    Adds safe-zone overlays using Pillow + MoviePy.
    """

    from moviepy import (
        ImageClip,
        VideoFileClip,
        CompositeVideoClip,
        concatenate_videoclips,
    )

    os.makedirs(output_dir, exist_ok=True)

    clips = []
    temp_dir = os.path.join(output_dir, "_tmp")
    os.makedirs(temp_dir, exist_ok=True)

    for scene in script["scenes"]:
        n = scene["scene_num"]
        total = len(script["scenes"])
        duration = _nearest_veo_duration(scene.get("duration_sec", 6))

        if progress_cb:
            progress_cb(f"Scene {n}/{total}: generating base clip…")

        clip_path = os.path.join(temp_dir, f"scene_{n}.mp4")
        base_clip = None

        if use_veo:
            video_prompt = scene.get("video_prompt") or scene.get("visual") or scene.get("image_prompt", "")
            generated = generate_scene_video(
                video_prompt=video_prompt,
                product_name=script.get("product", ""),
                output_path=clip_path,
                duration_sec=duration,
                progress_cb=progress_cb,
            )

            if generated:
                base_clip = VideoFileClip(generated)
                base_clip = _resize_video_cover(base_clip, 1080, 1920)
                if base_clip.duration and base_clip.duration > duration:
                    base_clip = base_clip.subclipped(0, duration) if hasattr(base_clip, "subclipped") else base_clip.subclip(0, duration)

        # Fallback: still image animated as a static clip
        if base_clip is None:
            if progress_cb:
                progress_cb(f"Scene {n}/{total}: using image fallback…")

            img_prompt = scene.get("image_prompt") or scene.get("visual", "")
            bg_image = generate_scene_image(img_prompt, script.get("product", ""))
            frame = create_scene_frame(script, scene, bg_image=bg_image)
            arr = np.array(frame.convert("RGB"))
            base_clip = ImageClip(arr)
            base_clip = _set_duration(base_clip, duration)

        # Add transparent text overlay
        overlay_img = create_text_overlay(script, scene)
        overlay_arr = np.array(overlay_img)
        overlay_clip = ImageClip(overlay_arr, transparent=True)
        overlay_clip = _set_duration(overlay_clip, duration)

        final_scene = CompositeVideoClip(
            [base_clip, overlay_clip],
            size=(1080, 1920),
        )
        final_scene = _set_duration(final_scene, duration)
        clips.append(final_scene)

    final = concatenate_videoclips(clips, method="compose")

    sku = script.get("sku", "ALK").replace("-", "_")
    angle = script.get("angle", "reel")
    filename = f"reel_{sku}_{angle}.mp4"
    path = os.path.join(output_dir, filename)

    if progress_cb:
        progress_cb("Writing final MP4…")

    final.write_videofile(
        path,
        fps=24,
        codec="libx264",
        audio=True,
        preset="medium",
        logger=None,
    )

    final.close()
    for c in clips:
        try:
            c.close()
        except Exception:
            pass

    return filename


# ── Step 3: Storyboard thumbnails ───────────────────────────────────────────
def generate_storyboard_frames(script: dict) -> list:
    SCALE = 0.25
    W = int(1080 * SCALE)
    H = int(1920 * SCALE)
    sz = {k: int(v * SCALE) for k, v in SAFE_ZONES.items() if isinstance(v, int)}

    f_xs = _font(9)
    f_sm = _font(11)
    f_md = _font(13)
    frames_b64 = []

    for scene in script.get("scenes", []):
        img = Image.new("RGBA", (W, H), (14, 16, 26, 255))
        draw = ImageDraw.Draw(img)

        draw.rectangle([sz["left"], sz["top"], sz["right"], sz["bottom"]], fill=(28, 38, 65))

        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ovd = ImageDraw.Draw(ov)
        ovd.rectangle([0, 0, W, sz["top"]], fill=(160, 30, 30, 140))
        ovd.rectangle([0, sz["bottom"], W, H], fill=(160, 30, 30, 140))
        ovd.rectangle([sz["right"], 0, W, H], fill=(160, 30, 30, 100))
        ovd.rectangle([0, 0, sz["left"], H], fill=(160, 30, 30, 55))
        img = Image.alpha_composite(img, ov)
        draw = ImageDraw.Draw(img)

        draw.rectangle([sz["left"], sz["top"], sz["right"], sz["bottom"]], outline=(30, 210, 100), width=2)

        draw.text(
            (sz["left"] + 4, sz["top"] + 4),
            f"Scene {scene['scene_num']} · {scene['duration_sec']}s",
            fill=(110, 130, 155),
            font=f_sm,
        )

        for i, ln in enumerate(textwrap.wrap(scene.get("visual", ""), 30)[:3]):
            draw.text((sz["left"] + 4, sz["top"] + 20 + i * 12), ln, fill=(140, 165, 210), font=f_xs)

        ov_data = scene.get("overlay", {})
        headline = (ov_data.get("headline") or "").strip()

        if headline:
            y_raw = int((ov_data.get("y_position") or 700) * SCALE)
            y_pos = max(sz["top"] + 55, min(y_raw, sz["bottom"] - 45))
            color = (255, 220, 40) if "yellow" in ov_data.get("style", "") else (255, 255, 255)

            for i, part in enumerate(textwrap.wrap(headline, 22)[:2]):
                draw.text((sz["left"] + 6, y_pos + i * 16), part, fill=color, font=f_md)

            sub = ov_data.get("subtext", "")
            if sub:
                draw.text((sz["left"] + 6, y_pos + 36), sub[:34], fill=(200, 210, 230), font=f_xs)

        cta_y = max(sz["bottom"] - 28, sz["top"] + 10)
        draw.text((sz["left"] + 4, cta_y), script.get("cta", "")[:30], fill=(245, 197, 24), font=f_sm)

        draw.text((4, 2), f"TOP {SAFE_ZONES['top']}px", fill=(220, 100, 100), font=f_xs)
        draw.text((4, sz["bottom"] + 18), f"BTM {1920 - SAFE_ZONES['bottom']}px", fill=(220, 100, 100), font=f_xs)

        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        frames_b64.append(base64.b64encode(buf.getvalue()).decode())

    return frames_b64


# ── Step 4: Caption package ─────────────────────────────────────────────────
def generate_caption_package(script: dict, platform: str) -> dict:
    prompt = f"""
Ready-to-post caption package.

Product: {script['product']}
Title: {script['title']}
Angle: {script['angle']}
Platform: {platform}
HWG: cosmetics only, no medical claims.

Return ONLY valid JSON:
{{
  "tiktok_caption": "<=150 chars, 2-3 emojis, one CTA",
  "instagram_caption": "<=220 chars, 2-3 emojis, conversational",
  "hashtags": ["10 relevant hashtags without #"],
  "alt_ctas": ["variant 1", "variant 2", "variant 3"]
}}
"""

    response = client.models.generate_content(model=MODEL_LITE, contents=prompt)
    return json.loads(_strip_json_fence(response.text))


# ── Main orchestrator ───────────────────────────────────────────────────────
def run_filmmaker_agent(
    sku: str,
    angle: str,
    platform: str,
    language: str,
    video_dir: str = "static/videos",
    use_veo: bool = True,
):
    product = next((p for p in PRODUCTS if p["sku"] == sku), PRODUCTS[0])

    yield {"step": f"Generating reel script for {product['name']}…", "done": False}
    script = generate_script(product, angle, platform, language)

    yield {
        "step": f"Script ready — {len(script.get('scenes', []))} scenes · {script.get('duration_sec')}s",
        "done": False,
    }

    yield {"step": "Generating storyboard thumbnails…", "done": False}
    frames = generate_storyboard_frames(script)
    yield {"step": f"{len(frames)} storyboard thumbnails ready", "done": False}

    yield {"step": "Generating caption package…", "done": False}
    captions = generate_caption_package(script, platform)
    yield {"step": "Captions ready", "done": False}

    step_queue = []

    def _cb(msg):
        step_queue.append(msg)

    import threading

    video_result = [None]
    video_error = [None]

    def _gen():
        try:
            video_result[0] = generate_video(script, video_dir, progress_cb=_cb, use_veo=use_veo)
        except Exception as e:
            video_error[0] = str(e)

    t = threading.Thread(target=_gen, daemon=True)
    t.start()

    while t.is_alive() or step_queue:
        if step_queue:
            yield {"step": step_queue.pop(0), "done": False}
        else:
            t.join(timeout=0.3)

    if video_error[0]:
        raise RuntimeError(video_error[0])

    yield {"step": f"Video ready — {video_result[0]}", "done": False}

    yield {
        "step": "Done — reel brief + video complete",
        "done": True,
        "result": {
            "script": script,
            "frames_b64": frames,
            "captions": captions,
            "safe_zones": SAFE_ZONES,
            "product": product,
            "video_filename": video_result[0],
        },
    }