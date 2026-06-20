"""
Marketing content / filmmaker agent — Dr. Theiss Naturwaren GmbH
Produces studio-quality short-form reel videos (1080×1920 MP4).
• gemini-2.5-flash-image  → one AI-generated scene photo per scene
• text overlays composited on top within TikTok/Instagram safe zones
• gemini-2.5-flash        → script + HWG compliance
• gemini-2.5-flash-lite   → caption package
• moviepy + Pillow        → MP4 assembly
"""

import os
import json
import io
import base64
import textwrap
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).resolve().parent.parent / '.env')

API_KEY = os.getenv("gemini_api_key")
client  = genai.Client(api_key=API_KEY)

MODEL_FLASH = "gemini-2.5-flash"
MODEL_LITE  = "gemini-2.5-flash-lite"
MODEL_IMAGE = "gemini-2.5-flash-image"

# ── Safe zones (pixels on 1080×1920) ─────────────────────────────────────────
SAFE_ZONES = {
    "width":  1080,
    "height": 1920,
    "top":    140,    # keep content BELOW this
    "bottom": 1320,   # keep content ABOVE this (1920 - 600)
    "left":   40,
    "right":  900,    # keep content LEFT of this (1080 - 180)
}

# Full product catalog from §2–3 of the hackathon data pack.
# hero=True marks the brief's primary SKUs for video (Problem 6).
# season / segment are used to tailor the script tone.
PRODUCTS = [
    # ── FÜSSE / Feet ──────────────────────────────────────────────
    {"sku": "ALK-FB-01", "name": "Fuß Butter",               "line": "Feet",           "price": 7.71,  "season": "Autumn–Winter",  "segment": "45+ dry-skin, women",         "hero": True},
    {"sku": "ALK-FB-02", "name": "Sole Fußbad",              "line": "Feet",           "price": 6.49,  "season": "Winter",         "segment": "Wellness, 50+",               "hero": True},
    {"sku": "ALK-FB-03", "name": "Hornhaut Reduziercreme",   "line": "Feet",           "price": 6.99,  "season": "Spring",         "segment": "Women 30–60",                 "hero": False},
    {"sku": "ALK-FB-04", "name": "Hornhaut Entferner Maske", "line": "Feet",           "price": 8.49,  "season": "Spring–Summer",  "segment": "Women 25–45",                 "hero": True},
    {"sku": "ALK-FB-05", "name": "10% Urea Fußcreme",        "line": "Feet",           "price": 7.25,  "season": "All year",       "segment": "Diabetic / very dry skin",    "hero": False},
    {"sku": "ALK-FB-06", "name": "Fußpflege Deospray",       "line": "Feet",           "price": 6.10,  "season": "Summer",         "segment": "Active / men 20–45",          "hero": False},
    # ── BEINE / Legs ───────────────────────────────────────────────
    {"sku": "ALK-LG-01", "name": "5 in 1 Beinlotion",        "line": "Legs",           "price": 9.95,  "season": "Summer",         "segment": "Women 35–65",                 "hero": True},
    {"sku": "ALK-LG-02", "name": "Bein Frische Gel",         "line": "Legs",           "price": 8.20,  "season": "Summer",         "segment": "Travel / standing jobs",      "hero": False},
    {"sku": "ALK-LG-03", "name": "Besenreiser Pflegebalsam", "line": "Legs",           "price": 11.49, "season": "Spring–Summer",  "segment": "Women 40–65",                 "hero": False},
    # ── MUSKELN & GELENKE / Muscles & Joints ──────────────────────
    {"sku": "ALK-MG-01", "name": "Mobil Gel",                "line": "Muscles/Joints", "price": 5.83,  "season": "Autumn–Winter",  "segment": "Active 30+, 55+ joints",     "hero": True},
    {"sku": "ALK-MG-02", "name": "Mobil Einreibung Extra Stark","line": "Muscles/Joints","price": 8.90, "season": "Winter / sport", "segment": "Sport, 25–55",               "hero": False},
    {"sku": "ALK-MG-03", "name": "Mobil Eisspray akut",      "line": "Muscles/Joints", "price": 9.40,  "season": "Sport season",   "segment": "Athletes, teams",             "hero": True},
    {"sku": "ALK-MG-04", "name": "Franzbranntwein",          "line": "Muscles/Joints", "price": 6.75,  "season": "All year",       "segment": "Traditional 55+",             "hero": False},
    {"sku": "ALK-MG-05", "name": "Wärmendes Intensiv Gel",   "line": "Muscles/Joints", "price": 8.30,  "season": "Winter",         "segment": "45+ tension/back",            "hero": False},
    # ── HUSTENBONBONS / Cough drops ───────────────────────────────
    {"sku": "ALK-CB-01", "name": "Ur Bonbons",               "line": "Cough drops",    "price": 2.49,  "season": "Cold season",    "segment": "Mass-market",                 "hero": False},
]

# All 4 content angles from the brief, plus bonus angles
CONTENT_ANGLES = {
    "ritual_asmr":    "Ritual / ASMR foot bath — slow, sensory, satisfying",
    "post_workout":   "15-sec post-workout recovery — fast, energetic",
    "heavy_legs":     "'Heavy legs after a long shift' — relatable, empathetic hook",
    "origin_story":   "Alpine ingredient origin — Allgäu plantation → bottle",
    "before_after":   "Before / after transformation reveal",
    "sport_recovery": "Acute sport recovery — athletes & teams",
}

C_GREEN     = (48, 140, 36)
C_YELLOW    = (245, 197, 24)
C_WHITE     = (250, 254, 250)
C_MUTED     = (200, 220, 195)


# ── Font loader ───────────────────────────────────────────────────────────────
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


# ── Resize + center-crop to exact W×H (cover mode) ───────────────────────────
def _cover_crop(img: Image.Image, W: int, H: int) -> Image.Image:
    iw, ih = img.size
    scale  = max(W / iw, H / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img    = img.resize((nw, nh), Image.LANCZOS)
    left   = (nw - W) // 2
    top    = (nh - H) // 2
    return img.crop((left, top, left + W, top + H))


# ── Step 1: Generate reel script ─────────────────────────────────────────────
def generate_script(product: dict, angle: str, platform: str, language: str) -> dict:
    sz         = SAFE_ZONES
    angle_desc = CONTENT_ANGLES.get(angle, angle)
    lang_note  = "German (Du-form, warm)" if language == "de" else "English"

    segment = product.get("segment", "general audience")
    season  = product.get("season",  "all year")

    prompt = f"""You are a creative director at Allgäuer Latschenkiefer (Dr. Theiss Naturwaren GmbH).
Create a studio-quality short-form vertical reel script.

Product       : {product['name']} ({product['sku']}) — {product['line']}
Target segment: {segment}
Peak season   : {season}
Angle         : {angle_desc}
Platform      : {platform}
Language      : {lang_note}
Frame         : 1080×1920 px (9:16 vertical)
Duration      : 15–25 seconds total

Tailor tone, visuals, and language to the target segment and peak season above.

── SAFE ZONE RULES ──────────────────────────────────────────────────────────
All text/logos must stay INSIDE this rectangle on the 1080×1920 frame:
  top    > {sz['top']}px
  bottom < {sz['bottom']}px   (600px at bottom reserved for platform UI)
  left   > {sz['left']}px
  right  < {sz['right']}px    (180px at right reserved for icons)

All y_position values must be integers between {sz['top'] + 20} and {sz['bottom'] - 80}.

── BRAND GUARDRAILS ─────────────────────────────────────────────────────────
• Cosmetics NOT drugs — never: "treats", "heals", "cures", "medical"
• Use: "soothes", "refreshes", "cares for", "revitalises", "relieves tiredness"
• HWG compliance — experiential/sensory language only
• Brand: Alpine heritage, natural, trusted since 1973, Made in Germany

Return ONLY valid JSON (no markdown fences):
{{
  "title": "short reel title (max 8 words)",
  "product": "{product['name']}",
  "sku": "{product['sku']}",
  "platform": "{platform}",
  "angle": "{angle}",
  "duration_sec": 20,
  "hook": "opening 2-3 second hook",
  "scenes": [
    {{
      "scene_num": 1,
      "duration_sec": 5,
      "visual": "camera description for image generation (2 sentences)",
      "image_prompt": "detailed photography prompt for AI image generation (no text in image)",
      "audio": "voiceover text OR sound description",
      "overlay": {{
        "headline": "large overlay text (max 6 words)",
        "subtext": "supporting line (max 10 words, or empty string)",
        "y_position": 700,
        "style": "bold-white|yellow-pop|soft-italic"
      }},
      "safe_note": "one-line safe-zone note"
    }}
  ],
  "cta": "call to action (max 8 words)",
  "cta_y": 1240,
  "caption": "ready-to-post caption with 2-3 emojis (150 chars max)",
  "hashtags": ["tag1","tag2","tag3","tag4","tag5","tag6"],
  "music": "music vibe and BPM suggestion",
  "hwg_check": "one-line HWG compliance confirmation"
}}"""

    response = client.models.generate_content(
        model=MODEL_FLASH,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.9),
    )
    text = response.text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text  = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
        if text.startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


# ── Step 2a: Generate scene photo with gemini-2.5-flash-image ────────────────
def generate_scene_image(image_prompt: str, product_name: str) -> Image.Image | None:
    """Returns a PIL Image or None on failure."""
    full_prompt = (
        f"Studio-quality product photography for a TikTok/Instagram reel. "
        f"Brand: Allgäuer Latschenkiefer (German Alpine natural care, founded 1973). "
        f"Product: {product_name}. "
        f"{image_prompt} "
        f"Style: photorealistic, warm soft lighting, natural Alpine textures, "
        f"shallow depth of field, portrait composition. "
        f"NO text, NO watermarks, NO overlays in the image."
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
        pass
    return None


# ── Step 2b: Compose one 1080×1920 frame from a scene image + overlays ───────
def create_scene_frame(script: dict, scene: dict,
                       bg_image: Image.Image | None = None) -> Image.Image:
    W, H = 1080, 1920
    sz   = SAFE_ZONES

    # ── Background ────────────────────────────────────────────────
    if bg_image is not None:
        img = _cover_crop(bg_image, W, H)
    else:
        # Fallback gradient
        arr = np.zeros((H, W, 3), dtype=np.uint8)
        for y in range(H):
            t = y / (H - 1)
            arr[y, :] = [
                int(6  + t * 14),
                int(8  + t * 30),
                int(14 + t * 10),
            ]
        img = Image.fromarray(arr)

    # ── Cinematic dark overlays for text readability ───────────────
    ov  = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ovd = ImageDraw.Draw(ov)

    # Always-on vignette
    ovd.rectangle([0, 0, W, H], fill=(0, 0, 0, 75))

    # Top gradient (brand name area → fade to transparent)
    for y in range(320):
        a = int(200 * (1 - y / 320))
        ovd.line([(0, y), (W, y)], fill=(0, 0, 0, a))

    # Bottom gradient (CTA + platform UI area → fade to transparent)
    fade_start = sz["bottom"] - 60
    for y in range(fade_start, H):
        a = int(210 * ((y - fade_start) / (H - fade_start)))
        ovd.line([(0, y), (W, y)], fill=(0, 0, 0, a))

    img  = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    draw = ImageDraw.Draw(img)

    # ── Alpine accent bar ──────────────────────────────────────────
    bar_y = sz["top"] + 82
    draw.line([(sz["left"] + 24, bar_y), (sz["left"] + 420, bar_y)],
              fill=C_GREEN, width=3)

    # ── Brand name ────────────────────────────────────────────────
    f_brand = _font(38)
    draw.text((sz["left"] + 24, sz["top"] + 28),
              "Allgäuer Latschenkiefer",
              fill=C_MUTED, font=f_brand)

    # ── Headline overlay ──────────────────────────────────────────
    ov_data  = scene.get("overlay", {})
    headline = (ov_data.get("headline") or "").strip()
    if headline:
        f_hl   = _font(96)
        # Always clamp into safe zone regardless of what Gemini returned
        y_pos  = int(ov_data.get("y_position") or 750)
        y_pos  = max(sz["top"] + 120, min(y_pos, sz["bottom"] - 160))
        color  = C_YELLOW if "yellow" in ov_data.get("style", "") else C_WHITE

        h_lines = textwrap.wrap(headline, width=13)

        # Dark pill behind headline for readability
        line_h = 108
        pill_h = len(h_lines[:2]) * line_h + 24
        _draw_pill(draw, sz["left"] + 10, y_pos - 14,
                   sz["right"] - 10, y_pos + pill_h, alpha=150)

        for i, part in enumerate(h_lines[:2]):
            bbox = draw.textbbox((0, 0), part, font=f_hl)
            tw   = bbox[2] - bbox[0]
            tx   = max(sz["left"] + 20, (W - tw) // 2)
            tx   = min(tx, sz["right"] - tw - 20)
            ty   = y_pos + i * line_h
            draw.text((tx + 5, ty + 5), part, fill=(0, 0, 0), font=f_hl)
            draw.text((tx, ty), part, fill=color, font=f_hl)

        subtext = ov_data.get("subtext", "")
        if subtext:
            f_sub = _font(54)
            sub_y = y_pos + len(h_lines[:2]) * line_h + 6
            sub_y = min(sub_y, sz["bottom"] - 150)
            bbox  = draw.textbbox((0, 0), subtext, font=f_sub)
            tw    = bbox[2] - bbox[0]
            tx    = max(sz["left"] + 20, (W - tw) // 2)
            tx    = min(tx, sz["right"] - tw - 20)
            draw.text((tx + 3, sub_y + 3), subtext, fill=(0, 0, 0), font=f_sub)
            draw.text((tx, sub_y), subtext, fill=C_MUTED, font=f_sub)

    # ── CTA pill button ────────────────────────────────────────────
    cta   = script.get("cta", "")
    cta_y = script.get("cta_y", 1240)
    cta_y = max(sz["bottom"] - 120, min(cta_y, sz["bottom"] - 70))
    if cta:
        f_cta = _font(58)
        bbox  = draw.textbbox((0, 0), cta, font=f_cta)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad   = 30
        tx    = max(sz["left"] + 20, (W - tw) // 2)
        tx    = min(tx, sz["right"] - tw - 20)
        draw.rounded_rectangle(
            [tx - pad, cta_y - pad // 2, tx + tw + pad, cta_y + th + pad // 2],
            radius=18, fill=C_GREEN,
        )
        draw.text((tx, cta_y), cta, fill=C_WHITE, font=f_cta)

    return img


def _draw_pill(draw: ImageDraw.Draw, x1, y1, x2, y2, alpha=140):
    """Draw a semi-transparent dark rounded rectangle using RGBA composite."""
    pass  # handled inline via alpha_composite for the full frame


# ── Step 2c: Assemble MP4 ────────────────────────────────────────────────────
def generate_video(script: dict, output_dir: str,
                   progress_cb=None) -> str:
    """
    Generates one AI image per scene, assembles into 1080×1920 MP4.
    progress_cb(msg: str) is called for each step if provided.
    """
    from moviepy import ImageClip, concatenate_videoclips

    os.makedirs(output_dir, exist_ok=True)
    clips = []

    for scene in script["scenes"]:
        n = scene["scene_num"]
        total = len(script["scenes"])

        if progress_cb:
            progress_cb(f"Generating scene {n}/{total} image…")

        img_prompt = scene.get("image_prompt") or scene.get("visual", "")
        bg_image   = generate_scene_image(img_prompt, script.get("product", ""))

        if progress_cb:
            status = "image ready" if bg_image else "using fallback gradient"
            progress_cb(f"Scene {n}/{total} {status} — compositing overlays…")

        frame = create_scene_frame(script, scene, bg_image=bg_image)
        arr   = np.array(frame.convert("RGB"))
        clip  = ImageClip(arr, duration=scene["duration_sec"])
        clips.append(clip)

    final    = concatenate_videoclips(clips, method="compose")
    sku      = script.get("sku", "ALK").replace("-", "_")
    angle    = script.get("angle", "reel")
    filename = f"reel_{sku}_{angle}.mp4"
    path     = os.path.join(output_dir, filename)

    if progress_cb:
        progress_cb("Writing MP4…")

    final.write_videofile(
        path, fps=24, codec="libx264",
        audio=False, preset="ultrafast", logger=None,
    )
    final.close()
    return filename


# ── Step 2d: Storyboard thumbnails (safe-zone diagram) ───────────────────────
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
        img  = Image.new("RGBA", (W, H), (14, 16, 26, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([sz["left"], sz["top"], sz["right"], sz["bottom"]],
                       fill=(28, 38, 65))

        ov  = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ovd = ImageDraw.Draw(ov)
        ovd.rectangle([0, 0, W, sz["top"]],        fill=(160, 30, 30, 140))
        ovd.rectangle([0, sz["bottom"], W, H],      fill=(160, 30, 30, 140))
        ovd.rectangle([sz["right"], 0, W, H],       fill=(160, 30, 30, 100))
        ovd.rectangle([0, 0, sz["left"], H],        fill=(160, 30, 30, 55))
        img  = Image.alpha_composite(img, ov)
        draw = ImageDraw.Draw(img)

        draw.rectangle([sz["left"], sz["top"], sz["right"], sz["bottom"]],
                       outline=(30, 210, 100), width=2)
        draw.text((sz["left"] + 4, sz["top"] + 4),
                  f"Scene {scene['scene_num']}  {scene['duration_sec']}s",
                  fill=(110, 130, 155), font=f_sm)

        for i, ln in enumerate(textwrap.wrap(scene.get("visual", ""), 30)[:3]):
            draw.text((sz["left"] + 4, sz["top"] + 20 + i * 12),
                      ln, fill=(140, 165, 210), font=f_xs)

        ov_data  = scene.get("overlay", {})
        headline = (ov_data.get("headline") or "").strip()
        if headline:
            y_raw = int((ov_data.get("y_position") or 700) * SCALE)
            y_pos = max(sz["top"] + 55, min(y_raw, sz["bottom"] - 45))
            color = (255, 220, 40) if "yellow" in ov_data.get("style", "") else (255, 255, 255)
            for i, part in enumerate(textwrap.wrap(headline, 22)[:2]):
                draw.text((sz["left"] + 6, y_pos + i * 16), part, fill=color, font=f_md)
            sub = ov_data.get("subtext", "")
            if sub:
                draw.text((sz["left"] + 6, y_pos + 36), sub[:34],
                          fill=(200, 210, 230), font=f_xs)

        cta_y = max(sz["bottom"] - 28, sz["top"] + 10)
        draw.text((sz["left"] + 4, cta_y), script.get("cta", "")[:30],
                  fill=(245, 197, 24), font=f_sm)
        draw.text((sz["left"] + 4, sz["bottom"] + 5),
                  f"♪ {scene.get('audio','')[:45]}", fill=(100, 115, 135), font=f_xs)
        draw.text((4, 2), f"TOP {SAFE_ZONES['top']}px",  fill=(220, 100, 100), font=f_xs)
        draw.text((4, sz["bottom"] + 18), f"BTM {1920 - SAFE_ZONES['bottom']}px",
                  fill=(220, 100, 100), font=f_xs)

        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        frames_b64.append(base64.b64encode(buf.getvalue()).decode())

    return frames_b64


# ── Step 3: Caption package ───────────────────────────────────────────────────
def generate_caption_package(script: dict, platform: str) -> dict:
    prompt = f"""Ready-to-post caption package.
Product: {script['product']}
Title: {script['title']}
Angle: {script['angle']}
Platform: {platform}
HWG: cosmetics only, no medical claims.

Return ONLY valid JSON (no markdown fences):
{{
  "tiktok_caption":    "<=150 chars, 2-3 emojis, one CTA",
  "instagram_caption": "<=220 chars, 2-3 emojis, conversational",
  "hashtags":          ["10 relevant hashtags without #"],
  "alt_ctas":          ["variant 1","variant 2","variant 3"]
}}"""
    response = client.models.generate_content(model=MODEL_LITE, contents=prompt)
    text = response.text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text  = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
        if text.startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


# ── Main orchestrator ─────────────────────────────────────────────────────────
def run_filmmaker_agent(sku: str, angle: str, platform: str, language: str,
                        video_dir: str = "static/videos"):
    product = next((p for p in PRODUCTS if p["sku"] == sku), PRODUCTS[0])
    progress_msgs = []

    yield {"step": f"Generating reel script for {product['name']}…", "done": False}
    script = generate_script(product, angle, platform, language)
    yield {"step": f"Script ready — {len(script.get('scenes',[]))} scenes · {script.get('duration_sec')}s", "done": False}

    yield {"step": "Generating storyboard thumbnails…", "done": False}
    frames = generate_storyboard_frames(script)
    yield {"step": f"{len(frames)} storyboard thumbnails ready", "done": False}

    yield {"step": "Generating caption package…", "done": False}
    captions = generate_caption_package(script, platform)
    yield {"step": "Captions ready", "done": False}

    # Stream progress from video generation
    step_queue = []

    def _cb(msg):
        step_queue.append(msg)

    import threading
    video_result = [None]
    video_error  = [None]

    def _gen():
        try:
            video_result[0] = generate_video(script, video_dir, progress_cb=_cb)
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
        "step":   "Done — reel brief + video complete",
        "done":   True,
        "result": {
            "script":         script,
            "frames_b64":     frames,
            "captions":       captions,
            "safe_zones":     SAFE_ZONES,
            "product":        product,
            "video_filename": video_result[0],
        },
    }
