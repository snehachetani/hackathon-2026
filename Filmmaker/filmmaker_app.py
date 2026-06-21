"""
Flask UI for the Marketing Filmmaker Agent — Dr. Theiss Naturwaren GmbH.
Run: python filmmaker_app.py  →  http://localhost:5003
"""

import os
import json
import uuid
import queue
import threading
from flask import Flask, render_template, request, jsonify, Response, send_from_directory

from filmmaker_agent import run_filmmaker_agent, PRODUCTS, CONTENT_ANGLES, SAFE_ZONES

app       = Flask(__name__)
jobs: dict = {}
VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "videos")
os.makedirs(VIDEO_DIR, exist_ok=True)


@app.route("/")
def index():
    return render_template(
        "filmmaker.html",
        products=PRODUCTS,
        angles=CONTENT_ANGLES,
        safe_zones=SAFE_ZONES,
    )


def _run_job(job_id: str, sku: str, angle: str, platform: str, language: str):
    q = jobs[job_id]
    try:
        for update in run_filmmaker_agent(
            sku,
            angle,
            platform,
            language,
            video_dir=VIDEO_DIR,
            use_veo=True,
        ):
            q.put(update)
    except Exception as exc:
        q.put({"step": f"Error: {exc}", "done": True, "error": str(exc)})
    finally:
        q.put(None)
        jobs.pop(job_id, None)


@app.route("/generate", methods=["POST"])
def generate():
    data     = request.json or {}
    sku      = data.get("sku",      "ALK-MG-01")
    angle    = data.get("angle",    "post_workout")
    platform = data.get("platform", "TikTok")
    language = data.get("language", "de")

    job_id       = str(uuid.uuid4())
    jobs[job_id] = queue.Queue()
    threading.Thread(
        target=_run_job, args=(job_id, sku, angle, platform, language), daemon=True
    ).start()
    return jsonify({"job_id": job_id})


@app.route("/stream/<job_id>")
def stream(job_id: str):
    q = jobs.get(job_id)
    if not q:
        return "Not found", 404

    def event_gen():
        while True:
            item = q.get()
            if item is None:
                yield "data: [DONE]\n\n"
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

    return Response(event_gen(), mimetype="text/event-stream")


@app.route("/videos/<filename>")
def serve_video(filename: str):
    return send_from_directory(VIDEO_DIR, filename, mimetype="video/mp4")


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5003)
