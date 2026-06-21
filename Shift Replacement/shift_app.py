"""
Flask UI for the UKS Shift Replacement Agent.
Run: python shift_app.py   then open http://localhost:5001
"""

import os
import sys
import json
import queue
import threading
import subprocess
import uuid
from flask import Flask, render_template, request, jsonify, Response

app = Flask(__name__)

# job_id -> queue of output lines (None = sentinel / done)
jobs: dict[str, queue.Queue] = {}

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shift_agent.py")


def _stream_agent(job_id: str, hr_message: str):
    q = jobs[job_id]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", SCRIPT, hr_message],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=os.path.dirname(SCRIPT) or ".",
        )
        for line in proc.stdout:
            q.put(line.rstrip("\n"))
        proc.wait()
    except Exception as exc:
        q.put(f"  ERROR: {exc}")
    finally:
        q.put(None)  # sentinel signals completion


@app.route("/")
def index():
    return render_template("shift.html")


@app.route("/api/run", methods=["POST"])
def api_run():
    data = request.get_json(force=True) or {}
    msg = (data.get("message") or "").strip()
    if not msg:
        return jsonify({"error": "No message provided"}), 400

    job_id = uuid.uuid4().hex[:8]
    jobs[job_id] = queue.Queue()
    threading.Thread(target=_stream_agent, args=(job_id, msg), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/stream/<job_id>")
def api_stream(job_id):
    q = jobs.get(job_id)
    if q is None:
        return jsonify({"error": "Job not found"}), 404

    def generate():
        while True:
            try:
                line = q.get(timeout=120)
            except queue.Empty:
                yield f"data: {json.dumps({'done': True, 'timeout': True})}\n\n"
                break
            if line is None:
                yield f"data: {json.dumps({'done': True})}\n\n"
                break
            yield f"data: {json.dumps({'line': line})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    print("UKS Shift Agent UI -> http://0.0.0.0:5001")
    app.run(host="0.0.0.0", debug=False, port=5001, threaded=True)
