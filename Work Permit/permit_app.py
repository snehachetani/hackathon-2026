import os, tempfile, uuid, threading
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from permit_agent import validate_permit

load_dotenv(Path(__file__).resolve().parent.parent / '.env')

app = Flask(__name__)
jobs: dict[str, dict] = {}


@app.route("/")
def index():
    return render_template("permits.html")


@app.route("/api/validate", methods=["POST"])
def validate():
    f = request.files.get("file")
    if not f or f.filename == "":
        return jsonify({"error": "No file uploaded"}), 400
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    job_id = uuid.uuid4().hex[:8]
    jobs[job_id] = {"state": "processing", "result": None}

    tmp = Path(tempfile.mkdtemp()) / f.filename
    f.save(str(tmp))

    def run():
        try:
            result = validate_permit(tmp)
            jobs[job_id]["result"] = result
            jobs[job_id]["state"] = "done"
        except Exception as e:
            jobs[job_id]["state"] = "error"
            jobs[job_id]["error"] = str(e)

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/validate/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    return jsonify(job)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5001)
