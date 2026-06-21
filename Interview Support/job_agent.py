"""
Interview Support Agent — Kohlpharma GmbH (Merzig)
Helps a non-technical hiring manager generate interview questions and spot red flags.
"""

import os, sys, json
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from google import genai
from google.genai import types

load_dotenv(Path(__file__).resolve().parent.parent / '.env')

API_KEY = os.getenv("gemini_api_key")
if not API_KEY:
    sys.exit("ERROR: gemini_api_key not found in .env")

client = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.5-flash-lite"  # no thinking overhead, fast structured output

# ── Single prompt — parse + generate in one call ───────────────────────────────

INTERVIEW_PROMPT = """You are an expert technical recruiter coaching a NON-TECHNICAL hiring manager.
They have limited domain knowledge but need to run a fair, structured interview.

Read the job posting below and generate a complete interview guide.
Return JSON only (no markdown fences):
{
  "role_title": "exact title from posting",
  "interview_overview": "2-sentence plain-English summary of what this role actually does, for the hiring manager",
  "question_categories": [
    {
      "category": "Role & Motivation",
      "icon": "🎯",
      "questions": [
        {
          "question": "exact question to ask",
          "purpose": "why this question matters (1 sentence, plain English)",
          "what_good_looks_like": "what a strong answer sounds/looks like",
          "follow_up": "one follow-up to probe deeper"
        }
      ]
    },
    { "category": "Technical Skills & Experience", "icon": "⚙️", "questions": [] },
    { "category": "Problem Solving & Judgment",    "icon": "🧠", "questions": [] },
    { "category": "Collaboration & Communication", "icon": "🤝", "questions": [] },
    { "category": "Culture & Growth",              "icon": "🌱", "questions": [] }
  ],
  "red_flags": [
    {
      "flag": "short label",
      "what_to_listen_for": "specific phrasing or behavior that signals this",
      "why_it_matters": "plain-English explanation of the risk",
      "how_to_probe": "follow-up question to confirm or rule out"
    }
  ],
  "scoring_rubric": [
    { "dimension": "e.g. Technical Depth", "weight": "high/medium/low", "description": "one sentence" }
  ],
  "hiring_manager_tips": ["tip 1", "tip 2", "tip 3"]
}

Rules:
- 3 questions per category (15 total)
- 5 red flags explained in plain English for a non-technical manager
- Questions must be specific to THIS role, not generic
- All questions open-ended behavioral or situational (no yes/no)
- Keep each field concise (1-2 sentences max)

Job posting:
"""


def stream_interview_guide(job_text: str):
    """Yields SSE chunks; final chunk is 'data: [DONE]\n\n'."""
    buffer = []
    for chunk in client.models.generate_content_stream(
        model=MODEL,
        contents=[INTERVIEW_PROMPT + job_text],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    ):
        text = chunk.text or ""
        if text:
            buffer.append(text)
            yield f"data: {json.dumps(text)}\n\n"
    full = "".join(buffer)
    # parse and send the final structured result
    try:
        guide = json.loads(full)
        yield f"data: [DONE] {json.dumps(guide)}\n\n"
    except json.JSONDecodeError as e:
        yield f"data: [ERROR] {json.dumps(str(e))}\n\n"


# ── Sample jobs from the PDF ───────────────────────────────────────────────────

SAMPLE_JOBS = {
    "hiring_manager": {
        "title": "Hiring Manager — People & Talent",
        "text": """Hiring Manager — People & Talent
Own end-to-end hiring for a fast-scaling AI product company

Company: MONA AI GmbH — applied AI agents for enterprise
Location: Saarbrücken (hybrid, 2–3 days on-site)
Team: People & Talent · reports to Head of People
Type: Full-time, permanent

About the role:
We're hiring across engineering, GTM and operations and need a Hiring Manager who can run structured, fair and fast processes. You'll partner with technical and commercial leads, design scorecards, and protect candidate experience while keeping time-to-hire low.

What you'll do:
- Run full-cycle recruiting: intake, sourcing strategy, screening, scheduling, offer and close.
- Design structured interview kits and scorecards with hiring leads; standardise rubrics.
- Own the ATS, pipeline hygiene and weekly hiring metrics (funnel, time-to-fill, pass-through).
- Coach interviewers on bias-aware, competency-based interviewing.
- Manage employer-branding basics and an inclusive, GDPR-compliant candidate experience.

Must-have qualifications:
- 3+ years in-house recruiting or talent acquisition, ideally in tech/startups.
- Track record closing roles across functions (technical and non-technical).
- Hands-on with an ATS (e.g. Greenhouse, Personio, Join) and structured interviewing.
- Fluent German and English; strong written communication.
- Working knowledge of German labour-law basics and GDPR in recruiting.

Nice to have:
- Experience hiring AI/ML or data talent.
- Familiarity with competency frameworks and work-sample assessments.
- Comfort building simple hiring dashboards.

Tools & stack: Personio / Join ATS · LinkedIn Recruiter · structured-interview scorecards · spreadsheet or BI for funnel metrics · basic German employment-law & GDPR knowledge.

What success looks like (first 6 months): A documented, repeatable interview process per function; median time-to-hire down; interviewer scorecard adoption above 80%; positive candidate-experience feedback.""",
    },
    "gtm_engineer": {
        "title": "Go-to-Market (GTM) Engineer",
        "text": """Go-to-Market (GTM) Engineer
Where revenue meets engineering: build the systems that scale sales

Company: MONA AI GmbH — applied AI agents for enterprise
Location: Saarbrücken / remote (EU time zones)
Team: Revenue · works across Sales, Marketing & Product
Type: Full-time, permanent

About the role:
A hybrid technical-commercial role. You'll automate the GTM motion end-to-end: enrich and route leads, build outbound and lifecycle workflows, wire the data between CRM and product, and ship internal tools (often AI-assisted) that make the revenue team faster.

What you'll do:
- Design and maintain lead enrichment, scoring and routing pipelines.
- Build outbound/lifecycle automations and integrations across CRM, product and billing data.
- Develop internal tools and lightweight apps (incl. LLM-powered workflows) for sales & CS.
- Instrument the funnel: event tracking, attribution, and revenue dashboards.
- Run experiments on messaging, sequencing and conversion; report what actually moves pipeline.

Must-have qualifications:
- 2+ years in GTM/RevOps/sales-engineering or software engineering touching go-to-market.
- Strong with APIs, webhooks and scripting (Python or JavaScript/TypeScript).
- Hands-on CRM automation (HubSpot or Salesforce) and data plumbing (SQL).
- Comfortable building with LLM APIs and prompt-based workflows.
- Clear communicator who can sit between technical and commercial teams.

Nice to have:
- Experience with iPaaS / workflow tools (Zapier, Make, n8n) and reverse-ETL.
- Familiarity with product-led growth instrumentation and attribution modelling.
- Prior startup 0→1 GTM tooling experience.

Tools & stack: Python / TypeScript · HubSpot or Salesforce · SQL & warehouse (BigQuery/Postgres) · REST/webhooks · LLM APIs · n8n/Make/Zapier · analytics (e.g. Looker/Metabase).

What success looks like (first 6 months): Lead routing and enrichment fully automated; a working revenue dashboard the team trusts; at least two shipped internal tools that measurably cut manual sales work.""",
    },
    "fde": {
        "title": "Forward Deployed Engineer (FDE)",
        "text": """Forward Deployed Engineer (FDE)
Embed with customers and turn their hardest problems into shipped AI solutions

Company: MONA AI GmbH — applied AI agents for enterprise
Location: Saarbrücken HQ + on-site at customers (travel up to ~30%)
Team: Delivery / Solutions Engineering · reports to Head of Delivery
Type: Full-time, permanent

About the role:
FDEs are senior engineers who deploy directly into customer environments, scope ambiguous problems, and build and integrate AI-agent solutions against real data and systems. You own delivery from discovery to production hand-off, and you're the technical face to the customer.

What you'll do:
- Scope customer problems on-site; translate vague requirements into a concrete technical plan.
- Build, integrate and deploy agentic workflows against customer data, APIs and internal systems.
- Design retrieval / RAG pipelines and evaluate LLM output quality with real test sets.
- Harden integrations: auth, error handling, observability, and security/PII handling.
- Run production hand-off, documentation and enablement; feed learnings back to Product.

Must-have qualifications:
- 4+ years software engineering with strong Python (and SQL); production systems experience.
- Built and shipped LLM/agent or data-integration systems against messy real-world data.
- Solid on APIs, cloud (AWS/GCP/Azure), containers, and CI/CD.
- Customer-facing maturity: can run a technical workshop and say 'no' diplomatically.
- Fluent English; German a strong plus for on-site work in DE.

Nice to have:
- Experience with RAG, vector databases, and LLM evaluation/guardrails.
- Background in regulated/enterprise environments (security, GDPR, audit).
- Prior consulting, solutions-engineering or FDE-style role.

Tools & stack: Python · SQL · LLM & agent frameworks · vector DBs (pgvector/Pinecone/Weaviate) · RAG & eval tooling · AWS/GCP/Azure · Docker · REST/gRPC · observability (logs/traces).

What success looks like (first 6 months): At least one customer taken from discovery to production; a reusable integration pattern contributed back; measurable quality bar on agent outputs (eval pass-rate, not vibes).""",
    },
}

# ── Flask app ──────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("job.html", sample_jobs=SAMPLE_JOBS)


@app.route("/api/sample/<job_key>")
def get_sample(job_key):
    job = SAMPLE_JOBS.get(job_key)
    if not job:
        return jsonify({"error": "Not found"}), 404
    return jsonify(job)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.json
    job_text = (data.get("job_text") or "").strip()
    if not job_text or len(job_text) < 50:
        return jsonify({"error": "Please provide a job description (at least 50 characters)."}), 400

    return Response(
        stream_with_context(stream_interview_guide(job_text)),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5005)
