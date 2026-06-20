# hackathon-2026

## Project Structure

```
q1/   — Invoice Web UI (Globus Group)
q2/   — Shift Replacement Agent (UKS Hospital)
q3/   — Work Permit Validator
q5/   — Interview Support Agent (Kohlpharma)
q6/   — Marketing Filmmaker Agent (Dr. Theiss)
q7/   — Customer Targeting Analytics (Dr. Theiss)
```

## How to Run

| Question | Command | Port |
|----------|---------|------|
| Question 1 | `python q1/app.py` | 5000 |
| Question 2 | `python q2/shift_app.py` | 5001 |
| Question 3 | `python q3/permit_app.py` | 5002 |
| Question 4 | `python q4/job_agent.py` | — |
| Question 5 | `python q5/invoice_agent.py` | CLI |
| Question 6 | `python q6/filmmaker_app.py` | 5003 |
| Question 7 | `python q7/targeting_app.py` | 5004 |

> All apps read `.env` from the project root automatically.
