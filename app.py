"""
Supersonic Assessment Platform — Backend API
Handles: face detection (OpenCV), candidate management, scoring, proctoring logs
All connected to Supabase for permanent storage — using direct REST API calls
(no supabase-py library) to avoid key-format compatibility issues.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
import secrets
import requests
from datetime import datetime, timedelta
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

app = Flask(__name__)
CORS(app)

# ── SUPABASE CONNECTION (via direct REST API, no client library) ──
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

SUPABASE_REST = f"{SUPABASE_URL}/rest/v1" if SUPABASE_URL else ""
# ── EMAIL CONFIG (Microsoft Outlook / Office 365) ──
SMTP_HOST = "smtp.office365.com"
SMTP_PORT = 587
SMTP_USER = os.environ.get("SMTP_USER", "")   # paneri.prajapati@supersonicindia.com
SMTP_PASS = os.environ.get("SMTP_PASS", "")   # your Outlook password

SUPABASE_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}


def db_select(table, params=None):
    url = f"{SUPABASE_REST}/{table}"
    r = requests.get(url, headers=SUPABASE_HEADERS, params=params or {})
    r.raise_for_status()
    return r.json()


def db_insert(table, data):
    url = f"{SUPABASE_REST}/{table}"
    r = requests.post(url, headers=SUPABASE_HEADERS, json=data)
    r.raise_for_status()
    return r.json()


def db_update(table, match_params, data):
    url = f"{SUPABASE_REST}/{table}"
    r = requests.patch(url, headers=SUPABASE_HEADERS, params=match_params, json=data)
    r.raise_for_status()
    return r.json()


def db_configured():
    return bool(SUPABASE_URL and SUPABASE_KEY)


def db_health_check():
    if not db_configured():
        return False, "not configured"
    try:
        r = requests.get(f"{SUPABASE_REST}/hr_users", headers=SUPABASE_HEADERS, params={"limit": 1})
        if r.status_code == 200:
            return True, "connected"
        return False, f"error {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, str(e)


# ── FACE DETECTION (OpenCV, free, no external API) ──
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')


# ============================================================
# EMAIL SENDING
# ============================================================

def send_email(to_email, to_name, role, company, invite_link, expiry_days=3):
    """Send assessment invitation email via Microsoft Outlook SMTP."""
    if not SMTP_USER or not SMTP_PASS:
        return False, "Email not configured — set SMTP_USER and SMTP_PASS in Render environment"

    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"Assessment Invitation — {role} | {company}"
        msg['From'] = f"HR Team — {company} <{SMTP_USER}>"
        msg['To'] = to_email

        # Plain text version
        text_body = f"""Hi {to_name},

You have been invited to complete an online assessment for the role of {role} at {company}.

Click the link below to begin your assessment:
{invite_link}

Important:
- The assessment takes approximately 30 minutes
- You will need a working webcam (proctored)
- This link expires in {expiry_days} days
- Do not share this link with anyone

If you have any questions, reply to this email.

Best regards,
HR Team
{company}
"""

        # HTML version
        html_body = f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#F9FAFB;margin:0;padding:0">
  <div style="max-width:560px;margin:40px auto;background:#fff;border-radius:12px;border:1px solid #E5E7EB;overflow:hidden">
    <div style="background:#0a4d8c;padding:28px 32px">
      <h1 style="color:#fff;font-size:20px;font-weight:700;margin:0">{company}</h1>
      <p style="color:rgba(255,255,255,.6);font-size:13px;margin:4px 0 0">Assessment Invitation</p>
    </div>
    <div style="padding:32px">
      <p style="font-size:15px;color:#374151;margin-bottom:16px">Hi <strong>{to_name}</strong>,</p>
      <p style="font-size:14px;color:#6B7280;line-height:1.6;margin-bottom:20px">
        You have been invited to complete an online assessment for the role of 
        <strong style="color:#111827">{role}</strong> at <strong style="color:#111827">{company}</strong>.
      </p>
      <div style="background:#F3F4F6;border-radius:8px;padding:16px;margin-bottom:24px">
        <p style="font-size:13px;font-weight:600;color:#374151;margin:0 0 8px">Before you begin:</p>
        <p style="font-size:13px;color:#6B7280;margin:4px 0">&#128247; Working webcam required (proctored)</p>
        <p style="font-size:13px;color:#6B7280;margin:4px 0">&#9201; Takes approximately 30 minutes</p>
        <p style="font-size:13px;color:#6B7280;margin:4px 0">&#128279; Link expires in {expiry_days} days</p>
        <p style="font-size:13px;color:#6B7280;margin:4px 0">&#128683; Do not share this link</p>
      </div>
      <a href="{invite_link}" 
         style="display:block;background:#E8541C;color:#fff;text-decoration:none;padding:14px 24px;border-radius:8px;font-size:15px;font-weight:700;text-align:center;margin-bottom:24px">
        Start Assessment &#8594;
      </a>
      <p style="font-size:12px;color:#9CA3AF;line-height:1.6">
        If the button doesn't work, copy and paste this link into your browser:<br>
        <span style="color:#6B7280;word-break:break-all">{invite_link}</span>
      </p>
    </div>
    <div style="background:#F9FAFB;border-top:1px solid #E5E7EB;padding:16px 32px">
      <p style="font-size:12px;color:#9CA3AF;margin:0">
        This invitation was sent by the HR team at {company}. 
        Reply to this email if you have any questions.
      </p>
    </div>
  </div>
</body>
</html>"""

        msg.attach(MIMEText(text_body, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, to_email, msg.as_string())

        return True, "Email sent successfully"

    except smtplib.SMTPAuthenticationError:
        return False, "Email authentication failed — check SMTP_USER and SMTP_PASS in Render environment"
    except Exception as e:
        return False, f"Email error: {str(e)}"


@app.route('/api/send-invite', methods=['POST'])
def send_invite():
    """HR sends an assessment invitation email to a candidate."""
    try:
        data = request.json
        to_email = data.get('email', '').strip()
        to_name = data.get('name', 'Candidate').strip()
        role = data.get('role', '')
        company = data.get('company', 'Supersonic')
        invite_link = data.get('invite_link', '')

        if not to_email or not invite_link:
            return jsonify({"success": False, "error": "Email and invite link are required"}), 400

        success, message = send_email(to_email, to_name, role, company, invite_link)

        return jsonify({"success": success, "message": message})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/')
def health():
    db_ok, db_msg = db_health_check()
    return jsonify({
        "status": "ok",
        "message": "Supersonic assessment backend is running",
        "database": db_msg
    })


# ============================================================
# FACE DETECTION ENDPOINT
# ============================================================
@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        img_b64 = data.get('image', '')
        candidate_id = data.get('candidate_id')

        if not img_b64:
            return jsonify({"error": "No image provided"}), 400

        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]

        img_bytes = base64.b64decode(img_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frontal_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        profile_faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        face_count = len(frontal_faces)
        status = "ok"
        reason = None

        if face_count == 0:
            if len(profile_faces) > 0:
                status = "looking_away"
                reason = "Face turned away from screen"
            else:
                status = "no_face"
                reason = "No face detected in frame"
        elif face_count > 1:
            status = "multiple_faces"
            reason = f"{face_count} faces detected — only one person should be visible"

        if status != "ok" and candidate_id and db_configured():
            try:
                db_insert("proctoring_events", {
                    "candidate_id": candidate_id,
                    "event_type": status,
                    "reason": reason
                })
                summary = db_select("proctoring_summary", {"candidate_id": f"eq.{candidate_id}"})
                if summary:
                    current = summary[0].get("face_alerts", 0)
                    db_update("proctoring_summary", {"candidate_id": f"eq.{candidate_id}"},
                              {"face_alerts": current + 1})
                else:
                    db_insert("proctoring_summary", {"candidate_id": candidate_id, "face_alerts": 1})
            except Exception as db_err:
                print(f"DB logging error (non-fatal): {db_err}")

        return jsonify({"status": status, "face_count": face_count, "reason": reason})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# CANDIDATE MANAGEMENT
# ============================================================

@app.route('/api/candidates/invite', methods=['POST'])
def invite_candidate():
    if not db_configured():
        return jsonify({"error": "Database not configured"}), 500
    try:
        data = request.json
        token = secrets.token_urlsafe(16)

        result = db_insert("candidates", {
            "full_name": data.get("full_name"),
            "email": data.get("email"),
            "phone": data.get("phone", ""),
            "role_applied": data.get("role_applied"),
            "company": data.get("company", "Supersonic"),
            "invite_token": token,
            "status": "invited",
            "invited_by": data.get("hr_user_id"),
            "expires_at": (datetime.utcnow() + timedelta(days=7)).isoformat()
        })

        return jsonify({
            "success": True,
            "candidate_id": result[0]["id"],
            "invite_link": f"/assess?token={token}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/candidates/by-token/<token>', methods=['GET'])
def get_candidate_by_token(token):
    if not db_configured():
        return jsonify({"error": "Database not configured"}), 500
    try:
        result = db_select("candidates", {"invite_token": f"eq.{token}"})
        if not result:
            return jsonify({"error": "Invalid or expired link"}), 404

        candidate = result[0]
        expires = datetime.fromisoformat(candidate["expires_at"].replace("Z", "+00:00"))
        if expires < datetime.utcnow().replace(tzinfo=expires.tzinfo):
            return jsonify({"error": "This link has expired"}), 410

        return jsonify({"success": True, "candidate": candidate})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/candidates/<candidate_id>/start', methods=['POST'])
def start_assessment(candidate_id):
    if not db_configured():
        return jsonify({"error": "Database not configured"}), 500
    try:
        db_update("candidates", {"id": f"eq.{candidate_id}"}, {
            "status": "in_progress",
            "started_at": datetime.utcnow().isoformat()
        })
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# ASSESSMENT SUBMISSION & SCORING
# ============================================================

@app.route('/api/assessment/submit', methods=['POST'])
def submit_assessment():
    if not db_configured():
        return jsonify({"error": "Database not configured"}), 500
    try:
        data = request.json
        candidate_id = data.get("candidate_id")
        responses = data.get("responses", {})

        cognitive_score = calculate_section_score(responses.get("cognitive", []))
        behavioural_score = calculate_section_score(responses.get("behavioural", []))
        personality_score = calculate_section_score(responses.get("personality", []))
        org_culture_score = calculate_section_score(responses.get("org_culture", []))

        role = data.get("role_applied", "")
        overall_score = calculate_weighted_overall(
            cognitive_score, behavioural_score, personality_score, org_culture_score, role
        )
        fit_tier = determine_fit_tier(overall_score)

        db_insert("assessment_results", {
            "candidate_id": candidate_id,
            "cognitive_score": cognitive_score,
            "behavioural_score": behavioural_score,
            "personality_score": personality_score,
            "org_culture_score": org_culture_score,
            "overall_score": overall_score,
            "fit_tier": fit_tier,
            "detailed_responses": responses,
        })

        db_update("candidates", {"id": f"eq.{candidate_id}"}, {
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat()
        })

        return jsonify({
            "success": True,
            "scores": {
                "cognitive": cognitive_score,
                "behavioural": behavioural_score,
                "personality": personality_score,
                "org_culture": org_culture_score,
                "overall": overall_score,
                "fit_tier": fit_tier
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def calculate_section_score(answers):
    if not answers:
        return 0
    total = len(answers)
    correct = sum(1 for a in answers if a.get("is_correct"))
    return round((correct / total) * 100, 2)


ROLE_WEIGHTS = {
    "Finance Executive": {"cognitive": 0.40, "behavioural": 0.20, "personality": 0.20, "org_culture": 0.20},
    "Sales Executive": {"cognitive": 0.20, "behavioural": 0.35, "personality": 0.25, "org_culture": 0.20},
    "Operations Executive": {"cognitive": 0.25, "behavioural": 0.25, "personality": 0.20, "org_culture": 0.30},
    "default": {"cognitive": 0.25, "behavioural": 0.25, "personality": 0.25, "org_culture": 0.25},
}


def calculate_weighted_overall(cog, beh, per, org, role):
    weights = ROLE_WEIGHTS.get(role, ROLE_WEIGHTS["default"])
    overall = (
        cog * weights["cognitive"] +
        beh * weights["behavioural"] +
        per * weights["personality"] +
        org * weights["org_culture"]
    )
    return round(overall, 2)


def determine_fit_tier(score):
    if score >= 80:
        return "Strong Fit"
    elif score >= 65:
        return "Good Fit"
    elif score >= 50:
        return "Moderate Fit"
    else:
        return "Not a Fit"


# ============================================================
# HR DASHBOARD DATA
# ============================================================

@app.route('/api/dashboard/candidates', methods=['GET'])
def get_all_candidates():
    if not db_configured():
        return jsonify({"error": "Database not configured"}), 500
    try:
        candidates = db_select("candidates", {"select": "*,assessment_results(*),proctoring_summary(*)"})
        return jsonify({"success": True, "candidates": candidates})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
