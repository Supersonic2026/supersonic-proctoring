"""
Supersonic Assessment Platform — Backend API
Handles: face detection (OpenCV), candidate management, scoring, proctoring logs
All connected to Supabase for permanent storage.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
import secrets
from datetime import datetime, timedelta
from supabase import create_client, Client

app = Flask(__name__)
CORS(app)

# ── SUPABASE CONNECTION ──
# Set these as environment variables on Render (Settings > Environment)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")  # use service_role key, not anon key

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ── FACE DETECTION (OpenCV, free, no external API) ──
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')


@app.route('/')
def health():
    db_status = "connected" if supabase else "not configured"
    return jsonify({
        "status": "ok",
        "message": "Supersonic assessment backend is running",
        "database": db_status
    })


# ============================================================
# FACE DETECTION ENDPOINT (unchanged from before, still free)
# ============================================================
@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        img_b64 = data.get('image', '')
        candidate_id = data.get('candidate_id')  # optional, logs to DB if provided

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

        # Log to database if this is a flagged event and we have a candidate ID + DB connected
        if status != "ok" and candidate_id and supabase:
            try:
                supabase.table("proctoring_events").insert({
                    "candidate_id": candidate_id,
                    "event_type": status,
                    "reason": reason
                }).execute()

                # Update running summary count
                summary = supabase.table("proctoring_summary").select("*").eq("candidate_id", candidate_id).execute()
                if summary.data:
                    current = summary.data[0]
                    supabase.table("proctoring_summary").update({
                        "face_alerts": current.get("face_alerts", 0) + 1
                    }).eq("candidate_id", candidate_id).execute()
                else:
                    supabase.table("proctoring_summary").insert({
                        "candidate_id": candidate_id,
                        "face_alerts": 1
                    }).execute()
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
    """HR creates a new candidate invite. Returns a unique link token."""
    if not supabase:
        return jsonify({"error": "Database not configured"}), 500
    try:
        data = request.json
        token = secrets.token_urlsafe(16)

        result = supabase.table("candidates").insert({
            "full_name": data.get("full_name"),
            "email": data.get("email"),
            "phone": data.get("phone", ""),
            "role_applied": data.get("role_applied"),
            "company": data.get("company", "Supersonic"),
            "invite_token": token,
            "status": "invited",
            "invited_by": data.get("hr_user_id"),
            "expires_at": (datetime.utcnow() + timedelta(days=7)).isoformat()
        }).execute()

        return jsonify({
            "success": True,
            "candidate_id": result.data[0]["id"],
            "invite_link": f"/assess?token={token}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/candidates/by-token/<token>', methods=['GET'])
def get_candidate_by_token(token):
    """Candidate's browser checks if their invite link is valid."""
    if not supabase:
        return jsonify({"error": "Database not configured"}), 500
    try:
        result = supabase.table("candidates").select("*").eq("invite_token", token).execute()
        if not result.data:
            return jsonify({"error": "Invalid or expired link"}), 404

        candidate = result.data[0]
        expires = datetime.fromisoformat(candidate["expires_at"].replace("Z", "+00:00"))
        if expires < datetime.utcnow().replace(tzinfo=expires.tzinfo):
            return jsonify({"error": "This link has expired"}), 410

        return jsonify({"success": True, "candidate": candidate})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/candidates/<candidate_id>/start', methods=['POST'])
def start_assessment(candidate_id):
    """Mark assessment as started when candidate begins."""
    if not supabase:
        return jsonify({"error": "Database not configured"}), 500
    try:
        supabase.table("candidates").update({
            "status": "in_progress",
            "started_at": datetime.utcnow().isoformat()
        }).eq("id", candidate_id).execute()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# ASSESSMENT SUBMISSION & SCORING
# ============================================================

@app.route('/api/assessment/submit', methods=['POST'])
def submit_assessment():
    """
    Receives full assessment answers, calculates scores, saves to database.
    """
    if not supabase:
        return jsonify({"error": "Database not configured"}), 500
    try:
        data = request.json
        candidate_id = data.get("candidate_id")
        responses = data.get("responses", {})  # {cognitive: [...], behavioural: [...], ...}

        # ── SCORING LOGIC ──
        # Each section's score = (correct/total) * 100, weighted by role
        cognitive_score = calculate_section_score(responses.get("cognitive", []))
        behavioural_score = calculate_section_score(responses.get("behavioural", []))
        personality_score = calculate_section_score(responses.get("personality", []))
        org_culture_score = calculate_section_score(responses.get("org_culture", []))

        role = data.get("role_applied", "")
        overall_score = calculate_weighted_overall(
            cognitive_score, behavioural_score, personality_score, org_culture_score, role
        )

        fit_tier = determine_fit_tier(overall_score)

        # Save results
        result = supabase.table("assessment_results").insert({
            "candidate_id": candidate_id,
            "cognitive_score": cognitive_score,
            "behavioural_score": behavioural_score,
            "personality_score": personality_score,
            "org_culture_score": org_culture_score,
            "overall_score": overall_score,
            "fit_tier": fit_tier,
            "detailed_responses": responses,
        }).execute()

        # Mark candidate as completed
        supabase.table("candidates").update({
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat()
        }).eq("id", candidate_id).execute()

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
    """answers = list of {question_id, selected, correct} or open-ended scored items"""
    if not answers:
        return 0
    total = len(answers)
    correct = sum(1 for a in answers if a.get("is_correct"))
    return round((correct / total) * 100, 2)


# Role-based weighting (matches the frontend logic you already have)
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
    """Returns all candidates with their scores for the HR dashboard."""
    if not supabase:
        return jsonify({"error": "Database not configured"}), 500
    try:
        candidates = supabase.table("candidates").select("*, assessment_results(*), proctoring_summary(*)").execute()
        return jsonify({"success": True, "candidates": candidates.data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
