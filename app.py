from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64

app = Flask(__name__)
CORS(app)  # Allow requests from your HTML file's domain

# Load OpenCV's built-in face detector (free, no API key, no download needed at runtime)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')


@app.route('/')
def health():
    """Simple health check so you know the server is alive."""
    return jsonify({"status": "ok", "message": "Proctoring server is running"})


@app.route('/detect', methods=['POST'])
def detect():
    """
    Receives a single webcam frame (base64 JPEG) from the browser.
    Returns: face count, whether someone is looking away, and a flag reason if any.
    """
    try:
        data = request.json
        img_b64 = data.get('image', '')

        if not img_b64:
            return jsonify({"error": "No image provided"}), 400

        # Strip the "data:image/jpeg;base64," prefix if present
        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]

        # Decode base64 to OpenCV image
        img_bytes = base64.b64decode(img_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect frontal faces
        frontal_faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        # Detect side-profile faces (catches "looking away" cases)
        profile_faces = profile_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        face_count = len(frontal_faces)
        looking_away = False
        status = "ok"
        reason = None

        if face_count == 0:
            if len(profile_faces) > 0:
                # Face detected but turned to the side
                looking_away = True
                status = "looking_away"
                reason = "Face turned away from screen"
            else:
                status = "no_face"
                reason = "No face detected in frame"
        elif face_count > 1:
            status = "multiple_faces"
            reason = f"{face_count} faces detected — only one person should be visible"
        else:
            status = "ok"

        return jsonify({
            "status": status,
            "face_count": face_count,
            "looking_away": looking_away,
            "reason": reason
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # For local testing
    app.run(host='0.0.0.0', port=5000, debug=True)
