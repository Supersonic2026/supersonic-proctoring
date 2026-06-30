# Supersonic Proctoring Backend — Free Deployment Guide

This is a Python + OpenCV server that detects:
- Whether a face is present
- Whether multiple faces are in frame
- Whether the candidate is looking away from the screen

Everything here uses free, open-source tools. Hosting on Render's free tier costs $0.

---

## Step 1 — Create a free Render account

1. Go to https://render.com
2. Sign up (you can use your GitHub account to sign up in one click)

## Step 2 — Put this code on GitHub

1. Create a new repository on GitHub (e.g. `supersonic-proctoring`)
2. Upload these 3 files into it:
   - `app.py`
   - `requirements.txt`
   - `render.yaml`

## Step 3 — Deploy on Render

1. In Render, click **New +** → **Blueprint**
2. Connect your GitHub account and select the `supersonic-proctoring` repo
3. Render will read `render.yaml` automatically and set everything up
4. Click **Apply** — it will build and deploy (takes 2-3 minutes)
5. You'll get a URL like: `https://supersonic-proctoring.onrender.com`

## Step 4 — Test it's working

Open this URL in your browser:
`https://supersonic-proctoring.onrender.com/`

You should see:
```json
{"status": "ok", "message": "Proctoring server is running"}
```

If you see that — your backend is live and free.

---

## Important note about the free tier

Render's free tier "sleeps" the server after 15 minutes of no traffic. The first request after sleeping takes about 20-30 seconds to wake up — after that it responds instantly. This is completely fine for occasional assessment use. If you start running many assessments daily and want zero wake-up delay, the next tier up is about $7/month.

---

## How your HTML file talks to this server

Once deployed, you give me the Render URL (e.g. `https://supersonic-proctoring.onrender.com`) and I'll wire up the assessment page to:

1. Capture a webcam frame every 2-3 seconds
2. Send it to `https://your-url.onrender.com/detect`
3. Get back a status (ok / no_face / multiple_faces / looking_away)
4. Show the live indicator and log flags into the HR report exactly like before

No API keys needed. No paid services. Completely free to run.
