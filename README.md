# 🎯 AI Interview Behavior Analyzer

> A real-time AI-powered interview behavior analysis system that continuously monitors observable behavioral signals through a webcam and generates a comprehensive interview performance report.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%20%7C%20Face%20Mesh%20%7C%20Hands-orange)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-blue)

---

# 📌 Overview

AI Interview Behavior Analyzer is a **real-time behavioral analytics system** designed for mock interviews.

Instead of attempting to determine personality or psychological traits, the system analyzes **observable behavioral signals** such as posture, eye contact, gestures, movement, facial expressions, and engagement throughout an interview session.

The application continuously evaluates these signals, logs behavioral events, and generates a structured interview performance report with statistical insights and graphical analytics.

---

# 🚀 Features

## ✅ Real-Time Behavioral Analysis

The system continuously monitors:

- 🧍 Posture Analysis
- 🧠 Head Tilt Detection
- 👁 Eye Contact Tracking
- 🙂 Facial Expression Analysis
- 🖐 Hand Gesture Detection
- 🔁 Movement & Restlessness Analysis
- 📊 Behavioral Scoring
- ⚡ Temporal Smoothing
- ⚠ Live Behavioral Warnings

---

## 📊 Session Analytics

During an interview session, the system records:

- Behavioral timeline
- Movement trends
- Eye-contact consistency
- Posture consistency
- Gesture frequency
- Behavioral events
- Session statistics

---

## 📄 Interview Report Generation

After the interview, the system automatically generates:

- Overall Interview Summary
- Behavioral Scorecard
- Posture Analysis
- Eye Contact Analysis
- Facial Behavior Summary
- Gesture Analysis
- Movement Analysis
- Timeline Events
- Statistical Summary
- Improvement Recommendations

---

## 📈 Graphical Analytics

Automatically generates graphs for:

- Overall Score Timeline
- Posture Score Trend
- Eye Contact Trend
- Movement Trend
- Engagement Timeline
- Attention Stability

---

# 🏗 System Architecture

```text
                  Webcam
                     │
                     ▼
          OpenCV Frame Capture
                     │
                     ▼
        ┌─────────────────────────┐
        │   MediaPipe Pose        │
        │   Face Mesh             │
        │   Hand Tracking         │
        └─────────────────────────┘
                     │
                     ▼
          Feature Extraction Layer
                     │
                     ▼
        Behavioral Signal Analysis
                     │
                     ▼
      Temporal Smoothing & Scoring
                     │
                     ▼
        Session Logger & Analytics
                     │
                     ▼
       Report + Graph Generation
```

---

# 🧠 Behavioral Signals Analyzed

## 🧍 Posture

- Straight posture
- Slight slouch
- Heavy slouch
- Leaning forward
- Leaning backward
- Left / Right lean

---

## 👁 Eye Contact

- Looking at camera
- Looking away
- Downward gaze
- Attention stability
- Gaze consistency

---

## 🙂 Facial Analysis

- Neutral expression
- Smile detection
- Facial engagement
- Blink monitoring
- Eye openness

---

## 🖐 Gesture Analysis

- Open hand
- Closed hand
- Hidden hands
- Gesture stability
- Hand visibility

---

## 🔁 Movement Analysis

- Stable
- Moderate movement
- Restlessness
- Movement spikes
- Behavioral consistency

---

# 📊 Generated Scores

The system continuously computes:

- Posture Score
- Eye Contact Score
- Gesture Score
- Facial Engagement Score
- Movement Score
- Attention Score
- Stability Score
- Overall Behavioral Score

---

# ⚠ Real-Time Warnings

Examples:

- Looking Away From Camera
- Low Eye Engagement
- Slouching Detected
- Excessive Movement
- Face Not Visible
- Eyes Closed
- Attention Drift
- Frequent Distraction

---

# 📈 Statistical Analysis

The report includes:

- Mean Scores
- Standard Deviation
- Variance
- Stability Metrics
- Signal Volatility
- Behavioral Consistency
- Session Analytics

---

# 🧰 Tech Stack

## Computer Vision

- OpenCV
- MediaPipe Pose
- MediaPipe Face Mesh
- MediaPipe Hands

## Scientific Computing

- NumPy

## Visualization

- Matplotlib

## Programming Language

- Python

---

# 🧠 Core Concepts

This project combines multiple AI and Computer Vision concepts:

- Computer Vision
- Human Pose Estimation
- Face Landmark Detection
- Hand Landmark Detection
- Behavioral Signal Analysis
- Temporal Signal Processing
- Human Computer Interaction (HCI)
- Real-Time Perception Systems
- Statistical Analytics
- Session Intelligence
- Event Detection
- Behavioral Computing

---

# 📂 Project Structure

```text
AI-Interview-Behavior-Analyzer/
│
├── interview_analyzer.py
├── reports/
│   ├── report.txt
│   ├── report.json
│   ├── graphs/
│   └── session_logs/
│
├── assets/
│
├── requirements.txt
│
└── README.md
```

---

# ▶ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/AI-Interview-Behavior-Analyzer.git
```

Move into the project

```bash
cd AI-Interview-Behavior-Analyzer
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python interview_analyzer.py
```

---

# 📷 Workflow

```text
Start Interview
      │
      ▼
Live Webcam Analysis
      │
      ▼
Behavior Detection
      │
      ▼
Real-Time Scoring
      │
      ▼
Behavior Event Logging
      │
      ▼
Session Analytics
      │
      ▼
Graph Generation
      │
      ▼
Interview Report
```

---

# 🎯 Future Improvements

- Voice Analysis
- Speech Fluency
- Filler Word Detection
- Pause Detection
- Interview Question Understanding
- LLM Feedback Generation
- PDF Reports
- Resume-aware Evaluation
- Recruiter Dashboard
- Multi-Candidate Analytics
- Cloud Deployment

---

# ⚠ Disclaimer

This project **does not determine personality, honesty, confidence, emotions, or psychological traits**.

It analyzes **observable behavioral signals** extracted from computer vision models and provides analytical feedback intended for interview practice and self-improvement.

Behavioral scores should not be interpreted as objective measures of a person's abilities or personality.

---

# 👨‍💻 Author

**Vishwas Namdeo**

AI • Computer Vision • Full Stack Development • Behavioral Analytics

GitHub: https://github.com/yourusername

---

## ⭐ If you found this project interesting, consider giving it a star!