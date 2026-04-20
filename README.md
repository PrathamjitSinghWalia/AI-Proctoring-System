# 🎓 AI Proctoring System

An explainable AI-powered exam monitoring system built using Computer Vision and Machine Learning. 

## 🚀 Overview
This proof-of-concept system provides real-time monitoring for online examinations. Unlike commercial "black-box" proctoring software, this system emphasizes **Explainable AI** by providing transparent reasoning and severity weights for every flagged event.

## 🛠️ Technology Stack
* **Python** — Core logic
* **OpenCV (Haar Cascades)** — Real-time face tracking and counting
* **MediaPipe** — 468-point facial landmark detection for complex head pose and iris-gaze tracking
* **Ultralytics YOLOv8** — Object detection for unauthorized items (phones, laptops, books)
* **Streamlit** — Interactive, data-driven local dashboard
* **Pandas & Matplotlib** — Behavioral analytics and timeline generation
* **FPDF** — Automated session report generation

## 🎯 Core Features
1. **Multi-Model Vision Pipeline:** Runs Haar Cascades, MediaPipe, and YOLOv8 simultaneously on a live webcam feed.
2. **Dynamic Gaze Tracking:** Combines head pose ratios with iris positioning to accurately detect screen deviation.
3. **Smart Cooldowns & Consecutive Frame Logic:** Eliminates false positives by requiring sustained detection before flagging an event.
4. **Weighted Suspicion Scoring:** Assigns specific mathematical risk values to different behaviors (e.g., Phone = +5, Looking Away = +1).
5. **Automated Analytics:** Generates real-time dashboards and downloadable PDF reports containing event timelines and human-readable summaries.

## ⚙️ How to Run Locally
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the monitoring module: `python main.py`
4. Launch the analytics dashboard: `streamlit run dashboard.py`