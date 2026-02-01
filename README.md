# AI COSS AutoCar3G Autonomous Driving Robot — Winner 🏆

**Winner solution** for the **AI COSS Autonomous Driving Robot Competition**,  
focusing on real-time autonomous driving and reward-maximizing decision-making  
under embedded system constraints.

---

## 🏁 Overview

This project implements an **end-to-end autonomous driving robot system** developed for the  
**AI COSS Autonomous Driving Competition**, where the objective was to **maximize total reward
within a limited time** while maintaining stable autonomous driving.

The system combines:
- **Camera-based line tracing** for continuous driving
- **Decision-making algorithms** to select optimal paths at intersections
- **Object detection** to identify reward signals

All components were deployed and validated on a **Jetson Nano–based AutoCar3G platform**,  
reflecting real-world constraints such as limited compute resources and real-time inference.

🏆 **Final Result: Winner (1st Place, External Competition)**

---

## 🧠 Core Problem

Beyond basic line following, the key challenge was **decision-making under uncertainty**:

- Which path (left/right) yields higher expected reward?
- How to balance exploration and exploitation in a short time window?
- How to integrate perception, control, and decision logic on embedded hardware?

To address this, we designed a system that **explicitly separates perception, control, and decision layers**, enabling flexible and robust behavior during competition runs.

---

## ⚙️ System Architecture
Camera Input
↓
CNN-based Line Tracing (Steering Control)
↓
Intersection Detection
↓
Two-Armed Bandit Decision Module (Left / Right)
↓
YOLO Object Detection (Reward Signal)
↓
Reward Update & Policy Adjustment


- **Jetson Nano** performs real-time control and inference
- **YOLO-based detection** provides sparse but high-impact reward signals
- Cooldown logic prevents duplicate reward counting within a single lap

---

## ✨ Key Features

- 🚗 **CNN-based Line Tracing**
  - Lightweight model optimized for Jetson Nano
  - Stable multi-lap autonomous driving without collisions

- 🎯 **Two-Armed Bandit Decision Making**
  - UCB-based and ensemble policies
  - Adaptive path selection based on accumulated rewards

- 👁️ **YOLO-based Object Detection**
  - Detects reward objects (green signals)
  - Serves as reinforcement feedback rather than dense supervision

- 🔗 **Real-World System Integration**
  - Deployed on AutoCar3G platform
  - Designed under real-time and hardware constraints

---

## 🛠 Tech Stack

- **Language**: Python
- **Embedded Platform**: NVIDIA Jetson Nano
- **Deep Learning**: CNN, YOLOv8
- **Decision Algorithms**: Two-Armed Bandit (UCB, Ensemble Policy)
- **Frameworks**: PyTorch / Keras, OpenCV

---

## 📊 Results

- Stable autonomous driving across continuous laps
- Effective reward maximization through adaptive decision-making
- Robust performance in a real competition environment
- 🏆 **Winner of AI COSS Autonomous Driving Robot Competition**

---

## 📄 Documentation

Detailed project documentation, experimental analysis, and the official competition report are available on Notion:

👉 https://www.notion.so/AI-COSS-2dd16f83064c813ebde2cc93412979f2

---

## 👤 Author

- **Affiliation**: Seoul National University of Science and Technology (SeoulTech)
- **Role**: Autonomous Driving, Decision Algorithms, System Integration



