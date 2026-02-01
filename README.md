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

