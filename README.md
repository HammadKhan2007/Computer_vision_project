# 🏏 AI Cricket Bowling Action Analyzer (Fair vs Unfair / Chucking)

Computer Vision aur Biomechanical Pose Estimation par mabni system jo cricket bowling action ko analyze karta hai aur decide karta hai k bowling action **Legal (Fair)** hai ya **Illegal (Unfair / Chucking / Throwing)**.

---

## 📌 Cricket Rule: ICC Law 21.2 (15-Degree Elbow Extension Rule)
International Cricket Council (ICC) regulations ke mutabiq:
- **Delivery Phase Definition**: Arm jab horizontal (shoulder height) par punchti hai wahan se le kar ball release point tak ka phase.
- **Allowed Tolerance**: Is delivery window ke dauran bowler ki kohni (elbow) ka extension **15 degrees** se zyada nahi hona chahiye.
- **Verdict**:
  - Agar $\text{Elbow Extension} \le 15^\circ \implies$ **FAIR / LEGAL ACTION** ✅
  - Agar $\text{Elbow Extension} > 15^\circ \implies$ **UNFAIR / ILLEGAL (CHUCKING)** 🚨

---

## 🚀 Features
1. **Pose Landmark Extraction (MediaPipe & OpenCV)**:
   - Shoulder, elbow, aur wrist ke 3D spatial coordinates track hote hain.
2. **Auto-Detect Bowling Arm**:
   - Right-arm ya Left-arm bowler ko automatically identify karta hai.
3. **Delivery Phase Segmentation**:
   - Horizontal entry point aur ball release keyframe accurately detect karta hai.
4. **Kinematic Angle Tracking**:
   - Frame-by-frame 3D interior elbow angle compute hota hai:
     $$\theta = \arccos\left(\frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|\|\vec{v}\|}\right)$$
5. **Interactive Streamlit Web Dashboard**:
   - Video upload (MP4, AVI, MOV).
   - Side-by-side annotated video with broadcast HUD overlay.
   - Live Elbow Angle gauge and skeleton visualizer.
   - Interactive Matplotlib timeline plot.
   - Instant Built-in Test Samples (Legal & Chucking).

---

## 🛠️ Installation & Quickstart

### 1. Requirements install karein:
```bash
pip install -r requirements.txt
```

### 2. Streamlit Web App chalayein:
```bash
streamlit run app.py
```
App browser me `http://localhost:8501` par open ho jaye gi.

### 3. Test Script chalayein:
```bash
python test_analyzer.py
```
