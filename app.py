"""
Cricket Bowling Action Analyzer — Streamlit Application
========================================================
Biomechanical Computer Vision system for detecting legal vs illegal
(chucking / throwing) cricket bowling actions based on ICC Law 21.2.

Supported camera angles:
  - Side-On  (Square-leg / Pitch-side camera)
  - Front-On (Straight / Umpire / Batsman camera)

Capabilities:
  - Dual-arm simultaneous pose tracking (Right & Left)
  - Automatic bowling-arm detection via angular sweep & wrist velocity
  - Spline interpolation over occluded frames
  - Broadcast-style HUD annotated video output
"""

import streamlit as st
import tempfile
import os
import time
import importlib

import bowling_analyzer
import utils
importlib.reload(bowling_analyzer)
importlib.reload(utils)

from bowling_analyzer import BowlingActionAnalyzer, BowlingActionResult
from utils import render_annotated_video, generate_angle_plot_figure
from sample_generator import generate_bowling_video


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BowlCheck — Cricket Bowling Action Analyzer",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLESHEET
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    .main { background-color: #0b0f19; }

    /* ── Top hero banner ── */
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1a2540 60%, #0f2027 100%);
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 32px 36px 28px;
        margin-bottom: 28px;
    }
    .hero h1 {
        font-size: 2.1rem;
        font-weight: 800;
        color: #f0f6ff;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .hero p {
        font-size: 0.97rem;
        color: #94a3b8;
        margin: 0;
        line-height: 1.6;
    }
    .hero .icc-badge {
        display: inline-block;
        background: #172554;
        border: 1px solid #1d4ed8;
        color: #93c5fd;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 12px;
        letter-spacing: 0.3px;
    }

    /* ── Section headings ── */
    .section-title {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        color: #475569;
        margin: 28px 0 10px 0;
    }

    /* ── Verdict cards ── */
    .verdict-card {
        border-radius: 14px;
        padding: 26px 28px;
        margin: 18px 0 22px 0;
        text-align: center;
    }
    .verdict-fair {
        background: linear-gradient(135deg, #052e16 0%, #14532d 100%);
        border: 1.5px solid #16a34a;
    }
    .verdict-unfair {
        background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 100%);
        border: 1.5px solid #dc2626;
    }
    .verdict-card h2 {
        font-size: 1.85rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 0 0 8px 0;
        letter-spacing: -0.3px;
    }
    .verdict-card p {
        font-size: 0.97rem;
        color: #cbd5e1;
        margin: 0 0 14px 0;
        line-height: 1.6;
    }
    .conf-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        background: rgba(255,255,255,0.08);
        color: #e2e8f0;
        border: 1px solid rgba(255,255,255,0.15);
        letter-spacing: 0.3px;
    }

    /* ── Info pills (camera / arm detection) ── */
    .info-pill-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 20px;
    }
    .info-pill {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 0.83rem;
        color: #94a3b8;
        flex: 1;
        min-width: 160px;
    }
    .info-pill strong { color: #e2e8f0; }

    /* ── Divider ── */
    .custom-divider {
        border: none;
        border-top: 1px solid #1e293b;
        margin: 24px 0;
    }

    /* ── Expander clean style ── */
    details summary { font-size: 0.9rem; color: #94a3b8; }

    /* ── Metric delta override ── */
    [data-testid="metric-container"] { background: #111827; border-radius: 10px; padding: 12px 16px; border: 1px solid #1f2937; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "video_path": None,
        "video_name": "",
        "analysis_result": None,
        "active_arm": "right",
        "annotated_videos": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def _render_sidebar():
    sample_triggered = False

    st.sidebar.markdown(
        "<div class='section-title' style='margin-top:0'>Quick-Start Samples</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Load a pre-generated synthetic delivery for instant testing.")

    col1, col2 = st.sidebar.columns(2)
    if col1.button("Legal Action", use_container_width=True, type="secondary"):
        _load_sample("fair")
        sample_triggered = True
    if col2.button("Illegal (Chuck)", use_container_width=True, type="secondary"):
        _load_sample("unfair")
        sample_triggered = True

    st.sidebar.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<div class='section-title'>Analysis Parameters</div>",
        unsafe_allow_html=True,
    )

    camera_view = st.sidebar.selectbox(
        "Camera Angle",
        options=["auto", "side_on", "front_on"],
        format_func=lambda x: {
            "auto": "Auto-Detect",
            "side_on": "Side-On  (Square-Leg / Pitch Side)",
            "front_on": "Front-On  (Umpire / Straight)",
        }[x],
        help=(
            "Side-On applies 2D planar projection, removing monocular depth noise. "
            "Front-On uses full 3D spatial geometry."
        ),
    )

    arm_selection = st.sidebar.selectbox(
        "Bowling Arm",
        options=["auto", "right", "left"],
        format_func=lambda x: {
            "auto": "Auto-Detect",
            "right": "Right-Arm Bowler",
            "left": "Left-Arm Bowler",
        }[x],
        help=(
            "Explicitly selecting the bowling arm prevents the model from "
            "confusing the guide/non-bowling hand."
        ),
    )

    threshold = st.sidebar.slider(
        "ICC Elbow Extension Limit (°)",
        min_value=5.0,
        max_value=30.0,
        value=15.0,
        step=0.5,
        help="ICC Law 21.2 mandates a maximum of 15.0° elbow extension during delivery.",
    )

    st.sidebar.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div style='font-size:0.78rem; color:#475569; line-height:1.7'>
        <strong style='color:#64748b'>ICC Law 21.2</strong><br>
        A delivery is unfair if the bowler's elbow extends by more than
        <strong style='color:#94a3b8'>15°</strong> from the point at which the
        arm reaches the horizontal until ball release.
        </div>
        """,
        unsafe_allow_html=True,
    )

    return camera_view, arm_selection, threshold, sample_triggered


def _load_sample(action_type: str):
    path_map = {
        "fair": "samples/fair_action_sample.mp4",
        "unfair": "samples/unfair_action_sample.mp4",
    }
    label_map = {
        "fair": "Legal Bowling Action — Synthetic Sample",
        "unfair": "Illegal Bowling Action (Chuck) — Synthetic Sample",
    }
    sample_path = path_map[action_type]
    os.makedirs("samples", exist_ok=True)
    if not os.path.exists(sample_path):
        with st.spinner("Generating synthetic sample video…"):
            generate_bowling_video(sample_path, action_type)
    st.session_state.video_path = sample_path
    st.session_state.video_name = label_map[action_type]
    st.session_state.analysis_result = None
    st.session_state.annotated_videos = {}


# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
def _render_hero():
    st.markdown(
        """
        <div class="hero">
          <h1>🏏 BowlCheck — Cricket Bowling Action Analyzer</h1>
          <p>
            Computer Vision system that performs frame-by-frame biomechanical analysis of cricket
            bowling deliveries. Detects elbow hyper-extension (chucking / throwing) from both
            side-on and front-on camera perspectives using MediaPipe pose estimation and
            dual-arm kinematic tracking.
          </p>
          <span class="icc-badge">ICC Law 21.2 — 15° Elbow Extension Rule</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# VERDICT BANNER
# ─────────────────────────────────────────────────────────────────────────────
def _render_verdict(arm_data):
    css_cls = "verdict-fair" if arm_data.is_fair else "verdict-unfair"
    icon = "✅" if arm_data.is_fair else "🚨"
    st.markdown(
        f"""
        <div class="verdict-card {css_cls}">
          <h2>{icon} {arm_data.verdict}</h2>
          <p>{arm_data.verdict_explanation}</p>
          <span class="conf-pill">Model Confidence: {arm_data.confidence}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ARM TAB CONTENT
# ─────────────────────────────────────────────────────────────────────────────
def _render_arm_tab(arm_key: str, result: BowlingActionResult, threshold: float):
    arm_data = result.right_arm if arm_key == "right" else result.left_arm
    if arm_data is None:
        st.warning(f"No kinematic data available for the {arm_key} arm.")
        return

    # Mark active arm button
    if st.session_state.active_arm != arm_key:
        if st.button(
            f"Set {arm_key.capitalize()}-Arm as Active Bowling Arm",
            key=f"set_{arm_key}",
        ):
            st.session_state.active_arm = arm_key
            result.chosen_arm = arm_key
            result.active_arm_data = arm_data
            st.rerun()

    _render_verdict(arm_data)

    # ── Metrics row ──────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Max Elbow Extension",
        f"{arm_data.max_elbow_extension}°",
        delta=f"{arm_data.max_elbow_extension - threshold:+.1f}° vs ICC limit",
        delta_color="inverse",
    )
    m2.metric(
        "ICC Legal Limit",
        f"{threshold}°",
        help="ICC Law 21.2 mandates ≤ 15° extension during delivery.",
    )
    m3.metric(
        "Peak Arm Flexion",
        f"{arm_data.min_elbow_angle}°",
        help="Minimum elbow angle (maximum bend) recorded during the delivery stride.",
    )
    m4.metric(
        "Angle at Release",
        f"{arm_data.release_elbow_angle}°",
        help="Elbow angle at the moment of ball release.",
    )

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ── Annotated video + kinematics plot ────────────────────────────────────
    vid_col, plot_col = st.columns([1, 1], gap="large")

    with vid_col:
        st.markdown(
            f"<div class='section-title'>{arm_key.capitalize()}-Arm — Annotated Video</div>",
            unsafe_allow_html=True,
        )
        vid_path = st.session_state.annotated_videos.get(arm_key)
        if not vid_path or not os.path.exists(vid_path):
            with st.spinner("Rendering HUD overlay…"):
                vid_path = os.path.join(
                    tempfile.gettempdir(),
                    f"annotated_{arm_key}_{int(time.time())}.mp4",
                )
                result.active_arm_data = arm_data
                render_annotated_video(st.session_state.video_path, vid_path, result)
                st.session_state.annotated_videos[arm_key] = vid_path

        if vid_path and os.path.exists(vid_path):
            st.video(vid_path)
            with open(vid_path, "rb") as fh:
                st.download_button(
                    label=f"Download Annotated Video — {arm_key.capitalize()} Arm",
                    data=fh,
                    file_name=f"bowlcheck_{arm_key}_arm_analysis.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                    key=f"dl_{arm_key}",
                )

    with plot_col:
        st.markdown(
            f"<div class='section-title'>{arm_key.capitalize()}-Arm — Elbow Angle Trajectory</div>",
            unsafe_allow_html=True,
        )
        result.active_arm_data = arm_data
        fig = generate_angle_plot_figure(result)
        st.pyplot(fig, use_container_width=True)

    # ── Technical breakdown ───────────────────────────────────────────────────
    with st.expander("Technical Delivery Breakdown", expanded=False):
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"**Bowling Arm Analyzed:** `{arm_key.capitalize()}`")
            st.markdown(f"**Total Frames:** `{result.total_frames}` @ `{result.fps:.1f} fps`")
            st.markdown(f"**Camera Geometry:** `{result.detected_camera_view.replace('_', ' ').title()}`")
            st.markdown(f"**Circular Angular Sweep:** `{arm_data.angular_sweep}°`")
        with cols[1]:
            st.markdown(f"**Peak Wrist Velocity:** `{arm_data.peak_wrist_speed:.3f} norm/s`")
            st.markdown(f"**Horizontal Entry Frame:** `{arm_data.horizontal_idx}`")
            st.markdown(f"**Ball Release Frame:** `{arm_data.release_idx}`")
            st.markdown(f"**Angle @ Horizontal:** `{arm_data.horizontal_elbow_angle}°`")
            st.markdown(f"**Angle @ Release:** `{arm_data.release_elbow_angle}°`")
            st.markdown(f"**Net Extension (Δθ):** `{arm_data.max_elbow_extension}°`")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    _init_state()
    _render_hero()

    camera_view, arm_selection, threshold, sample_triggered = _render_sidebar()

    # ── Upload area ───────────────────────────────────────────────────────────
    st.markdown(
        "<div class='section-title'>Upload Bowling Video</div>",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Drag and drop a bowling video file (MP4 · AVI · MOV · MKV)",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        tmp_path = os.path.join(tempfile.gettempdir(), f"upload_{uploaded_file.name}")
        with open(tmp_path, "wb") as fh:
            fh.write(uploaded_file.read())
        if st.session_state.video_path != tmp_path:
            st.session_state.video_path = tmp_path
            st.session_state.video_name = uploaded_file.name
            st.session_state.analysis_result = None
            st.session_state.annotated_videos = {}

    if st.session_state.video_path is None:
        st.info(
            "Upload a bowling video above, or load a built-in synthetic sample from the sidebar to get started."
        )
        return

    # ── Loaded file indicator + controls ─────────────────────────────────────
    loaded_col, btn_col, reset_col = st.columns([4, 1.4, 0.8])
    loaded_col.success(f"Loaded: **{st.session_state.video_name}**")
    analyze_btn = btn_col.button(
        "Run Analysis", type="primary", use_container_width=True
    )
    if reset_col.button("Reset", use_container_width=True):
        for k in ("video_path", "video_name", "analysis_result", "annotated_videos"):
            st.session_state[k] = None if k != "annotated_videos" else {}
        st.session_state.video_name = ""
        st.rerun()

    # ── Run analysis ──────────────────────────────────────────────────────────
    should_run = (
        analyze_btn
        or sample_triggered
        or (
            st.session_state.analysis_result is None
            and st.session_state.video_path is not None
        )
    )

    if should_run:
        bar = st.progress(0, text="Initializing Dual-Arm Pose Kinematics Engine…")
        try:
            analyzer = BowlingActionAnalyzer(
                bowling_arm=arm_selection,
                camera_view=camera_view,
                extension_threshold=threshold,
            )
            bar.progress(18, text="Extracting pose landmarks from video frames…")
            result = analyzer.analyze_video(
                st.session_state.video_path,
                progress_callback=lambda p: bar.progress(
                    int(18 + p * 47),
                    text=f"Tracking arm kinematics — {int(p * 100)}% complete",
                ),
            )
            bar.progress(66, text="Rendering broadcast HUD overlay on video…")
            tmp_out = os.path.join(
                tempfile.gettempdir(),
                f"bowlcheck_{result.chosen_arm}_{int(time.time())}.mp4",
            )
            render_annotated_video(
                input_path=st.session_state.video_path,
                output_path=tmp_out,
                analysis_result=result,
                progress_callback=lambda p: bar.progress(
                    int(66 + p * 34),
                    text=f"Rendering annotated video — {int(p * 100)}%",
                ),
            )
            bar.progress(100, text="Analysis complete.")
            time.sleep(0.35)
            bar.empty()

            st.session_state.analysis_result = result
            st.session_state.active_arm = result.chosen_arm
            st.session_state.annotated_videos[result.chosen_arm] = tmp_out

        except Exception as exc:
            bar.empty()
            st.error(f"Processing error: {exc}")
            import traceback
            st.code(traceback.format_exc())
            return

    # ── Results ───────────────────────────────────────────────────────────────
    result: BowlingActionResult = st.session_state.analysis_result
    if result is None:
        return

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'>Analysis Report</div>",
        unsafe_allow_html=True,
    )

    # Detection metadata strip
    st.markdown(
        f"""
        <div class="info-pill-row">
          <div class="info-pill">📷 <strong>Camera Angle:</strong> {result.detected_camera_view.replace("_", " ").title()}</div>
          <div class="info-pill">🤖 <strong>AI-Detected Bowling Arm:</strong> {result.auto_detected_arm.capitalize()}-arm</div>
          <div class="info-pill">↻ <strong>Angular Sweep R / L:</strong> {result.right_arm.angular_sweep}° / {result.left_arm.angular_sweep}°</div>
          <div class="info-pill">🎯 <strong>Active Arm:</strong> {st.session_state.active_arm.capitalize()}-arm</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dual-arm tabs
    tab_right_label = (
        f"Right Arm {'— Active' if st.session_state.active_arm == 'right' else ''}"
    )
    tab_left_label = (
        f"Left Arm {'— Active' if st.session_state.active_arm == 'left' else ''}"
    )
    tab_right, tab_left = st.tabs([tab_right_label, tab_left_label])

    with tab_right:
        _render_arm_tab("right", result, threshold)

    with tab_left:
        _render_arm_tab("left", result, threshold)


if __name__ == "__main__":
    main()
