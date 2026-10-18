"""
Streamlit Web Application: Cricket Bowling Action Analyzer (V2)
===============================================================
Biomechanical Computer Vision application for detecting Fair vs Unfair
(Chucking / Throwing) bowling actions based on ICC Law 21.2.
Supports:
- Side-On View (Square-Leg / Pitch Side)
- Front-On View (Straight / Umpire / Batsman)
- Simultaneous Dual-Arm Tracking with 1-Click Arm Switching
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


# Streamlit Page Config
st.set_page_config(
    page_title="Cricket Bowling Action Analyzer",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Cricket Broadcast Theme)
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .verdict-fair {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 2px solid #34d399;
        margin-bottom: 20px;
    }
    .verdict-unfair {
        background: linear-gradient(135deg, #991b1b 0%, #b91c1c 100%);
        color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 2px solid #f87171;
        margin-bottom: 20px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: #1e293b;
        color: #38bdf8;
        border: 1px solid #0284c7;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    if "video_name" not in st.session_state:
        st.session_state.video_name = ""
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "active_arm_selection" not in st.session_state:
        st.session_state.active_arm_selection = "right"
    if "annotated_videos" not in st.session_state:
        st.session_state.annotated_videos = {}


def main():
    init_session_state()

    st.title("🏏 AI Cricket Bowling Action Analyzer (V2)")
    st.markdown(
        "**Multi-Angle Biomechanical Computer Vision System**  \n"
        "Evaluates bowling delivery actions against **ICC Law 21.2 (15° elbow extension limit)** with dedicated support for **Side-On** & **Front-On** camera angles."
    )

    # --- TOP CONTROLS: QUICK VIEW & ARM SELECTION ---
    st.markdown("### 🎯 Video & Camera Settings")
    col_cam, col_arm, col_thresh = st.columns([1.2, 1.2, 1.0])

    with col_cam:
        camera_view = st.selectbox(
            "📹 Camera Angle",
            options=["auto", "side_on", "front_on"],
            format_func=lambda x: "🤖 Auto-Detect Angle" if x == "auto" else ("📐 Side-On View (Square-Leg / Pitch Side)" if x == "side_on" else "🎯 Front-On / Straight View (Umpire)"),
            help="Side-On view applies 2D planar projection to eliminate Z-depth jitter. Front-On applies 3D spatial geometry."
        )

    with col_arm:
        arm_selection = st.selectbox(
            "🏏 Bowler Bowling Arm",
            options=["auto", "right", "left"],
            format_func=lambda x: "🤖 Auto-Detect (AI decides)" if x == "auto" else ("👉 Right-Arm Bowler (e.g. Bumrah)" if x == "right" else "👈 Left-Arm Bowler (e.g. Shaheen)"),
            help="Select bowler's arm directly so the model never confuses the guide/lead hand!"
        )

    with col_thresh:
        threshold = st.slider(
            "⚖️ ICC Limit (°)",
            min_value=5.0,
            max_value=30.0,
            value=15.0,
            step=0.5,
            help="Maximum elbow extension permitted under ICC regulations is 15.0 degrees."
        )

    # --- SIDEBAR SAMPLES ---
    st.sidebar.header("🧪 Built-in Test Samples")
    st.sidebar.markdown("Instant verification samples:")

    col_btn1, col_btn2 = st.sidebar.columns(2)
    sample_triggered = False

    if col_btn1.button("✅ Legal Sample", use_container_width=True):
        sample_path = "samples/fair_action_sample.mp4"
        if not os.path.exists(sample_path):
            generate_bowling_video(sample_path, "fair")
        st.session_state.video_path = sample_path
        st.session_state.video_name = "Legal Bowling Action Sample"
        st.session_state.analysis_result = None
        st.session_state.annotated_videos = {}
        sample_triggered = True

    if col_btn2.button("🚨 Chucking Sample", use_container_width=True):
        sample_path = "samples/unfair_action_sample.mp4"
        if not os.path.exists(sample_path):
            generate_bowling_video(sample_path, "unfair")
        st.session_state.video_path = sample_path
        st.session_state.video_name = "Chucking (Illegal) Action Sample"
        st.session_state.analysis_result = None
        st.session_state.annotated_videos = {}
        sample_triggered = True

    # --- FILE UPLOADER ---
    st.markdown("### 📤 Upload Bowling Video")
    uploaded_file = st.file_uploader(
        "Upload bowling video from side or front angle (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file is not None:
        temp_dir = tempfile.gettempdir()
        temp_input = os.path.join(temp_dir, f"upload_{uploaded_file.name}")
        with open(temp_input, "wb") as f:
            f.write(uploaded_file.read())
        if st.session_state.video_path != temp_input:
            st.session_state.video_path = temp_input
            st.session_state.video_name = uploaded_file.name
            st.session_state.analysis_result = None
            st.session_state.annotated_videos = {}

    if st.session_state.video_path is None:
        st.info("👆 Upar apni bowling video upload karein ya left sidebar se sample button press karein.")
        return

    st.success(f"Video Loaded: **{st.session_state.video_name}**")

    # --- RUN ANALYSIS ---
    col_run, col_clear = st.columns([4, 1])
    analyze_btn = col_run.button("🚀 Analyze Bowling Action", type="primary", use_container_width=True)
    if col_clear.button("🔄 Reset", use_container_width=True):
        st.session_state.video_path = None
        st.session_state.video_name = ""
        st.session_state.analysis_result = None
        st.session_state.annotated_videos = {}
        st.rerun()

    if analyze_btn or sample_triggered or (st.session_state.analysis_result is None and st.session_state.video_path is not None):
        progress_bar = st.progress(0, text="Initializing Dual-Arm Pose Kinematics Engine...")

        try:
            analyzer = BowlingActionAnalyzer(
                bowling_arm=arm_selection,
                camera_view=camera_view,
                extension_threshold=threshold
            )

            progress_bar.progress(20, text="Tracking 3D human pose landmarks and arm kinematics...")
            result = analyzer.analyze_video(
                st.session_state.video_path,
                progress_callback=lambda p: progress_bar.progress(int(20 + p * 45), text=f"Analyzing frames... {int(p*100)}%")
            )

            progress_bar.progress(65, text="Rendering broadcast overlay & skeletal HUD video...")
            temp_out = os.path.join(tempfile.gettempdir(), f"annotated_{result.chosen_arm}_{int(time.time())}.mp4")
            render_annotated_video(
                input_path=st.session_state.video_path,
                output_path=temp_out,
                analysis_result=result,
                progress_callback=lambda p: progress_bar.progress(int(65 + p * 35), text=f"Rendering HUD video... {int(p*100)}%")
            )

            progress_bar.progress(100, text="Analysis Complete!")
            time.sleep(0.4)
            progress_bar.empty()

            st.session_state.analysis_result = result
            st.session_state.active_arm_selection = result.chosen_arm
            st.session_state.annotated_videos[result.chosen_arm] = temp_out

        except Exception as err:
            st.error(f"Error during video processing: {err}")
            import traceback
            st.code(traceback.format_exc())
            return

    # --- DISPLAY DUAL-ARM RESULTS ---
    result: BowlingActionResult = st.session_state.analysis_result

    if result is not None:
        st.markdown("---")
        st.header("📊 Bowling Action Verdict & Kinematic Report")

        # Camera & Arm Auto-Detection Info Bar
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.info(f"📷 **Camera Angle:** `{result.detected_camera_view.replace('_', ' ').title()}`")
        with info_col2:
            st.info(f"🤖 **AI Auto-Detected Arm:** `{result.auto_detected_arm.capitalize()}-arm` (Sweep: {result.right_arm.angular_sweep}° vs {result.left_arm.angular_sweep}°)")
        with info_col3:
            st.info(f"🎯 **Active Analyzed Arm:** `{st.session_state.active_arm_selection.capitalize()}-arm`")

        # Interactive Arm Switcher Tabs
        arm_tab_right, arm_tab_left = st.tabs([
            f"👉 Right Arm {'(Selected 🏏)' if st.session_state.active_arm_selection == 'right' else ''}",
            f"👈 Left Arm {'(Selected 🏏)' if st.session_state.active_arm_selection == 'left' else ''}"
        ])

        # Helper renderer for chosen tab
        def render_arm_view(arm_key: str):
            arm_data = result.right_arm if arm_key == 'right' else result.left_arm
            if arm_data is None:
                st.warning(f"No kinematic data available for {arm_key} arm.")
                return

            # Button to make this arm active
            if st.session_state.active_arm_selection != arm_key:
                if st.button(f"⚡ Set {arm_key.capitalize()} Arm as Active Delivery Arm", key=f"btn_set_{arm_key}"):
                    st.session_state.active_arm_selection = arm_key
                    result.chosen_arm = arm_key
                    result.active_arm_data = arm_data
                    st.rerun()

            # Big Verdict Banner
            if arm_data.is_fair:
                st.markdown(f"""
                <div class="verdict-fair">
                    <h2 style="margin:0; font-size:2.2rem;">✅ {arm_data.verdict}</h2>
                    <p style="font-size:1.15rem; margin-top:8px;">{arm_data.verdict_explanation}</p>
                    <span class="badge">Model Confidence: {arm_data.confidence}%</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-unfair">
                    <h2 style="margin:0; font-size:2.2rem;">🚨 {arm_data.verdict}</h2>
                    <p style="font-size:1.15rem; margin-top:8px;">{arm_data.verdict_explanation}</p>
                    <span class="badge" style="background-color:#450a0a; color:#fca5a5; border-color:#ef4444;">Model Confidence: {arm_data.confidence}%</span>
                </div>
                """, unsafe_allow_html=True)

            # Key Metric Cards (4 Columns)
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                col1.metric(
                    label="Max Elbow Extension",
                    value=f"{arm_data.max_elbow_extension}°",
                    delta=f"{arm_data.max_elbow_extension - threshold:+.1f}° vs limit",
                    delta_color="inverse"
                )

            with col2:
                col2.metric(
                    label="ICC Limit",
                    value=f"{result.threshold_degrees}°",
                    help="Maximum extension permitted by ICC regulations"
                )

            with col3:
                col3.metric(
                    label="Peak Arm Flexion",
                    value=f"{arm_data.min_elbow_angle}°",
                    help="Minimum elbow angle (greatest bend) reached during delivery stride"
                )

            with col4:
                col4.metric(
                    label="Release Angle",
                    value=f"{arm_data.release_elbow_angle}°",
                    help="Elbow angle recorded at the instant of ball release"
                )

            # Video & Graphs Layout
            vid_col, plot_col = st.columns([1.1, 1.2])

            with vid_col:
                st.subheader(f"🎥 {arm_key.capitalize()}-Arm Annotated Video")
                # Check if annotated video rendered for this arm
                vid_path = st.session_state.annotated_videos.get(arm_key)
                if not vid_path or not os.path.exists(vid_path):
                    with st.spinner(f"Generating overlay for {arm_key} arm..."):
                        vid_path = os.path.join(tempfile.gettempdir(), f"annotated_{arm_key}_{int(time.time())}.mp4")
                        result.active_arm_data = arm_data
                        render_annotated_video(st.session_state.video_path, vid_path, result)
                        st.session_state.annotated_videos[arm_key] = vid_path

                if vid_path and os.path.exists(vid_path):
                    st.video(vid_path)
                    with open(vid_path, "rb") as f:
                        st.download_button(
                            label=f"⬇️ Download {arm_key.capitalize()}-Arm Video",
                            data=f,
                            file_name=f"bowling_action_{arm_key}.mp4",
                            mime="video/mp4",
                            use_container_width=True,
                            key=f"dl_{arm_key}"
                        )

            with plot_col:
                st.subheader(f"📈 {arm_key.capitalize()}-Arm Kinematics Plot")
                result.active_arm_data = arm_data
                fig = generate_angle_plot_figure(result)
                st.pyplot(fig)

            # Detailed Breakdown Accordion
            with st.expander(f"📋 Detailed {arm_key.capitalize()}-Arm Delivery Breakdown"):
                st.write(f"**Total Frames Processed:** `{result.total_frames}` at `{result.fps:.1f} FPS`")
                st.write(f"**Camera Geometry:** `{result.detected_camera_view.replace('_', ' ').title()}`")
                st.write(f"**Circular Angular Sweep:** `{arm_data.angular_sweep}°`")
                st.write(f"**Peak Wrist Velocity:** `{arm_data.peak_wrist_speed} norm/s`")
                st.write(f"**Horizontal Arm Entry Frame:** `{arm_data.horizontal_idx}`")
                st.write(f"**Ball Release Frame:** `{arm_data.release_idx}`")
                st.write(f"**Angle at Horizontal:** `{arm_data.horizontal_elbow_angle}°`")
                st.write(f"**Angle at Release:** `{arm_data.release_elbow_angle}°`")
                st.write(fr"**Net Extension ($\Delta\theta$):** `{arm_data.max_elbow_extension}°`")

        with arm_tab_right:
            render_arm_view('right')

        with arm_tab_left:
            render_arm_view('left')


if __name__ == "__main__":
    main()
