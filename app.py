"""
Streamlit Web Application: Cricket Bowling Action Analyzer
==========================================================
Interactive Computer Vision application for detecting Fair vs Unfair
(Chucking / Throwing) bowling actions based on ICC Law 21.2.
"""

import streamlit as st
import tempfile
import os
import time
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
    if "annotated_video_path" not in st.session_state:
        st.session_state.annotated_video_path = None


def main():
    init_session_state()

    st.title("🏏 Cricket Bowling Action Analyzer")
    st.markdown(
        "**AI Computer Vision System for Biomechanical Action Legal / Illegal (Chucking) Classification**  \n"
        "Evaluates the delivery phase from horizontal arm level to ball release against **ICC Law 21.2 (15° elbow extension limit)**."
    )

    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.header("⚙️ Analysis Settings")

    arm_selection = st.sidebar.selectbox(
        "Bowling Arm",
        options=["auto", "right", "left"],
        format_func=lambda x: "🤖 Auto-Detect" if x == "auto" else ("👉 Right Arm" if x == "right" else "👈 Left Arm"),
        help="Select bowling arm or let AI detect automatically based on overhead swing."
    )

    threshold = st.sidebar.slider(
        "ICC Extension Threshold (°)",
        min_value=5.0,
        max_value=30.0,
        value=15.0,
        step=0.5,
        help="Standard ICC rule allows a maximum of 15.0 degrees of elbow extension."
    )

    st.sidebar.markdown("---")
    st.sidebar.header("🧪 Quick Test Samples")
    st.sidebar.markdown("Video nahi hai? Instant test ke liye sample button click karein:")

    col_btn1, col_btn2 = st.sidebar.columns(2)
    sample_triggered = False

    if col_btn1.button("✅ Legal Sample", use_container_width=True):
        sample_path = "samples/fair_action_sample.mp4"
        if not os.path.exists(sample_path):
            generate_bowling_video(sample_path, "fair")
        st.session_state.video_path = sample_path
        st.session_state.video_name = "Legal Bowling Action Sample"
        st.session_state.analysis_result = None
        st.session_state.annotated_video_path = None
        sample_triggered = True

    if col_btn2.button("🚨 Chucking Sample", use_container_width=True):
        sample_path = "samples/unfair_action_sample.mp4"
        if not os.path.exists(sample_path):
            generate_bowling_video(sample_path, "unfair")
        st.session_state.video_path = sample_path
        st.session_state.video_name = "Chucking (Illegal) Action Sample"
        st.session_state.analysis_result = None
        st.session_state.annotated_video_path = None
        sample_triggered = True

    # --- FILE UPLOADER ---
    st.markdown("### 📤 Upload Bowling Video")
    uploaded_file = st.file_uploader(
        "Apni cricket bowling video upload karein (MP4, AVI, MOV)",
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
            st.session_state.annotated_video_path = None

    if st.session_state.video_path is None:
        st.info("👆 Upar apni bowling video upload karein ya left sidebar se **Legal Sample** ya **Chucking Sample** button press karein.")
        return

    st.success(f"Loaded: **{st.session_state.video_name}**")

    # --- RUN ANALYSIS ---
    col_run, col_clear = st.columns([4, 1])
    analyze_btn = col_run.button("🚀 Analyze Bowling Action", type="primary", use_container_width=True)
    if col_clear.button("🔄 Reset", use_container_width=True):
        st.session_state.video_path = None
        st.session_state.video_name = ""
        st.session_state.analysis_result = None
        st.session_state.annotated_video_path = None
        st.rerun()

    if analyze_btn or sample_triggered or (st.session_state.analysis_result is None and st.session_state.video_path is not None):
        progress_bar = st.progress(0, text="Initializing Computer Vision & Pose Detection Models...")

        try:
            analyzer = BowlingActionAnalyzer(
                bowling_arm=arm_selection,
                extension_threshold=threshold
            )

            progress_bar.progress(20, text="Tracking 3D human pose landmarks and arm kinematics...")
            result = analyzer.analyze_video(
                st.session_state.video_path,
                progress_callback=lambda p: progress_bar.progress(int(20 + p * 40), text=f"Analyzing frames... {int(p*100)}%")
            )

            progress_bar.progress(65, text="Rendering broadcast overlay & skeletal HUD video...")
            temp_out = os.path.join(tempfile.gettempdir(), f"annotated_{int(time.time())}.mp4")
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
            st.session_state.annotated_video_path = temp_out

        except Exception as err:
            st.error(f"Error during video processing: {err}")
            import traceback
            st.code(traceback.format_exc())
            return

    # --- DISPLAY RESULTS ---
    result = st.session_state.analysis_result
    annotated_path = st.session_state.annotated_video_path

    if result is not None:
        st.markdown("---")
        st.header("📊 Bowling Action Verdict & Kinematic Report")

        # Big Verdict Banner
        if result.is_fair:
            st.markdown(f"""
            <div class="verdict-fair">
                <h2 style="margin:0; font-size:2.2rem;">✅ {result.verdict}</h2>
                <p style="font-size:1.15rem; margin-top:8px;">{result.verdict_explanation}</p>
                <span class="badge">Model Confidence: {result.confidence}%</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-unfair">
                <h2 style="margin:0; font-size:2.2rem;">🚨 {result.verdict}</h2>
                <p style="font-size:1.15rem; margin-top:8px;">{result.verdict_explanation}</p>
                <span class="badge" style="background-color:#450a0a; color:#fca5a5; border-color:#ef4444;">Model Confidence: {result.confidence}%</span>
            </div>
            """, unsafe_allow_html=True)

        # Key Metric Cards (4 Columns)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            col1.metric(
                label="Max Elbow Extension",
                value=f"{result.max_elbow_extension}°",
                delta=f"{result.max_elbow_extension - threshold:+.1f}° vs limit",
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
                value=f"{result.min_elbow_angle}°",
                help="Minimum elbow angle (greatest bend) reached during delivery stride"
            )

        with col4:
            col4.metric(
                label="Release Angle",
                value=f"{result.release_elbow_angle}°",
                help="Elbow angle recorded at the instant of ball release"
            )

        # Video & Graphs Layout
        vid_col, plot_col = st.columns([1.1, 1.2])

        with vid_col:
            st.subheader("🎥 Annotated Analysis Video")
            if annotated_path and os.path.exists(annotated_path):
                st.video(annotated_path)
                with open(annotated_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Annotated Video",
                        data=f,
                        file_name="bowling_action_analyzed.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )

        with plot_col:
            st.subheader("📈 Elbow Angle Kinematics")
            fig = generate_angle_plot_figure(result)
            st.pyplot(fig)

        # Detailed Breakdown Accordion
        with st.expander("📋 Detailed Delivery Phase Breakdown"):
            st.write(f"**Detected Bowling Arm:** `{result.bowling_arm.capitalize()}-arm`")
            st.write(f"**Total Frames Processed:** `{result.total_frames}` at `{result.fps:.1f} FPS`")
            st.write(f"**Horizontal Arm Entry Frame:** `{result.detected_phases.get('horizontal_entry')}`")
            st.write(f"**Ball Release Frame:** `{result.detected_phases.get('ball_release')}`")
            st.write(f"**Angle at Horizontal:** `{result.horizontal_elbow_angle}°`")
            st.write(f"**Angle at Release:** `{result.release_elbow_angle}°`")
            st.write(f"**Net Extension ($\Delta\\theta$):** `{result.max_elbow_extension}°`")


if __name__ == "__main__":
    main()
