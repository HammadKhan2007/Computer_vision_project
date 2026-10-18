"""
Cricket Bowling Action Visualizer and HUD Overlay Utilities
===========================================================
Draws skeletons, elbow angle arcs, live metric gauges,
delivery phase labels, and generates performance graphs.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from bowling_analyzer import BowlingActionResult, FrameAnalysis, LANDMARKS


# Color Palette (BGR format)
COLOR_NEON_GREEN = (50, 220, 50)
COLOR_CRIMSON_RED = (50, 50, 230)
COLOR_CYAN = (230, 200, 40)
COLOR_AMBER = (30, 165, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_DARK_GRAY = (35, 35, 35)
COLOR_ACCENT_BLUE = (255, 120, 40)


def draw_hud(
    frame: np.ndarray,
    analysis: FrameAnalysis,
    max_extension_so_far: float,
    threshold: float = 15.0
) -> np.ndarray:
    """
    Overlays a professional broadcast-style HUD on the video frame.
    """
    h, w, _ = frame.shape
    overlay = frame.copy()

    # 1. Top HUD Bar (Semi-transparent black banner)
    hud_height = 90
    cv2.rectangle(overlay, (0, 0), (w, hud_height), COLOR_DARK_GRAY, -1)
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 2. Project Title / Branding
    cv2.putText(
        frame,
        "CRICKET BOWLING ACTION ANALYZER (ICC LAW 21.2)",
        (20, 28),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        COLOR_WHITE,
        1,
        cv2.LINE_AA
    )

    # 3. Status Badge (Delivery Phase)
    phase_bg_color = COLOR_AMBER if analysis.is_delivery_phase else (120, 120, 120)
    if "Release" in analysis.phase_label:
        phase_bg_color = COLOR_ACCENT_BLUE

    phase_text = f"PHASE: {analysis.phase_label.upper()}"
    cv2.putText(
        frame,
        phase_text,
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        phase_bg_color,
        2,
        cv2.LINE_AA
    )

    # 4. Live Metrics: Elbow Angle & Max Extension
    angle_text = f"ELBOW ANGLE: {analysis.elbow_angle:.1f}\xba"
    cv2.putText(
        frame,
        angle_text,
        (w // 2 - 120, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        COLOR_CYAN,
        2,
        cv2.LINE_AA
    )

    # Extension Indicator (Red if > 15, Green if <= 15)
    is_exceeded = max_extension_so_far > threshold
    ext_color = COLOR_CRIMSON_RED if is_exceeded else COLOR_NEON_GREEN
    ext_text = f"EXT: {max_extension_so_far:.1f}\xba / {threshold:.0f}\xba"
    status_label = "[ UNFAIR - CHUCK ]" if is_exceeded else "[ FAIR ACTION ]"

    cv2.putText(
        frame,
        f"{ext_text}  {status_label}",
        (w - 380, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        ext_color,
        2,
        cv2.LINE_AA
    )

    # 5. Bowling Arm Skeleton Highlight & Angle Arc
    if analysis.is_valid_detection:
        sh = analysis.shoulder_pt
        el = analysis.elbow_pt
        wr = analysis.wrist_pt

        arm_color = ext_color if analysis.is_delivery_phase else COLOR_CYAN

        # Draw Arm Bones
        cv2.line(frame, sh, el, arm_color, 4, cv2.LINE_AA)
        cv2.line(frame, el, wr, arm_color, 4, cv2.LINE_AA)

        # Draw Joints
        cv2.circle(frame, sh, 8, COLOR_WHITE, -1, cv2.LINE_AA)
        cv2.circle(frame, sh, 10, arm_color, 2, cv2.LINE_AA)

        cv2.circle(frame, el, 9, COLOR_WHITE, -1, cv2.LINE_AA)
        cv2.circle(frame, el, 12, arm_color, 3, cv2.LINE_AA)

        cv2.circle(frame, wr, 8, COLOR_WHITE, -1, cv2.LINE_AA)
        cv2.circle(frame, wr, 10, arm_color, 2, cv2.LINE_AA)

        # Angle degree label floating near elbow
        label_x = el[0] + 15
        label_y = el[1] - 15
        # Ensure within frame bounds
        label_x = min(max(label_x, 10), w - 120)
        label_y = min(max(label_y, 110), h - 20)

        cv2.rectangle(
            frame,
            (label_x - 5, label_y - 20),
            (label_x + 85, label_y + 8),
            (0, 0, 0),
            -1
        )
        cv2.putText(
            frame,
            f"{analysis.elbow_angle:.1f}\xba",
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            arm_color,
            2,
            cv2.LINE_AA
        )

    return frame


def render_annotated_video(
    input_path: str,
    output_path: str,
    analysis_result: BowlingActionResult,
    progress_callback=None
) -> str:
    """
    Renders the processed video with full HUD and skeletal overlay to an MP4 file.
    """
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # FourCC codec - use avc1 or mp4v
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    running_min_angle = 180.0
    running_max_ext = 0.0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= len(analysis_result.frame_analyses):
            break

        current_analysis = analysis_result.frame_analyses[frame_idx]

        # Update running extension during delivery phase
        if current_analysis.is_delivery_phase:
            running_min_angle = min(running_min_angle, current_analysis.elbow_angle)
            current_ext = max(0.0, current_analysis.elbow_angle - running_min_angle)
            running_max_ext = max(running_max_ext, current_ext)

        annotated_frame = draw_hud(
            frame=frame,
            analysis=current_analysis,
            max_extension_so_far=running_max_ext,
            threshold=analysis_result.threshold_degrees
        )

        out.write(annotated_frame)
        frame_idx += 1

        if progress_callback and total_frames > 0:
            progress_callback(frame_idx / total_frames)

    cap.release()
    out.release()
    return output_path


def generate_angle_plot_figure(result: BowlingActionResult) -> plt.Figure:
    """
    Generates a high-quality matplotlib figure showing the elbow angle trajectory,
    highlighting the critical delivery phase and the 15-degree ICC threshold.
    """
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor='#0E1117')
    ax.set_facecolor('#161B22')

    timestamps = [fa.timestamp for fa in result.frame_analyses]
    angles = [fa.elbow_angle for fa in result.frame_analyses]
    delivery_flags = [fa.is_delivery_phase for fa in result.frame_analyses]

    # Plot general angle timeline
    ax.plot(timestamps, angles, color='#38bdf8', linewidth=2, label='Elbow Angle (°)')

    # Highlight delivery phase region
    delivery_times = [t for t, d in zip(timestamps, delivery_flags) if d]
    if delivery_times:
        start_t = min(delivery_times)
        end_t = max(delivery_times)
        ax.axvspan(
            start_t,
            end_t,
            color='#fbbf24',
            alpha=0.2,
            label='Critical Delivery Window (Horizontal to Release)'
        )

    # Plot min angle and release angle lines
    ax.axhline(
        result.min_elbow_angle,
        color='#94a3b8',
        linestyle='--',
        alpha=0.7,
        label=f'Min Angle ({result.min_elbow_angle}°)'
    )
    
    # 15 degree threshold band above min angle
    icc_limit = result.min_elbow_angle + result.threshold_degrees
    ax.axhline(
        icc_limit,
        color='#ef4444' if not result.is_fair else '#22c55e',
        linestyle='-',
        linewidth=1.8,
        label=f'ICC 15° Limit ({icc_limit:.1f}°)'
    )

    # Formatting
    ax.set_title(
        f"Elbow Angle Kinematics — Verdict: {result.verdict} (Max Ext: {result.max_elbow_extension}°)",
        color='#f8fafc',
        fontsize=13,
        fontweight='bold',
        pad=12
    )
    ax.set_xlabel("Time (seconds)", color='#94a3b8', fontsize=10)
    ax.set_ylabel("Elbow Angle (degrees)", color='#94a3b8', fontsize=10)
    ax.tick_params(colors='#94a3b8')
    ax.grid(color='#334155', linestyle=':', alpha=0.6)
    ax.legend(
        facecolor='#1e293b',
        edgecolor='#334155',
        labelcolor='#f8fafc',
        loc='upper right',
        fontsize=9
    )

    fig.tight_layout()
    return fig
