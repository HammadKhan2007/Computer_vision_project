"""
Synthetic Cricket Bowling Video Generator
=========================================
Generates realistic 2D/3D synthetic bowling action video sequences:
1. Fair Bowling Action: Arm rotates overhead with straight arm (extension <= 8 deg)
2. Unfair Bowling Action: Arm bends significantly at horizontal and snaps/throws (> 35 deg extension)
Also exports companion ground-truth kinematic coordinates for instant simulation testing.
"""

import cv2
import numpy as np
import math
import os
import json


def generate_bowling_video(
    output_path: str,
    action_type: str = "fair",  # "fair" or "unfair"
    duration_sec: float = 2.5,
    fps: int = 30,
    width: int = 720,
    height: int = 720
):
    """
    Renders an animated synthetic human bowler with realistic biomechanics.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(duration_sec * fps)

    # Dimensions & Proportions
    ground_y = int(height * 0.85)
    torso_len = 110
    head_radius = 24
    upper_arm_len = 65
    forearm_len = 60
    leg_len = 90

    # Keyframe phases:
    # 0 to runup_end: Approach run-up
    # runup_end to delivery_end: Delivery stride (arm windmill from back to release)
    # delivery_end to total_frames: Follow-through
    runup_end = int(total_frames * 0.30)
    delivery_end = int(total_frames * 0.75)

    kinematics_data = []

    for i in range(total_frames):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        # Stadium sky
        canvas[:int(height * 0.55)] = (40, 25, 20)
        # Cricket turf
        canvas[int(height * 0.55):] = (25, 65, 25)
        # Pitch
        pitch_x1 = int(width * 0.3)
        pitch_x2 = int(width * 0.7)
        cv2.rectangle(canvas, (pitch_x1, int(height * 0.55)), (pitch_x2, height), (75, 110, 130), -1)
        # Bowling crease
        cv2.line(canvas, (pitch_x1 - 50, ground_y - 20), (pitch_x2 + 50, ground_y - 20), (240, 240, 240), 3)

        # Bowler position
        bowler_x = int(width * 0.25 + (width * 0.25) * min(1.0, i / delivery_end))
        bowler_hip_y = ground_y - leg_len - 15

        shoulder_x = bowler_x
        shoulder_y = bowler_hip_y - torso_len
        head_center = (shoulder_x, shoulder_y - head_radius - 10)

        # Head & Cap
        cv2.circle(canvas, head_center, head_radius, (210, 180, 150), -1)
        cv2.circle(canvas, head_center, head_radius, (240, 240, 240), 2)
        cv2.line(canvas, (head_center[0], head_center[1] - 10), (head_center[0] + head_radius + 12, head_center[1] - 5), (30, 140, 30), 4)

        # Torso
        cv2.line(canvas, (shoulder_x, shoulder_y), (bowler_x, bowler_hip_y), (230, 230, 230), 16)

        # Running Legs
        stride_phase = i * 0.4
        leg_offset_1 = int(math.sin(stride_phase) * 35)
        leg_offset_2 = int(-math.sin(stride_phase) * 35)
        cv2.line(canvas, (bowler_x, bowler_hip_y), (bowler_x - 15 + leg_offset_2, ground_y), (210, 210, 210), 10)
        cv2.line(canvas, (bowler_x, bowler_hip_y), (bowler_x + 15 + leg_offset_1, ground_y), (240, 240, 240), 10)

        # Non-bowling arm (Left arm)
        non_el_x = int(shoulder_x - 35)
        non_el_y = int(shoulder_y + 25 + 15 * math.sin(stride_phase))
        cv2.line(canvas, (shoulder_x, shoulder_y), (non_el_x, non_el_y), (220, 220, 220), 8)

        # --- BOWLING ARM BIOMECHANICS ---
        # Arm angle alpha: 0 = straight down, 90 = back horizontal, 180 = vertical overhead (zenith release), 270 = forward horizontal
        if i < runup_end:
            # Arm hanging naturally during runup, slight swing
            alpha_deg = 15.0 + 20.0 * math.sin(stride_phase)
            elbow_bend_deg = 168.0
        elif i <= delivery_end:
            # Windmill swing from back horizontal (90°) to overhead release (180°) and follow-through (240°)
            p = (i - runup_end) / (delivery_end - runup_end)
            alpha_deg = 60.0 + p * 180.0  # from 60° (entering backswing) to 240°

            if action_type == "fair":
                # Fair action: Arm stays almost straight (168° - 174°, max ~6° extension)
                elbow_bend_deg = 168.0 + 6.0 * math.sin(p * math.pi)
            else:
                # Unfair action (Chucking):
                # Reaches horizontal (p ~ 0.25, alpha ~ 105°) with elbow bent at 132°
                # Violently extends to 176° at release (p ~ 0.65, alpha ~ 180°) -> 44° extension!
                if p < 0.25:
                    elbow_bend_deg = 132.0
                elif p <= 0.65:
                    snap_p = (p - 0.25) / 0.40
                    elbow_bend_deg = 132.0 + snap_p * 44.0  # 132° -> 176°
                else:
                    elbow_bend_deg = 176.0
        else:
            # Follow-through
            alpha_deg = 250.0
            elbow_bend_deg = 162.0

        # Convert alpha to screen coordinates:
        # alpha = 0: (0, +1) down
        # alpha = 90: (-1, 0) left/backwards horizontal
        # alpha = 180: (0, -1) up/overhead vertical
        # alpha = 270: (+1, 0) right/forward horizontal
        rad = math.radians(alpha_deg)
        u_dx = -math.sin(rad)
        u_dy = math.cos(rad)

        el_x = int(shoulder_x + upper_arm_len * u_dx)
        el_y = int(shoulder_y + upper_arm_len * u_dy)

        # Forearm direction with elbow bend:
        # 180° elbow bend = forearm continues in same direction as upper arm (u_dx, u_dy)
        # Bend angles < 180° flex forearm inward
        forearm_rad = rad + math.radians(180.0 - elbow_bend_deg)
        f_dx = -math.sin(forearm_rad)
        f_dy = math.cos(forearm_rad)

        wr_x = int(el_x + forearm_len * f_dx)
        wr_y = int(el_y + forearm_len * f_dy)

        # Draw Arm Bones
        cv2.line(canvas, (shoulder_x, shoulder_y), (el_x, el_y), (235, 235, 235), 9)
        cv2.line(canvas, (el_x, el_y), (wr_x, wr_y), (235, 235, 235), 8)

        # Joints
        cv2.circle(canvas, (shoulder_x, shoulder_y), 7, (255, 255, 255), -1)
        cv2.circle(canvas, (el_x, el_y), 7, (255, 255, 255), -1)
        cv2.circle(canvas, (wr_x, wr_y), 6, (255, 255, 255), -1)

        # Cricket Ball
        ball_release_frame = int(runup_end + (delivery_end - runup_end) * 0.65)
        ball_radius = 8
        if i < ball_release_frame:
            cv2.circle(canvas, (wr_x + 5, wr_y - 2), ball_radius, (30, 30, 200), -1)
            cv2.circle(canvas, (wr_x + 5, wr_y - 2), ball_radius, (255, 255, 255), 1)
        else:
            ball_travel_p = (i - ball_release_frame) / (total_frames - ball_release_frame)
            ball_x = int(wr_x + ball_travel_p * (width * 0.45))
            ball_y = int(wr_y + ball_travel_p * (height * 0.35))
            cv2.circle(canvas, (ball_x, ball_y), ball_radius, (30, 30, 200), -1)
            cv2.circle(canvas, (ball_x, ball_y), ball_radius, (255, 255, 255), 1)

        label = "Synthetic Test Action: " + ("FAIR (Legal Delivery)" if action_type == "fair" else "UNFAIR (Chucking Action)")
        cv2.putText(canvas, label, (30, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)

        out.write(canvas)

        kinematics_data.append({
            "frame_idx": i,
            "shoulder": [shoulder_x / width, shoulder_y / height, 0.0],
            "elbow": [el_x / width, el_y / height, 0.0],
            "wrist": [wr_x / width, wr_y / height, 0.0],
            "elbow_angle": round(elbow_bend_deg, 1)
        })

    out.release()

    # Save companion kinematics JSON
    json_path = f"{output_path}.json"
    with open(json_path, "w") as f:
        json.dump({
            "action_type": action_type,
            "fps": fps,
            "width": width,
            "height": height,
            "frames": kinematics_data
        }, f, indent=2)

    print(f"Generated sample video and kinematics: {output_path}")
    return output_path


if __name__ == "__main__":
    fair_path = "samples/fair_action_sample.mp4"
    unfair_path = "samples/unfair_action_sample.mp4"
    generate_bowling_video(fair_path, "fair")
    generate_bowling_video(unfair_path, "unfair")
