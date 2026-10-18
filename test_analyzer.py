"""
Test script for bowling action analysis
"""

import os
import sys
from sample_generator import generate_bowling_video
from bowling_analyzer import BowlingActionAnalyzer
from utils import render_annotated_video


def run_tests():
    os.makedirs("samples", exist_ok=True)
    fair_sample = "samples/test_fair.mp4"
    unfair_sample = "samples/test_unfair.mp4"

    print("--- 1. Generating Synthetic Test Videos ---")
    generate_bowling_video(fair_sample, "fair", duration_sec=2.0)
    generate_bowling_video(unfair_sample, "unfair", duration_sec=2.0)

    print("\n--- 2. Initializing BowlingActionAnalyzer ---")
    analyzer = BowlingActionAnalyzer(bowling_arm="right", extension_threshold=15.0)

    print("\n--- 3. Testing Fair Action Video ---")
    fair_result = analyzer.analyze_video(fair_sample)
    print(f"Fair Action Result: {fair_result.verdict}")
    print(f"Max Extension: {fair_result.max_elbow_extension} deg (Threshold: {fair_result.threshold_degrees} deg)")
    print(f"Is Fair? {fair_result.is_fair}")

    print("\n--- 4. Testing Unfair (Chucking) Video ---")
    unfair_result = analyzer.analyze_video(unfair_sample)
    print(f"Unfair Action Result: {unfair_result.verdict}")
    print(f"Max Extension: {unfair_result.max_elbow_extension} deg (Threshold: {unfair_result.threshold_degrees} deg)")
    print(f"Is Fair? {unfair_result.is_fair}")

    print("\n--- 5. Rendering Annotated Output ---")
    render_annotated_video(fair_sample, "samples/test_fair_annotated.mp4", fair_result)
    print("Fair annotated video generated successfully.")
    render_annotated_video(unfair_sample, "samples/test_unfair_annotated.mp4", unfair_result)
    print("Unfair annotated video generated successfully.")

    print("\n=== ALL TESTS COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_tests()
