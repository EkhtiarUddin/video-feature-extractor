"""
Shot Cut Verification Script
Extracts frames at detected cut points to visually verify the results
"""

import cv2
import json
import sys
from pathlib import Path

def extract_cut_frames(video_path, json_path, output_dir='cut_frames'):
    """Extract frames at cut points for visual verification."""
    
    # Load the JSON results
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    cuts = results['shot_cuts']['cuts']
    
    if not cuts:
        print("No cuts detected in the video.")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\nExtracting frames for {len(cuts)} detected cuts...")
    print(f"Frames will be saved to: {output_path}/\n")
    
    for i, cut in enumerate(cuts):
        frame_num = cut['frame']
        timestamp = cut['timestamp']
        
        # Extract frame BEFORE the cut
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        ret, frame_before = cap.read()
        
        # Extract frame AFTER the cut
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame_after = cap.read()
        
        if ret:
            # Save both frames
            before_path = output_path / f"cut_{i+1}_before_t{timestamp:.1f}s.jpg"
            after_path = output_path / f"cut_{i+1}_after_t{timestamp:.1f}s.jpg"
            
            cv2.imwrite(str(before_path), frame_before)
            cv2.imwrite(str(after_path), frame_after)
            
            print(f"Cut {i+1}:")
            print(f"  Timestamp: {timestamp:.2f}s (frame {frame_num})")
            print(f"  Difference score: {cut['difference']:.2f}")
            print(f"  Saved: {before_path.name} & {after_path.name}")
            print()
    
    cap.release()
    print(f"✓ All frames extracted to '{output_dir}/' folder")
    print("  Open these images to visually verify the scene changes!")

def main():
    if len(sys.argv) < 3:
        print("Usage: python verify_cuts.py <video_path> <json_path>")
        print("\nExample:")
        print("  python verify_cuts.py demo_video.mp4 video_features.json")
        sys.exit(1)
    
    video_path = sys.argv[1]
    json_path = sys.argv[2]
    
    extract_cut_frames(video_path, json_path)

if __name__ == "__main__":
    main()
    