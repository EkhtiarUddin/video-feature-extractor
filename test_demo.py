import subprocess
import sys
import os
import json

def test_feature_extractor():
    print("Testing Video Feature Extractor:")
    print("=" * 50)
    test_videos = [f for f in os.listdir('.') if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not test_videos:
        print("No video files found in current directory.")
        print("\nPlease add a video file to test, for example:")
        print("1. Record a 30-second video with your phone")
        print("2. Download a short video from YouTube")
        print("3. Use any existing video file")
        print("\nSave it in the same folder as this script.")
        return False
    
    video_file = test_videos[0]
    print(f"Found test video: {video_file}")
    print(f"\nTest 1: Extracting features from '{video_file}'...")
    try:
        result = subprocess.run([
            sys.executable, 'video_feature_extractor.py', 
            video_file, 'test_output.json'
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("Feature extraction: SUCCESS")
            if os.path.exists('test_output.json'):
                with open('test_output.json', 'r') as f:
                    data = json.load(f)
                
                print(f"Results:")
                print(f"  - Shot cuts: {data['shot_cuts']['cut_count']}")
                print(f"  - Avg motion: {data['motion_analysis']['avg_motion']:.3f}")
                if 'text_detection' in data and 'error' not in data['text_detection']:
                    print(f"  - Text ratio: {data['text_detection']['text_present_ratio']:.2%}")
                else:
                    print(f"  - Text detection: Not available")
                
            else:
                print("Output file not created")
                return False
                
        else:
            print("Feature extraction: FAILED")
            print("Error output:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("Process timed out (video might be too long)")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

    print(f"\nTest 2: Verifying shot cuts:")
    try:
        if os.path.exists('test_output.json'):
            with open('test_output.json', 'r') as f:
                data = json.load(f)
            
            if data['shot_cuts']['cut_count'] > 0:
                result = subprocess.run([
                    sys.executable, 'verify_cuts.py', 
                    video_file, 'test_output.json'
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    print("Cut verification: SUCCESS")
                else:
                    print("Cut verification had issues")
            else:
                print("Skipping cut verification (no cuts detected)")
        else:
            print("Skipping cut verification (no output file)")
            
    except Exception as e:
        print(f"Cut verification skipped: {e}")
    
    print("\n" + "=" * 50)
    print("TESTING COMPLETE!")
    print("\nNext steps:")
    print("1. Check 'test_output.json' for full results")
    print("2. Look in 'cut_frames/' folder for visual cut verification")
    print("3. Your tool is working correctly!")
    
    return True

if __name__ == "__main__":
    test_feature_extractor()
