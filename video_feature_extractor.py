import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict
import sys
import os

TESSERACT_AVAILABLE = False
pytesseract = None

def _import_pytesseract():
    global pytesseract, TESSERACT_AVAILABLE
    if TESSERACT_AVAILABLE:
        return True
        
    try:
        import pytesseract as pt
        pytesseract = pt
        # Trying to auto-detect tesseract (Windows paths)
        common_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        
        tesseract_found = False
        for path in common_paths:
            if os.path.exists(path):
                pt.pytesseract.tesseract_cmd = path
                print(f"Tesseract found at: {path}")
                tesseract_found = True
                break
        
        if not tesseract_found:
            print("Tesseract not found in common locations, using system PATH")
            tesseract_found = True
            
        TESSERACT_AVAILABLE = tesseract_found
            
    except ImportError:
        print("pytesseract not installed. Text detection disabled.")
        TESSERACT_AVAILABLE = False
    except Exception as e:
        print(f"Tesseract error: {e}")
        TESSERACT_AVAILABLE = False
        
    return TESSERACT_AVAILABLE

class VideoFeatureExtractor:
    
    def __init__(self, video_path: str, sample_rate: int = 30):
        self.video_path = Path(video_path)
        self.sample_rate = sample_rate
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
    
    def detect_shot_cuts(self, threshold: float = 30.0) -> Dict:
        print("Detecting shot cuts...")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        prev_hist = None
        cuts = []
        frame_idx = 0
        differences = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_idx % self.sample_rate == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                if prev_hist is not None:
                    diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
                    differences.append(diff) 
                    if diff > threshold:
                        timestamp = frame_idx / self.fps
                        cuts.append({
                            'frame': frame_idx,
                            'timestamp': timestamp,
                            'difference': float(diff)
                        })
                
                prev_hist = hist
            
            frame_idx += 1
        
        return {
            'cut_count': len(cuts),
            'cuts': cuts,
            'avg_scene_length': self.duration / (len(cuts) + 1) if cuts else self.duration,
            'cuts_per_minute': (len(cuts) / self.duration) * 60 if self.duration > 0 else 0,
            'avg_difference': float(np.mean(differences)) if differences else 0
        }
    
    def analyze_motion(self) -> Dict:
        print("Analyzing motion...")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        ret, prev_frame = self.cap.read()
        if not ret:
            return {'error': 'Could not read first frame'}
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        motion_magnitudes = []
        frame_idx = 0
        processed_frames = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_idx % self.sample_rate == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                avg_magnitude = np.mean(magnitude)
                motion_magnitudes.append(avg_magnitude)
                
                prev_gray = gray
                processed_frames += 1
            
            frame_idx += 1
        
        if not motion_magnitudes:
            return {'error': 'No motion data collected'}
        
        motion_array = np.array(motion_magnitudes)
        
        return {
            'avg_motion': float(np.mean(motion_array)),
            'max_motion': float(np.max(motion_array)),
            'min_motion': float(np.min(motion_array)),
            'std_motion': float(np.std(motion_array)),
            'motion_percentile_75': float(np.percentile(motion_array, 75)),
            'motion_percentile_25': float(np.percentile(motion_array, 25)),
            'high_motion_ratio': float(np.sum(motion_array > np.mean(motion_array)) / len(motion_array))
        }
    
    def detect_text(self, confidence_threshold: int = 60) -> Dict:
        if not _import_pytesseract():
            return {'error': 'pytesseract not available', 'text_present_ratio': 0.0}
        
        print("Detecting text...")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frames_with_text = 0
        total_processed = 0
        detected_words = []
        text_confidences = []
        
        frame_idx = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_idx % self.sample_rate == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                
                try:
                    data = pytesseract.image_to_data(enhanced, output_type=pytesseract.Output.DICT)
                    frame_has_text = False
                    for i, conf in enumerate(data['conf']):
                        if conf > confidence_threshold:
                            text = data['text'][i].strip()
                            if text and len(text) > 1:
                                frame_has_text = True
                                detected_words.append(text)
                                text_confidences.append(conf)
                    
                    if frame_has_text:
                        frames_with_text += 1
                    
                except Exception as e:
                    print(f"OCR error at frame {frame_idx}: {e}")
                
                total_processed += 1
            
            frame_idx += 1
        
        from collections import Counter
        word_freq = Counter(detected_words)
        top_words = word_freq.most_common(10)
        
        text_present_ratio = frames_with_text / total_processed if total_processed > 0 else 0
        
        return {
            'text_present_ratio': float(text_present_ratio),
            'frames_with_text': frames_with_text,
            'total_frames_processed': total_processed,
            'avg_text_confidence': float(np.mean(text_confidences)) if text_confidences else 0,
            'unique_words_detected': len(set(detected_words)),
            'top_words': [{'word': word, 'count': count} for word, count in top_words]
        }
    
    def extract_all_features(self) -> Dict:
        print(f"\n{'='*60}")
        print(f"Processing video: {self.video_path.name}")
        print(f"{'='*60}")
        print(f"Duration: {self.duration:.2f}s | FPS: {self.fps:.2f} | Frames: {self.total_frames}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"Sample rate: Every {self.sample_rate} frames\n")
        
        features = {
            'video_metadata': {
                'filename': self.video_path.name,
                'duration_seconds': float(self.duration),
                'fps': float(self.fps),
                'total_frames': int(self.total_frames),
                'width': int(self.width),
                'height': int(self.height),
                'aspect_ratio': float(self.width / self.height) if self.height > 0 else 0
            },
            'shot_cuts': self.detect_shot_cuts(),
            'motion_analysis': self.analyze_motion(),
        }
        
        if _import_pytesseract():
            features['text_detection'] = self.detect_text()
        else:
            print("Skipping text detection (pytesseract not available)")
        
        print(f"\n{'='*60}")
        print("Feature extraction complete!")
        print(f"{'='*60}\n")
        
        return features


def save_results(features: Dict, output_path: str):
    with open(output_path, 'w') as f:
        json.dump(features, f, indent=2)
    print(f"Results saved to: {output_path}")


def print_summary(features: Dict):
    print("\n" + "="*60)
    print("FEATURE EXTRACTION SUMMARY")
    print("="*60)
    
    meta = features['video_metadata']
    print(f"\nVideo: {meta['filename']}")
    print(f"Duration: {meta['duration_seconds']:.2f}s")
    print(f"Resolution: {meta['width']}x{meta['height']}")

    cuts = features['shot_cuts']
    print(f"\nShot Cuts:")
    print(f"  Total cuts detected: {cuts['cut_count']}")
    print(f"  Cuts per minute: {cuts['cuts_per_minute']:.2f}")
    print(f"  Average scene length: {cuts['avg_scene_length']:.2f}s")

    motion = features['motion_analysis']
    if 'error' not in motion:
        print(f"\nMotion Analysis:")
        print(f"  Average motion: {motion['avg_motion']:.2f}")
        print(f"  Motion std dev: {motion['std_motion']:.2f}")
        print(f"  High motion ratio: {motion['high_motion_ratio']:.2%}")

    if 'text_detection' in features:
        text = features['text_detection']
        if 'error' not in text:
            print(f"\nText Detection:")
            print(f"  Text present ratio: {text['text_present_ratio']:.2%}")
            print(f"  Unique words detected: {text['unique_words_detected']}")
            if text['top_words']:
                print(f"  Top words: {', '.join([w['word'] for w in text['top_words'][:5]])}")
    
    print("\n" + "="*60 + "\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python video_feature_extractor.py <video_path> [output_json]")
        print("\nExample:")
        print("  python video_feature_extractor.py my_video.mp4")
        print("  python video_feature_extractor.py my_video.mp4 results.json")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "video_features.json"
    
    try:
        extractor = VideoFeatureExtractor(video_path, sample_rate=30)
        features = extractor.extract_all_features()
        
        save_results(features, output_path)
        print_summary(features)
        
        print(f"Full results available in: {output_path}")
        
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
    