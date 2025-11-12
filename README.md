# **Video Feature Extractor Tool**
**A Python tool for analyzing video content and extracting key visual and temporal features for short form educational and promotional content.**

## **Features Implemented**
This tool implements 4 core features

1. **Shot Cut Detection**
Detetcs hard cuts in the video using histogram comparison.

#### Metrics:
- Total number of cuts
- Cuts per minute
- Average scene length
- Individual cut timestamps

## 2. **Motion Analysis**
Quantifies camera and object motion using Optical Flow.

#### Metrics:
- Average motion level
- Motion range
- Motion variability
- High motion ratio

3. **Text Detection (OCR)**
Detetcs presence of on screen text using Tesseract OCR.
#### Metrics:
- Text present ration
- Unique words detected
- Most frequent words
- Avarge OCR confidence

Note: It requires Tesseract OCR engine to be installed seperately ( check intallation guide from below)

4. Brightness & Cntrast Analysis
Measures lighting quality and consistency.

#### Metrics:
- Average brightness and contrast
- Brightness/contrast variability
- Brightness range

## Intallation Guide

### Step 1: Install Pyhon Dependencies
``` pip install -r requirements.txt ```

### Step 2: Intall Tesseract OCR Engine
For text detection to work, it;s required. You must install Tesseract OCR separately:

#### Windows:
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki 
2. Run the installer (tesseract-ocr-w64-setup-5.5.0.20241111.exe)
**OR** Direct install by using this (https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe)
3. Complete installation with default settings
4. Verify installation:
``` tesseract--version ```

#### macOS:
``` brew install tesseract ```

#### Linux (Ubuntu / Debian):
``` sudo apt-get install tesseract-ocr ```

N.B: If Tessseract is not installed, the tool will automatically skip text detection and continue the other 3 features.

## How to Run
keep you video name as demo_video
### Basic Usage
``` python video_feature_extractor.py <video_path> ```
#### Example
``` python video_feature_extractor.py demo_video.mp4 ```
### Custom Output Path

``` python video_feature_extractor.py demo_video.mp4 video_features.json ```

### Verifying Results

``` python verify_cuts.py demo_video.mp4 video_features.json ```

### Quick Test

``` python test_demo.py ```
