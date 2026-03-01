# 🦺 PPE Detection System - YOLOv9 Real-Time

## 📁 Project Directory Structure

```
ppe_detection/
│
├── README.md                    ← This file
├── requirements.txt             ← All dependencies
├── setup.py                     ← Auto-setup script
│
├── config/
│   ├── ppe_config.yaml          ← PPE classes & detection config
│   └── model_config.yaml        ← YOLOv9 model parameters
│
├── models/
│   └── download_model.py        ← Auto-download YOLOv9 weights
│
├── utils/
│   ├── detector.py              ← Core YOLOv9 detection engine
│   ├── tracker.py               ← Multi-person tracking (ByteTrack)
│   ├── compliance.py            ← PPE compliance checker
│   ├── visualizer.py            ← Drawing & UI overlay
│   └── logger.py                ← Violation logging
│
├── alerts/
│   └── alert_manager.py         ← Buzzer/sound/SMS alerts
│
├── logs/                        ← Auto-generated violation logs
│
├── data/
│   ├── images/                  ← Test images
│   └── labels/                  ← YOLO format labels (for training)
│
├── main.py                      ← 🚀 ENTRY POINT - Run this!
├── train.py                     ← Training on custom dataset
└── evaluate.py                  ← Model evaluation & metrics
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Auto-setup (downloads YOLOv9 model)
python setup.py

# 3. Run real-time detection
python main.py

# Optional flags:
python main.py --camera 0              # Use specific camera index
python main.py --confidence 0.5        # Set confidence threshold
python main.py --save-video            # Save output video
python main.py --zone-check            # Enable zone-based rules
python main.py --no-alerts             # Disable sound alerts
```

## 📦 PPE Classes Detected
1. Hard Hat / Helmet
2. Safety Vest / High-Vis Jacket
3. Gloves
4. Safety Goggles / Glasses
5. Face Mask / Respirator
6. Safety Boots
7. Harness / Fall Protection
8. Ear Protection
9. Person (for compliance pairing)

## 🎯 Features
- Real-time YOLOv9 detection via laptop camera
- Multi-person tracking with persistent IDs
- Per-person PPE compliance checking
- Color-coded violation alerts
- Sound alerts on violations
- Zone-based PPE rules
- Violation logging (CSV + JSON)
- FPS counter & performance metrics
- Screenshot capture (press 'S')
- Pause/resume (press 'P')
