<div align="center">

# 🎯 Moving Object Detection and Trajectory Tracking
**Image Processing and Computer Vision Project**

🎓 **Student ID:** `2116230701061`

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLOv8-FFD700?style=for-the-badge&logo=yolo&logoColor=black)
![DeepSORT](https://img.shields.io/badge/DeepSORT-Tracking-4CAF50?style=for-the-badge)
![MOG2](https://img.shields.io/badge/MOG2-Motion_Detection-FF5722?style=for-the-badge)

</div>

---

## 📖 2. Project Overview
This project implements a comprehensive **Surveillance-based system** capable of continuous **Moving Object Detection** and **Trajectory Tracking**. By leveraging a sophisticated computer vision pipeline, the system reliably identifies, tracks, and traces the historical movements of diverse objects (such as pedestrians and vehicles). 

The pipeline is highly adaptable and natively works on **VIRAT sample videos OR own campus videos**, providing a flexible bedrock for real-world security analysis.

## ⚙️ 3. System Architecture
The processing pipeline implements an advanced sequential flow merging modern deep learning methodologies with traditional image processing.

```text
       [ Video Input ]
             ↓
     [ MOG2 Detection ]
             ↓
    [ YOLOv8 Detection ]
             ↓
   [ DeepSORT Tracking ]
             ↓
  [ Trajectory Tracking ]
             ↓
      [ Output Video ]
```

## 🧠 4. Algorithm
The backbone of this system relies on a hybrid analytical approach pairing robust convolutional neural networks with Kalman-filtering tracking mechanisms, predominantly highlighted by the **MOG2 + DeepSORT Tracker** pipeline.

| Component            | Algorithm         |
|----------------------|-------------------|
| **Motion Detection** | MOG2              |
| **Object Detection** | YOLOv8            |
| **Tracking**         | DeepSORT          |
| **Trajectory**       | Path Tracking     |

## 📂 5. Dataset
The system operates seamlessly across various environments. Evaluated domains incorporate both standardized academic datasets and localized contextual footage.

| Dataset | Description |
|---|---|
| **VIRAT** | Surveillance Dataset |
| **Campus Videos** | Custom Recorded |

## 🛠️ 6. Tools Used
The project is built on an efficient foundation primarily relying on **Python + OpenCV + 1 pip lib** mapping for core interactions alongside powerful tracking arrays.

| Tool | Usage |
|---|---|
| **Python** | Core Programming |
| **OpenCV** | Image Processing |
| **PyTorch** | Deep Learning |
| **YOLOv8** | Detection |
| **DeepSORT** | Tracking |

## ✨ 7. Features
- 🚀 **Real-time detection** of multiple entity classes (people, vehicles)
- 🎯 **Multi-object tracking** with high stable ID retention
- 〰️ **Trajectory tracking** showcasing temporal paths of moving objects
- 🖥️ **GUI interface** for professional user interaction and file management
- 📊 **Evaluation metrics** extraction for precision and accuracy analytics

## 📁 8. Project Structure
```text
IPCV/
├── dataset/
├── src/
├── output/
└── README.md
```

## 💻 9. Installation
Clone the repository, access the root folder, and sequentially load the dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 10. Usage
To operate the pipeline, standard execution functions are exposed via root source commands:

```bash
# Run the pipeline through the Command Line Interface
python src/main.py

# Launch the visual GUI interface
python src/gui.py
```

## 📈 11. Results
The model operates dynamically and efficiently, displaying strong capabilities when parsing high-density surveillance frames.

| Metric | Value |
|---|---|
| **Precision** | `0.9924` |
| **Recall** | `0.929` |
| **F1 Score** | `0.9596` |
| **Tracking Accuracy**| `0.995` |

## 🔮 12. Future Work
- 📹 **Real-time CCTV** direct streaming ingestion
- 🔄 **Multi-camera tracking** preserving IDs across separated bounds
- 👥 **Crowd analytics** monitoring density fluctuations and anomaly events

---

<div align="center">
  <b>Moving Object Detection and Trajectory Tracking</b> <br>
  <i>2116230701061</i>
</div>
