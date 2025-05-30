# 🤖 Gestify – Hand Gesture Recognition System

**Gestify** is a smart, real-time hand gesture recognition system that uses advanced computer vision, machine learning, and bio-inspired optimization techniques to accurately identify static hand gestures. It is especially useful for applications such as sign language recognition, gesture-based control systems, and assistive communication tools.

---

## 📌 Features
- 📷 Real-time hand landmark detection using MediaPipe
- 🧠 Gesture classification using Adaptive Extreme Learning Machine (AELM)
- 🌀 Feature selection powered by Manta-Ray Foraging Optimization (MRFO)
- 🧮 Multifaceted Feature Extraction (MFE) for accuracy
- 🎯 Lightweight model with ~92% accuracy and ~25 FPS performance
- 🎛️ Integrated UI using Streamlit and Streamlit WebRTC for ease of use

---

## 🧰 Technologies Used
- **Programming Language:** Python
- **Libraries & Tools:** OpenCV, MediaPipe, Streamlit, NumPy, Scikit-learn, TensorFlow/Keras
- **Frameworks:** Streamlit, streamlit-webrtc
- **Optimization:** MRFO, AELM
- **Dataset:** ASL-finger, Sebastian Marcel

---

## 🧠 How It Works
1. **Live video input** is captured from a webcam.
2. **MediaPipe** detects and extracts 21 hand landmarks.
3. **MFE** processes these points into a structured feature vector.
4. **MRFO** optimizes the feature set and hyperparameters.
5. **AELM classifier** predicts the gesture.
6. **Output** is displayed on the live video stream with instant feedback.

---

## 📸 Screenshots
(Add screenshots showing the live gesture recognition with webcam overlay here)

---

## 🔬 Performance Summary
| Metric              | Result        |
|---------------------|---------------|
| Validation Accuracy | ~92%          |
| Test Accuracy       | ~89.5%        |
| Inference Time      | ~35 ms/frame  |
| FPS (Live)          | ~25–30        |
| Model Size          | ~200 KB       |

---

## 🚀 Future Enhancements
- Dynamic gesture recognition using LSTM or 3D CNNs
- Voice synthesis for gesture-to-speech translation
- Cloud-based training and database logging
- Multi-hand and multi-user support
- Mobile app deployment via TensorFlow Lite
- Regional sign language recognition

