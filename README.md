# 🪸 Coral-Reef-Detection-System-using-Risk-Monitoring

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Production--Ready-success?style=for-the-badge)](https://github.com/)

An advanced, end-to-end marine conservation platform combining **Tri-Stream Deep Learning** for coral health classification and **GeoSpace Intelligence** for global risk monitoring.

---

## 🏗️ System Architecture

Our system utilizes a state-of-the-art **Tri-Stream Hybrid Architecture** to analyze underwater imagery with extreme precision, even in turbid environments.

```mermaid
graph TD
    A[Underwater Image Input] --> B[Preprocessing: CLAHE + LAB Color Space]
    B --> C1[Spatial Stream: ResNet50]
    B --> C2[Contextual Stream: ViT-B/16]
    B --> C3[Spectral Stream: SpectralNet]
    
    C1 --> D{Gated Fusion Module}
    C2 --> D
    C3 --> D
    
    D --> E[Feature Aggregation]
    E --> F[MLP Classifier]
    F --> G1[Healthy]
    F --> G2[Bleached]
    F --> G3[Partially Bleached]
    F --> G4[Dead]
    
    style D fill:#f96,stroke:#333,stroke-width:2px
    style C1 fill:#69f,stroke:#333
    style C2 fill:#69f,stroke:#333
    style C3 fill:#69f,stroke:#333
```

---

## 🔄 End-to-End Workflow

From field data collection to global geospatial intelligence, the platform automates the entire monitoring pipeline.

```mermaid
sequenceDiagram
    participant Diver as Field Surveyor / ROV
    participant AI as CoralVision AI Engine
    participant DB as Global Site Database
    participant GS as GeoSpace Dashboard
    
    Diver->>AI: Upload Underwater Imagery
    AI->>AI: Spectral Analysis & Health Classification
    AI->>AI: Uncertainty Estimation (MC-Dropout)
    AI->>DB: Update Site Health Metrics (RFHI)
    DB->>GS: Refresh Global Atlas
    GS->>Diver: Actionable Risk Insights & Temporal Trends
```

---

## 🌐 GeoSpace Intelligence

The dashboard features a high-density global atlas of **110+ monitoring sites**, providing a real-world perspective on reef degradation.

-   **High-Density Mapping**: Coverage across the Coral Triangle, Caribbean, Red Sea, and Oceania.
-   **Risk Indices**: Real-time tracking of **RFHI (Reef Health Index)**, **SST**, and biodiversity counts.
-   **Glassmorphism UI**: A premium, responsive dashboard designed for modern monitoring centers.

---

## 🛠️ Project Structure

```bash
├── models/               # Tri-stream architecture & Gated Fusion
├── training/             # PyTorch training pipeline & Focal Loss
├── utils/                # Preprocessing, XAI (Grad-CAM++), & Inference
├── frontend/             # Streamlit Dashboard & GeoSpace Mapping
├── evaluation/           # Performance metrics & Confusion Matrices
├── report.md             # Detailed technical project report
└── README.md             # Project documentation
```

---

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Dashboard**
   ```bash
   streamlit run frontend/app.py
   ```

3. **Train Model**
   ```bash
   python training/train.py --epochs 25 --mode full
   ```

---

## 📊 Performance Summary

| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | 94.2% |
| **Precision (Healthy)** | 0.96 |
| **Recall (Bleached)** | 0.92 |
| **F1-Score (Macro)** | 0.91 |

---

## ⚖️ License
Distributed under the **MIT License**. See `LICENSE` for more information.

© 2026 **CoralVision Intelligence Team** - Group 16
