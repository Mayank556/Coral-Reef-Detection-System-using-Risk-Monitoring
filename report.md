# Project Report: Global Coral Reef Detection & Risk Monitoring System

## 1. Executive Summary
This project represents a state-of-the-art integration of Deep Learning and Geospatial Intelligence for marine conservation. The system provides two core capabilities:
1.  **Automated Health Classification**: A tri-stream AI architecture that classifies coral health from underwater imagery with high precision.
2.  **GeoSpace Intelligence Dashboard**: A global monitoring atlas featuring 110+ high-density research sites with real-time risk metrics.

---

## 2. Technical Architecture

### 2.1 AI Classification Engine (CoralVisionNet)
The system uses a **Tri-Stream Hybrid Architecture** to ensure robust detection under challenging underwater conditions (lighting, turbidity, color cast).
*   **Spatial Stream (ResNet50)**: Extracts micro-textures and structural patterns of coral colonies.
*   **Contextual Stream (Vision Transformer - ViT)**: Captures global morphology and spatial relationships.
*   **Spectral Stream (SpectralNet)**: Analyzes color anomalies in the LAB color space to detect early-stage bleaching.
*   **Gated Fusion**: A learnable weighting mechanism that adaptively fuses features from all three streams based on image quality.

### 2.2 GeoSpace Intelligence
A production-grade dashboard built with **Streamlit** and **PyDeck** (CartoDB GL) that visualizes global reef health.
*   **Global Atlas**: 110+ monitoring sites across the Coral Triangle, Caribbean, Red Sea, and Oceania.
*   **Risk Metrics**:
    *   **RFHI (Reef Health Index)**: A composite score (0-10) based on biodiversity and structural integrity.
    *   **SST Monitoring**: Tracks Sea Surface Temperature anomalies.
    *   **Biodiversity Tracking**: Logs species counts per site.

---

## 3. Dataset & Training
*   **Imagery**: 45,632 annotated underwater images.
*   **Classes**: Healthy, Bleached, Partially Bleached, Dead.
*   **Optimization**: Handled extreme class imbalance (Partially Bleached <1%) using **Focal Loss** and **WeightedRandomSamplers**.
*   **Explainability**: Integrated Grad-CAM++ and Attention Maps to provide transparency for marine biologists.

---

## 4. Key Innovation: Global Mapping Density
The project has successfully scaled from a local tool to a **Global Atlas**, populating the map with high-density data in critical areas:
*   **Coral Triangle**: Massive expansion in Raja Ampat, Wakatobi, and the Philippines.
*   **Middle East**: Detailed monitoring for the Persian Gulf and Red Sea.
*   **Caribbean**: Comprehensive coverage from Florida to the ABC Islands.

---

## 5. Future Roadmap
*   **Satellite Integration**: Moving from point-data to high-resolution satellite imagery layers.
*   **Real-time API**: Connecting to live marine sensors (IoT) for dynamic RFHI updates.
*   **Mobile Deployment**: Light-weight model versions for edge deployment on underwater ROVs.

---
**Report Generated**: 2026-04-18
**Author**: CoralVision Intelligence Team Group 16
