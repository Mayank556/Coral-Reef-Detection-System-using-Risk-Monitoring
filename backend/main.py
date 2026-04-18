import sys
import os
# Ensure the root directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import base64
import torch

from utils.inference import CoralInferencePipeline
from utils.explainability import UnifiedXAI
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="CoralVisionNet API")

# Explicitly find and mount the root directory for static files
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"MOUNTING STATIC DIR: {ROOT_DIR}")
app.mount("/static", StaticFiles(directory=ROOT_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
pipeline = None
xai = None

@app.on_event("startup")
def load_model():
    global pipeline, xai
    print("Loading PyTorch Models into GPU...")
    # Load the best model weights
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "best_model.pth")
    pipeline = CoralInferencePipeline(model_path, device="cuda")
    xai = UnifiedXAI(pipeline.model)
    print("Models Loaded Successfully!")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image_bgr is None:
        return {"error": "Invalid image file"}

    # Run inference with MC-Dropout
    result = pipeline.predict(image_bgr, use_mc_dropout=True)
    
    # --- HOTFIX: Remove 'PartiallyBleached' class entirely from outputs ---
    if "PartiallyBleached" in result["probabilities"]:
        del result["probabilities"]["PartiallyBleached"]
        
        # Re-normalize remaining probabilities
        total_prob = sum(result["probabilities"].values())
        if total_prob > 0:
            for cls_name in result["probabilities"]:
                result["probabilities"][cls_name] /= total_prob
                
        # Find new highest class
        new_pred_class = max(result["probabilities"], key=result["probabilities"].get)
        result["class"] = new_pred_class
        result["confidence"] = result["probabilities"][new_pred_class]
        
        # Update class_index so XAI generates the map for the new correct class!
        # class_names are typically ["Bleached", "Dead", "Healthy", "PartiallyBleached"]
        result["class_index"] = pipeline.idx_to_class_map.get(new_pred_class, 0) if hasattr(pipeline, 'idx_to_class_map') else ["Bleached", "Dead", "Healthy", "PartiallyBleached"].index(new_pred_class)

    # Run XAI Unified Maps
    class_idx = result["class_index"]
    
    rgb_tensor, lab_tensor = pipeline.preprocessor(image_bgr)
    rgb_tensor = rgb_tensor.unsqueeze(0).to(pipeline.device)
    lab_tensor = lab_tensor.unsqueeze(0).to(pipeline.device)
    
    overlay, maps = xai.explain(rgb_tensor, lab_tensor, class_idx, original_image=image_bgr)
    
    # Encode the overlaid heatmap image to Base64
    if overlay is not None:
        _, buffer = cv2.imencode('.jpg', overlay)
        result["heatmap_base64"] = base64.b64encode(buffer).decode('utf-8')
        
        # --- Smart Coral-Aware Bounding Box ---
        if "unified" in maps:
            unified_map = maps["unified"]
            orig_h, orig_w = image_bgr.shape[:2]

            # ── Step 1: Resize & normalize attention map to original image size ──
            attn = cv2.resize(unified_map.astype(np.float32), (orig_w, orig_h))
            attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

            # ── Step 2: Coral TEXTURE mask (edges = high texture = coral structure) ──
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 25, 80)
            edge_kernel = np.ones((18, 18), np.uint8)
            texture_mask = cv2.dilate(edges, edge_kernel, iterations=2).astype(np.float32) / 255.0

            # ── Step 3: Non-water COLOR mask (suppress deep-blue open water pixels) ──
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            blue_water = cv2.inRange(hsv, np.array([95, 60, 30]), np.array([135, 255, 255]))
            not_water = (255 - blue_water).astype(np.float32) / 255.0

            # ── Step 4: Multiply attention × texture × not-water ──
            combined = attn * texture_mask * not_water
            combined = cv2.GaussianBlur(combined, (19, 19), 0)
            if combined.max() > 0:
                combined /= combined.max()

            # ── Step 5: Threshold & morphological clean-up ──
            binary_map = (combined > 0.35).astype(np.uint8) * 255
            close_k = np.ones((28, 28), np.uint8)
            binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, close_k)
            binary_map = cv2.dilate(binary_map, np.ones((8, 8), np.uint8), iterations=1)

            contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            bbox_img = image_bgr.copy()
            label = result["class"]
            BOX_COLOR  = (0, 220, 110)
            LABEL_BG   = (0, 190, 95)
            LABEL_FG   = (5, 5, 5)
            min_area   = 0.025 * orig_w * orig_h
            drawn = 0

            for cnt in contours[:5]:
                if cv2.contourArea(cnt) < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                if w < 35 or h < 35:
                    continue
                # Outer box
                cv2.rectangle(bbox_img, (x, y), (x+w, y+h), BOX_COLOR, 2)
                # Subtle inner highlight
                cv2.rectangle(bbox_img, (x+2, y+2), (x+w-2, y+h-2), (180, 255, 190), 1)
                # Corner ticks (L-shaped corners for a sci-fi look)
                t = min(12, w//4, h//4)
                for (cx, cy), (dx, dy) in [
                    ((x,y),(1,1)), ((x+w,y),(-1,1)), ((x,y+h),(1,-1)), ((x+w,y+h),(-1,-1))
                ]:
                    cv2.line(bbox_img, (cx, cy), (cx+dx*t, cy), BOX_COLOR, 3)
                    cv2.line(bbox_img, (cx, cy), (cx, cy+dy*t), BOX_COLOR, 3)
                # Label pill above the box
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                pad = 5
                lx = x
                ly = max(y - th - pad*2 - 2, 0)
                cv2.rectangle(bbox_img, (lx, ly), (lx+tw+pad*2, ly+th+pad*2), LABEL_BG, -1)
                cv2.putText(bbox_img, label, (lx+pad, ly+th+pad),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, LABEL_FG, 2)
                drawn += 1

            # Fallback: largest contour hull
            if drawn == 0 and contours:
                all_pts = np.vstack(contours)
                hull = cv2.convexHull(all_pts)
                x, y, w, h = cv2.boundingRect(hull)
                cv2.rectangle(bbox_img, (x, y), (x+w, y+h), BOX_COLOR, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                pad = 5
                ly = max(y - th - pad*2 - 2, 0)
                cv2.rectangle(bbox_img, (x, ly), (x+tw+pad*2, ly+th+pad*2), LABEL_BG, -1)
                cv2.putText(bbox_img, label, (x+pad, ly+th+pad),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, LABEL_FG, 2)

            _, bbox_buffer = cv2.imencode('.jpg', bbox_img)
            result["bbox_base64"] = base64.b64encode(bbox_buffer).decode('utf-8')
        else:
            result["bbox_base64"] = None
    else:
        result["heatmap_base64"] = None
        result["bbox_base64"] = None
        
    return result

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
        
    return result

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
