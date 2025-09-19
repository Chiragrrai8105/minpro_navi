import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import shutil
import tempfile
from pathlib import Path
import os

# ================= CONFIG =================
from pathlib import Path
import tempfile
import os

APP_DIR = Path(__file__).parent

MODEL_INFOS = [
    {"path": APP_DIR / "ecoli_detector.pt", "name": "E_coli"},
    {"path": APP_DIR / "camper_model_best.pt", "name": "Campher"},
    {"path": APP_DIR / "staphy_model_best.pt", "name": "Staphylococcus"},
    {"path": APP_DIR / "bacillus_model_best.pt", "name": "Bacillus"},
]

# Validate model files
AVAILABLE_MODELS = []
for model_info in MODEL_INFOS:
    if Path(model_info["path"]).exists():
        AVAILABLE_MODELS.append(model_info)
    else:
        st.warning(f"Model file not found: {model_info['path']}")

if not AVAILABLE_MODELS:
    st.error("No model files found. Please check the model paths.")
    st.stop()

CONF_THR = 0.25
IOU_DEDUPE = 0.5

# Create a temporary directory for uploads
TEMP_DIR = Path(tempfile.mkdtemp())
# ===========================================

# Utility functions
def xyxy_area(xyxy):
    x1,y1,x2,y2 = xyxy
    return max(0, x2-x1) * max(0, y2-y1)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB-xA); interH = max(0, yB-yA)
    interArea = interW*interH
    boxAArea = xyxy_area(boxA); boxBArea = xyxy_area(boxB)
    union = boxAArea + boxBArea - interArea
    return interArea / union if union>0 else 0

@st.cache_resource
def load_models():
    """Load all available models once and cache them"""
    models = {}
    for m in AVAILABLE_MODELS:
        try:
            models[m["name"]] = {"model": YOLO(str(m["path"])), "info": m}
        except Exception as e:
            st.warning(f"Failed to load model {m['name']}: {str(e)}")
    return models

def run_detection(image_path):
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError("Failed to load image")
        
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            raise ValueError("Invalid image dimensions")

        all_dets = []
        models = load_models()

        # Run all models
        for model_name, model_data in models.items():
            try:
                results = model_data["model"].predict(str(image_path), conf=CONF_THR, save=False)
                for r in results:
                    if not hasattr(r.boxes, "xyxy") or len(r.boxes) == 0:
                        continue
                        
                    xyxy_arr = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    
                    for i, box in enumerate(xyxy_arr):
                        conf = float(confs[i])
                        if conf < CONF_THR:
                            continue
                        all_dets.append({
                            "name": model_data["info"]["name"],
                            "conf": conf,
                            "xyxy": [float(v) for v in box],
                        })
            except Exception as e:
                st.warning(f"Error running detection with model {model_name}: {str(e)}")
                continue

        # Deduplicate
        final_dets = []
        used = [False]*len(all_dets)
        for i, d in enumerate(all_dets):
            if used[i]: continue
            boxA = d["xyxy"]
            best = d
            used[i] = True
            for j in range(i+1, len(all_dets)):
                if used[j]: continue
                boxB = all_dets[j]["xyxy"]
                if iou(boxA, boxB) > IOU_DEDUPE:
                    if all_dets[j]["conf"] > best["conf"]:
                        best = all_dets[j]
                    used[j] = True
            final_dets.append(best)

        # Draw detections
        for det in final_dets:
            x1,y1,x2,y2 = map(int, det["xyxy"])
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, f"{det['name']} {det['conf']:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Analysis
        counts = {}
        areas = {}
        for det in final_dets:
            counts[det["name"]] = counts.get(det["name"], 0) + 1
            areas[det["name"]] = areas.get(det["name"], 0) + xyxy_area(det["xyxy"])
        total_area = sum(areas.values())

        analysis = {}
        for name in counts:
            perc = (areas[name]/total_area*100) if total_area>0 else 0
            analysis[name] = {"count": counts[name], "percentage": perc}

        return img, analysis
        
    except Exception as e:
        st.error(f"Error in detection process: {str(e)}")
        return None, {}

# Cleanup function for temporary files
def cleanup_temp_files():
    try:
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
    except Exception as e:
        st.warning(f"Failed to cleanup temporary files: {str(e)}")

# ===== Streamlit UI =====
try:
    st.title("🦠 Multi-Bacteria Detection App")

    # Show available models
    st.sidebar.subheader("Available Models:")
    for model in AVAILABLE_MODELS:
        st.sidebar.text(f"✓ {model['name']}")

    uploaded_file = st.file_uploader("Upload Petri Dish Image", type=["jpg","png","jpeg"])

    if uploaded_file:
        try:
            # Create a unique temporary file path
            temp_path = TEMP_DIR / f"{uploaded_file.name}"
            
            # Save uploaded image temporarily
            img = Image.open(uploaded_file)
            img.save(temp_path)

            st.image(img, caption="Uploaded Image", use_column_width=True)

            if st.button("Run Detection"):
                with st.spinner("Running detection..."):
                    try:
                        result_img, analysis = run_detection(temp_path)
                        
                        if not analysis:
                            st.warning("No bacteria colonies detected in the image.")
                        else:
                            st.image(result_img, caption="Detection Result", use_column_width=True)

                            st.subheader("Analysis:")
                            for name, info in analysis.items():
                                st.write(f"{name}: {info['count']} colonies, {info['percentage']:.2f}% of total area")
                    except Exception as e:
                        st.error(f"Error during detection: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")
            
finally:
    # Cleanup temporary files when the app exits
    cleanup_temp_files()
