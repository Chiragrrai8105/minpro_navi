import streamlit as st
from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import tempfile
import shutil

try:
    # Load model from file in same folder - using Path for proper path handling
    model_path = Path(__file__).parent / "ecoli_detector.pt"
    st.write(f"Looking for model at: {model_path}")
    
    if not model_path.exists():
        st.error(f"Model file not found at {model_path}. Please check if the model file exists in the application folder.")
        st.stop()
    
    # Print model file size for verification
    st.write(f"Model file size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Load the model with debug info
    st.write("Loading model...")
    model = YOLO(str(model_path))
    st.write("Model loaded successfully!")
    
except Exception as e:
    st.error(f"Error loading model: {str(e)}\nModel path: {model_path}")
    st.stop()

st.title("🧫 Bacteria Detection & Growth Estimation")

# Create a temporary directory for our files
temp_dir = tempfile.mkdtemp()
try:
    uploaded_file = st.file_uploader("Upload a Petri Dish Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # Save uploaded file temporarily with proper path handling
        temp_image_path = Path(temp_dir) / "input_image.jpg"
        
        try:
            # Save the uploaded file
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Validate image
            img = cv2.imread(str(temp_image_path))
            if img is None:
                st.error("Failed to load the uploaded image. Please try again with a different image.")
                st.stop()

            # Run detection with error handling
            try:
                st.write("Running detection...")
                st.write(f"Input image path: {temp_image_path}")
                
                results = model(str(temp_image_path), save=True, project="runs", name="detect")
                st.write("Detection completed!")
                st.write(f"Results type: {type(results)}")
                st.write(f"Number of detections: {len(results) if results else 0}")
                
                if not results or len(results) == 0:
                    st.warning("No colonies detected in the image.")
                    st.image(str(temp_image_path), caption="Original Image")
                    st.stop()

                # Calculate growth %
                h, w = img.shape[:2]
                dish_area = h * w

                # Get detection boxes
                if hasattr(results[0].boxes, 'xywh') and len(results[0].boxes.xywh) > 0:
                    total_area = sum([bw * bh for _,_,bw,bh in results[0].boxes.xywh])
                    percentage = (total_area / dish_area) * 100

                    # Show annotated output
                    # YOLO saves results in a numbered directory like 'predict', 'predict2', etc.
                    predict_dir = None
                    for i in range(1, 10):  # Check predict1 through predict9
                        suffix = '' if i == 1 else str(i)
                        test_path = Path("runs/detect/predict" + suffix)
                        if test_path.exists() and any(test_path.iterdir()):
                            predict_dir = test_path
                            break
                    
                    if predict_dir:
                        # Get the latest image in the predict directory
                        output_files = list(predict_dir.glob("*"))
                        st.write(f"Found {len(output_files)} files in output directory: {predict_dir}")
                        for f in output_files:
                            st.write(f"Output file: {f}")
                            
                        if output_files:
                            latest_output = max(output_files, key=lambda x: x.stat().st_mtime)
                            st.write(f"Selected output file: {latest_output}")
                            st.image(str(latest_output), caption="Detected Colonies")
                            st.success(f"✅ Bacterial Growth: {percentage:.2f}%")
                        else:
                            st.error("No output image found in the results directory.")
                    else:
                        st.error("Could not find the processed image output directory.")
                else:
                    st.warning("No colonies detected in the image.")
                    st.image(str(temp_image_path), caption="Original Image")

            except Exception as e:
                st.error(f"Error during detection: {str(e)}")

        except Exception as e:
            st.error(f"Error processing the image: {str(e)}")

finally:
    # Clean up temporary files
    try:
        shutil.rmtree(temp_dir)
        if Path("runs").exists():
            shutil.rmtree("runs")
    except Exception as e:
        st.warning(f"Warning: Could not clean up temporary files: {str(e)}")

