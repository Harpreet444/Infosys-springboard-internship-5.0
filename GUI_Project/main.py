import streamlit as st
import torch
import torchvision
import cv2
import numpy as np
import os
import tempfile
from torchvision import transforms
from PIL import Image
import imageio
from imageai.Detection import VideoObjectDetection
from object_detection_utils import ObjectDetection
import math
from ultralytics import YOLO


def process_video_tracking(video_path):
    """Process video with tracking feature."""
    od = ObjectDetection()
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_output:
        temp_path = temp_output.name
    
    writer = imageio.get_writer(temp_path, fps=fps, codec='libx264', pixelformat='yuv420p')
    st_frame = st.empty()
    
    tracking_objects = {}
    track_id = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        center_points_cur_frame = []
        (class_ids, scores, boxes) = od.detect(frame)
        
        for box in boxes:
            x, y, w, h = box
            cx, cy = int(x + w/2), int(y + h/2)
            center_points_cur_frame.append((cx, cy))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        for pt in center_points_cur_frame:
            same_object_detected = False
            for object_id, prev_pt in tracking_objects.items():
                distance = math.hypot(prev_pt[0] - pt[0], prev_pt[1] - pt[1])
                if distance < 60:
                    tracking_objects[object_id] = pt
                    same_object_detected = True
                    break
            if not same_object_detected:
                tracking_objects[track_id] = pt
                track_id += 1
        
        for object_id, pt in tracking_objects.items():
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)
            cv2.putText(frame, str(object_id), (pt[0] - 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        st_frame.image(frame, channels="RGB")
    
    cap.release()
    writer.close()
    return temp_path

def process_video_masked_tracking(video_path):
    """Process video with masked tracking feature."""
    od = ObjectDetection()
    cap = cv2.VideoCapture(video_path)
    mask = cv2.imread("images/mask.png", 0)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Resize the mask to match frame size
    mask = cv2.resize(mask, (frame_width, frame_height))
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_output:
        temp_path = temp_output.name
    
    writer = imageio.get_writer(temp_path, fps=fps, codec='libx264', pixelformat='yuv420p')
    st_frame = st.empty()
    
    tracking_objects = {}
    track_id = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        center_points_cur_frame = []
        (class_ids, scores, boxes) = od.detect(masked_frame)
        
        for box in boxes:
            x, y, w, h = box
            cx, cy = int(x + w/2), int(y + h/2)
            center_points_cur_frame.append((cx, cy))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        for pt in center_points_cur_frame:
            same_object_detected = False
            for object_id, prev_pt in tracking_objects.items():
                distance = math.hypot(prev_pt[0] - pt[0], prev_pt[1] - pt[1])
                if distance < 60:
                    tracking_objects[object_id] = pt
                    same_object_detected = True
                    break
            if not same_object_detected:
                tracking_objects[track_id] = pt
                track_id += 1
        
        for object_id, pt in tracking_objects.items():
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)
            cv2.putText(frame, str(object_id), (pt[0] - 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        st_frame.image(frame, channels="RGB")
    
    cap.release()
    writer.close()
    return temp_path

def apply_preprocessing(image, method):
    if method == "Grayscale":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif method == "Histogram Equalization":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)
    elif method == "Edge Detection":
        return cv2.Canny(image, 100, 200)
    elif method == "Blue Channel":
        red = image.copy()
        red[:, :, 0] = 0  # Zero out Blue
        red[:, :, 1] = 0  # Zero out Green
        return red
    elif method == "Green Channel":
        green = image.copy()
        green[:, :, 0] = 0  # Zero out Blue
        green[:, :, 2] = 0  # Zero out Red
        return green
    elif method == "Red Channel":
        blue = image.copy()
        blue[:, :, 1] = 0  # Zero out Green
        blue[:, :, 2] = 0  # Zero out Red
        return blue
    return image  # Default: No processing


def load_yolov4():
    """Load YOLOv4 model."""
    model = ObjectDetection(
        weights_path="models/dnn_model/yolov4.weights",
        cfg_path="models/dnn_model/yolov4.cfg",
        classes_path="models/dnn_model/classes.txt",
        image_size=608,
        conf_threshold=0.5,
        nms_threshold=0.4,
        use_gpu=False  # Change to True if using CUDA
    )
    return model


def load_retinanet():
    """Load the built-in RetinaNet model from torchvision."""
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def load_tinyyolov3():
    """Initialize TinyYOLOv3 from ImageAI."""
    detector = VideoObjectDetection()
    detector.setModelTypeAsTinyYOLOv3()
    # Use relative path and fix path separator
    model_path = os.path.join("models", "tiny-yolov3.pt")
    if os.path.isfile(model_path):
        detector.setModelPath(model_path)
        detector.loadModel()
        return detector
    else:
        st.error(f"TinyYOLOv3 model not found at: {model_path}")
        return None

def detect_objects_retinanet(frame, model, threshold=0.5):
    """Run object detection on a video frame using RetinaNet."""
    transform = transforms.Compose([transforms.ToTensor()])
    frame_tensor = transform(frame).unsqueeze(0)
    
    with torch.no_grad():
        predictions = model(frame_tensor)
    
    return predictions

def draw_boxes(frame, predictions, threshold=0.5):
    """Draw bounding boxes on video frames based on RetinaNet's predictions."""
    labels = predictions[0]['labels'].numpy()
    scores = predictions[0]['scores'].numpy()
    boxes = predictions[0]['boxes'].numpy()
    
    for label, score, box in zip(labels, scores, boxes):
        if score >= threshold:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def process_video_yolov4(video_path, model, threshold=0.5):
    """Process video using YOLOv4 model."""
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()['fps']

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_output:
        temp_path = temp_output.name

    writer = imageio.get_writer(temp_path, fps=fps, codec='libx264', pixelformat='yuv420p')
    st_frame = st.empty()

    for frame in reader:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        class_ids, scores, boxes = model.detect(frame_rgb)

        # Draw bounding boxes
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            label = f"{model.classes[class_ids[i]]}: {scores[i]:.2f}"
            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_rgb, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        writer.append_data(frame_rgb)
        st_frame.image(frame_rgb, channels="RGB")

    writer.close()
    return temp_path


def process_video_retinanet(video_path, model, threshold=0.5):
    """Process a video frame by frame and apply object detection using RetinaNet."""
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()['fps']
    
    # Create temp file with proper closure
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_output:
        temp_path = temp_output.name
    
    # Use proper video codec
    writer = imageio.get_writer(temp_path, fps=fps, codec='libx264', pixelformat='yuv420p')
    
    st_frame = st.empty()
    
    for frame in reader:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        predictions = detect_objects_retinanet(frame_pil, model, threshold)
        output_frame = draw_boxes(frame_rgb, predictions, threshold)
        writer.append_data(output_frame)
        
        # Show processed frame
        st_frame.image(output_frame, channels="RGB")
    
    writer.close()
    return temp_path

def process_video_tinyyolo(video_path, detector, frames_per_second, min_probability):
    """Process video using TinyYOLOv3 from ImageAI."""
    # Create temp file with proper closure
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_output:
        output_video_path = temp_output.name
    
    detector.detectObjectsFromVideo(
        input_file_path=video_path,
        output_file_path=output_video_path,
        frames_per_second=frames_per_second,
        minimum_percentage_probability=min_probability,
        save_detected_video=True
    )
    return output_video_path

def load_yolov8():
    """Load YOLOv8 model from Ultralytics"""
    model_path = os.path.join("models", "yolov8s.pt")
    if os.path.isfile(model_path):
        model = YOLO(model_path)
        return model
    else:
        st.error(f"YOLOv8 model not found at: {model_path}")
        return None

def process_video_yolov8(video_path, model, threshold=0.5):
    """Process video using YOLOv8 model"""
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()['fps']
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_output:
        temp_path = temp_output.name
    
    writer = imageio.get_writer(temp_path, fps=fps, codec='libx264', pixelformat='yuv420p')
    st_frame = st.empty()

    for frame in reader:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(frame_rgb, conf=threshold)
        
        # Draw bounding boxes
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.putText(frame_rgb, label, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        writer.append_data(frame_rgb)
        st_frame.image(frame_rgb, channels="RGB")
    
    writer.close()
    return temp_path

def main():
    st.title("Surveillance Video Object Detection 🎥")
    st.sidebar.image(r"D:\Git\Infosys-springboard-internship-5.0\GUI_Project\assist\CCTV.png")
    st.sidebar.header("Configuration")
    mode = st.sidebar.radio("Select Mode", ["Image Preprocessing", "Video Object Detection"])
    
    if mode == "Image Preprocessing":
        preprocessing = st.sidebar.selectbox("Select Preprocessing Method", 
                                            ["None", "Grayscale", "Histogram Equalization", "Edge Detection", "Red Channel", "Green Channel", "Blue Channel"])
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])


        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = np.array(image)
            st.image(image, caption="Original Image", use_container_width=True)
            
            if st.button("Apply Preprocessing"):
                processed_image = apply_preprocessing(image, preprocessing)
                st.image(processed_image, caption="Processed Image", use_container_width=True)
    elif mode == "Video Object Detection":
        model_type = st.sidebar.selectbox("Select Model", ["YOLOv8", "YOLOv4","RetinaNet","Object Tracking", "Masked Tracking", "TinyYOLOv3"], index=0)

        min_probability = st.sidebar.slider("Minimum probability for detection (%)", 10, 100, 30)
        frames_per_second = st.sidebar.slider("Frames per second for processing", 1, 30, 10)
        
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
        
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                temp_file.write(uploaded_file.read())
                temp_video_path = temp_file.name
            
            st.video(temp_video_path)
            
            if st.button("Run Object Detection"):
                if model_type == "YOLOv4":
                    st.write("Processing video with YOLOv4, please wait...")
                    model = load_yolov4()
                    processed_video_path = process_video_yolov4(temp_video_path, model, threshold=min_probability / 100)
                    st.video(processed_video_path)
                elif model_type == "RetinaNet":
                    st.write("Processing video with RetinaNet, please wait...")
                    model = load_retinanet()
                    processed_video_path = process_video_retinanet(temp_video_path, model, threshold=min_probability / 100)
                    st.video(processed_video_path)
                elif model_type == "YOLOv8":
                    st.write("Processing video with YOLOv8, please wait...")
                    model = load_yolov8()
                    if model:
                        processed_video_path = process_video_yolov8(temp_video_path, model, threshold=min_probability / 100)
                        st.video(processed_video_path)
                elif model_type == "Object Tracking":
                    st.write("Object Tracking, please wait...")
                    process_video_tracking(temp_video_path)
                    st.video(processed_video_path)   
                    
                elif model_type == "Masked Tracking":                
                    st.write("Processing video with Masked Tracking, please wait...")
                    processed_video_path = process_video_masked_tracking(temp_video_path)
                    st.video(processed_video_path)     
                
                else:
                    st.write("Processing video with TinyYOLOv3, please wait...")
                    detector = load_tinyyolov3()
                    if detector:
                        processed_video_path = process_video_tinyyolo(temp_video_path, detector, frames_per_second, min_probability)
                        st.video(processed_video_path)

if __name__ == "__main__":
    main()