import cv2
import numpy as np

class ObjectDetection:
    def __init__(self, weights_path="models/dnn_model/yolov4.weights", 
                 cfg_path="models/dnn_model/yolov4.cfg", 
                 classes_path="models/dnn_model/classes.txt", 
                 image_size=608, 
                 conf_threshold=0.5, 
                 nms_threshold=0.4, 
                 use_gpu=False):  # Changed default to False for safety
        """
        Initialize the Object Detection model with YOLOv4.
        """
        print("Initializing Object Detection...")
        
        self.nmsThreshold = nms_threshold
        self.confThreshold = conf_threshold
        self.image_size = image_size

        # Load YOLO Network
        try:
            self.net = cv2.dnn.readNet(weights_path, cfg_path)
        except Exception as e:
            raise FileNotFoundError(f"Error loading YOLO files: {e}")

        # Improved backend handling
        self.backend = cv2.dnn.DNN_BACKEND_OPENCV
        self.target = cv2.dnn.DNN_TARGET_CPU
        self.use_gpu = use_gpu
        
        # Handle GPU/CPU configuration
        if self.use_gpu:
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("Successfully configured CUDA backend")
                self.backend = cv2.dnn.DNN_BACKEND_CUDA
                self.target = cv2.dnn.DNN_TARGET_CUDA
            except Exception as e:
                print(f"CUDA initialization failed: {str(e)}")
                print("Falling back to CPU")
                self.use_gpu = False
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        else:
            print("Using CPU by configuration")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Initialize detection model with proper parameters
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(
            size=(self.image_size, self.image_size), 
            scale=1/255, 
            swapRB=True,  # Add RGB swapping
            crop=False
        )

        # Load class names
        self.classes = self.load_class_names(classes_path)
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def load_class_names(self, classes_path):
        """
        Load class names from a file with improved error handling.
        """
        try:
            with open(classes_path, "r") as file:
                classes = [line.strip() for line in file if line.strip()]
                print(f"Successfully loaded {len(classes)} classes")
                return classes
        except Exception as e:
            raise RuntimeError(f"Error loading class names: {str(e)}")

    def detect(self, frame):
        """
        Perform object detection with enhanced error handling.
        """
        try:
            # Convert frame if necessary
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
                
            # Perform detection
            class_ids, scores, boxes = self.model.detect(
                frame, 
                confThreshold=self.confThreshold,
                nmsThreshold=self.nmsThreshold
            )
            
            # Handle empty detections
            if len(class_ids) == 0:
                return np.array([]), np.array([]), np.array([])
                
            return class_ids.flatten(), scores.flatten(), boxes
        except Exception as e:
            raise RuntimeError(f"Detection error: {str(e)}")