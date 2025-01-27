import cv2
import numpy as np

class ObjectDetection:
    def __init__(self, weights_path="models/dnn_model/yolov4.weights", 
                 cfg_path="models/dnn_model/yolov4.cfg", 
                 classes_path="models/dnn_model/classes.txt", 
                 image_size=608, 
                 conf_threshold=0.5, 
                 nms_threshold=0.4, 
                 use_gpu=True):
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
        
        # Enable GPU CUDA if available and specified
        if use_gpu:
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("Using GPU for computation.")
            except Exception:
                print("CUDA is not available. Falling back to CPU.")
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        else:
            print("Using CPU for computation.")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(self.image_size, self.image_size), scale=1 / 255)

        # Load class names
        self.classes = self.load_class_names(classes_path)
        self.colors = self.generate_colors()

    def load_class_names(self, classes_path):
        """
        Load class names from a file.
        """
        try:
            with open(classes_path, "r") as file:
                classes = [line.strip() for line in file.readlines()]
                print(f"Loaded {len(classes)} class names.")
            return classes
        except FileNotFoundError:
            raise FileNotFoundError(f"Class file not found: {classes_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading class names: {e}")

    def generate_colors(self):
        """
        Generate unique colors for each class.
        """
        num_classes = len(self.classes)
        return np.random.uniform(0, 255, size=(num_classes, 3))

    def detect(self, frame):
        """
        Perform object detection on a single frame.
        """
        try:
            class_ids, scores, boxes = self.model.detect(
                frame, 
                nmsThreshold=self.nmsThreshold, 
                confThreshold=self.confThreshold
            )
            return class_ids, scores, boxes
        except Exception as e:
            raise RuntimeError(f"Error during detection: {e}")
