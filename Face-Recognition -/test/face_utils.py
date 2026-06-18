"""
Utility functions for face recognition operations (OpenCV DNN version)
Uses OpenCV's deep neural networks instead of dlib to avoid compilation issues
Includes: loading encodings, recognizing faces, drawing bounding boxes
"""

import os
import pickle
import cv2
import numpy as np
from pathlib import Path
import config
import face_recognition


# Initialize OpenCV DNN models
MODEL_DIR = "models"

def get_face_detector():
    """Load OpenCV DNN face detector"""
    prototxt = os.path.join(MODEL_DIR, "deploy.prototxt")
    model = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
    
    # Download models if not present
    if not os.path.exists(model) or not os.path.exists(prototxt):
        print("Downloading face detection models...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Use OpenCV's built-in FaceDetectorYN instead (no download needed)
        print("Note: Using OpenCV's built-in face detector")
    
    try:
        net = cv2.dnn.readNetFromCaffe(prototxt, model)
        return net
    except:
        print("Using OpenCV YuNet face detector (built-in)")
        return None


def detect_faces_opencv(image, conf_threshold=0.5):
    """
    Detect faces using OpenCV's DNN or YuNet detector
    
    Args:
        image: Input image (BGR format)
        conf_threshold: Confidence threshold for face detection
    
    Returns:
        list: List of (x, y, w, h) bounding boxes
    """
    h, w = image.shape[:2]
    
    # Try using YuNet detector (built-in, no model download needed)
    try:
        detector = cv2.FaceDetectorYN.create(
            "face_detection_yunet_2023mar.onnx",
            "",
            input_size=(320, 320),
            score_threshold=conf_threshold,
            nms_threshold=0.3,
            top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )
        detector.setInputSize((w, h))

        result = detector.infer(image)
        print("DEBUG RESULT:", result)

        if isinstance(result, tuple):
            faces = result[1]
        else:
            faces = result

        face_rects = []
        if faces is not None:
            for face in faces:
                x, y, bw, bh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                face_rects.append((x, y, bw, bh))
        return face_rects
    except:
        # Fallback: Use Haar Cascade (always available, less accurate)
        print("Warning: Using Haar Cascade face detector (lower accuracy)")
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(image, 1.3, 5)
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


def extract_face_features(image, face_rect):
    """
    Extract features from a face region using HOG
    
    Args:
        image: Input image (BGR format)
        face_rect: (x, y, w, h) face rectangle
    
    Returns:
        numpy array: Feature vector for the face
    """
    x, y, w, h = face_rect
    
    # Ensure coordinates are within image bounds
    x = max(0, x)
    y = max(0, y)
    w = min(image.shape[1] - x, w)
    h = min(image.shape[0] - y, h)
    
    if w <= 0 or h <= 0:
        return None
    
    # Crop face region
    face_img = image[y:y+h, x:x+w]
    
    # Resize to standard size
    face_img = cv2.resize(face_img, (128, 128))
    
    # Convert to grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Extract HOG features
    hog = cv2.HOGDescriptor()
    features = hog.compute(gray)
    
    # Normalize features
    if features is not None and len(features) > 0:
        features = features.flatten()
        features = features / (np.linalg.norm(features) + 1e-6)
    else:
        # Fallback: use simple histogram features
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        features = hist.flatten()
        features = features / (np.linalg.norm(features) + 1e-6)
    
    return features


def load_encoded_faces():
    """
    Load pre-encoded face database from pickle file
    
    Returns:
        tuple: (known_encodings, known_names) or ([], []) if file doesn't exist
    """
    if not os.path.exists(config.ENCODINGS_FILE):
        print(f"Warning: Encodings file not found at {config.ENCODINGS_FILE}")
        print("Run encode_faces.py first to create the face database.")
        return [], []
    
    try:
        with open(config.ENCODINGS_FILE, 'rb') as f:
            data = pickle.load(f)
            return data.get('encodings', []), data.get('names', [])
    except Exception as e:
        print(f"Error loading encodings: {e}")
        return [], []


def save_encoded_faces(encodings, names):
    """
    Save encoded faces to pickle file
    
    Args:
        encodings: List of face feature vectors (numpy arrays)
        names: List of corresponding person names
    """
    os.makedirs(config.ENCODINGS_DIR, exist_ok=True)
    data = {'encodings': encodings, 'names': names}
    
    try:
        with open(config.ENCODINGS_FILE, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {len(encodings)} face encodings to {config.ENCODINGS_FILE}")
    except Exception as e:
        print(f"Error saving encodings: {e}")


def recognize_faces_in_image(image, known_encodings, known_names):

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_locations_fr = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(
        rgb_image,
        face_locations_fr
    )

    face_locations = []
    face_labels = []
    face_distances = []

    for (top, right, bottom, left), face_encoding in zip(
        face_locations_fr,
        face_encodings
    ):

        w = right - left
        h = bottom - top

        face_locations.append((left, top, w, h))

        if len(known_encodings) == 0:
            face_labels.append("Unknown")
            face_distances.append(1.0)
            continue

        distances = face_recognition.face_distance(
            known_encodings,
            face_encoding
        )

        best_idx = np.argmin(distances)
        best_distance = distances[best_idx]

        if best_distance <= 0.6:
            label = known_names[best_idx]
        else:
            label = "Unknown"

        face_labels.append(label)
        face_distances.append(float(best_distance))

    return face_locations, face_labels, face_distances

def draw_face_boxes(image, face_locations, face_labels, show_distance=False, face_distances=None):
    """
    Draw bounding boxes and labels on image
    
    Args:
        image: Input image (numpy array, BGR format)
        face_locations: List of (x, y, w, h) tuples
        face_labels: List of face label strings
        show_distance: Whether to show distance score
        face_distances: List of distance scores (required if show_distance=True)
    
    Returns:
        image: Image with drawn boxes and labels
    """
    image_copy = image.copy()
    
    for i, ((x, y, w, h), label) in enumerate(zip(face_locations, face_labels)):
        # Draw rectangle around face
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), config.BOX_COLOR, 2)
        
        # Prepare label text
        label_text = label
        if show_distance and face_distances is not None and i < len(face_distances):
            label_text = f"{label} ({face_distances[i]:.2f})"
        
        # Draw label background and text
        label_y = y - 10 if y > 30 else y + h + 25
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, config.LABEL_FONT_SIZE, config.LABEL_THICKNESS)[0]
        cv2.rectangle(image_copy, (x, label_y - text_size[1] - 5), (x + text_size[0] + 5, label_y + 5), config.BOX_COLOR, cv2.FILLED)
        cv2.putText(image_copy, label_text, (x + 3, label_y - 2),
                   cv2.FONT_HERSHEY_DUPLEX, config.LABEL_FONT_SIZE, (255, 255, 255), config.LABEL_THICKNESS)
    
    return image_copy


def get_face_files_from_directory(directory):
    """
    Get all image files from directory, organized by subdirectory (person)
    
    Args:
        directory: Path to directory with person subdirectories
    
    Returns:
        dict: {person_name: [list of image paths]}
    """
    face_files = {}
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return face_files
    
    # Look for subdirectories (one per person)
    for person_dir in os.listdir(directory):
        person_path = os.path.join(directory, person_dir)
        
        if not os.path.isdir(person_path):
            continue
        
        face_files[person_dir] = []
        
        # Get all image files from this person's directory
        for filename in os.listdir(person_path):
            if os.path.splitext(filename)[1].lower() in valid_extensions:
                face_files[person_dir].append(os.path.join(person_path, filename))
    
    return face_files


def load_image(image_path):
    """
    Load image from file
    
    Args:
        image_path: Path to image file
    
    Returns:
        image: Image array (BGR format) or None if failed
    """
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    return image


def save_image(image, output_path):
    """
    Save image to file
    
    Args:
        image: Image array (BGR format)
        output_path: Path to save image to
    
    Returns:
        bool: True if successful, False otherwise
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    success = cv2.imwrite(output_path, image)
    if success:
        print(f"Saved image to {output_path}")
    else:
        print(f"Failed to save image to {output_path}")
    return success
