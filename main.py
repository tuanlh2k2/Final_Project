import cv2
import torch
import numpy as np
import time

# Load YOLOv5 model
model = torch.hub.load('yolov5-master', path='yolov5s-fp16.tflite', source='local', model='custom', force_reload=True)

# Open video file for reading
cap = cv2.VideoCapture('Vehicle_Detection.mp4')

# Define class names and corresponding colors
class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
               "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
               "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
               "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
               "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
               "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
               "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
               "hair drier", "toothbrush"]

# Generate unique colors for each class
class_colors = np.random.uniform(0, 255, size=(len(class_names), 3))

# Initialize variables for calculating FPS
start_time = time.time()
frame_count = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Resize the frame
    frame = cv2.resize(frame, (640, 640))

    # Perform inference with YOLOv5
    results = model(frame)

    # Extract bounding box coordinates, class predictions, and confidences
    bboxes = results.xyxy[0].cpu().numpy()
    classes = results.names

    # Loop through each detected object
    for bbox in bboxes:
        # Extract object coordinates
        x_min, y_min, x_max, y_max = bbox[:4]

        # Convert to integer values
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

        # Get class index and confidence
        class_index = int(bbox[5])
        confidence = bbox[4]

        # Get class label and color
        class_label = class_names[class_index]
        color = class_colors[class_index]

        # Draw bounding box on the frame with unique color
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

        # Add label to the bounding box with unique color
        label = f'{class_label}: {confidence:.2f}'
        cv2.putText(frame, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('OBJECT_DETECTION_UNDER_RAINY_WEATHER_CONDITION', frame)

    # Break the loop if the 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
