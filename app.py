from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO

app = Flask(name)

# Define relevant class IDs (based on COCO dataset)
DESIRED_CLASSES = [
    39,  # Bottle
    67,  # Cell phone
    73,  # Book
    75,  # Remote
]

# Load the YOLOv8 model (using yolov8x.pt for highest accuracy)
model = YOLO("yolov8x.pt")

# Flags to control the camera state
camera_active = False
detect_objects = False

def generate_frames():
    global camera_active, detect_objects
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while camera_active:
        ret, frame = cap.read()
        if not ret:
            break

        # If object detection is active, process the frame
        if detect_objects:
            # Perform inference with confidence and IOU thresholding
            results = model(frame, conf=0.5, iou=0.4)
            detections = results[0].boxes
            boxes = detections.xyxy.cpu().numpy()  # Bounding boxes
            class_ids = detections.cls.cpu().numpy().astype(int)  # Class IDs

            # Filter only desired classes
            filtered_boxes = [box for box, class_id in zip(boxes, class_ids) if class_id in DESIRED_CLASSES]

            # Count the detected objects
            num_objects = len(filtered_boxes)

            # Draw bounding boxes without labels
            for box in filtered_boxes:
                x1, y1, x2, y2 = box.astype(int)
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display the object count
            cv2.putText(frame, f"Objects Detected: {num_objects}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    # Render the index.html page
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global camera_active
    camera_active = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detect_objects
    detect_objects = True
    return jsonify({"status": "Detection started."})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detect_objects
    detect_objects = False
    return jsonify({"status": "Detection stopped."})


if name == "main":
    app.run(debug=True)