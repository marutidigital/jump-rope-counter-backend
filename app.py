from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Jump counter variables
counter1 = 0
counter2 = 0
jump = 0
stage1 = None
stage2 = None

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

@app.route('/process-frame', methods=['POST'])
def process_frame():
    global counter1, counter2, jump, stage1, stage2

    # Receive frame as a base64-encoded string
    frame_data = request.json.get("frame")
    if not frame_data:
        return jsonify({"error": "No frame data provided"}), 400

    # Decode the image (example assumes base64 encoding)
    nparr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Recolor the image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    try:
        landmarks = results.pose_landmarks.landmark

        # Get coordinates for left side
        lh = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        lk = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        la = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # Get coordinates for right side
        rh = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        rk = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ra = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # Calculate angles
        left_knee_angle = calculate_angle(lh, lk, la)
        right_knee_angle = calculate_angle(rh, rk, ra)

        # Jump logic
        if right_knee_angle > 150:
            stage1 = "down"
        if right_knee_angle < 140 and stage1 == "down":
            stage1 = "up"
            counter1 += 1

        if left_knee_angle > 150:
            stage2 = "down"
        if left_knee_angle < 140 and stage2 == "down":
            stage2 = "up"
            counter2 += 1

        if counter1 > 0 and counter2 > 0:
            jump += 1
            counter1, counter2 = 0, 0

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"jumps": jump})

if __name__ == '__main__':
    app.run(debug=True)
