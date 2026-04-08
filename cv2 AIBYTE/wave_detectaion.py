import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prev_x = None
movement_counter = 0

while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark

        wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Convert normalized to pixel
        h, w, _ = frame.shape
        wrist_y = wrist.y * h
        shoulder_y = shoulder.y * h
        wrist_x = wrist.x * w

        # Check hand is above shoulder
        if wrist_y < shoulder_y:
            if prev_x is not None and abs(wrist_x - prev_x) > 20:
                movement_counter += 1
            prev_x = wrist_x
        else:
            movement_counter = 0

        if movement_counter > 5:
            cv2.putText(frame, "WAVING!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            print("Wave detected")

    cv2.imshow("Waving Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()