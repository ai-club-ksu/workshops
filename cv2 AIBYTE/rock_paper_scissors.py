import cv2
import random
import time
from ultralytics import YOLO

model = YOLO("wighets/best.pt")
cap = cv2.VideoCapture(0)

moves = ["Rock", "Paper", "Scissors"]
result_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()

    cv2.putText(annotated, "Press S to play", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    if result_text:
        cv2.putText(annotated, result_text, (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("Rock Paper Scissors", annotated)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):

        # countdown
        for i in ["3","2","1","Show!"]:
            ret, frame = cap.read()
            temp = frame.copy()

            cv2.putText(temp, i, (300,200),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,255), 5)

            cv2.imshow("Rock Paper Scissors", temp)
            cv2.waitKey(1000)

        # detect hand
        results = model(frame)

        if len(results[0].boxes.cls) > 0:

            player = model.names[int(results[0].boxes.cls[0])]
            computer = random.choice(moves)

            if player == computer:
                result = "Draw"
            elif (player == "Rock" and computer == "Scissors") or \
                 (player == "Paper" and computer == "Rock") or \
                 (player == "Scissors" and computer == "Paper"):
                result = "You Win!"
            else:
                result = "Computer Wins!"

            result_text = f"You: {player} | CPU: {computer} | {result}"

        else:
            result_text = "No hand detected"

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()