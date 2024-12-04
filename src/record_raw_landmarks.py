# dump raw mediapipe landmarks to csv for testing
# just want to see if the coordinates are stable

import csv
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

cap = cv2.VideoCapture(0)
outfile = open("raw_landmarks.csv", "w", newline="")
writer = csv.writer(outfile)
header = ["frame"] + [f"lm_{i}_{c}" for i in range(21) for c in "xyz"]
writer.writerow(header)

frame_num = 0
print("recording landmarks... press q to stop")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        row = [frame_num]
        for lm in hand.landmark:
            row.extend([lm.x, lm.y, lm.z])
        writer.writerow(row)
        frame_num += 1

    cv2.imshow("recording", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
outfile.close()
print(f"saved {frame_num} frames")
