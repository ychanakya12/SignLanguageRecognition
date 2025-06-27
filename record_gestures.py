import os
import numpy as np
import cv2
import mediapipe as mp

# --- CONFIG ---
PHRASE = input("Enter the phrase which you want to add in the database")                   # <- Change this per phrase
SAVE_DIR = f"data/{PHRASE}"        # Folder to save images
IMAGE_SIZE = 64                    # Resize to 64x64
MAX_IMAGES = 300                   # Total images to collect (across both hands)

# --- SETUP ---
os.makedirs(SAVE_DIR, exist_ok=True)
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,              # Allow 2 hands
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame = cv2.flip(frame, 1) #horizontal flip , Webcam feeds often appear mirrored (like a selfie). Flipping corrects this
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #Converts the frame from BGR (OpenCV format) to RGB (MediaPipe format)
    result = hands.process(img_rgb) # checks if hands were detected and handedness (left/right) is available

    if result.multi_hand_landmarks and result.multi_handedness:# if result
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            label = result.multi_handedness[i].classification[0].label  # 'Left' or 'Right'

#result.multi_hand_landmarks is a list of hand landmark objects (one per hand, up to max_num_hands=2).
#hand_landmarks: Contains 21 landmarks (e.g., fingertips, joints) with x, y, z coordinates.
#i: Index of the current hand (0 or 1).

            # Get bounding box around hand
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min = int(min(x_coords) * w) - 20
            y_min = int(min(y_coords) * h) - 20
            x_max = int(max(x_coords) * w) + 20
            y_max = int(max(y_coords) * h) + 20

            # Clamp values
            x_min, y_min = max(x_min, 0), max(y_min, 0)
            x_max, y_max = min(x_max, w), min(y_max, h)

            # Draw hand rectangle and label
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} Hand", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 255), 2)

            # Extract and preprocess hand ROI
            hand_img = frame[y_min:y_max, x_min:x_max]
            hand_gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
            hand_resized = cv2.resize(hand_gray, (IMAGE_SIZE, IMAGE_SIZE))

            # Save on 's' key
            key = cv2.waitKey(1)
            if key == ord('s') and count < MAX_IMAGES:
                filename = f"{SAVE_DIR}/{count:04d}_{label}.jpg"
                cv2.imwrite(filename, hand_resized)
                count += 1
                print(f"[+] Saved: {filename}")

    # Show Preview Frame
    cv2.putText(frame, f"Phrase: {PHRASE} | Saved: {count}/{MAX_IMAGES}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow("Gesture Recorder", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= MAX_IMAGES:
        break

cap.release()
cv2.destroyAllWindows()
