import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3
import time

# Load model and labels
model = load_model('gesture_model.h5')
classes = ['super','call me','hello','i want to eat']
IMAGE_SIZE = 64

# TTS setup
engine = pyttsx3.init()

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

# Webcam
cap = cv2.VideoCapture(0)

# Variables to track gesture state
current_prediction = None
start_time = 0
spoken = False
required_hold_time = 2  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture_detected = False

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get bounding box
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min = int(min(x_coords) * w) - 20
            y_min = int(min(y_coords) * h) - 20
            x_max = int(max(x_coords) * w) + 20
            y_max = int(max(y_coords) * h) + 20

            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            hand_img = frame[y_min:y_max, x_min:x_max]
            try:
                gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
                input_img = resized.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1) / 255.0

                pred = model.predict(input_img, verbose=0)
                class_id = np.argmax(pred)
                class_name = classes[class_id]
                gesture_detected = True

                # If it's a new gesture, reset timer
                if class_name != current_prediction:
                    current_prediction = class_name
                    start_time = time.time()
                    spoken = False

                # If same gesture is held for 3 seconds and not yet spoken
                elif not spoken and (time.time() - start_time) >= required_hold_time:
                    engine.say(class_name)
                    engine.runAndWait()
                    spoken = True

                # Show prediction on screen
                cv2.putText(frame, class_name, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            except:
                pass

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # If no gesture detected, reset
    if not gesture_detected:
        current_prediction = None
        start_time = 0
        spoken = False

    cv2.imshow("Sign to Voice Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

