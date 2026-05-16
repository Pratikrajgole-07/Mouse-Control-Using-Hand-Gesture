import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

screen_width, screen_height = pyautogui.size()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(False, 2, 1, 0.7, 0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

smoothening = 7
prev_x, prev_y = 0, 0

frame_reduction = 100

last_click_time = 0
last_right_click_time = 0
last_double_click_time = 0
last_volume_time = 0

click_delay = 0.4
right_click_delay = 0.5
double_click_delay = 0.7
volume_delay = 0.3

screenshot_delay = 4
both_hands_start_time = None
screenshot_taken = False

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    cv2.rectangle(frame, (frame_reduction, frame_reduction),
                  (w - frame_reduction, h - frame_reduction),
                  (255, 0, 255), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Screenshot
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        if both_hands_start_time is None:
            both_hands_start_time = time.time()

        if time.time() - both_hands_start_time >= screenshot_delay and not screenshot_taken:
            pyautogui.screenshot(f"screenshot_{int(time.time())}.png")
            screenshot_taken = True
    else:
        both_hands_start_time = None
        screenshot_taken = False

    if results.multi_hand_landmarks:

        thumb_count = 0

        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x1, y1 = int(lm[8].x * w), int(lm[8].y * h)
            x2, y2 = int(lm[4].x * w), int(lm[4].y * h)
            x3_m, y3_m = int(lm[12].x * w), int(lm[12].y * h)

            screen_x = np.interp(x1, (frame_reduction, w - frame_reduction), (0, screen_width))
            screen_y = np.interp(y1, (frame_reduction, h - frame_reduction), (0, screen_height))

            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening

            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Left Click
            if np.hypot(x2 - x1, y2 - y1) < 30:
                if time.time() - last_click_time > click_delay:
                    pyautogui.click()
                    last_click_time = time.time()

            # Right Click
            if np.hypot(x3_m - x1, y3_m - y1) < 30:
                if time.time() - last_right_click_time > right_click_delay:
                    pyautogui.rightClick()
                    last_right_click_time = time.time()

            # Finger States
            thumb_up = lm[4].y < lm[3].y
            index_up = lm[8].y < lm[6].y
            middle_up = lm[12].y < lm[10].y
            ring_up = lm[16].y < lm[14].y
            pinky_up = lm[20].y < lm[18].y

            # Double Click
            if thumb_up and index_up and not middle_up and not ring_up and not pinky_up:
                if time.time() - last_double_click_time > double_click_delay:
                    pyautogui.doubleClick()
                    last_double_click_time = time.time()

            # Count thumbs-up gesture
            if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
                thumb_count += 1

        # Volume Control
        if thumb_count == 1:
            if time.time() - last_volume_time > volume_delay:
                pyautogui.press("volumeup")
                last_volume_time = time.time()
                cv2.putText(frame, "Volume Up", (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

        elif thumb_count == 2:
            if time.time() - last_volume_time > volume_delay:
                pyautogui.press("volumedown")
                last_volume_time = time.time()
                cv2.putText(frame, "Volume Down", (50, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    cv2.imshow("Hand Mouse AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()