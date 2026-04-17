import cv2
import mediapipe as mp
import pyautogui

pyautogui.FAILSAFE = False

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

screen_w, screen_h = pyautogui.size()

def finger_status(hand_landmarks):
    """Return list of fingers: 1 = open, 0 = closed"""
    tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    fingers = []

    # Thumb
    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other four fingers
    for id in range(1, 5):
        if hand_landmarks.landmark[tips[id]].y < hand_landmarks.landmark[tips[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for lmSet in results.multi_hand_landmarks:
            lm = lmSet.landmark

            # Finger detection
            fingers = finger_status(lmSet)

            index_x = int(lm[8].x * screen_w)
            index_y = int(lm[8].y * screen_h)

            # 1) Mouse Movement (Index Finger only)
            if fingers[1] == 1 and sum(fingers) == 1:
                pyautogui.moveTo(index_x, index_y)

            # 2) Left Click (Thumb + Index Finger Pinch)
            thumb = (lm[4].x, lm[4].y)
            index = (lm[8].x, lm[8].y)
            pinch_distance = abs(thumb[0] - index[0]) + abs(thumb[1] - index[1])

            if pinch_distance < 0.07:
                pyautogui.click()

            # 3) Right Click → Two Fingers Up (Index + Middle)
            if fingers[1] == 1 and fingers[2] == 1 and sum(fingers) == 2:
                pyautogui.rightClick()

            # ===== SCROLL UP (Index Finger Up Only) =====
            if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
                pyautogui.scroll(30) 

            # # ===== SCROLL DOWN (Index Finger Pointing Down) =====
            index_tip_y = lm[8].y
            index_mcp_y = lm[5].y
            # When index finger tip is LOWER than knuckle → pointing down
            if index_tip_y > index_mcp_y:
                pyautogui.scroll(-30)  # scroll down
            mp_draw.draw_landmarks(frame, lmSet, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
