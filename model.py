import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Function to determine if hand is Left or Right
def get_hand_label(hand_landmarks):
    """
    Determines if the detected hand is Left or Right based on landmark positions.
    Returns 'Left' or 'Right'.
    """
    if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x:
        return "Right"
    else:
        return "Left"

# Function to count the number of open fingers
def count_open_fingers(hand_landmarks):
    """
    Counts the number of extended fingers.
    Returns an integer count.
    """
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

    # Thumb special case (compared with Index MCP)
    if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x:
        fingers.append(1)  # Thumb extended
    else:
        fingers.append(0)  # Thumb folded

    # Other four fingers
    for i in range(1, 5):  
        if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
            fingers.append(1)  # Finger extended
        else:
            fingers.append(0)  # Finger folded

    return sum(fingers)

# Start webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert image for MediaPipe processing
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Convert image back for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get hand label (Left/Right)
                hand_label = get_hand_label(hand_landmarks)

                # Count open fingers
                finger_count = count_open_fingers(hand_landmarks)

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Get bounding box for hand
                x_min = min([lm.x for lm in hand_landmarks.landmark])
                y_min = min([lm.y for lm in hand_landmarks.landmark])
                x_max = max([lm.x for lm in hand_landmarks.landmark])
                y_max = max([lm.y for lm in hand_landmarks.landmark])

                # Convert to pixel coordinates
                h, w, _ = image.shape
                x_min, y_min, x_max, y_max = int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)

                # Draw rectangle around hand
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)

                # Display hand label
                cv2.putText(image, hand_label, (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)

                # Display finger count
                cv2.putText(image, str(finger_count), (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)

        # Show output
        cv2.imshow('Hand Recognition', cv2.flip(image, 1))

        # Press 'ESC' to exit
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()

#only include to close the opened wabcam tab
cv2.destroyAllWindows()
