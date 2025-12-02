import csv
import cv2
import mediapipe as mp
import numpy as np
from features.feature_extractor import extract_features

def save_sample(label: str, feature_vector):
    with open("dataset.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label] + list(feature_vector))

def main():
    label = input("Enter gesture label (A, B, C, etc.): ").strip()
    total_samples = int(input("How many samples to record? (recommended: 30): "))

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)

    cap = cv2.VideoCapture(0)
    saved = 0

    while saved < total_samples:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand_lm = result.multi_hand_landmarks[0].landmark

            #convert to numpy array shape (21, 3)
            hand_np = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm], dtype=np.float32)

            ml_vector = extract_features(hand_np)

            # Save to dataset.csv
            save_sample(label, ml_vector)
            saved += 1

            print(f"Saved sample {saved}/{total_samples}")

            mp.solutions.drawing_utils.draw_landmarks(
                frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
            )

        cv2.putText(frame, f"Collecting '{label}' {saved}/{total_samples}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Gesture Data Collection", frame)

        #press esc to stop
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n data collection complete")
    print("saved to dataset.csv")


if __name__ == "__main__":
    main()