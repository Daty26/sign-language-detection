import cv2
from landmarks import MediaPipeHandDetector, extract_first_hand

def main():
    cap = cv2.VideoCapture(0)  
    detector = MediaPipeHandDetector()

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр с камеры")
            break

        results = detector.process(frame)
        hand = extract_first_hand(results, frame.shape)

        if hand is not None:
            print(
                "shape:", hand.landmarks.shape,
                "hand:", hand.handedness,
                "score:", round(hand.score, 2),
            )
        else:
            print("no hand")

        # Покажем картинку для контроля
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
