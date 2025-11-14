class RuleBasedClassifier:
    def __init__(self):
        pass

    def classify(self, features):
        """
        features = {
            "thumb": 0/1,
            "index": 0/1,
            "middle": 0/1,
            "ring": 0/1,
            "pinky": 0/1,
            "palm_orientation": "front" / "side" / "down",
            "hand_movement": "forward" / "none"
        }
        """

        thumb = features["thumb"]
        index = features["index"]
        middle = features["middle"]
        ring = features["ring"]
        pinky = features["pinky"]
        palm = features["palm_orientation"]
        move = features["hand_movement"]

        if all([thumb, index, middle, ring, pinky]) and palm == "front" and move == "none":
            return "Hello"

        if all([thumb, index, middle, ring, pinky]) and palm == "front" and move == "forward":
            return "Thank You"

        if not thumb and not index and not middle and not ring and not pinky:
            return "Yes"

        if index and middle and not ring and not pinky:
            return "No"

        if all([thumb, index, middle, ring, pinky]) and palm == "front":
            return "Stop"

        return "Unknown"
