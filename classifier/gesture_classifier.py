from classifier.rule_based import RuleBasedClassifier
from classifier.ml_classifier import MLClassifier

class GestureClassifier:
    def __init__(self, confidence_threshold=0.8, use_rule_based=True):
        self.use_rule_based = use_rule_based
        self.rule = RuleBasedClassifier() if use_rule_based else None

        self.ml = MLClassifier(path="classifier/model.pkl")
        self.confidence_threshold = confidence_threshold

        try:
            self.ml.load()
            self.use_ml = True
        except Exception as e:
            print(f"ML not loaded: {e}. Using rule-based only.")
            self.use_ml = False

    def classify(self, features, ml_vector=None):
        if self.use_rule_based and self.rule is not None:
            result = self.rule.classify(features)
            if result != "Unknown":
                return result

        if self.use_ml and ml_vector is not None:
            try:
                prediction, confidence = self.ml.predict_with_confidence(ml_vector)
                if confidence >= self.confidence_threshold:
                    return prediction
            except Exception as e:
                print(f"ML prediction error: {e}")

        return "Unknown"