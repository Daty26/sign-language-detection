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
    
    def classify_with_confidence(self, features, ml_vector=None):
        """
        Returns: (label, confidence_percent_int)
        confidence is 0..100
        """
        # Rule-based result (treat as 100% confidence if it fires)
        if self.use_rule_based and self.rule is not None:
            result = self.rule.classify(features)
            if result != "Unknown":
                return result, 100

        # ML result
        if self.use_ml and ml_vector is not None:
            try:
                prediction, conf01 = self.ml.predict_with_confidence(ml_vector)  # 0..1
                conf_pct = int(round(conf01 * 100))
                if conf01 >= self.confidence_threshold:
                    return prediction, conf_pct
                else:
                    # below threshold => return Unknown but still show confidence if you want
                    return "Unknown", conf_pct
            except Exception as e:
                print(f"ML prediction error: {e}")

        return "Unknown", 0