from classifier.rule_based import RuleBasedClassifier
from classifier.ml_classifier import MLClassifier

class GestureClassifier:
    def __init__(self):
        self.rule = RuleBasedClassifier()
        self.ml = MLClassifier()

        try:
            self.ml.load()
            self.use_ml = True
        except:
            self.use_ml = False

    def classify(self, features, ml_vector=None):
        result = self.rule.classify(features)
        if result != "Unknown":
            return result

        if self.use_ml and ml_vector is not None:
            return self.ml.predict(ml_vector)

        return "Unknown"
