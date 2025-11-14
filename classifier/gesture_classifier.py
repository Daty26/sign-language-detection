from classifier.rule_based import RuleBasedClassifier

class GestureClassifier:
    def __init__(self):
        self.rule_classifier = RuleBasedClassifier()

    def classify(self, features):
        result = self.rule_classifier.classify(features)

        if result is None:
            return "Unknown"

        return result
