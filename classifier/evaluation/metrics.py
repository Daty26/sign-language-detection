from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def print_report(y_true, y_pred, labels):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))))
