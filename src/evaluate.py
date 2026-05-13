from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(best_model, X_test, Y_test):

    Y_pred = best_model.predict(X_test)

    acc = accuracy_score(Y_test, Y_pred)
    cm = confusion_matrix(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred)

    print('Accuracy_Score:',acc)
    print('confusion_matrix:',cm)
    print('classification_report:',report)


