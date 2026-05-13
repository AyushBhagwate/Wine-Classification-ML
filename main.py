# Lib
from src.data_preprocessing import get_pipeline
from src.train_model import train_model
from src.evaluate import evaluate_model
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# Loading Dataset :
df = pd.read_csv('data/wine_classification.csv')

X = df.drop('target',axis=1)
Y = df['target']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

#Pipeline :
model = get_pipeline()

#  Use RandomForest model + Hyperparameter Tuning :
best_model = train_model(model, X_train, Y_train)


# Evaluate the model using ('Accuracy', 'Confusion_matrix', 'Classification_report')
evaluate_model(best_model, X_test, Y_test)


# Important Features :
tree = best_model.named_steps['model']

importances = pd.Series(tree.feature_importances_, index=X_train.columns)
print('Important_Features: \n',importances.sort_values(ascending=False).head())


# Saving Model:
with open ('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)


# Saving in Outputs :
# 1 Saving predictions
preds = best_model.predict(X_test)

df_outputs = pd.DataFrame({
    'Actual:' : Y_test,
    'predictions:' : preds
})

df_outputs.to_csv('outputs/Predictions.csv', index=False)

# 2 Saving metrics :
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

acc = accuracy_score(Y_test, preds)
cm = confusion_matrix(Y_test, preds)
report = classification_report(Y_test, preds)

with open('outputs/metrics.txt', 'w') as f:
    f.write(f'Acc : {acc}\n')
    f.write(f'Cm : {cm}\n')
    f.write(f'Report : {report}\n')


# 3 Save plotting 
# Feature Importance :
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.barplot(x=importances.values, y=importances.index)
plt.title('Feature Importance (RandomForest)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

plt.savefig('outputs/feature_importance.png')
plt.close()

print('Plot-Saved in the outputs/folder')