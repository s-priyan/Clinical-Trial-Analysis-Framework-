from sklearn.metrics import f1_score
import pandas as pd
from sklearn.metrics import confusion_matrix

# Load the CSV 
df = pd.read_csv('../complete_dataset/CSV_files/results_ada.csv')
print(len(df))

# Drop rows with missing labels 
df = df.dropna(subset=["label", "prediction"])

print(len(df))

# set labels true and prediction
y_true = df["label"]
y_pred = df["prediction"]

micro_f1 = f1_score(y_true, y_pred, average="macro")

print(micro_f1)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)


print(cm)
