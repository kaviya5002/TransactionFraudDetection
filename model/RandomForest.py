
import sys
sys.path.append("../dbpreprocessing")  # ensure custom modules are accessible

from data_loader import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

dataset_path = "../dataset/creditcard.csv"
loader = DataLoader(dataset_path)
data = loader.load_data()
loader.summarize_data(data)  # optional summary

X = data.drop("Class", axis=1)
y = data["Class"]


scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X["Amount"].values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n🚀 Training Random Forest Model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_test)

print("\n✅ Model Evaluation Results:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 6))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


feat_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = feat_importances.nlargest(10)

plt.figure(figsize=(10,6))
top_features.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("🔍 Top 10 Important Features in Fraud Detection")
plt.ylabel("Feature Importance Score")
plt.xlabel("Features")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


plt.savefig("../static/feature_importance.png")

def predict_transaction(transaction):
    """
    Predicts if a single transaction is fraudulent (1) or not (0).
    Example:
        transaction = {
            'Time': 1000, 'V1': -1.5, 'V2': 0.7, ..., 'Amount': 50
        }
    """
    df = pd.DataFrame([transaction])
    df["Amount"] = scaler.transform(df[["Amount"]])
    prediction = rf_model.predict(df)

    if prediction[0] == 1:
        print("\n🚨 ALERT: Fraudulent Transaction Detected!")
    else:
        print("\n✅ Legitimate Transaction.")


if __name__ == "__main__":
    sample_transaction = {
        'Time': 1000,
        'V1': -1.3598071336738,
        'V2': -0.0727811733098497,
        'V3': 2.53634673796914,
        'V4': 1.37815522427443,
        'V5': -0.338320769942518,
        'V6': 0.462387777762292,
        'V7': 0.239598554061257,
        'V8': 0.0986979012610507,
        'V9': 0.363786969611213,
        'V10': 0.0907941719789316,
        'V11': -0.551599533260813,
        'V12': -0.617800855762348,
        'V13': -0.991389847235408,
        'V14': -0.311169353699879,
        'V15': 1.46817697209427,
        'V16': -0.470400525259478,
        'V17': 0.207971241929242,
        'V18': 0.0257905801985591,
        'V19': 0.403992960255733,
        'V20': 0.251412098239705,
        'V21': -0.018306777944153,
        'V22': 0.277837575558899,
        'V23': -0.110473910188767,
        'V24': 0.0669280749146731,
        'V25': 0.128539358273528,
        'V26': -0.189114843888824,
        'V27': 0.133558376740387,
        'V28': -0.0210530534538215,
        'Amount': 50
    }

    predict_transaction(sample_transaction)
