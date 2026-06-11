import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import(
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from xgboost import XGBClassifier
df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)
df.drop(
    "customerID",
    axis=1,
    inplace=True
)
y=df["Churn"].map({
    "Yes": 1,
    "No": 0
})
X=df.drop("Churn", axis=1)
categorical_cols = X.select_dtypes(
    include=["object"]
).columns

numerical_cols = X.select_dtypes(
    exclude=["object"]
).columns
print("Categorical columns:", categorical_cols
      )
print("Numerical columns:", numerical_cols
      )
numeric_pipeline=Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")
        )
    ]
)
categorical_pipeline=Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)
#Combine pipelines
preprocessor=ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numerical_cols
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_cols
        )
    ]

)
#Build Xgboost Pipeline
pipeline=Pipeline(
    steps=[
        ("preprocessor", preprocessor),
    ('model',XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    ))
    ]

)
#Split data
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.2,
                                                    random_state=42,stratify=y)
#Train model
pipeline.fit(X_train, y_train)
#Predict
pred=pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]
#Evaluate
print("Accuracy:", accuracy_score(y_test, pred))    
print("Classification Report:\n", classification_report(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
auc_score = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {auc_score:.4f}")
#Plot ROC curve
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc_score:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Customer Churn Prediction")

plt.legend()
plt.grid(True)

plt.show()
#Save model
joblib.dump(pipeline,"models/customer_churn_model.pkl")
print("Model saved to models/customer_churn_model.pkl")
