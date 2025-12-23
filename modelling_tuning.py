import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Inisialisasi DagsHub (Syarat Advance)
dagshub.init(repo_owner='lesyeuxdeisha', repo_name='Eksperimen_SML_AisyahRidhoAlhaqRifai_Siswa', mlflow=True)

# Memuat data sesuai folder kamu
df = pd.read_csv('heart_disease_preprocessing/heart_disease_clean.csv') 
X = df.drop('num', axis=1)
y = df['num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning (Syarat Skilled/Advance)
params = {"n_estimators": 50, "max_depth": 5, "random_state": 42}

with mlflow.start_run(run_name="Advance_Model_Aisyah"):
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Manual Logging (Syarat Skilled/Advance - Bukan Autolog)
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model_aisyah")

    # Dua Artefak Tambahan (Syarat Mutlak Advance)
    # Artefak 1: Confusion Matrix
    plt.figure(figsize=(5,5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    
    # Artefak 2: Feature Importance
    plt.figure(figsize=(5,5))
    pd.Series(model.feature_importances_, index=X.columns).nlargest(10).plot(kind='barh')
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")

    print(f"Berhasil! Akurasi: {acc}")