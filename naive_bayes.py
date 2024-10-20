import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Fungsi untuk memuat data
def load_data(file_path):
    data = pd.read_csv(file_path)
    features = data.iloc[:, [2, 3]].values
    target = data.iloc[:, -1].values
    return features, target

# Fungsi untuk mempersiapkan data
def prepare_data(features, target, test_size=0.2, random_seed=42):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_seed)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

# Fungsi untuk melatih model dan melakukan prediksi
def train_and_predict(X_train, y_train, X_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model.predict(X_test)

# Fungsi untuk mengevaluasi model
def evaluate_model(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    acc_score = accuracy_score(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)
    return conf_matrix, acc_score, class_report

# Fungsi untuk visualisasi hasil
def visualize_results(X_test, y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], c='blue', label='Class 0 Sesungguhnya')
    plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], c='orange', label='Class 1 Sesungguhnya')
    plt.scatter(X_test[y_pred != y_test][:, 0], X_test[y_pred != y_test][:, 1], c='red', marker='x', s=100, label='Miss klasifikasi')
    plt.title('Hasil Klasifikasi Naive Bayes')
    plt.xlabel('Fitur Skala 1')
    plt.ylabel('Fitur Skala 2')
    plt.legend()
    plt.show()

# Eksekusi utama
if __name__ == "__main__":
    # Memuat data
    X, y = load_data('Social_Network_Ads2.csv')
    
    # Siapkan data
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    
    # Train dan prediksi
    y_pred = train_and_predict(X_train, y_train, X_test)
    
    # Evaluasi Model
    conf_matrix, acc_score, class_report = evaluate_model(y_test, y_pred)
    
    # Menampilkan hasil
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"\nAccuracy: {acc_score:.4f}")
    print("\nClassification Report:")
    print(class_report)
    
    # Visualisasi hasil
    visualize_results(X_test, y_test, y_pred)