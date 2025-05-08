import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from FR import load_img_features


# 划分训练集和测试集
def split_data(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# 训练和评估模型


def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_name):
    if model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_name == "SVM":
        model = SVC(kernel='linear')
    elif model_name == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError("Invalid model name!")

    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time

    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    testing_time = end_time - start_time

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    total_time = training_time + testing_time

    print(f"{model_name} Accuracy: {accuracy}, F1 Score: {f1}")
    print(f"{model_name} Training Time: {training_time} seconds")
    print(f"{model_name} Testing Time: {testing_time} seconds")
    print(f"{model_name} Total Time: {total_time} seconds")

    # 計算混淆矩陣
    cm = confusion_matrix(y_test, y_pred)

    # 繪製混淆矩陣
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()


if __name__ == "__main__":
    # 加载数据
    X_train, y_train, X_train_sift, X_train_orb, X_train_brisk, X_train_resnet = load_img_features(
        'train.txt')
    X_test, y_test, X_test_sift, X_test_orb, X_test_brisk, X_test_resnet = load_img_features(
        'test.txt')

    # 训练和评估模型
    models = ["KNN", "SVM", "RandomForest"]
    for model_name in models:

        print(f"\n{model_name} with sift Features:")
        train_and_evaluate_model(
            X_train_sift, X_test_sift, y_train, y_test, model_name)

        print(f"\n{model_name} with orb Features:")
        train_and_evaluate_model(
            X_train_orb, X_test_orb, y_train, y_test, model_name)

        print(f"\n{model_name} with brisk Features:")
        train_and_evaluate_model(
            X_train_brisk, X_test_brisk, y_train, y_test, model_name)

        print(f"\n{model_name} with resnet Features:")
        train_and_evaluate_model(
            X_train_resnet, X_test_resnet, y_train, y_test, model_name)
