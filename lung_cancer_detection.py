import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
import pickle

DATASET_PATH = 'lung_cancer_survey.csv'
MODEL_PATH = 'lung_cancer_model.pkl'


def load_dataset():
    dataset = pd.read_csv(DATASET_PATH)
    print('Dataset head:')
    print(dataset.head())
    print(f'\nShape: {dataset.shape}')
    print(f'Null values present: {dataset.isnull().values.any()}')
    return dataset


def plot_label_distribution(dataset):
    label_counts = dataset['LUNG_CANCER'].value_counts()
    plt.figure(figsize=(8, 8))
    label_counts.plot.pie(autopct='%1.1f%%')
    plt.title('Distribution of Label Column')
    plt.ylabel('')
    plt.show()


def prepare_data(dataset):
    df = pd.DataFrame(dataset)
    dataset2 = pd.get_dummies(df, columns=['GENDER', 'LUNG_CANCER'])
    print('\nEncoded dataset preview:')
    print(dataset2.head())

    X = dataset2.drop(columns=['LUNG_CANCER_YES'])
    y = dataset2['LUNG_CANCER_YES']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_knn(X_train_scaled, X_test_scaled, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    y_knn_pred = knn.predict(X_test_scaled)

    accuracy_knn = accuracy_score(y_test, y_knn_pred)
    print(f'\nKNN Accuracy: {accuracy_knn}')
    print('KNN Classification Report:')
    print(classification_report(y_test, y_knn_pred))

    cm_knn = confusion_matrix(y_test, y_knn_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_knn,
        annot=True,
        fmt='d',
        cmap='summer',
        xticklabels=np.unique(y_test),
        yticklabels=np.unique(y_test),
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('KNN Confusion Matrix')
    plt.show()

    return knn


def train_kmeans(X_train_scaled, X_test_scaled, y_train, y_test):
    kmeans = KMeans(n_clusters=len(np.unique(y_train)), random_state=42, n_init=10)
    kmeans.fit(X_train_scaled)
    y_kmeans_pred = kmeans.predict(X_test_scaled)

    mapping = {}
    for cluster_id in range(len(np.unique(kmeans.labels_))):
        mask = (kmeans.labels_ == cluster_id)
        if len(np.bincount(y_train[mask])) > 0:
            mapping[cluster_id] = np.bincount(y_train[mask]).argmax()

    y_kmeans_mapped = np.array([mapping[label] for label in y_kmeans_pred])

    accuracy_kmeans = accuracy_score(y_test, y_kmeans_mapped)
    print(f'\nK-Means Accuracy: {accuracy_kmeans}')
    print('K-Means Classification Report:')
    print(classification_report(y_test, y_kmeans_mapped, zero_division=1))

    cm_kmeans = confusion_matrix(y_test, y_kmeans_mapped)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_kmeans,
        annot=True,
        fmt='d',
        cmap='summer',
        xticklabels=np.unique(y_test),
        yticklabels=np.unique(y_test),
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('K-Means Confusion Matrix')
    plt.show()


def train_naive_bayes(X_train_scaled, X_test_scaled, y_train, y_test):
    nb = GaussianNB()
    nb.fit(X_train_scaled, y_train)
    y_nb_pred = nb.predict(X_test_scaled)

    accuracy_nb = accuracy_score(y_test, y_nb_pred)
    print(f'\nNaive Bayes Accuracy: {accuracy_nb}')
    print('Naive Bayes Classification Report:')
    print(classification_report(y_test, y_nb_pred))

    cm_nb = confusion_matrix(y_test, y_nb_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_nb,
        annot=True,
        fmt='d',
        cmap='summer',
        xticklabels=np.unique(y_test),
        yticklabels=np.unique(y_test),
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Naive Bayes Confusion Matrix')
    plt.show()

    return nb


def train_decision_tree(X_train_scaled, X_test_scaled, y_train, y_test):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_scaled, y_train)
    y_dt_pred = dt.predict(X_test_scaled)

    accuracy_dt = accuracy_score(y_test, y_dt_pred)
    print(f'\nDecision Tree Accuracy: {accuracy_dt}')
    print('Decision Tree Classification Report:')
    print(classification_report(y_test, y_dt_pred))

    cm_dt = confusion_matrix(y_test, y_dt_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_dt,
        annot=True,
        fmt='d',
        cmap='summer',
        xticklabels=np.unique(y_test),
        yticklabels=np.unique(y_test),
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Decision Tree Confusion Matrix')
    plt.show()

    return dt


def save_model(model, scaler):
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump({'model': model, 'scaler': scaler}, file)
    print(f'\nModel saved to {MODEL_PATH}')


if __name__ == '__main__':
    dataset = load_dataset()
    plot_label_distribution(dataset)

    X_train, X_test, y_train, y_test = prepare_data(dataset)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_knn = train_knn(X_train_scaled, X_test_scaled, y_train, y_test)
    train_kmeans(X_train_scaled, X_test_scaled, y_train, y_test)
    model_nb = train_naive_bayes(X_train_scaled, X_test_scaled, y_train, y_test)
    model_dt = train_decision_tree(X_train_scaled, X_test_scaled, y_train, y_test)

    # Save the best-performing model (Decision Tree in this notebook)
    save_model(model_dt, scaler)

    print('\nCompleted lung cancer detection training pipeline.')
