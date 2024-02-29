import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
def read_data():
    data = pd.read_csv('data/EFraud_Data_Country_Processed.csv')
    return data

# data partitioning
def partition_data(data):
    train = data.drop(['class'], axis=1)
    test = data['class']
    X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.2, random_state=47)
    return X_train, X_test, y_train, y_test
# feature imbalance class-SMOTE
def smote(X_train, y_train):
    sm = SMOTE(random_state=47)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    return X_train, y_train
# ADASYN
def adasyn(X_train, y_train):
    ad = ADASYN(random_state=47)
    X_train, y_train = ad.fit_resample(X_train, y_train)
    return X_train, y_train
def feature_scaling(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test

# evaluate model
def evaluate_model(y_test, y_pred):

    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # print results
    print('Confusion Matrix: ', cm)
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1 Score: ', f1)

if __name__ == '__main__':
    data = read_data()
    X_train, X_test, y_train, y_test = partition_data(data)
    X_train, X_test = feature_scaling(X_train, X_test)
    # X_train, y_train = smote(X_train, y_train)
    X_train, y_train = adasyn(X_train, y_train)
    X_train, X_test = feature_scaling(X_train, X_test)
    classifier = RandomForestClassifier(n_estimators=500, random_state=47, criterion='entropy', class_weight='balance', n_jobs=-1)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    evaluate_model(y_test, y_pred)





