import joblib
import pandas as pd
from imblearn.combine import SMOTEENN
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    classification_report, log_loss
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
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
# SMOTEENN
def smoteenn(X_train, y_train):
    sm = SMOTEENN(random_state=47)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    return X_train, y_train
def feature_scaling(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test
# tree visualization Random Forest Classifier
# def tree_visualization(estimator, columns):
#     plt.figure(figsize=(15, 20))
#     tree.plot_tree(estimator, filled=True, feature_names=columns, class_names=['1', '0'], rounded=True)
#     plt.show()
def heatmap(y_test, y_pred):
    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm_r')
    plt.title('Confusion Matrix for LSTM Model')
    plt.xlabel('Predicted Category')
    plt.ylabel('Actual Category')
    plt.xticks([0.5, 1.5], ['Legit', 'Fraud'])
    plt.yticks([0.5, 1.5], ['Legit', 'Fraud'], rotation=0)
    plt.show()
# evaluate model
def evaluate_model(y_test, y_pred, y_pred_probs):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_probs)
    print('Confusion Matrix: ', cm)
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1 Score: ', f1)
    print('Log Loss: ', logloss)


if __name__ == '__main__':
    data = read_data()
    X_train, X_test, y_train, y_test = partition_data(data)
################## Imbalance Algorithm ##################
    X_train, y_train = smote(X_train, y_train)
    # X_train, y_train = adasyn(X_train, y_train)
    # X_train, y_train = smoteenn(X_train, y_train)
################## Feature Scaling ##################
    X_train, X_test = feature_scaling(X_train, X_test)
################## Random Forest Classifier ##################
    # class_weight='balanced',
    classifier = RandomForestClassifier(n_estimators=500, random_state=47, criterion='entropy', n_jobs=-1)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred_probs = classifier.predict_proba(X_test)
    evaluate_model(y_test, y_pred, y_pred_probs)
################## CatBoost Classifier ##################
    classifier = CatBoostClassifier(iterations=500, depth=15, learning_rate=0.01, loss_function='Logloss', random_seed=47, l2_leaf_reg = 3)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred_probs = classifier.predict_proba(X_test)
    # evaluate_model(y_test, y_pred, y_pred_probs)

################ Long Short Term Memory ################
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    modellstm = Sequential()
    modellstm.add(LSTM(200, input_shape=(X_train.shape[1], X_train.shape[2])))
    modellstm.add(Dropout(0.2))
    modellstm.add(Dense(1, activation='sigmoid'))
    modellstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = modellstm.fit(X_train, y_train, epochs=50, batch_size=70, validation_data=(X_test, y_test), verbose=2, shuffle=False)
    # test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    # print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')
    y_pred_probs = modellstm.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)
    evaluate_model(y_test, y_pred, y_pred_probs)




################## Multilayer Perceptron ##################
    # model = Sequential()
    # model.add(Dense(64, input_dim=21, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # history = model.fit(X_train, y_train, epochs=10, batch_size=70, validation_data=(X_test, y_test), verbose=2, shuffle=False)
    # test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    # print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')
################## Autoencoder for label data ##################
    # define encoder
    # input_dim = X_train.shape[1]
    # input_layer = tf.keras.layers.Input(shape=(input_dim,))
    # encoder = tf.keras.layers.Dense(14, activation='relu')(input_layer)
    # encoder = tf.keras.layers.Dense(7, activation='relu')(encoder)
    # decoder = tf.keras.layers.Dense(14, activation='relu')(encoder)
    # decoder = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoder)
    # autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)
    # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    # autoencoder.summary()
    # history = autoencoder.fit(X_train, X_train, epochs=10, batch_size=70, validation_data=(X_test, X_test), verbose=2, shuffle=False)


    # save the model
    joblib.dump(classifier, 'RFC_model.joblib')


















