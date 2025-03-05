import pandas as pd

df= pd.read_csv('path/to/dataset.csv')
df.fillna(method='ffill', inplace=True)  # Forward fill
from sklearn.preprocessing import StandardScaler
#normalize/scale fetures
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
#encode categorical vars.
df_encoded = pd.get_dummies(df, columns=['categorical_column'])

#featurrenginering to new/change old fetures..

#dimensionality redn.

#model 1 sel
import xgboost as xgb

model = xgb.XGBClassifier()
model.fit(X_train, y_train)
#model2 sel
from keras.models import Sequential
from keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(50, input_shape=(timesteps, features)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#modelEval
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
##metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
##confusinMatrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.show()
