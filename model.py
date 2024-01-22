import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.under_sampling import NearMiss, TomekLinks, RandomUnderSampler

warnings.filterwarnings('ignore')

data = pd.read_csv('final_dataset.csv')
print('File loaded.')

# Since the arrival information won't be available when making real-time predictions, we drop the columns pertaining to such information.
# The column 'CANCELLED' is also to be dropped as it contains redundant values and has no further use.

data = data.drop(['CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY', 'CANCELLED'], axis=1)
data = data.dropna(axis=0)
print('quick debug msg')

x = data[['CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON', 'TAXI_IN', 'CARRIER_DELAY',
          'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']]
y = data['Delayed']

undersample = NearMiss()  # RandomUnderSampler(random_state=42, replacement=True)
x, y = undersample.fit_resample(x, y)

resultant_dataset = pd.DataFrame(x)
resultant_dataset['Delayed'] = y

print('Under-sampled dataframe created.')

resultant_dataset.to_csv('finale_undersampled_ds2.csv')

print('Under sampled dataset saved.')

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=69, test_size=0.2, shuffle=True)
classifiers = {
    'XGBoost Classifier': XGBClassifier(),
    'Decision Tree CLassifier': DecisionTreeClassifier(),
    'Random Forest Classifier': RandomForestClassifier(),
    'Hist-Gradient Boost Classifier': HistGradientBoostingClassifier(),
    'Logistic Regression Classifier': LogisticRegression(max_iter=int(data.shape[0]))
}

for name, model in classifiers.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    cm = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=model.classes_)
    color = 'white'
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.show()

    print(f'{name}:\nAccuracy = {accuracy:.2f}\n')
    print(report)
    print('=' * 80)

print('Program terminated.')
