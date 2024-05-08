import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv('./data.csv')

categoricals = ['Gender', 
                'CAEC', 
                'CALC', 
                'MTRANS', ]
#'NObeyesdad']

X = dataset.drop('NObeyesdad', axis=1)
print(dataset['NOBeyesdad'])
Y = OrdinalEncoder().fit_transform(dataset['NObeyesdad'])

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)

print(X.describe())

models = [
        ('Naive Bayes', GaussianNB),
        ('Multi-Layer Perceptron', MLPClassifier),
        ('Decision Tree', DecisionTreeClassifier),
        ('k-Nearest Neighbours', KNeighborsClassifier),
]

encoding = ColumnTransformer(
        [('categorical', OrdinalEncoder(), categoricals)],
        remainder='passthrough',
)

preprocessing = [
        [('Encode', encoding)],
        #[('Encode', OrdinalEncoder()), ('Normalise', StandardScaler()), ('Project', PCA()),],
]

pipelines = []

for name, klass in models:
    for pre_steps in preprocessing:
        steps = [
            *pre_steps,
            (name, klass()),
        ]

        pipelines.append(steps)

pipelines = map(lambda steps: Pipeline(steps), pipelines)

for pipeline in pipelines:
    pipeline.fit(train_x, train_y)

    print(classification_report(test_y, pipeline.predict(test_x)))

