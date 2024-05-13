import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold, RFE

dataset = pd.read_csv('./data.csv')
OUTPUT_COLUMN = 'NObeyesdad'

categoricals = ['Gender', 'CAEC', 'CALC', 'MTRANS', 'SMOKE', 
                'SCC', 'family_history_with_overweight', 'FAVC', ]

numericals = ['Age', 'Height', 'Weight', 'FCVC', 
              'NCP', 'CH2O',  'FAF', 'TUE']

X = dataset.drop(OUTPUT_COLUMN, axis=1)
Y = pd.DataFrame(data=dataset[OUTPUT_COLUMN], columns=[OUTPUT_COLUMN])

X.boxplot(['Age'])
plt.savefig('age-boxplot.png')
plt.cla()

X.boxplot(['Height'])
plt.savefig('height-boxplot.png')
plt.cla()

X.boxplot(['Weight'])
plt.savefig('weight-boxplot.png')
plt.cla()

categorical_pipeline = Pipeline([
    ('Missing Values', SimpleImputer(strategy='most_frequent')),
    ('encoding', OrdinalEncoder()),
])

numerical_pipeline = Pipeline([
    ('Missing Values', SimpleImputer()),
])

encoding = ColumnTransformer([
    ('Categorical', categorical_pipeline, categoricals), 
    ('Numerical', numerical_pipeline, numericals),
])

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)
train_y = train_y.values.ravel()
test_y =  test_y.values.ravel()

models = [
    ('Naive Bayes', GaussianNB),
    ('Decision Tree', DecisionTreeClassifier),
    ('Multi-Layer Perceptron', MLPClassifier),
    ('k-Nearest Neighbours', KNeighborsClassifier),
]

preprocessing = [
        [
            ('Initial', encoding), 
            ('Normalise', StandardScaler()), 
            ('Project', PCA()), 
        ],
        [
            ('Initial', encoding),
            ('filter', VarianceThreshold()),
        ],
        [
            ('Initial', encoding),
            ('wrap', RFE(RandomForestClassifier())),
        ],
]

preprocessing_kinds = ['PCA', 'Variance Threshold', 'RFE']

pipelines = []

for name, klass in models:
    for i, pre_steps in enumerate(preprocessing):
        steps = [
            *pre_steps,
            (name, klass()),
        ]

        pipelines.append((name, steps, preprocessing_kinds[i]))

pipelines = map(lambda t: (t[0], t[2], Pipeline(t[1])), pipelines)

scores = []

for (i, (name, preprocessing_kind, pipeline)) in enumerate(pipelines):
    plt.title(f'{name}\nwith {preprocessing_kind}')

    pipeline.fit(train_x, train_y)

    score = cross_val_score(pipeline, test_x, test_y, n_jobs=1)
    scores.append((name, preprocessing_kind, score))

    ConfusionMatrixDisplay.from_estimator(
            pipeline,
            test_x,
            test_y,
            normalize='true',
    )

    print(name)

    plt.savefig(f'{name}-{preprocessing_kind}.png')
    plt.cla()

L = 15

print('=' * L + '***' + '=' * L)

for n, k, s in scores:
    
    s = np.nan_to_num(s)
    print(f'{n} - {k}')
    print(f'Scores: {s}\nMean: {s.mean()}, std: {s.std()}')

    print('=' * L + '***' + '=' * L)


