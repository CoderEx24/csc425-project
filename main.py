import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import ConfusionMatrixDisplay
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

sns.heatmap(dataset[numericals].corr(method='spearman', min_periods=1), annot=True)
plt.savefig('heatmap.png')
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
    ('Naive Bayes', GaussianNB, {}),
    ('Decision Tree', DecisionTreeClassifier, {}),
    ('Multi-Layer Perceptron', MLPClassifier, {'max_iter': 10000}),
    ('k-Nearest Neighbours', KNeighborsClassifier, {}),
]

preprocessing = [
        [
            'PCA',
            ('Initial', encoding), 
            ('Normalise', StandardScaler()), 
            ('Project', PCA()), 
        ],
        [
            'Variance Threshold',
            ('Initial', encoding),
            ('filter', VarianceThreshold()),
        ],
        [
            'RFE',
            ('Initial', encoding),
            ('wrap', RFE(RandomForestClassifier())),
        ],
]

pipelines = []

for name, klass, kwargs in models:
    for pre_steps in preprocessing:
        preprocessing_kind = pre_steps[0]
        steps = [
            *pre_steps[1:],
            (name, klass(**kwargs)),
        ]

        pipelines.append((name, steps, preprocessing_kind))

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

    plt.savefig(f'{name}-{preprocessing_kind}-cm.png')
    plt.cla()

    plt.title(f'{name}\nwith {preprocessing_kind}')
    
L = 30

print('=' * L + '***' + '=' * L)

for n, k, s in scores:
    s = s[~np.isnan(s)]
    print(f'{n} - {k}')
    print(f'Scores: {s}\nMean: {s.mean()}, std: {s.std()}')

    print('=' * L + '***' + '=' * L)


