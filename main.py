import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


dataset = pd.read_csv('./data.csv')
OUTPUT_COLUMN = 'NObeyesdad'

categoricals = ['Gender', 'CAEC', 'CALC', 'MTRANS', 'SMOKE', 'SCC',
                'family_history_with_overweight', 'FAVC', ]

numericals = ['Age', 'Height', 'Weight', 'FCVC', 
              'NCP', 'CH2O',  'FAF', 'TUE']

X = dataset.drop(OUTPUT_COLUMN, axis=1)
Y = pd.DataFrame(data=dataset[OUTPUT_COLUMN], columns=[OUTPUT_COLUMN])

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
]

preprocessing_kinds = ['PCA']

pipelines = []

for name, klass in models:
    for pre_steps in preprocessing:
        steps = [
            *pre_steps,
            (name, klass()),
        ]

        pipelines.append((name, steps))

pipelines = map(lambda t: (t[0], Pipeline(t[1])), pipelines)

SIZE = (3, 3)
plt.subplots(*SIZE)
plt.figure(figsize=(20.0, 20.0))

for (i, (name, pipeline)) in enumerate(pipelines):
    ax = plt.subplot(*SIZE, i + 1)
    ax.set_title(f'{name} with {preprocessing_kinds[i // len(models)]}')

    pipeline.fit(train_x, train_y)

    ConfusionMatrixDisplay.from_estimator(
            pipeline,
            test_x,
            test_y,
            normalize='true',
            ax=ax,
    )

    print(name)

plt.savefig('figure.png')
