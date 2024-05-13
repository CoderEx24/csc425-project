#align(center, text(20pt)[
  Predicting Obesity
])

#set par(justify: true)

= Abstract
Obesity is a global health concern with severe implications. This project aims to classify obesity
levels using advanced machine learning techniques. We developed a model that processes
demographic, dietary, and physical activity data to predict obesity levels. Our findings indicate
that machine learning can effectively predict obesity levels, which could be instrumental in
preventive health measures.
/*
We explore a dataset concerning obesity levels in Mexico, Colombia and Peru.
The dataset consists of data regarding physical attributes like weight, height and gender,
and data regarding habits like smoking.
We compare a number of feature reduction techniques, feature selection techniques, wrapper methods
and models. Then, all combinations of the aforementioned are evaluated by cross-validation.
*/
= Introduction
The main problem addressed in this project is the accurate classification of obesity levels based
on various factors. We utilized several machine learning techniques, including Naive Bayes,
Decision Trees, and Neural Networks, to tackle this issue. The main contribution of this project
is the development of a robust model that outperforms existing benchmarks in accuracy and
efficiency. The rest of the document is organized into sections detailing related work,
methodology, proposed model, results, discussion, conclusion, and future work.
= Related Work
= Methodology
We employed various methods for data preprocessing, feature selection, and model training.
Each method’s brief description is provided, highlighting its significance in the project.
= Proposed Model
Our model consists of several phases:
- Preprocessing: Data normalization, missing value treatment, and encoding.
- Feature Selection: Techniques like Variance Threshold and RFE were used.
- Feature Reduction: PCA was implemented to reduce dimensionality.
- Classification Methods: GaussianNB, DecisionTreeClassifier, MLPClassifier, and
  KNeighborsClassifier were used.
- Evaluation Metrics: Accuracy, precision, recall, F-measure, and ROC were calculated.
/*
We use data that's been published by Fabio Mendoza Palechor and Alexis De la Hoz Manotas.
23% of the data are original, while the rest has been synthesised using Synthetic Minority Oversampling Technique
Filter (SMOTE). 
*/
= Results and Discussion
- Data Set Description: The dataset comprises demographic, dietary, and physical
  activity data.
- Preprocessing Results: We performed data visualization and handled missing values.
  Statistical analysis included calculating min, max, mean, variance, standard deviation,
  skewness, and kurtosis.
- Feature Reduction Results: We compared LDA, PCA, and SVD, interpreting their
  effectiveness.
- Classification Results: We presented the results in tables and figures, comparing the
  performance of different classifiers.
- Evaluation Metrics: We applied K-fold cross-validation and confusion matrix analysis to
  evaluate the models.

= Conclusion and Future Work
The project successfully demonstrated the use of machine learning in classifying obesity levels.
Future work could explore the integration of more diverse datasets and the application of deep
learning techniques for improved accuracy.
= References


