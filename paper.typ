#align(center, text(20pt)[
  Predicting Obesity
])

#set par(justify: true)

= Abstract
Obesity is a global health concern with severe implications. This project aims to classify obesity
levels using advanced machine learning techniques. 
We explore a dataset concerning obesity levels in Mexico, Colombia and Peru.
We developed a model that processes
demographic, dietary, and physical activity data to predict obesity levels. 
We compare a number of feature reduction techniques, feature selection techniques, wrapper methods
and models.
Our findings indicate
that machine learning can effectively predict obesity levels, which could be instrumental in
preventive health measures.
/*
The dataset consists of data regarding physical attributes like weight, height and gender,
and data regarding habits like smoking.
We compare a number of feature reduction techniques, feature selection techniques, wrapper methods
and models. Then, all combinations of the aforementioned are evaluated by cross-validation.
*/
= Introduction
Obesity is a disease characterise by excessive fat deposits 
that can lead to Type-2 diabetes, heart disease and certain types of cancers. 
According to the World Health Organisation, in 2022, 890 million adults suffer from obesity.
Obesity can dramatically impairs one's quality of life, and contributes to other health complications.
We aim to make a machine learning mode that's able to predict obesity levels based on eating habits
and physical conditions.
In this project, we compare a number of methods in order to select the most performant one.
/*
The main problem addressed in this project is the accurate classification of obesity levels based
on various factors. We utilized several machine learning techniques, including Naive Bayes,
Decision Trees, and Neural Networks, to tackle this issue. The main contribution of this project
is the development of a robust model that outperforms existing benchmarks in accuracy and
efficiency. The rest of the document is organized into sections detailing related work,
methodology, proposed model, results, discussion, conclusion, and future work.
*/
= Related Work
= Methodology
We select a number of feature reduction and feature selection techniques.
For each such technique, we pair them with a classifier model.
The resulting pipeline is then trained, evaluated and compared with other pipelines in order to 
select the most performant one.
= Proposed Model
The models consists of a pipeline, the pipeline is divided into 3 stages.
In the first stage, Categorical data is encoded using ordinal encoding and missing values of both
categorical and numerical variables are handled. For numerical variables, missing values are replaced
with the mean, while missing values in categorical variables are replaced with the most frequent value.
In the second stage, Either feature selection or feature reduction is performed.
For feature reduction, Principle Component Analysis has been chosen.
For feature selection, Variance Threshold or Recursive Feature Elemenation have been chosen.
Once a pipeline is constructed, it's trained and evaluated using cross-validation.
The best model is then selected.
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


