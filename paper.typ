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

#grid(
  rows: 2,
  columns: 2,
  grid.cell(
    colspan: 2,
    figure(
      image("./hist.png", width: 45%),
      caption: [Distribution of classes in the dataset]
    )
  ),
  figure(
    image("./age-boxplot.png", width: 65%),
    caption: [Ages]
  ),
  figure(
    image("./height-boxplot.png", width: 65%),
    caption: [Heights]
  )
)

= Related work
@dataset is the dataset used, it consists of 17 attributes and 2111 instances. 23% of the dataset are original,
the rest has been generated using SMOTE filter.

#pagebreak()

/*
The main problem addressed in this project is the accurate classification of obesity levels based
on various factors. We utilized several machine learning techniques, including Naive Bayes,
Decision Trees, and Neural Networks, to tackle this issue. The main contribution of this project
is the development of a robust model that outperforms existing benchmarks in accuracy and
efficiency. The rest of the document is organized into sections detailing related work,
methodology, proposed model, results, discussion, conclusion, and future work.
*/
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
For feature selection, Variance Threshold or Recursive Feature Elimination have been chosen.
Once a pipeline is constructed, it's trained and evaluated using cross-validation.
The best model is then selected.

#figure(
  image("./pipeline_model.png", width: 35%),
  caption: [Pipeline],
)

/*
We use data that's been published by Fabio Mendoza Palechor and Alexis De la Hoz Manotas.
23% of the data are original, while the rest has been synthesised using Synthetic Minority Oversampling Technique
Filter (SMOTE). 
*/
= Results and Discussion
#table(
  columns: 3,
  [Pipeline], [Mean Accuracy], [Standard Deviation],
  [Naive Bayes with PCA], [0.6517507002801121], [0.04107792042513687],
  [Naive Bayes with Variance Threshold], [0.5603991596638656], [0.029137943560944846],
  [Naive Bayes with RFE], [0.5960434173669468], [0.04209660664344372],
  [Decision Tree with PCA], [0.5812324929971989], [0.04008735136514084],
  [Decision Tree with Variance Threshold], [0.8466036414565826], [0.03328495167706323],
  [Decision Tree with RFE], [0.819922969187675], [0.04010574265762946],
  [Multi-Layer Perceptron with PCA], [0.8140406162464986], [0.04086377474357136],
  [Multi-Layer Perceptron with Variance Threshold], [0.7698529411764706], [0.038029707473487516],
  [Multi-Layer Perceptron with RFE], [0.6935224089635854], [0.08116535759280616],
  [k-Nearest Neighbours with PCA], [0.6872899159663866], [0.02590835312810534],
  [k-Nearest Neighbours with Variance Threshold], [0.7551470588235294], [0.010054995831748011],
  [k-Nearest Neighbours with RFE], [0.7316176470588235], [0.02347189661692996],
)

#figure(
  image(
    "./Decision Tree-RFE-cm.png",
    width: 70%
  ),
  caption: [Confusion Matrix of Decision Tree with Variance Threshold]
) <dtvth>

The results shows that decision trees with variance threshold is the most performant model
given the current dataset.

= Conclusion and Future Work
The project successfully demonstrated the use of machine learning in classifying obesity levels.
Future work could explore the integration of more diverse datasets and the application of deep
learning techniques for improved accuracy.


#bibliography("bib.yml", style: "apa")
