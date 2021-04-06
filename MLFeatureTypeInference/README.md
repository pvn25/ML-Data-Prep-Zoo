# Task: ML Feature Type Inference

This [project](https://adalabucsd.github.io/sortinghat.html) is about inferring ML feature types over tabular data. Please refer to our [tech report](https://adalabucsd.github.io/papers/TR_2021_SortingHat.pdf) for more details.

## Changelog

t1v2.0 Corrected 32 examples in our labeled dataset.


## Benchmark Labeled Data

Benchmark-Labeled-Data/ contains our labeled dataset with the train/test partitions, corresponding metadata, raw CSV files, and our base featurization for ML models.

## Source code

Models/ contain the source code (jupyter notebooks) of different ML models and the apis we use to benchmark feature type inference for AutoML platforms.

## Pre-trained Models

Pre-trained Models/ contain the trained ML models ready for inference.

## Library

Library/ contain our models and featurization routines wrapped under functions in a Python library. It explains how to load the pre-trained models for inference.

## Downstream Benchmark

Downstream-Benchmark/ contain links to the datasets, their source details, and downstream model source code


## AutoML Benchmark

The following table presents the binarized class-specific accuracy, precision, and recall of different approaches on our **benchmark labeled held-out test dataset.**

| Feature Type          | Metric           |     TFDV          |     Pandas    |     TransmogrifAI    |     AutoGluon     |     Log Reg       |     CNN           |     Rand   Forest    |
|-----------------------|------------------|-------------------|---------------|----------------------|-------------------|-------------------|-------------------|----------------------|
|     Numeric           |     Precision    |     0.657         |     0.614     |     0.605            |     0.646         |     0.909         |     0.929         |     0.934            |
|                       |     Recall       |     1             |     1         |     1                |     1             |     0.943         |     0.941         |     0.984            |
|                       |     Accuracy     |     0.814         |     0.776     |     0.767            |     0.805         |     0.946         |     0.953         |     0.97             |
|                       |                  |                   |               |                      |                   |                   |                   |                      |
|     Categorical       |     Precision    |     0.396         |     -         |     -                |     0.667         |     0.808         |     0.846         |     0.913            |
|                       |     Recall       |     0.652         |               |                      |     0.534         |     0.884         |     0.928         |     0.943            |
|                       |     Accuracy     |     0.691         |               |                      |     0.831         |     0.925         |     0.945         |     0.966            |
|                       |                  |                   |               |                      |                   |                   |                   |                      |
|     Datetime          |     Precision    |     0.985         |     0.956     |     1                |     1             |     0.951         |     0.925         |     0.945            |
|                       |     Recall       |     0.475         |     0.915     |     0.454            |     0.844         |     0.972         |     0.965         |     0.972            |
|                       |     Accuracy     |     0.962         |     0.991     |     0.961            |     0.989         |     0.994         |     0.992         |     0.994            |
|                       |                  |                   |               |                      |                   |                   |                   |                      |
|     Sentence          |     Precision    |     0.472         |     -         |     -                |     0.516         |     0.913         |     0.725         |     0.865            |
|                       |     Recall       |     0.457         |               |                      |     0.902         |     0.793         |     0.804         |     0.902            |
|                       |     Accuracy     |     0.951         |               |                      |     0.956         |     0.987         |     0.977         |     0.989            |
|                       |                  |                   |               |                      |                   |                   |                   |                      |
|     Not-Generalizable |     Precision    |     -             |     -         |     -                |     0.465         |     0.732         |     0.81          |     0.934            |
|                       |     Recall       |                   |               |                      |     0.53          |     0.732         |     0.66          |     0.86             |
|                       |     Accuracy     |                   |               |                      |     0.883         |     0.947         |     0.937         |     0.978            |
|                       |                  |                   |               |                      |                   |                   |                   |                      |
|     Context-Specific  |     Precision    |     -             |     0.08      |     0.074            |     -             |     0.747         |     0.741         |     0.859            |
|                       |     Recall       |                   |     0.295     |     0.295            |                   |     0.621         |     0.663         |     0.705            |
|                       |     Accuracy     |                   |     0.609     |     0.582            |                   |     0.944         |     0.946         |     0.961            |
|                       |                  |                   |               |                      |                   |                   |                   |                      |

<!-- ![TableComparison](images/table_comparison.png) -->


## Leaderboard on our Labeled Data

We invite researchers and practitioners to use our datasets and contribute to create better featurizations and models. By submitting results, you acknowledge that your holdout test results (data_test.csv) are obtained purely by training on the training set (data_train.csv).

<!-- ![TableAccuracy](images/table_models_all.png) -->

|                                                 Approaches                                               |     9-class      Accuracy    |   |      Numeric     |               |   |     Categorical    |               |   |      Datetime    |               |   |      Sentence    |               |   |        URL       |               |   |     Embedded   Number    |               |   |        List      |               |   |     Not-Generalizable    |               |   |     Context-Specific    |               |   |
|:--------------------------------------------------------------------------------------------------------:|:----------------------------:|---|:----------------:|:-------------:|---|:------------------:|:-------------:|---|:----------------:|:-------------:|---|:----------------:|:-------------:|---|:----------------:|:-------------:|---|:------------------------:|:-------------:|---|:----------------:|:-------------:|---|:------------------------:|:-------------:|---|:-----------------------:|:-------------:|---|
|                                                                                                          |                              |   |     Precision    |     Recall    |   |      Precision     |     Recall    |   |     Precision    |     Recall    |   |     Precision    |     Recall    |   |     Precision    |     Recall    |   |         Precision        |     Recall    |   |     Precision    |     Recall    |   |         Precision        |     Recall    |   |         Precision       |     Recall    |   |
|     [Random Forest (Shah et al. 2020)](https://adalabucsd.github.io/papers/TR_2020_SortingHat.pdf)       |             0.9259           |   |       0.934      |      0.984    |   |        0.913       |      0.943    |   |       0.945      |      0.972    |   |       0.865      |      0.902    |   |       0.968      |      0.938    |   |           0.929          |      0.929    |   |         1        |      0.827    |   |           0.934          |      0.86     |   |           0.859         |      0.705    |   |
|     [k-NN (Shah et al. 2020)](https://adalabucsd.github.io/papers/TR_2020_SortingHat.pdf)                |             0.8796           |   |       0.946      |      0.94     |   |        0.874       |      0.884    |   |       0.914      |      0.952    |   |       0.841      |      0.796    |   |         1        |      0.909    |   |           0.842          |      0.885    |   |        0.87      |      0.769    |   |           0.838          |      0.801    |   |           0.681         |      0.722    |   |
|     [CNN (Shah et al. 2020)](https://adalabucsd.github.io/papers/TR_2020_SortingHat.pdf)                 |             0.8788           |   |       0.929      |      0.941    |   |        0.846       |      0.928    |   |       0.925      |      0.965    |   |       0.725      |      0.804    |   |       0.828      |      0.75     |   |           0.747          |      0.717    |   |       0.732      |      0.577    |   |            0.81          |      0.693    |   |           0.741         |      0.663    |   |
|     [RBF-SVM (Shah et al. 2020)](https://adalabucsd.github.io/papers/TR_2020_SortingHat.pdf)             |             0.8761           |   |       0.921      |      0.944    |   |        0.855       |      0.885    |   |         1        |      0.963    |   |       0.879      |      0.624    |   |       0.967      |      0.879    |   |           0.955          |      0.972    |   |       0.542      |      0.907    |   |           0.832          |      0.796    |   |           0.768         |      0.676    |   |
|     [Logistic Regression (Shah et al. 2020)](https://adalabucsd.github.io/papers/TR_2020_SortingHat.pdf) |             0.8643           |   |       0.909      |      0.943    |   |        0.808       |      0.884    |   |       0.951      |      0.972    |   |       0.913      |      0.793    |   |       0.939      |      0.969    |   |           0.919          |      0.919    |   |        0.93      |      0.769    |   |           0.732          |      0.66     |   |           0.747         |      0.621    |   |