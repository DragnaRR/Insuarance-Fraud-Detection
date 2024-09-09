# Insuarance-Fraud-Detection

 With the growth of the population of automobiles, auto insaurance has progressively become an essential sector linked to gobal economic growth and people's lives. It's primarily responsible for covering the cost of damages due to natural catastrophes and vehicle accidents, which includes auto insaurance and third party motor car liability insaurance. But insurance fraud accounts for a significant proportion of insurance expenses. Insurance fraud not only lowers earnings in the insurance business, leading to substantial losses but also impacts the price strategy and social-economic advantages of the insurance firm in the long run. 

 The proposed framework for fraud detection in the auto insurance industry by using artificial neural netwrok. The model is built utilizing a publicly available car insurance dataset. Six metrics are computed from the confusion matrix to assess the performance of the predictive model. It is revealed that the fault, base policy, and age of the policyholder are the most influential features. The findings of this study are beneficial for fraud detection in the auto insurance industry. Additionally, the underlying framework holds a functionality for real-time problem-solving in the auto insurance industry.


## Requirement

- Python 3.7.3

Some of the basic libraries that is been used in the project: 
- numpy 1.21.5 
- pandas 1.4.2
- tensorflow 2.9.2
- keras 2.12.0
- keras-tuner
- sklearn 1.0.2
- matplotlib 3.5.1
- seaborn 0.11.2
- flask 1.1.2
- werkzeug 2.0.3

## Model Architecture

- Total parameters

![parameters](https://github.com/DragnaRR/Insuarance-Fraud-Detection/blob/main/Screenshot/model_arch.PNG)

## Confusion Matrix

Confusion matrix is a tabular visualisation of hte ground-truth labels versus model predictions. It's a tool for summarizing the performance of a classification algorithm. A confusion matrix will give us a clear picture of classification model performance and the types of errors produced by the model. Each row of matrix represents the instances in a predicted class and each column represents the instances in an actual class. It gives us a summary of correct and incorrect predictions broken down by each category. The summary is represented in a tabular form.

Four types of outcomes are possible while evaluating a classification model performance. These four outcomes are described below:-

**True Positives (TP)** – True Positives occur when we predict an observation belongs to a certain class and the observation actually belongs to that class.

**True Negatives (TN)** – True Negatives occur when we predict an observation does not belong to a certain class and the observation actually does not belong to that class.

**False Positives (FP)** – False Positives occur when we predict an observation belongs to a    certain class but the observation actually does not belong to that class. This type of error is called **Type I error.**

**False Negatives (FN)** – False Negatives occur when we predict an observation does not belong to a certain class but the observation actually belongs to that class. This is a very serious error and it is called **Type II error.**

| True Positive (TP) | True Negative (TN) | False Positive (FP) | False Negative (FN) |
| :-------- | :-------- | :-------- | :-------- |
| 211 | 22 | 15 | 52 |

![Confusion Matrix](https://github.com/DragnaRR/Insuarance-Fraud-Detection/blob/main/Screenshot/confusion_matrix.PNG)

The confusion matrix shows `211 + 22 = 233 correct predictions` and `15 + 52 = 67 incorrect predictions`.


In this case, we have


- `True Positives` (Actual Positive:1 and Predict Positive:1) - 211


- `True Negatives` (Actual Negative:0 and Predict Negative:0) - 22


- `False Positives` (Actual Negative:0 but Predict Positive:1) - 15 `(Type I error)`


- `False Negatives` (Actual Positive:1 but Predict Negative:0) - 52 `(Type II error)`

## Performance Metrics

Performance metrices are a part of every machine learning pipeline. They tell whether the model is making any progress or not.  Metrics are used to monitor and measure the performance of a model (during training and testing), and don't need to be differentiable. 

- precision

Precision can be defined as the percentage of correctly predicted positive outcomes out of all the predicted positive outcomes. It can be given as the ratio of true positives (TP) to the sum of true and false positives (TP + FP).

So, Precision identifies the proportion of correctly predicted positive outcome. It is more concerned with the positive class than the negative class.

Mathematically, precision can be defined as the ratio of TP to (TP + FP).

```
precision = True Positive / (True Positive + False Positive)

```

- Recall / True Positive Rate / Sensitivity / Hit-Rate

Recall can be defined as the percentage of correctly predicted positive outcomes out of all the actual positive outcomes. It can be given as the ratio of true positives (TP) to the sum of true positives and false negatives (TP + FN). Recall is also called Sensitivity.

Recall identifies the proportion of correctly predicted actual positives.

Mathematically, recall can be defined as the ratio of TP to (TP + FN).

```
Recall = True Positive / (True Positive + False Negative)

```

- F1 Score

f1-score is the weighted harmonic mean of precision and recall. The best possible f1-score would be 1.0 and the worst would be 0.0. f1-score is the harmonic mean of precision and recall. So, f1-score is always lower than accuracy measures as they embed precision and recall into their computation. The weighted average of f1-score should be used to compare classifier models, not global accuracy.

```
F1 Score = 2 X Precision X Recall / (Precision + Recall)

```
| Precision | Recall | F1 Score |
| :-------- | :-------- | :-------- |
| 0.93 | 0.80 | 0.86 |

## Area under Curve (AUC)

Better known as Area under Receiver operating characteristics curve (AUROC) is a graph between True Positive Rate also known as Recall & False Positive Rate also known as Fallout
ROC AUC is a single number summary of classifier performance. The higher the value, the better the classifier. A perfect classifier will have a ROC AUC equal to 1, whereas a purely random classifier will have a ROC AUC equal to 0.5.
Another tool to measure the classification model performance visually is ROC Curve. ROC Curve stands for Receiver Operating Characteristic Curve. An ROC Curve is a plot which shows the performance of a classification model at various classification threshold levels.

The ROC Curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold levels.

True Positive Rate (TPR) is also called Recall. It is defined as the ratio of TP to (TP + FN).

False Positive Rate (FPR) is defined as the ratio of FP to (FP + TN).

In the ROC Curve, we will focus on the TPR (True Positive Rate) and FPR (False Positive Rate) of a single point. This will give us the general performance of the ROC curve which consists of the TPR and FPR at various threshold levels. So, an ROC Curve plots TPR vs FPR at different classification threshold levels. If we lower the threshold levels, it may result in more items being classified as positve. It will increase both True Positives (TP) and False Positives (FP).

![AUC](https://github.com/DragnaRR/Insuarance-Fraud-Detection/blob/main/Screenshot/AUC.PNG)

## Accuracy

The simplest metric to use and implement and is defined as the number of correct predictions divided by the total number of predictions, multiplied by 100.

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

```

- Model Accuracy and Loss percentage

| Model | Train Accuracy | Validation Accuracy | 
| :-------- | :-------- | :-------- |
| ANN | 79.71 | 77.66 |

- Train loss VS Validation loss
  
![Loss](https://github.com/DragnaRR/Insuarance-Fraud-Detection/blob/main/Screenshot/loss.PNG)
