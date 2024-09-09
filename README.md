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

![parameters](https://github.com/DragnaRR/Deepfake-Detection-System/assets/95096810/13311b56-3f01-43cc-8b8d-0768c3a13969)

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

![Confusion Matrix](https://github.com/DragnaRR/Deepfake-Detection-System/assets/95096810/baeecd44-6d1a-4d52-89b6-3b795247fabf)

The confusion matrix shows `211 + 22 = 233 correct predictions` and `15 + 52 = 67 incorrect predictions`.


In this case, we have


- `True Positives` (Actual Positive:1 and Predict Positive:1) - 211


- `True Negatives` (Actual Negative:0 and Predict Negative:0) - 22


- `False Positives` (Actual Negative:0 but Predict Positive:1) - 15 `(Type I error)`


- `False Negatives` (Actual Positive:1 but Predict Negative:0) - 52 `(Type II error)`

## Performance Metrics

Performance metrices are a part of every machine learning pipeline. They tell whether the model is making any progress or not.  Metrics are used to monitor and measure the performance of a model (during training and testing), and don't need to be differentiable. 

- precision

```
precision = True Positive / (True Positive + False Positive)

```

- Recall / Sensitivity / Hit-Rate

```
Recall = True Positive / (True Positive + False Negative)

```

- F1 Score

```
F1 Score = 2 X Precision X Recall / (Precision + Recall)

```
| Precision | Recall | F1 Score |
| :-------- | :-------- | :-------- |
| 0.93 | 0.87 | 0.90 |

## Area under Curve (AUC)

Better known as Area under Receiver operating characteristics curve (AUROC) is a graph between True Positive Rate also known as Recall & False Positive Rate also known as Fallout

![AUC](https://github.com/DragnaRR/Deepfake-Detection-System/assets/95096810/07e7ab3e-1ffa-4162-b08f-f0ff08a6e3a9)

## Accuracy

The simplest metric to use and implement and is defined as the number of correct predictions divided by the total number of predictions, multiplied by 100.

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

```

- Model Accuracy and Loss percentage

| Model | Train Accuracy | Validation Accuracy | Training loss | Validation loss | Accuracy Average |
| :-------- | :-------- | :-------- | :-------- | :-------- | :-------- |
| Meso 4 | 96.21 | 83.95 | 07.89 | 15.66 | 88.8 |

- Train Accuracy VS Validation Accuracy

![Accuracy](https://github.com/DragnaRR/Deepfake-Detection-System/assets/95096810/180bd55b-8613-4389-bf2e-13af7a451ff3)

- Train loss VS Validation loss
  
![Loss](https://github.com/DragnaRR/Deepfake-Detection-System/assets/95096810/9945e2d4-fff4-47a0-bfc4-88a733b96312)

## Web Application

![Home Page](https://github.com/DragnaRR/Deepfake-Detection-System/assets/95096810/5127f6a7-6dfa-4b77-9d25-85577ef30164)

![Upload](https://github.com/DragnaRR/Deepfake-Detection-System/assets/95096810/97f6a6d9-1cad-4818-ba0e-db54c861e1c2)

![Result](https://github.com/DragnaRR/Deepfake-Detection-System/assets/95096810/2a983e59-126b-47f9-a4ba-c025079f012e)

[Download Paper](https://github.com/DragnaRR/Deepfake-Detection-System/files/12874285/paper.pdf)




 
