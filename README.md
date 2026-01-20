#  Diabetes Prediction System (Machine Learning)

##  Project Overview

This project is a **Diabetes Prediction System** built using **Machine Learning classification models** to predict whether a person is likely to have diabetes based on medical input features.

The main focus of this project is to **train multiple ML models**, compare their performance, and identify the **best-performing model**.

---

##  Objectives

 - Predict diabetes cases using supervised learning
 - Train and evaluate multiple classification models
 - Compare models using **Accuracy, Precision, Recall, and ROC Curve**
 - Select the best model for diabetes prediction

---

## ⚙️ Models Used

The following models were implemented and tested:

* Logistic Regression
* Random Forest Classifier
* Decision Tree Classifier
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)

---

## 📊 Model Evaluation Results

| Model                  | Accuracy   | Precision  | Recall     |
| ---------------------- | ---------- | ---------- | ---------- |
| Logistic Regression    | 0.7651     | 0.7222     | 0.5098     |
| Random Forest          | 0.7517     | 0.6750     | 0.5294     |
| Decision Tree          | 0.6644     | 0.5098     | 0.5098     |
| **SVM (Best Model)** | **0.7987** | **0.7838** | **0.5686** |
| KNN                    | 0.7315     | 0.6341     | 0.5098     |

---

##  Best Model

###  Support Vector Machine (SVM)

 **SVM performed best** because it achieved the highest:

* **Accuracy (79.87%)**
* **Precision (78.38%)**
* **Recall (56.86%)**

This means SVM predicts diabetes more reliably and reduces incorrect predictions compared to other models.

---

##  ROC Curve Analysis

The **ROC Curve (Receiver Operating Characteristic Curve)** helps evaluate how well a model separates:

* **Diabetes (Positive Class)**
* **No Diabetes (Negative Class)**

###  ROC Curve Interpretation

* A curve **closer to the top-left corner** indicates a better model.
* A curve close to the **diagonal line** indicates the model is close to random guessing.
* Better ROC performance means:

  * Higher **True Positive Rate (TPR)**
  * Lower **False Positive Rate (FPR)**

###  ROC Curve Observations

 **SVM shows the strongest ROC performance**, meaning it provides the best separation between diabetic and non-diabetic patients.
 **Decision Tree shows weaker ROC performance**, meaning it struggles to separate classes effectively.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

---

## How to Run

1. Clone this repository:

```bash
git clone <your-repo-link>
```

2. Open the notebook:

```bash
jupyter notebook
```

3. Run the project file:
   `Diabetes_Prediction_system.ipynb`

---

## Conclusion

This project compared multiple machine learning models for diabetes prediction.

 The **SVM model achieved the best overall results** and showed the strongest ROC performance, making it the most reliable model in this system.

---

