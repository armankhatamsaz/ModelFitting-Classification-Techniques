# Model Fitting and Classification Techniques

## Description

This project provides a practical demonstration and comparison of several fundamental model fitting and classification techniques commonly used in pattern recognition and machine learning. The Jupyter Notebook (`ModelFitting-Classification-Techniques.ipynb`) implements these methods from foundational principles and applies them to diverse datasets:

1.  **Iris Dataset:** Fitting Multivariate Gaussian distributions for continuous data.
2.  **SMS Spam Collection:** Fitting Bernoulli models for discrete data using the Bag-of-Words representation for text classification (spam detection).
3.  **Phoneme Dataset:** Applying and comparing Gaussian discriminative analysis techniques (LDA, QDA, Naive Bayes) for continuous data classification.

The notebook covers parameter estimation (MLE/MAP), classifier implementation, performance evaluation (Accuracy, ROC curves, EER), and feature selection based on Mutual Information.

## Key Features

*   **Model Fitting:**
    *   Multivariate Gaussian distribution fitting for continuous data (Iris dataset).
    *   Bernoulli model fitting for discrete Bag-of-Words data (SMS Spam dataset).
*   **Parameter Estimation:**
    *   Maximum Likelihood Estimation (MLE) for Gaussian parameters (mean, covariance).
    *   Estimation of prior probabilities (πc) and class-conditional probabilities (θjc) for Naive Bayes (Bernoulli).
    *   Comparison between MAP and MLE classification results.
*   **Classification Algorithms Implemented:**
    *   Naive Bayes Classifier (Bernoulli variant for BoW, Gaussian variant for Phoneme).
    *   Quadratic Discriminant Analysis (QDA).
    *   Linear Discriminant Analysis (LDA).
*   **Performance Evaluation:**
    *   Accuracy calculation for different classifiers on test data.
    *   Receiver Operating Characteristic (ROC) curve plotting.
    *   Equal Error Rate (EER) estimation from the ROC curve.
*   **Feature Selection (Optional Section):**
    *   Ranking features using Mutual Information for the Bag-of-Words model.
    *   Analyzing classifier accuracy as a function of the number of selected features (K).
*   **Visualization:**
    *   Scatter plots and histograms for data exploration.
    *   Visualization of 2D Gaussian probability density functions (PDFs).
    *   Bar plots for class-conditional densities (θjc).
    *   ROC curve plots.
    *   Accuracy vs. K features plot.

## Datasets Used

1.  **Iris Dataset Subset (`Lab2_Ex_1_Iris.hdf5`):** A subset of the classic Iris dataset, containing only two classes (Setosa, Versicolour) and two features (petal length, petal width). Used for demonstrating Multivariate Gaussian fitting. (Based on: [UCI Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris))
2.  **SMS Spam Collection (`SMSSpamCollection`):** A dataset of tagged SMS messages (ham/spam). Used for demonstrating Bag-of-Words representation and Bernoulli Naive Bayes classification. (Source: [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) / [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset))
3.  **Phoneme Dataset (`Lab2_Ex_6_phoneme.hdf5`):** A dataset used for phoneme recognition. Used for comparing QDA, LDA, and Gaussian Naive Bayes classifiers on continuous data.

*(Note: Ensure the data files are accessible, either included in the repository or provide instructions/links on how to obtain them if they are large or have licensing restrictions.)*

## Techniques Demonstrated

*   **Model Fitting for Continuous Distributions:** Multivariate Gaussian.
*   **Model Fitting for Discrete Distributions:** Bernoulli model within Bag-of-Words.
*   **Discriminative Analysis:** LDA and QDA.
*   **Probabilistic Classification:** Naive Bayes (Bernoulli & Gaussian).
*   **Parameter Estimation:** MLE and MAP principles.
*   **Performance Metrics:** Accuracy, TPR, FPR, ROC, EER.
*   **Feature Engineering/Selection:** Bag-of-Words, Mutual Information.

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/amankhatamsaz/ModelFitting-Classification-Techniques.git
    cd ModelFitting-Classification-Techniques
    ```
2.  **Ensure you have Python 3 installed.**
3.  **Install required libraries:** A `requirements.txt` file should ideally be included. If not, you'll need libraries like:
    ```bash
    pip install numpy matplotlib pandas seaborn scikit-learn h5py jupyter
    ```
    *(You might need to create a `requirements.txt` file listing these)*
4.  **Place dataset files** (`Lab2_Ex_1_Iris.hdf5`, `Lab2_Ex_6_phoneme.hdf5`, `SMSSpamCollection`) in the appropriate directory (e.g., the root or a `data/` subdirectory) if they are not included in the repo. Update file paths in the notebook if necessary.

## Usage

1.  Launch Jupyter Notebook or Jupyter Lab:
    ```bash
    jupyter notebook
    ```
    or
    ```bash
    jupyter lab
    ```
2.  Open the `ModelFitting-Classification-Techniques.ipynb` notebook.
3.  Run the cells sequentially to execute the code, visualize the results, and see the analysis.

## Results Summary

The notebook performs the following analyses:

*   Fits Gaussian models to the Iris subset and visualizes the PDFs.
*   Fits Bernoulli Naive Bayes parameters (priors, conditional probabilities) to the SMS Spam dataset using Bag-of-Words.
*   Identifies informative and uninformative features based on class-conditional probabilities and optionally via Mutual Information.
*   Compares MAP vs. MLE accuracy for the Naive Bayes classifier.
*   Plots the ROC curve and estimates the EER for the SMS spam classifier.
*   Applies LDA, QDA, and Gaussian Naive Bayes to the Phoneme dataset and compares their accuracy.
*   (Optional) Demonstrates the effect of feature selection (varying K) on classifier accuracy.

Refer to the output and markdown cells within the notebook for detailed results and commentary on model performance and characteristics.
