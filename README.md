Predictive Modeling of Developer Task Success (dev_pro Dataset)

1. Project Overview
This project focuses on building and evaluating machine learning models to predict task_success (whether a developer task is successful or not) based on various developer-related metrics and environmental factors. Through this exploration, we encountered and thoroughly investigated a common but critical challenge in predictive modeling: achieving suspiciously high, even 100%, accuracy.

2. Dataset
The analysis utilizes the dev_pro dataset (https://www.kaggle.com/datasets/atharvasoundankar/ai-developer-productivity-dataset/code), which contains 500 observations across 9 variables. The variables include:

hours_coding: Hours spent coding.
coffee_intake_mg: Coffee intake in milligrams.
distractions: Level of distractions experienced.
sleep_hours: Hours of sleep obtained.
commits: Number of code commits.
bugs_reported: Number of bugs reported.
ai_usage_hours: Hours spent using AI tools.
cognitive_load: Perceived cognitive load.
task_success: Binary outcome (0 for Fail, 1 for Success), our target variable.
task_success_numeric_temp: A temporary numerical representation of task_success (initially for correlation calculations).

3. Problem Statement & Initial Challenges
Our primary challenge revolved around consistently achieving 100% classification accuracy with initial models, which is highly unusual for real-world predictive tasks. This phenomenon prompted a deep dive into data integrity, potential data leakage, and the fundamental nature of the dataset.

4. Methodology
The project employed a standard machine learning pipeline, adapted and refined through iterative analysis:

Data Preparation:
Converted task_success to a factor with "Fail" and "Success" levels.
Explicitly removed task_success_numeric_temp to prevent data leakage.
Data Splitting: Dataset was split into 70% training and 30% testing sets using stratified sampling to ensure class balance in both sets.
Data Balancing (SMOTEWB): The SMOTEWB technique was applied to the training data to address class imbalance, ensuring models do not overly favor the majority class.
Model Exploration:
Random Forest: Initially used due to its robustness and ability to capture complex non-linear relationships.
Logistic Regression: Employed as a linear model to contrast its performance against Random Forest and further diagnose the nature of predictability.
Feature Scaling: For Logistic Regression, feature scaling (centering and scaling) was applied within the caret's train function to optimize model performance, as linear models are sensitive to feature magnitudes.
Model Evaluation: Performance was assessed using cross-validation metrics (ROC, Sensitivity, Specificity) on the training set and a final confusion matrix on the unseen test set.

5. Key Findings & Insights
The "Perfect Prediction" Phenomenon:

Our Random Forest model consistently achieved 100% accuracy on the test set, with cross-validation ROC scores near 0.997.
Initial investigations involved removing highly correlated features (hours_coding, cognitive_load, bugs_reported) and crucially, the task_success_numeric_temp column, which was a direct, albeit accidental, form of data leakage.
Despite these efforts, Random Forest's near-perfect performance persisted. Direct inspection of feature ranges revealed no single feature perfectly separated the classes, suggesting a complex, non-linear deterministic rule was at play within the dataset.
Random Forest vs. Logistic Regression:

The introduction of Logistic Regression provided a critical diagnostic:
It achieved a realistic ~87% accuracy on the test set (with a cross-validation ROC of ~0.956).
This significant difference in performance (100% vs. ~87%) strongly indicates that the dev_pro dataset is not perfectly linearly separable. The Random Forest model was exploiting a non-linear deterministic pattern that a linear model like Logistic Regression could not fully capture.
The continued high accuracy of Logistic Regression still points to very strong, predictable patterns within the dataset.
Feature Importance (from Logistic Regression):

Based on the standardized coefficients, the most influential features for predicting task_success in the Logistic Regression model are:

coffee_intake_mg: Strongest Positive Impact. Higher coffee intake is strongly associated with increased task_success.
hours_coding: Strong Positive Impact. More hours spent coding is associated with increased task_success.
bugs_reported: Strong Negative Impact. More bugs reported is associated with decreased task_success.
sleep_hours: Moderate Positive Impact. More sleep hours is associated with increased task_success.
ai_usage_hours: Moderate Negative Impact. Higher AI usage hours is associated with decreased task_success.
cognitive_load: Moderate Negative Impact. Higher cognitive load is associated with decreased task_success.
commits: Moderate Negative Impact. More commits is associated with decreased task_success.
distractions: Weak Negative Impact. More distractions is associated with decreased task_success.
Overall Conclusion:

The project's findings strongly suggest that the dev_pro dataset likely contains highly deterministic relationships governing task_success, possibly due to its synthetic generation or unusually clean nature. While models perform exceptionally well on this specific data, such perfect predictability is rare in real-world scenarios due to inherent noise and variability. This project serves as a valuable case study in diagnosing unusual model performance, understanding data leakage, and comparing the strengths of different machine learning algorithms.

6. How to Run the Code
The analysis was performed using R. To replicate the findings:

Ensure you have R installed.
Install necessary R packages:
R
install.packages(c("caret", "SMOTEWB", "dplyr", "ggplot2", "randomForest"))
Load your dev_pro dataset into your R environment.
Combine the R code snippets provided during our discussion. The pipeline involves:
Data loading and initial task_success factor conversion.
Creating dev_pro_model_ready by excluding task_success_numeric_temp.
Train-test splitting.
SMOTEWB for balancing the training data.
caret::train calls for both randomForest (method "rf") and Logistic Regression (method "glm", family = "binomial", with preProcess = c("center", "scale") in the train call).
Prediction and confusionMatrix calls for evaluation.

7. Dependencies
R (>= 4.0)
caret package
SMOTEWB package
dplyr package
ggplot2 package
randomForest package
