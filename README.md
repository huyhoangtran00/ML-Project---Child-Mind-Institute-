# Problematic Internet Use Prediction

## 1. Indirect Prediction of SII (via PCIAT)
To estimate the Self-Identified Internet Addiction Score (SII), we first predict the component columns of the PCIAT (Problematic Internet Use scale) using machine learning models. The SII is then derived indirectly by aggregating the predicted PCIAT scores.

### Steps:
1. **Data Preprocessing:**
   - Handle missing values.
   - Standardize or normalize feature values as necessary.
   - Split the dataset into training and testing sets.

2. **Model Training:**
   - Train regression models (e.g., Linear Regression, Random Forest) to predict individual PCIAT components.

3. **Aggregation:**
   - Combine the predicted PCIAT components to calculate the final SII.

4. **Evaluation:**
   - Compare the derived SII values with the ground truth SII values.
   - Metrics used: RMSE, MAE, and R-squared.

## 2. Direct Prediction of SII + Grid Search Optimization
In this approach, the SII is predicted directly from the input features using a single regression model.

### Steps:
1. **Data Preparation:**
   - Prepare features and labels (SII).
   - Perform feature engineering (e.g., dimensionality reduction if needed).

2. **Model Selection:**
   - Use models such as Random Forest, Gradient Boosting, or Support Vector Regression.

3. **Hyperparameter Tuning:**
   - Apply Grid Search with cross-validation to identify the optimal model parameters.
   - Optimize the model using the Quadratic Weighted Kappa (QWK) score.

4. **Evaluation:**
   - Assess the model's performance using metrics like QWK and RMSE.

## 3. XGBoost + Threshold Tuning
This approach uses XGBoost, a powerful gradient boosting algorithm, combined with threshold tuning to enhance prediction accuracy.

### Steps:
1. **Feature Selection and Engineering:**
   - Identify the most influential features for predicting SII.
   - Apply one-hot encoding or other transformations as needed.

2. **Model Training:**
   - Train an XGBoost regressor using training data.

3. **Threshold Tuning:**
   - Fine-tune the decision thresholds to optimize the model's predictive power.
   - Use the QWK score to evaluate and adjust the thresholds.

4. **Evaluation:**
   - Use validation data to measure performance and adjust thresholds accordingly.
   - Metrics used: QWK and RMSE.

## 4. Vote Regressor + Threshold Tuning
This method combines the predictions of multiple regression models through ensemble learning and tunes thresholds for better accuracy.

### Steps:
1. **Ensemble Creation:**
   - Train multiple regression models (e.g., Linear Regression, Random Forest, XGBoost).
   - Combine their predictions using a Voting Regressor.

2. **Threshold Tuning:**
   - Adjust thresholds to minimize prediction errors.
   - Optimize the ensemble performance using the QWK score.

3. **Evaluation:**
   - Compare ensemble performance with individual models.
   - Metrics used: QWK and RMSE.
