# Credit Card Fraud Detection System

## Overview
This project implements a machine learning solution for detecting fraudulent credit card transactions using a Random Forest Classifier. The system analyzes transaction patterns and identifies potentially fraudulent activities with high accuracy.

## Dataset
The analysis uses the Credit Card Fraud Detection dataset (`creditcard.csv`), which contains transactions made by European cardholders in September 2013. The dataset is highly imbalanced, with fraudulent transactions representing only a small fraction of all transactions.

### Key Features:
- **Time**: Seconds elapsed between the first transaction and subsequent transactions
- **Amount**: Transaction amount
- **Class**: Target variable (0 = valid, 1 = fraudulent)
- **V1-V28**: PCA-transformed features (anonymized for privacy)

## Requirements

### Python Libraries
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```
## Environment
- Google Colab (for Google Drive integration)
- Alternative: Local Python environment with adjusted file paths

## Code Structure
1. Data Loading
```python
# Mounts Google Drive and loads the dataset
drive.mount('/content/drive')
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/coisas/csv/creditcard.csv')
```
2. Exploratory Data Analysis (EDA)
- Dataset statistics and description
- Fraud vs. valid transaction distribution
- Amount analysis for both transaction types
- Correlation heatmap visualization
- Top 10 feature correlation analysis

3. Data Preparation
- 40% random sampling for efficient processing
- Feature-target separation
- Train-test split (80/20 ratio)

4. Model Training
- Algorithm: Random Forest Classifier
- Parameters:
    - n_estimators: 100
    - criterion: "gini"
    - max_depth: None (unlimited)

5. Model Evaluation
- Accuracy: Overall correct predictions
- Precision: Fraud detection reliability
- Recall: Fraud detection completeness
- F1-Score: Harmonic mean of precision and recall
- MCC (Matthews Correlation Coefficient): Balanced measure for imbalanced data
- Confusion Matrix: Visual representation of predictions

## Output Metrics
The system outputs:
- Dataset statistics (shape, balance ratio)
- Fraud vs. valid transaction counts
- Amount distribution details
- Top 10 features correlated with fraud
- Model performance metrics
- Confusion matrix visualization
- Performance Considerations:
    - Sampling: 40% of the data is sampled to optimize training time while maintaining representativeness
    - Feature Selection: Top 10 correlated features are identified for focused analysis
    - Imbalanced Data Handling: Random Forest naturally handles class imbalance; additional metrics (MCC) provide robust evaluation

## Usage Instructions
For Google Colab:
 -   Upload creditcard.csv to your Google Drive at: /MyDrive/
-  Run all cells sequentially
 -   Review the metrics and visualizations

For Local Environment:
  -  Remove Google Drive mounting code
   - Update the file path to your local creditcard.csv location
    -Remove from google.colab import drive import

## Expected Results
Based on typical credit card fraud detection scenarios:
-    Precision: High (most fraud alerts are correct)
-    Recall: Moderate to High (catches most fraudulent transactions)
-    F1-Score: Balanced metric between precision and recall
-    MCC: Values near 1 indicate strong correlation with actual fraud

## Limitations & Future Improvements
-    Imbalanced Data: Consider SMOTE or other oversampling techniques
-    Feature Engineering: Original features are PCA-transformed; consider dimensionality reduction alternatives
-    Real-time Processing: Optimize for streaming data scenarios
-    Model Tuning: Perform grid search for optimal hyperparameters
-    Cross-validation: Implement k-fold validation for more robust evaluation

## License

This project is for educational purposes. Ensure compliance with dataset usage terms and data privacy regulations when using real transaction data.

## Author Notes
This implementation prioritizes:
- Clear visualization of model performance
- Handling of imbalanced classification problems
- Efficient processing of large datasets through sampling
- Multiple evaluation metrics for comprehensive assessment
