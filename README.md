# README: Interpretable Sentiment Analysis of Financial Headlines with LIME

## Introduction
This notebook demonstrates how to perform sentiment analysis on financial news headlines and, more importantly, how to interpret the model's predictions using LIME (Local Interpretable Model-agnostic Explanations). We train a simple text classification model and then use LIME to understand which words or phrases contribute most to a specific prediction.

## Dataset
The dataset used is `FinanceMTEB/financial_phrasebank`, which consists of financial news headlines annotated with sentiment labels (Negative, Neutral, Positive).

## Methodology
1.  **Data Loading and Preparation**: The financial phrasebank dataset is loaded, and the text (headlines) and labels are extracted. The data is then split into training and testing sets.
2.  **Model Training**: A `TfidfVectorizer` is used to convert text data into numerical features, followed by a `LogisticRegression` classifier. These are combined into a `Pipeline` for efficient processing and training.
3.  **Prediction**: The trained model predicts the sentiment of a sample headline from the test set.
4.  **LIME Explainer Setup**: A `LimeTextExplainer` is initialized with the class names to provide local explanations for text predictions.
5.  **Explanation Generation**: A helper function `get_explanation` is defined to generate LIME explanations for a given text, target class, and explanation parameters (number of features, number of samples). These explanations provide weights for words/tokens, indicating their contribution to the predicted class.
6.  **Robustness Check of Explanations**: LIME is a sampling-based method, and its explanations can vary slightly between runs. To assess this variability, the explanation generation is performed multiple times (e.g., 3 runs). The results are then compared to see which features are consistently selected and how their weights vary.

## Key Findings and Interpretations
-   The notebook successfully trains a Logistic Regression model for financial sentiment analysis.
-   LIME provides insights into which specific words or phrases in a headline influenced the model's prediction for a given sentiment class.
-   By comparing multiple LIME runs for the same prediction, we can observe the stability (or instability) of the explanations. Some features might be consistently identified as important, while others might appear or disappear across runs, highlighting the inherent stochasticity of LIME.

This notebook serves as a practical guide to understanding model predictions in text classification tasks, particularly when working with financial data where interpretability is crucial.
