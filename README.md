# Sentiment Analysis of Technical Comments in GitHub Repo

## Overview
This project performs sentiment analysis on developer comments in GitHub repositories. It utilizes Supprt Vector Machine (SVM) and Natural Language Processing model BERT to classify comments into Positive, Negative, or Neutral sentiments. The goal is to analyze developer sentiment to assess code quality and project discussions.

## Features
- **Text Preprocessing:** Cleans GitHub comments by removing mentions, URLs, extra spaces, and special characters.
- **Sentiment Classification:** Uses TextBlob for basic sentiment detection.
- **Machine Learning Model (SVM):**
  - Extracts text features using TF-IDF.
  - Trains an SVM model with hyperparameter tuning via GridSearchCV.
- **Deep Learning Model (BERT):**
  - Uses pre-trained BERT for sequence classification.
  - Implements fine-tuning with TensorFlow.
- **Visualization:**
  - Pie charts, bar plots, and confusion matrices to analyze sentiment distribution and model performance.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-github.git
   cd sentiment-analysis-github
   ```

## Dataset
- **Dataset 1:** `github_comments_filtered.csv` (Used for SVM model)
- **Dataset 2:** `git_final_vadar.csv` (applied vadar for predicting sentiment labels and against that used for BERT model for fine tuning and prediction.)

Ensure the datasets contain the required columns:
- `Comment` (Raw text)
- `Processed_Comment` (Cleaned text)
- `Sentiment` (Labels for SVM)
- `cleaned_comments` (For BERT processing)
- `sentiment_category` (For BERT labels)

## Usage
### 1. Running the SVM Model
```bash
python svm.py
```
- Outputs sentiment classification results and saves them in `svm_sentiment_results.csv`.
- Displays sentiment distribution and confusion matrix.

### 2. Running the BERT Model
```bash
python bert.py
```
- Fine-tunes BERT on GitHub comments.
- Displays accuracy and training loss graphs.

## Results
- **SVM Accuracy:** 80%
- **BERT Accuracy:** 84.30%
- **Visualizations:** Generates sentiment distribution graphs.

## Future Improvements
- Fine-tune BERT with a larger dataset.
- Experiment with other transformer models like RoBERTa.
- Analyze sentiment trends over time in GitHub projects.

# feel free for any suggestions!
