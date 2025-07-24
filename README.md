# NLP

# 📊 Sentiment Analysis on Amazon Reviews

This project builds a complete text classification pipeline to analyze the sentiment of Amazon product reviews using different machine learning models.

[![Open in Colab] https://colab.research.google.com/drive/1BiukuRQH6uVcgEejz--nnC4VrjJyv14B#scrollTo=PyDmkvv15cna

---

## ✅ Project Overview

We implemented a sentiment analysis pipeline that includes:

- Data Preprocessing
- Feature Engineering (BoW, TF-IDF)
- Model Training (Naïve Bayes, Logistic Regression, SVM)
- Evaluation & Comparison using Accuracy, Precision, Recall, and F1 Score
- Final Visualization of Model Metrics

---

## 📈 Model Comparison Results

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Naïve Bayes         | 0.849    | 0.675     | 0.503  | 0.466    |
| Logistic Regression | 0.865    | 0.904     | 0.556  | 0.564    |
| SVM                 | 0.886    | 0.808     | 0.691  | 0.728    |

> ✅ **SVM outperformed other models**, delivering the highest F1 Score and overall accuracy.

---

## 🧠 Technologies Used

- Python
- Scikit-learn
- Pandas & NumPy
- Matplotlib & Seaborn
- Google Colab

---

## 🛠️ How to Run

1. Click the "Open in Colab" button above
2. Run each cell in order
3. Review outputs and plots

---

## 🔍 Key Takeaways

- Feature representation plays a crucial role in text classification.
- SVM is highly effective for sparse, high-dimensional data like TF-IDF features.
- Precision-Recall tradeoffs vary by model — choose based on use case (e.g., risk of false positives vs. false negatives).

---

## ✨ Future Improvements

- Add deep learning models (LSTM, BERT)
- Integrate Word Embeddings (Word2Vec, GloVe)
- Apply hyperparameter tuning (GridSearchCV)
