# 📰 News Article Recommendation System

A machine learning–based **News Article Recommendation System** that suggests relevant news articles to users based on content similarity. The system leverages **Natural Language Processing (NLP)** techniques such as **TF-IDF vectorization** and **cosine similarity** to provide accurate and personalized recommendations.

---

## 🚀 Features

* Content-based news recommendation
* Text preprocessing (cleaning, tokenization, stop-word removal)
* TF-IDF vectorization with unigrams & bigrams
* Similarity computation using cosine similarity
* Fast and scalable recommendation pipeline

---

## 🧠 Approach

1. **Data Collection**

   * News articles dataset with titles and content

2. **Text Preprocessing**

   * Lowercasing
   * Removing punctuation & stop words
   * Tokenization

3. **Feature Extraction**

   * TF-IDF Vectorizer
   * N-gram range: (1, 2)
   * Filtering rare and overly common words

4. **Similarity Calculation**

   * Cosine similarity between articles

5. **Recommendation**

   * Returns top-N most similar news articles

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLP (TF-IDF, Cosine Similarity)

---

## 📦 Installation

```bash
git clone https://github.com/your-username/news-article-recommendation-system.git
cd news-article-recommendation-system
pip install -r requirements.txt
```

---

## ▶️ Usage

```python
recommend_articles("Government announces new economic policy")
```

Output:

```
1. Economic reforms and market impact
2. Inflation trends and policy analysis
3. Finance ministry updates
```

---

## 📊 Example Parameters

```python
TfidfVectorizer(
    stop_words='english',
    max_df=0.8,
    min_df=5,
    ngram_range=(1, 2)
)
```

---

---

## 🔮 Future Improvements

* User-based and hybrid recommendation
* Deep learning models (BERT embeddings)
* Real-time recommendations
* Web interface using Flask or Streamlit

---

## 👨‍💻 Author

**Muhammad Rameez**
Data Scientist | ML Engineer
University of Engineering & Technology, Lahore

---

## ⭐ Acknowledgments

* Scikit-learn Documentation
* Kaggle News Datasets
* NLP research community

