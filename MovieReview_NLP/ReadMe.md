📊 Topic Modeling on IMDb Movie Reviews using NMF
🧠 Project Overview
This project demonstrates Topic Modeling on the IMDb movie reviews dataset using TF-IDF vectorization and Non-negative Matrix Factorization (NMF).

The goal is to automatically discover hidden themes (topics) present in large collections of text data.

📁 Dataset
Dataset: IMDb Movie Reviews (aclImdb)
Contains:
25,000 training reviews
25,000 test reviews
Categories:
pos → Positive reviews
neg → Negative reviews


⚙️ Steps Performed
1. Data Loading
Loaded text files from directory structure
Read all reviews into a Python list
Handled encoding issues using:

encoding='utf-8', errors='ignore'


2. Text Vectorization (TF-IDF)
Used TF-IDF (Term Frequency - Inverse Document Frequency) to convert text into numerical features.

from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(

    stop_words='english',

    max_features=5000,

    min_df=5,

    max_df=0.7

)

X = vect.fit_transform(reviews)
Key Parameters:
stop_words='english' → Removes common words
max_features=5000 → Limits vocabulary size
min_df=5 → Ignore rare words
max_df=0.7 → Ignore overly frequent words


3. Topic Modeling using NMF
Applied Non-negative Matrix Factorization (NMF) to extract topics.

from sklearn.decomposition import NMF

N_TOPICS = 15

nmf = NMF(

    n_components=N_TOPICS,

    init='nndsvd',

    max_iter=500,

    random_state=42

)

W = nmf.fit_transform(X)  # Document-topic matrix

H = nmf.components_       # Topic-term matrix
Outputs:
W (Document-Topic Matrix):

Shows topic distribution per document

H (Topic-Term Matrix):

Shows importance of words per topic


4. Extracting Top Words per Topic
Identified top contributing words for each topic:

import numpy as np

feature_names = vect.get_feature_names_out()

for i, topic in enumerate(H):

    top_words = [feature_names[j] for j in topic.argsort()[-10:][::-1]]

    print(f"Topic {i+1}: {top_words}")


5. Topic Interpretation
Each topic is represented by a group of keywords. Example:

Topic 1: ['film', 'movie', 'story', 'characters', 'director']

Topic 2: ['horror', 'scary', 'kill', 'blood', 'dead']

These keywords help interpret the underlying theme.


📈 Key Learnings
TF-IDF helps convert text into meaningful numerical representation
NMF is effective for extracting interpretable topics
Preprocessing significantly impacts topic quality
Proper parameter tuning improves convergence and results

How to Run
Install dependencies:

pip install numpy pandas scikit-learn

Load dataset
Run vectorization
Apply NMF
Extract topics

Conclusion
This project demonstrates how unsupervised learning techniques like NMF can uncover hidden structures in textual data and provide meaningful insights from large datasets.




