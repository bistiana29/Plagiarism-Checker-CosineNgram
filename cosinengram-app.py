import re
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("⚖️Plagiarism Checker For Law Faculty Thesis")

# Import dataset
df = pd.read_excel('skripsi UB ilmu hukum 2021-2024.xlsx')
df = df.drop(columns=["relation","type", "creator","format", "language", "identifier", "subject", "publisher", "date"])

# split dataset
test_df = df.tail(2)
train_df = df.drop(test_df.index)

# Preprocessing
def preprocessing(df, columns):
    # Membuat stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # List Stop word
    stop_words = set(stopwords.words('indonesian'))

    def preprocessing_txt(text):
        # Cleaning
        text = text.lower()
        text = re.sub(r"[\"'{}\-/\\%+=&:;,()!?\.]", "", text)
        text = re.sub(r'[^A-Za-z0-9 ]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenisasi
        words = word_tokenize(text)

        # Stopword removal
        filtered_words = [word for word in words if word not in stop_words]

        # Stemming
        text = ' '.join(filtered_words)
        text = stemmer.stem(text)
        
        return text
        
    for col in columns:
        df[col] = df[col].apply(preprocessing_txt)
    
    return df

# Preprocessing data train
#train_df = preprocessing(train_df, ['title', 'description'])
#train_df.to_excel('train_df.xlsx', index=False)

# Build model plagiarism checker
# Import train data
train_df = pd.read_excel('train_df.xlsx')
train_Df = train_df.drop_duplicates()

train_cosine = pd.DataFrame()
train_cosine['title'] = train_Df['title'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else str(x).lower())
train_cosine['description'] = train_Df['description'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else str(x).lower())

# Vectorization
vectorizer_title = TfidfVectorizer(analyzer='char', ngram_range=(1, 5))
vectorizer_description = TfidfVectorizer(analyzer='char', ngram_range=(1, 5))

tfidf_title = vectorizer_title.fit_transform(train_cosine['title'])
tfidf_description = vectorizer_description.fit_transform(train_cosine['description'])

# Calculate similarity
def calculate_similarity(vector1, vector2):
    return cosine_similarity(vector1, vector2)[0][0]

# Check plagiarism
def check_plagiarism_cosine(new_title, new_description):
    # Preprocessing new data
    test_df = pd.DataFrame({'title': [new_title], 'description': [new_description]})
    test_df = preprocessing(test_df, ['title', 'description'])
    
    # Ambil teks hasil preprocessing
    new_title = test_df['title'].iloc[0]
    new_description = test_df['description'].iloc[0]

    # Vectorizing new data
    vector_new_title = vectorizer_title.transform([new_title])
    vector_new_description = vectorizer_description.transform([new_description])

    similarity_title = 0
    similarity_description = 0
    index_title = 0
    index_description = 0

    for i in range(len(train_cosine)):
        similarity_title_temp = calculate_similarity(vector_new_title, tfidf_title[i])
        similarity_description_temp = calculate_similarity(vector_new_description, tfidf_description[i])
        
        if similarity_title_temp > similarity_title:
            similarity_title = similarity_title_temp
            index_title = i
            
        if similarity_description_temp > similarity_description:
            similarity_description = similarity_description_temp
            index_description = i
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f'Percentage of title plagiarism: {similarity_title*100:.2f}%')

    with col2:
        st.success(f'Percentage of abstract plagiarism: {similarity_description*100:.2f}%')
    
    # Visualisasi pie chart
    col3, col4 = st.columns(2)
    with col3:
        sizes = [similarity_title*100, 100 - (similarity_title*100)]
        plagiarism_title = similarity_title * 100
        unique_title = 100 - plagiarism_title
        labels = [f'Plagiarism: {plagiarism_title:.2f}%', f'Non-Plagiarism: {unique_title:.2f}%']
        colors = ['#CF0000', '#66b3ff']
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    
    with col4:
        sizes = [similarity_description*100, 100 - (similarity_description*100)]
        plagiarism_description = similarity_description * 100
        unique_description = 100 - plagiarism_description
        labels = [f'Plagiarism: {plagiarism_description:.2f}%', f'Non-Plagiarism: {unique_description:.2f}%']
        colors = ['#CF0000', '#66b3ff']
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

    if similarity_title > 0.5:
        st.error(f'The similarity of the new title is close to the title below🥺:')
        st.error(f'"{train_cosine["title"].iloc[index_title]}"')
    else:
        st.info(f'The new title is not classified as plagiarism😆')

    if similarity_description > 0.5:
        st.error(f'The similarity of the new abstract is close to the title below🥺:')
        st.error(f'"{train_cosine["description"].iloc[index_description]}"')
    else:
        st.info(f'The new abstract is not classified as plagiarism😆')

# Input new title and new description
input_title = st.text_area("🔖Input Title: ")
input_description = st.text_area("🔖Input Description: ")

if st.button("🔎Check Plagiarism"):
    start_time = time.time()
    check_plagiarism_cosine(input_title, input_description)
    end_time = time.time()

    execution_time = end_time - start_time
    st.success(f"⌛Execution time: {execution_time:.4f} s")