import streamlit as st
import nltk
import spacy
import string
import pandas as pd
import re
import matplotlib.pyplot as plt

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer  # class
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfVectorizer

# DOWNLOAD NLTK DATA
nltk.download("punkt")
nltk.download("stopwords")

# LOAD SPACY MODEL
nlp = spacy.load("en_core_web_sm")

# STREAMLIT PAGE CONFIG
st.set_page_config(
    page_title="NLP Preprocessing",
    layout="wide"
)




# APP TITLE
st.title("NLP Preprocessing App")
st.write("Tokenization, Text Cleaning, Stemming, Lemmatization , Bag of Words, TF-IDF, and  Word Embeddings")


# USER INPUT
text = st.text_area("Enter text for NLP processing", height=150,
        placeholder="Example: Satya is the BEST HOD of HIT and loves NLP.")

# SIDEBAR OPTIONS
option = st.sidebar.radio(
    "Select NLP Technique",
    [
        "Tokenization",
        "Text Cleaning",
        "Stemming",
        "Lemmatization",
        "Bag of Words",
        "TF-IDF",
        "Word Embeddings"
    ]
)

# PROCESS BUTTON
if st.button("Process Text"):
    if text.strip() == "":
        st.warning("Please enter some text.")

    
    # TOKENIZATION
    elif option == "Tokenization":
        st.subheader("Tokenization Output")
        col1, col2, col3 = st.columns(3)

        # Sentence Tokenization
        with col1:
            st.markdown("### Sentence Tokenization")
            sentences = sent_tokenize(text)
            st.write(sentences)

        # Word Tokenization
        with col2:
            st.markdown("### Word Tokenization")
            words = word_tokenize(text)
            st.write(words)

        # Character Tokenization
        with col3:
            st.markdown("### Character Tokenization")
            characters = list(text)
            st.write(characters)

    
    # TEXT CLEANING
    elif option == "Text Cleaning":
        st.subheader("Advanced Text Cleaning Output")

    # Step 1: Convert to lowercase
        text_lower = text.lower()

    # Step 2: Regex based cleaning
        text_clean = re.sub(r"http\S+|www\S+", "", text_lower)        # Remove URLs
        text_clean = re.sub(r"\S+@\S+", "", text_clean)              # Remove Emails
        text_clean = re.sub(r"@\w+", "", text_clean)                 # Remove Mentions (@user)
        text_clean = re.sub(r"#\w+", "", text_clean)                 # Remove Hashtags
        text_clean = re.sub(r"\d+", "", text_clean)                  # Remove numbers
        text_clean = re.sub(r"[^\w\s]", "", text_clean)              # Remove punctuation & emojis
        text_clean = re.sub(r"\s+", " ", text_clean).strip()         # Remove extra spaces

    # Step 3: Remove stopwords using spaCy
        doc = nlp(text_clean)
        final_words = [token.text for token in doc if not token.is_stop and token.text.strip() != ""]

        st.markdown("### Original Text")
        st.write(text)

        st.markdown("### After Regex Cleaning")
        st.write(text_clean)

        st.markdown("### After Stopword Removal (Final Cleaned Text)")
        st.write(" ".join(final_words))


    
    # STEMMING
    elif option == "Stemming":
        st.subheader("Stemming Output")

        words = word_tokenize(text)

        porter = PorterStemmer()
        lancaster = LancasterStemmer()

        # Apply stemming
        porter_stem = [porter.stem(word) for word in words]
        lancaster_stem = [lancaster.stem(word) for word in words]

        # Comparison table
        df = pd.DataFrame({
            "Original Word": words,
            "Porter Stemmer": porter_stem,
            "Lancaster Stemmer": lancaster_stem
        })

        st.dataframe(df, use_container_width=True)

    # LEMMATIZATION
    elif option == "Lemmatization":
        st.subheader("Lemmatization using spaCy")

        doc = nlp(text)
        data = [(token.text, token.pos_, token.lemma_) for token in doc]

        df = pd.DataFrame(data, columns=["Word", "POS", "Lemma"])
        st.dataframe(df, use_container_width=True)

    
    # BAG OF WORDS
    elif option == "Bag of Words":
        st.subheader("Bag of Words Representation")

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform([text])

        vocab = vectorizer.get_feature_names_out()
        freq = X.toarray()[0]

        df = pd.DataFrame({
            "Word": vocab,
            "Frequency": freq
        }).sort_values(by="Frequency", ascending=False)

        st.markdown("### BoW Frequency Table")
        st.dataframe(df, use_container_width=True)

        
        # PIE CHART (TOP-N WORDS)
        st.markdown("### Word Frequency Distribution (Top 10)")

        top_n = 10
        df_top = df.head(top_n)

        fig, ax = plt.subplots()
        ax.pie(
            df_top["Frequency"],
            labels=df_top["Word"],
            autopct="%1.1f%%",
            startangle=90
        )
        ax.axis("equal")  # Makes pie circular

        st.pyplot(fig) 
        
    #TF-IDF    
    elif option == "TF-IDF":
        st.subheader("TF-IDF Representation")

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform([text])

        words = vectorizer.get_feature_names_out()
        scores = X.toarray()[0]

        df = pd.DataFrame({
           "Word": words,
           "TF-IDF Score": scores
        }).sort_values(by="TF-IDF Score", ascending=False)

        st.dataframe(df, use_container_width=True)

        st.markdown("### Top Important Words")
        st.write(df.head(10)) 
        
    # WORD EMBEDDINGS
    elif option == "Word Embeddings":
        st.subheader("Word Embeddings (spaCy Vectors)")

        doc = nlp(text)

        data = []
        for token in doc:
            if token.has_vector:
                data.append((token.text, token.vector_norm))

        df = pd.DataFrame(data, columns=["Word", "Vector Magnitude"])
        st.dataframe(df, use_container_width=True)

        st.markdown("Higher vector value = more semantic information")
       
        
        
        