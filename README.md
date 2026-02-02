🧠 NLP Preprocessing & Feature Engineering App

An interactive NLP Preprocessing Web Application built using Streamlit that demonstrates core Natural Language Processing (NLP) techniques such as text cleaning, tokenization, stemming, lemmatization, Bag of Words, TF-IDF, and word embeddings.

This project is designed for students, beginners, and academic demonstrations to understand how raw text is converted into meaningful numerical features.

✨ Key Features

🔹 Clean and intuitive Streamlit UI

🔹 Real-time NLP processing on user-provided text

🔹 Visual and tabular representation of results

🔹 Covers both text preprocessing and feature extraction

🧪 NLP Techniques Implemented
1️⃣ Tokenization

Sentence Tokenization

Word Tokenization

Character Tokenization

2️⃣ Text Cleaning

Lowercasing

Removal of URLs, emails, mentions, hashtags

Removal of numbers & punctuation

Stopword removal using spaCy

Regex-based normalization

3️⃣ Stemming

Porter Stemmer

Lancaster Stemmer

Side-by-side comparison table

4️⃣ Lemmatization

POS tagging

Lemma extraction using spaCy NLP pipeline

5️⃣ Bag of Words (BoW)

Word frequency representation

Pie-chart visualization of top words

6️⃣ TF-IDF

Term importance scoring

Ranking of most important words

7️⃣ Word Embeddings

Word vector magnitude using spaCy embeddings

Semantic representation of words

🛠️ Technology Stack
Category	Tools
Language	Python
UI	Streamlit
NLP	NLTK, spaCy
ML	Scikit-learn
Data Handling	Pandas
Visualization	Matplotlib
📂 Project Structure
nlp-preprocessing-app/
│
├── app.py              # Main Streamlit application
├── README.md           # Project documentation
├── requirements.txt    # Project dependencies

⚙️ Installation & Setup
Step 1: Clone Repository
git clone https://github.com/your-username/nlp-preprocessing-app.git
cd nlp-preprocessing-app

Step 2: Install Dependencies
pip install streamlit nltk spacy pandas scikit-learn matplotlib

Step 3: Download spaCy Model
python -m spacy download en_core_web_sm

Step 4: Run the App
streamlit run app.py

🧾 Sample Input
Satya is the BEST HOD of HIT and loves NLP.

🎯 Learning Outcomes

Understand text preprocessing pipeline

Learn feature extraction techniques in NLP

Visualize word importance and frequency

Gain hands-on experience with NLP libraries

Build interactive ML applications using Streamlit

🎓 Academic Relevance

✔ NLP Lab
✔ Mini Project
✔ Semester Practical
✔ Resume / Portfolio Project
✔ Viva Demonstration

👨‍💻 Author

Amarjeet Kumar
Computer Science Student
Interests: NLP, Machine Learning, AI

📜 License

This project is open-source and intended for educational purposes.
