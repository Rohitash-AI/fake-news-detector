import streamlit as st
import pickle
import re
import os
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ── Train model if not exists ─────────────────────────
if not os.path.exists('model.pkl'):
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    st.info("Training model for first time... please wait 2-3 minutes!")

    fake = pd.read_csv('Fake.csv')
    real = pd.read_csv('True.csv')
    fake['label'] = 0
    real['label'] = 1

    df = pd.concat([fake, real], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    stop_words = set(stopwords.words('english'))
    def clean(text):
        text = str(text).lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        words = text.split()
        words = [w for w in words if w not in stop_words]
        return ' '.join(words)

    df['clean'] = (df['title'] + ' ' + df['text']).apply(clean)
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['clean'])
    model = LogisticRegression(max_iter=1000)
    model.fit(X, df['label'])

    pickle.dump(model, open('model.pkl', 'wb'))
    pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
    st.success("Model trained! Refreshing...")
    st.rerun()

# ── Load Model ────────────────────────────────────────
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# ── UI ────────────────────────────────────────────────
st.set_page_config(page_title="Fake News Detector", page_icon="🗞️", layout="centered")
st.title("🗞️ Fake News Detector")
st.markdown("### Detect whether a news article is **Real** or **Fake** instantly!")
st.markdown("---")

news_input = st.text_area("📝 Paste your news headline or article here:", height=200,
    placeholder="e.g. Scientists discover new treatment for cancer...")

if st.button("🔍 Detect Now", use_container_width=True):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some text first!")
    else:
        cleaned = clean_text(news_input)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized)[0]

        st.markdown("---")
        if prediction == 0:
            st.error("🚨 FAKE NEWS DETECTED!")
            st.markdown(f"### Confidence: {confidence[0]*100:.2f}%")
        else:
            st.success("✅ REAL NEWS!")
            st.markdown(f"### Confidence: {confidence[1]*100:.2f}%")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔴 Fake", f"{confidence[0]*100:.1f}%")
        with col2:
            st.metric("🟢 Real", f"{confidence[1]*100:.1f}%")

st.markdown("---")
st.markdown("*Built with Python, Scikit-learn & Streamlit*")