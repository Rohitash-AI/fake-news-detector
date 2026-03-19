# model.py
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# ── 1. Load Data ──────────────────────────────────────
fake = pd.read_csv('Fake.csv')
real = pd.read_csv('True.csv')

fake['label'] = 0
real['label'] = 1

df = pd.concat([fake, real], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ── 2. Clean Text ─────────────────────────────────────
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

print("Cleaning text... (takes 1-2 mins)")
df['clean_text'] = (df['title'] + ' ' + df['text']).apply(clean_text)
print("Cleaning done!")

# ── 3. Split Data ─────────────────────────────────────
X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

# ── 4. TF-IDF Vectorization ───────────────────────────
print("Vectorizing text...")
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print("Vectorization done!")

# ── 5. Train Model ────────────────────────────────────
print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
print("Training done!")

# ── 6. Evaluate ───────────────────────────────────────
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*40}")
print(f"✅ MODEL ACCURACY: {accuracy*100:.2f}%")
print(f"{'='*40}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

# ── 7. Save Model ─────────────────────────────────────
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
print("\n✅ Model saved as model.pkl")
print("✅ Vectorizer saved as tfidf.pkl")
print("\nDay 2 Complete! 🎉")