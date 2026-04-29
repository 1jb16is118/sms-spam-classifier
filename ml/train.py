import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
import joblib
import os
from preprocess import clean_batch

df = pd.read_csv("data/spam.csv",encoding='latin-1')

df = df[['v1','v2']]
df.columns = ["label",'text']

print("DataSet Loaded")
print(df.shape)
print(df['label'].value_counts())

# def clean_text(text):
#     text = text.lower()
#     text = text.strip()
#     return text

# df['text'] = df['text'].apply(clean_text)
df['text'] = clean_batch(df['text'])
print("sampled Cleaned")
print(df['text'].head(3))

X = df['text']
y = df['label']

# Split into training (80%) and testing (20%)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

vectorizer  = TfidfVectorizer(stop_words="english",max_features=3000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec,y_train)

print("\nModel trained!")
y_pred = model.predict(X_test_vec)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# Step 7: Save model and vectorizer to disk
os.makedirs('ml', exist_ok=True)

joblib.dump(model,      'ml/model.pkl')
joblib.dump(vectorizer, 'ml/vectorizer.pkl')

print("\nModel saved to ml/model.pkl")
print("Vectorizer saved to ml/vectorizer.pkl")