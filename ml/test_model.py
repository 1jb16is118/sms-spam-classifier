# ml/test_model.py

import joblib

# Step 1: Load saved model and vectorizer
model      = joblib.load('ml/model.pkl')
vectorizer = joblib.load('ml/vectorizer.pkl')

print("Model loaded successfully!\n")

# Step 2: Test messages
test_messages = [
    "Congratulations! You won a free prize! Click now!",   # spam
    "Hey, are you coming for dinner tonight?",              # ham
    "FREE entry! Win cash prizes worth $1000. Call now!",  # spam
    "Can you pick up some milk on your way home?",          # ham
    "URGENT: Your account has been compromised. Click here",# spam
    "Mom, I'll be late. Don't wait for me for dinner.",    # ham
]

# Step 3: Predict each message
print(f"{'Message':<50} {'Prediction'}")
print("-" * 65)

for msg in test_messages:
    vec        = vectorizer.transform([msg])
    prediction = model.predict(vec)[0]
    confidence = model.predict_proba(vec).max() * 100

    # add emoji for easy reading
    icon = "🔴 SPAM" if prediction == "spam" else "🟢 HAM"

    print(f"{msg[:50]:<50} {icon}  ({confidence:.1f}%)")