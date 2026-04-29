from fastapi import FastAPI
import joblib
from  app.schema import TextRequest

app = FastAPI()

model = joblib.load("ml/model.pkl")
vectorizer = joblib.load("ml/vectorizer.pkl")

@app.post("/spam")
def spamCheck(request:TextRequest):
    text=  request.text
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    confidence = round(model.predict_proba(vec).max() * 100,2)

    # add emoji for easy reading
    icon = "🔴 SPAM" if prediction == "spam" else "🟢 HAM"

    val = f"{text[:50]:<50} {icon}  ({confidence:.1f}%)"

    return {
        "text":text,
        "prediction":prediction,
        "confidence":confidence,
        "icon":icon
    }
