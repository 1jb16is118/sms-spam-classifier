# 📬 SMS Spam Classifier

A beginner-friendly Machine Learning project that detects whether an SMS message is **spam** or **ham (not spam)** using FastAPI and Naive Bayes.

---

## 📌 What This Project Does

- Takes an SMS message as input
- Cleans and converts the text into numbers
- Uses a trained ML model to predict spam or ham
- Returns the result with a confidence percentage via a REST API

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Programming language |
| FastAPI | Web framework for the API |
| Uvicorn | Server to run FastAPI |
| Scikit-learn | ML model and TF-IDF vectorizer |
| Pandas | Loading and handling the dataset |
| Joblib | Saving and loading the trained model |

---

## 📁 Folder Structure

```
spam-classifier/
├── app/
│   ├── main.py           # FastAPI app and /spam endpoint
│   ├── model.py          # Loads saved model and vectorizer
│   └── schemas.py        # Pydantic input/output models
├── ml/
│   ├── train.py          # Train and save the ML model
│   ├── preprocess.py     # Text cleaning functions
│   ├── test_model.py     # Test model before running API
│   ├── model.pkl         # Saved trained model (auto-generated)
│   └── vectorizer.pkl    # Saved TF-IDF vectorizer (auto-generated)
├── data/
│   └── spam.csv          # SMS Spam Collection dataset
├── tests/
│   └── test_api.py       # API tests
├── requirements.txt      # All dependencies
└── README.md             # This file
```

---

## ⚙️ Setup Instructions

### 1. Clone or Download the Project

```bash
cd Desktop
mkdir spam-classifier
cd spam-classifier
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install fastapi uvicorn scikit-learn pandas joblib python-multipart
```

### 5. Save Dependencies

```bash
pip freeze > requirements.txt
```

---

## 📦 Dataset

This project uses the **SMS Spam Collection Dataset**.

Download it from one of these sources:

- **Kaggle:** https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- **UCI Repository:** https://archive.ics.uci.edu/dataset/228/sms+spam+collection

After downloading, place the file here:
```
data/spam.csv
```

---

## 🤖 Train the Model

Run this command from the project root folder:

```bash
python ml/train.py
```

This will:
1. Load the dataset from `data/spam.csv`
2. Clean and preprocess the text
3. Train a Naive Bayes model
4. Save `model.pkl` and `vectorizer.pkl` inside the `ml/` folder

Expected output:
```
Dataset loaded!
(5572, 2)
Accuracy: 97.85%
Model and vectorizer saved!
```

---

## 🧪 Test the Model (Before Running API)

```bash
python ml/test_model.py
```

Expected output:
```
Message                                            Prediction
-----------------------------------------------------------------
Congratulations! You won a free prize! Click now   🔴 SPAM  (99.2%)
Hey, are you coming for dinner tonight?            🟢 HAM   (97.8%)
```

---

## 🚀 Run the API

```bash
uvicorn app.main:app --reload
```

The API will start at:
```
http://127.0.0.1:8000
```

---

## 📖 API Documentation (Swagger UI)

FastAPI generates interactive docs automatically. Open this in your browser:

```
http://127.0.0.1:8000/docs
```

You can test the endpoint directly from the browser — no extra tools needed.

---

## 📡 API Endpoint

### `POST /spam`

Checks whether an SMS message is spam or ham.

**Request Body:**
```json
{
  "text": "Congratulations! You won a free prize!"
}
```

**Response:**
```json
{
  "message"    : "Congratulations! You won a free prize!",
  "prediction" : "spam",
  "confidence" : "99.2%",
  "icon"       : "🔴 SPAM"
}
```

---

## 🧠 How the ML Works

```
Raw SMS Text
    ↓
Preprocessor       → lowercase, strip spaces
    ↓
TF-IDF Vectorizer  → converts text to numbers
    ↓
Naive Bayes Model  → predicts spam or ham
    ↓
JSON Response      → label + confidence %
```

The model is trained on 80% of the data and tested on the remaining 20%.
Typical accuracy: **96–98%**

---

## 📊 Model Performance

| Metric | Ham | Spam |
|---|---|---|
| Precision | ~99% | ~96% |
| Recall | ~99% | ~93% |
| F1-Score | ~99% | ~94% |

Overall Accuracy: **~97–98%**

---

## ❓ Common Errors

| Error | Cause | Fix |
|---|---|---|
| `Could not import module "main"` | Wrong uvicorn command | Use `uvicorn app.main:app --reload` |
| `FileNotFoundError: model.pkl` | Model not trained yet | Run `python ml/train.py` first |
| `UnicodeDecodeError` | Wrong file encoding | Add `encoding='latin-1'` when reading CSV |
| `ModuleNotFoundError` | Library not installed | Run `pip install -r requirements.txt` |

---

## 📝 Things to Update

- [ ] Add your name as author
- [ ] Update accuracy score after training
- [ ] Add any extra features you build
- [ ] Add your dataset source link

---

## 📚 What I Learned Building This

- How to train and save an ML model
- How TF-IDF converts text to numbers
- How `train_test_split` works and why we use it
- How to build a REST API with FastAPI
- How Naive Bayes classifies text

---

## 👤 Author

**Your Name Here**
Built for learning purposes — FastAPI + ML beginner project.