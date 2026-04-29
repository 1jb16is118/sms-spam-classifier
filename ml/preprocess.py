
import re
import string

def clean_text(text: str) -> str:
    """
    Cleans raw SMS text before feeding into the ML model.
    Same function is used during training AND prediction.
    """

    # Step 1: Convert to lowercase
    # "FREE PRIZE" and "free prize" should mean the same thing
    text = text.lower()

    # Step 2: Remove punctuation
    # "win!!!" becomes "win"
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Step 3: Remove numbers
    # "call 07984562" becomes "call"
    text = re.sub(r'\d+', '', text)

    # Step 4: Remove extra whitespace
    # "hello   world" becomes "hello world"
    text = re.sub(r'\s+', ' ', text)

    # Step 5: Strip leading and trailing spaces
    text = text.strip()

    return text


def clean_batch(texts):
    """
    Cleans a list of messages at once.
    Used in train.py to clean the entire dataset.
    """
    return [clean_text(text) for text in texts]