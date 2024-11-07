import re
import string
import spacy
from sklearn.metrics import classification_report

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """Preprocess and clean text data."""
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)  # Remove content within brackets
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text

def tokenize_text(text):
    """Tokenize and remove stop words from text."""
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return tokens

def evaluate_extraction(y_true, y_pred):
    """Evaluate extraction performance using classification report."""
    return classification_report(y_true, y_pred)

