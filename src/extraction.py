import spacy
from spacy import displacy
import re

# Load the English NLP model (fine-tune with a legal domain model if needed)
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    """Extracts key named entities from legal document text."""
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def extract_contract_details(text):
    """Extracts specific details like parties and dates from contract text using patterns."""
    parties = re.findall(r"(Party\s[A|B]:\s)([A-Za-z\s]+)", text)
    dates = re.findall(r"\b(?:\d{1,2}[/-])?(?:\d{1,2}[/-])?\d{2,4}\b", text)
    amounts = re.findall(r"\$\d+(?:,\d{3})*(?:\.\d{2})?", text)
    return {"parties": parties, "dates": dates, "amounts": amounts}

def display_entities(text):
    """Display entities in the document text."""
    doc = nlp(text)
    displacy.render(doc, style="ent", jupyter=True)

