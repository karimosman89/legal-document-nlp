from gensim.summarization import summarize
from transformers import pipeline

def textrank_summary(text, ratio=0.2):
    """Generate summary using TextRank from Gensim."""
    try:
        return summarize(text, ratio=ratio)
    except ValueError:
        return "Summary could not be generated. Text may be too short."

def transformer_summary(text):
    """Generate summary using a transformer model like BERT or GPT-2."""
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

