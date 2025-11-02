import re
from difflib import SequenceMatcher

STOPWORDS = {"the", "a", "an", "in", "for", "to", "is", "and", "of", "what", "are"}

def tokenize_and_filter_description(text):
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())  # words >= 3 letters
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

def retrieve_relevant_text(user_query, text_chunks, top_k=3):
    query_terms = set(re.findall(r"\b\w+\b", user_query.lower()))
    scored_chunks = []
    for chunk in text_chunks:
        words = set(re.findall(r"\b\w+\b", chunk.lower()))
        score = len(query_terms.intersection(words))
        scored_chunks.append((score, chunk))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored_chunks[:top_k] if score > 0]

def clean_course_data(courses):
    cleaned = []
    seen = set()

    for c in courses:
        code = c.get("course_code", "").strip().upper()
        title = c.get("title", "").strip().title()
        desc = re.sub(r"\s+", " ", c.get("description", "")).strip()
        desc = tokenize_and_filter_description(desc)
        ctype = c.get("course_type", "").strip().title()
        minor = c.get("minor_track", [])

        # Skip empty or placeholder rows
        if not code or desc.lower() in ["n/a", "no description available."]:
            continue

        key = (code, title)
        if key not in seen:
            seen.add(key)
            cleaned.append({
                "course_code": code,
                "title": title,
                "description": desc,
                "course_type": ctype,
                "minor_track": minor
            })

    return cleaned


def remove_duplicate_courses(courses, threshold=0.9):
    unique = []
    for c in courses:
        if not any(SequenceMatcher(None, c["title"], u["title"]).ratio() > threshold for u in unique):
            unique.append(c)
    return unique


def clean_pdf_text(text):
    text = re.sub(r"Page\s*\d+\s*of\s*\d+", "", text)  # Remove "Page 1 of 12"
    text = re.sub(r"\s+", " ", text)                    # Collapse spaces/newlines
    text = text.replace("•", "-")                       # Normalise bullet points
    text = text.strip()
    return text
