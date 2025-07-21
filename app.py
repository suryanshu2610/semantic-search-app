import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

# Load precomputed embeddings and data
df = pd.read_csv("patent_data.csv")
embeddings = np.load("embeddings.npy")

# Rebuild FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Load model 
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Function to highlight query words in text
def highlight_text(text, query):
    # Get unique lowercase query words
    query_words = set(query.lower().split())
    
    # Function to replace matched words with <mark> tags
    def replacer(match):
        word = match.group(0)
        if word.lower() in query_words:
            return f"<mark>{word}</mark>"
        else:
            return word

    # Regex to find words (case insensitive)
    pattern = re.compile(r'\b\w+\b', re.IGNORECASE)
    highlighted_text = pattern.sub(replacer, text)
    return highlighted_text

# UI

st.title("🔎 Patent Semantic Search Engine")
st.write("Search patents by meaning using AI-powered semantic search.")

sort_option = st.selectbox(
    "Sort results by:",
    ["Semantic Score", "Publication Date (newest)", "Abstract Length"],
    key="sort_selectbox"
)

query = st.text_input("Enter your search query:")
top_k = st.slider("Number of results", 1, 10, 3)

if st.button("Search") and query:
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    st.markdown(f"### 🔍 Top {top_k} Results for: `{query}`")
    for i in indices[0]:
        st.markdown("----")

        # Highlight title
        highlighted_title = highlight_text(df.iloc[i]["title"], query)
        st.markdown(f"### {highlighted_title}", unsafe_allow_html=True)

        st.write(f"**Patent Number:** {df.iloc[i].get('patent_number', 'N/A')}")
        st.write(f"**Publication Date:** {df.iloc[i].get('publication_date', 'N/A')}")
        st.write(f"**Inventors:** {df.iloc[i].get('inventors', 'N/A')}")
        st.write(f"**Assignees:** {df.iloc[i].get('assignees', 'N/A')}")

        st.markdown("**Abstract:**")
        # Highlight abstract
        highlighted_abstract = highlight_text(df.iloc[i]["abstract"], query)
        st.markdown(highlighted_abstract, unsafe_allow_html=True)
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def highlight_text(text, query):
    query_words = set(word.lower() for word in query.split() if word.lower() not in stop_words)
    
    def replacer(match):
        word = match.group(0)
        if word.lower() in query_words:
            return f"<mark>{word}</mark>"
        else:
            return word

    pattern = re.compile(r'\b\w+\b', re.IGNORECASE)
    return pattern.sub(replacer, text)
