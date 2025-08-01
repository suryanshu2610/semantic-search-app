import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re

# ----------------- Load data and model -----------------
df = pd.read_csv("patent_data.csv")

# 🛠️ Fill NaN values in relevant columns to avoid TypeErrors
df['title'] = df['title'].fillna("")
df['abstract'] = df['abstract'].fillna("")

embeddings = np.load("embeddings.npy")
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# ----------------- Helper function -----------------
def get_matching_sentences(text, query):
    if not isinstance(text, str):  # 🛡️ Defensive check
        return []
    
    sentences = re.split(r'(?<=[.!?]) +', text)
    query_words = set(word.lower() for word in query.split())
    
    matched_sentences = []
    for sentence in sentences:
        sentence_words = set(word.lower() for word in re.findall(r'\w+', sentence))
        if sentence_words.intersection(query_words):
            matched_sentences.append(sentence.strip())
    return matched_sentences

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Semantic Search App", layout="wide")
st.title("🔍 Patent Semantic Search")

query = st.text_input("Enter your search query:")
top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)

if st.button("Search") and query:
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    st.markdown(f"### 🔎 Top {top_k} Results for: `{query}`")
    
    for i in indices[0]:
        st.markdown("---")
        title = df.iloc[i]["title"]
        abstract = df.iloc[i]["abstract"]
        
        st.subheader(title)

        # Matching explanation
        matched_title_sents = get_matching_sentences(title, query)
        matched_abstract_sents = get_matching_sentences(abstract, query)

        if matched_title_sents or matched_abstract_sents:
            st.markdown("**📌 Why this result?**")
            if matched_title_sents:
                st.markdown("**Title Match:**")
                for sent in matched_title_sents:
                    st.write(f"> {sent}")
            if matched_abstract_sents:
                st.markdown("**Abstract Match:**")
                for sent in matched_abstract_sents:
                    st.write(f"> {sent}")
        else:
            st.markdown("_No direct match found in title or abstract sentences._")

        # Additional info
        st.markdown("**Abstract:**")
        st.write(abstract)
        st.write(f"**Patent Number:** {df.iloc[i].get('patent_number', 'N/A')}")
        st.write(f"**Publication Date:** {df.iloc[i].get('publication_date', 'N/A')}")
        st.write(f"**Inventors:** {df.iloc[i].get('inventors', 'N/A')}")
        st.write(f"**Assignees:** {df.iloc[i].get('assignees', 'N/A')}")
