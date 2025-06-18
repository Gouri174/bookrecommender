from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import re
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Changed to recursive splitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Load data and initialize components
print("Loading books data...")
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# Initialize embeddings with a specific model
print("Initializing embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# Load and process documents
print("Loading documents...")
try:
    raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
    
    # Use recursive splitter which handles long documents better
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Reduced chunk size
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    documents = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(documents)} documents")
    
    print(f"Creating Chroma vector store with {len(documents)} documents...")
    db_books = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Vector store created successfully!")
except Exception as e:
    print(f"Error initializing vector store: {str(e)}")
    raise

# Prepare dropdown options
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    try:
        recs = db_books.similarity_search(query, k=initial_top_k)
        books_list = []
        
        for rec in recs:
            # Extract ISBN more carefully
            content = rec.page_content.strip()
            # Find the first sequence of digits (ISBN)
            isbn_match = re.search(r'\d+', content)
            if isbn_match:
                try:
                    isbn = int(isbn_match.group())
                    books_list.append(isbn)
                except ValueError:
                    print(f"Could not convert to ISBN: {isbn_match.group()}")
                    continue
        
        if not books_list:
            print("No valid ISBNs found in recommendations")
            return pd.DataFrame()
            
        book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

        if category != "All":
            book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
        else:
            book_recs = book_recs.head(final_top_k)

        if tone == "Happy":
            book_recs.sort_values(by="joy", ascending=False, inplace=True)
        elif tone == "Surprising":
            book_recs.sort_values(by="surprise", ascending=False, inplace=True)
        elif tone == "Angry":
            book_recs.sort_values(by="anger", ascending=False, inplace=True)
        elif tone == "Suspenseful":
            book_recs.sort_values(by="fear", ascending=False, inplace=True)
        elif tone == "Sad":
            book_recs.sort_values(by="sadness", ascending=False, inplace=True)

        return book_recs
    except Exception as e:
        print(f"Error in recommendation: {str(e)}")
        return pd.DataFrame()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", categories=categories, tones=tones)

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    query = data.get("query", "")
    category = data.get("category", "All")
    tone = data.get("tone", "All")
    
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append({
            "image": row["large_thumbnail"],
            "caption": caption
        })
    
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)  # Changed debug to False