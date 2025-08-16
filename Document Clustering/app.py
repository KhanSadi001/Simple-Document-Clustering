from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
import re

app = Flask(__name__)

# -------------------- TEXT CLEANING --------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)  # keep only letters
    text = " ".join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    return text

# -------------------- ROUTE --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    clusters = {}
    error = None

    if request.method == "POST":
        # Step A: Collect documents
        docs = []

        # Uploaded files
        uploaded_files = request.files.getlist("files")
        for file in uploaded_files:
            if file and file.filename.endswith(".txt"):
                content = file.read().decode("utf-8").strip()
                if content:
                    docs.append(content)

        # Pasted text
        pasted_text = request.form.get("pasted_text")
        if pasted_text:
            docs.extend([t.strip() for t in pasted_text.split("\n") if t.strip()])

        # Number of clusters
        try:
            k = int(request.form.get("k"))
        except:
            k = 2

        # Step B: Validation
        if not docs:
            error = "Please provide at least one document (upload or paste)."
        elif len(docs) < k:
            error = f"Number of documents ({len(docs)}) must be greater than or equal to the number of clusters ({k})."
        else:
            # Step C: Clean text
            cleaned_docs = [clean_text(doc) for doc in docs]

            # Step D: TF-IDF Vectorization
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(cleaned_docs)

            # Step E: KMeans Clustering
            model = KMeans(n_clusters=k, random_state=42)
            model.fit(X)
            labels = model.labels_

            # Step F: Extract top keywords per cluster
            terms = vectorizer.get_feature_names_out()
            top_keywords = {}
            for i, center in enumerate(model.cluster_centers_):
                term_indices = center.argsort()[-5:][::-1]  # top 5
                top_keywords[i] = [terms[ind] for ind in term_indices]

            # Step G: Build cluster dictionary
            for cluster_id in range(k):
                doc_indices = [i for i, label in enumerate(labels) if label == cluster_id]
                clusters[cluster_id] = {
                    "documents": [docs[i] for i in doc_indices],
                    "keywords": top_keywords[cluster_id]
                }

    return render_template("index.html", clusters=clusters, error=error)

# -------------------- RUN APP --------------------
if __name__ == "__main__":
    app.run(debug=True)
