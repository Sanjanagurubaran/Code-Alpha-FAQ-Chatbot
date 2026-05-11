from flask import Flask, render_template, request, jsonify
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load FAQ data
data = pd.read_csv("faq.csv")

questions = data["question"].tolist()
answers = data["answer"].tolist()

# Convert text into vectors
vectorizer = TfidfVectorizer()

question_vectors = vectorizer.fit_transform(questions)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():

    user_message = request.json["message"].lower()

    # Greeting responses
    greetings = ["hi", "hello", "hey"]

    if user_message in greetings:
        return jsonify({
            "reply": "Hello! How can I help you today?"
        })

    # Convert message to vector
    user_vector = vectorizer.transform([user_message])

    # Similarity checking
    similarity = cosine_similarity(user_vector, question_vectors)

    best_match_index = similarity.argmax()

    best_score = similarity[0][best_match_index]

    # If similarity is too low
    if best_score < 0.3:
        response = "Sorry, I couldn't understand your question."

    else:
        response = answers[best_match_index]

    return jsonify({
        "reply": response
    })

if __name__ == "__main__":
    app.run(debug=True)