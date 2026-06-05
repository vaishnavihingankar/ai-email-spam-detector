from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

spam_words = ["free", "win", "winner", "lottery", "cash", "prize", "offer"]

def explain_spam(text):
    found = [w for w in spam_words if w in text.lower()]
    if found:
        return f"Contains suspicious words: {', '.join(found)}"
    return "No suspicious keywords detected."

def highlight_words(text):
    for word in spam_words:
        text = text.replace(word, f"<span class='highlight'>{word}</span>")
    return text

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    msg = data['message']

    vec = vectorizer.transform([msg])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][1]

    result = "Spam" if pred == 1 else "Ham"

    return jsonify({
        "result": result,
        "prob": round(prob * 100, 2),
        "explanation": explain_spam(msg),
        "highlighted": highlight_words(msg)
    })

if __name__ == "__main__":
    app.run(debug=True)