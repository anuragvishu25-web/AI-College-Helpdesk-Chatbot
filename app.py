from flask import Flask, render_template, request, jsonify
import pickle
import json
import random

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

with open("intents.json") as file:
    intents = json.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_message = request.json["message"]
    X = vectorizer.transform([user_message])
    prediction = model.predict(X)[0]

    for intent in intents["intents"]:
        if intent["tag"] == prediction:
            return jsonify({"reply": random.choice(intent["responses"])})

    return jsonify({"reply": "Sorry, I didn't understand that."})

if __name__ == "__main__":
    app.run(port=5001, debug=True)
    