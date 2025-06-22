from flask import Flask, render_template, request, jsonify
import torch
import random
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#from chatbot.model import ChatModel
#from chatbot.preprocessing import tokenize, bag_of_words

import openai
import os


app = Flask(__name__)

# Load model
"""with open('data/intents.json', 'r') as f:
    intents = json.load(f)

data = torch.load("chatbot_model.pth")
model = ChatModel(data["input_size"], data["hidden_size"], data["output_size"])
model.load_state_dict(data["model_state"])
model.eval()

all_words = data["all_words"]
tags = data["tags"]

def get_response(msg):
    tokens = tokenize(msg)
    bow = bag_of_words(tokens, all_words)
    bow = torch.tensor(bow).unsqueeze(0)

    output = model(bow)
    _, pred = torch.max(output, dim=1)
    tag = tags[pred.item()]
    probs = torch.softmax(output, dim=1)
    conf = probs[0][pred.item()]

    if conf.item() > 0.75:
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
    else:
        return "I'm not sure I understand 🤔"
"""

#openai.api_key = os.getenv("OPENAI_API_KEY")  # safer way (set it in environment)

import openai
from openai import OpenAI
import os

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # or hardcode for dev

def get_response(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4"
            messages=[
                {"role": "system", "content": "You are a helpful and friendly chatbot."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"


@app.route("/")
def home():
    return render_template("index.html")

"""@app.route("/get", methods=["GET"])
def get_bot_response():
    user_input = request.args.get("msg")
    response = get_response(user_input)
    return jsonify({"response": response})
"""

@app.route("/get", methods=["GET"])
def get_bot_response():
    user_input = request.args.get("msg")
    reply = get_response(user_input)
    return jsonify({"response": reply})


if __name__ == "__main__":
    app.run(debug=True)
