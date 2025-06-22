import random
import json
import torch

from chatbot.model import ChatModel
from chatbot.preprocessing import tokenize, stem, bag_of_words

# Load intents and model
with open('data/intents.json', 'r') as f:
    intents = json.load(f)

FILE = "chatbot_model.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = ChatModel(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Chat loop
print("🤖 Chatbot is online! (type 'quit' to exit)")
while True:
    sentence = input("You: ")
    if sentence.lower() == "quit":
        break

    tokens = tokenize(sentence)
    bow = bag_of_words(tokens, all_words)
    bow_tensor = torch.tensor(bow).unsqueeze(0)

    output = model(bow_tensor)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    confidence = probs[0][predicted.item()]

    if confidence.item() > 0.75:
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                print(f"{random.choice(intent['responses'])}")
    else:
        print("I'm not sure I understand 🤔")
