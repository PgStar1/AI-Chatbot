from chatbot.preprocessing import load_training_data

X, y, all_words, tags = load_training_data('data/intents.json')
print(X.shape)  # Should show shape of training data
print(tags)     # Should print all intent tags
