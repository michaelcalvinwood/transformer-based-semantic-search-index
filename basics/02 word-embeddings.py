# Importing necessary modules
from transformers import BertModel, AutoTokenizer
from scipy.spatial.distance import cosine

# Defining the model name
model_name = "bert-base-cased"

# Loading the pre-trained model and tokenizer
model = BertModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Defining a function to encode the input text and get model predictions
# Return the last hidden state (the final tokens)
def predict(text):
    encoded_inputs = tokenizer(text, return_tensors="pt")
    return model(**encoded_inputs)[0]

# Defining the sentences
sentence1 = "There was a fly drinking from my soup"
sentence2 = "There is a fly swimming in my juice"
# sentence2 = "To become a commercial pilot, he had to fly for 1500 hours." # second fly example

# Tokenizing the sentences
tokens1 = tokenizer.tokenize(sentence1)
tokens2 = tokenizer.tokenize(sentence2)

# Getting model predictions for the sentences
out1 = predict(sentence1)
out2 = predict(sentence2)

# Extracting embeddings for the word 'fly' in both sentences
emb1 = out1[0:, tokens1.index("fly"), :].detach()
emb2 = out2[0:, tokens2.index("fly"), :].detach()

# emb1 = out1[0:, 3, :].detach()
# emb2 = out2[0:, 3, :].detach()

# Calculating the cosine similarity between the embeddings
print("emb1: ", emb1)

import numpy as np
emb1 = np.array(emb1).flatten()
emb2 = np.array(emb2).flatten()

print("emb1: ", emb1)


print("\n\nCosine:\n", cosine(emb1, emb2))

