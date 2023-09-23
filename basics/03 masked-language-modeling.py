# Importing necessary modules
from transformers import AutoModelForMaskedLM, AutoTokenizer
import pandas as pd
import numpy as np
from scipy.special import softmax

# Defining the model name
model_name = "bert-base-cased"

# Loading the pre-trained model and tokenizer
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a mask token
mask = tokenizer.mask_token

# Defining the sentence
sentence = f"I want to {mask} pizza for tonight."

# Tokenize the sentence
tokens = tokenizer.tokenize(sentence)

# Encode the input sentence and getting model predictions
encoded_inputs = tokenizer(sentence, return_tensors="pt")
output = model(**encoded_inputs)

# Detach the logits from the model output and converting to numpy array
logits = output.logits.detach().numpy()[0]

# Extracting the logits for the masked token and calculating the confidence scores
masked_logits = logits[tokens.index(mask) + 1]
confidence_scores = softmax(masked_logits)

# Iterating over the top 5 predicted tokens and printing the sentences with the masked token replaced
for i in np.argsort(confidence_scores)[::-1][:5]:
    pred_token = tokenizer.decode(i)
    score = confidence_scores[i]

    # print(pred_token, score)
    print(sentence.replace(mask, pred_token))

