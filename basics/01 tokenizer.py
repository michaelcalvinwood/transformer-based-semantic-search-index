# Import required libraries
from transformers import BertModel, AutoTokenizer
import pandas as pd

# Specify the pre-trained model to use: BERT-base-cased
model_name = "bert-base-cased"

# Instantiate the model and tokenizer for the specified pre-trained model
model = BertModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set a sentence for analysis
sentence = "When life gives you lemons, don't make lemonade."

# Tokenize the sentence
tokens = tokenizer.tokenize(sentence)
print(tokens)

# Create a DataFrame with the tokenizer's vocabulary
vocab = tokenizer.vocab
vocab_df = pd.DataFrame({"token": vocab.keys(), "token_id": vocab.values()})
vocab_df = vocab_df.sort_values(by="token_id").set_index("token_id")

print("\nTokenizer Vocabulary\n", vocab_df)

# Encode the sentence into token_ids using the tokenizer
token_ids = tokenizer.encode(sentence)
print("\nToken IDs for the tokenized sentence\n", token_ids)


# Print the length of tokens and token_ids
print("\nNumber of tokens: ", len(tokens))
print("Number of token IDs:", len(token_ids))
print("The initial and last Token IDs mark the beginning and end as these tokens were used in training BERT.")

print("\nMapping Token IDs and tokens by removing the beginning/ending tokens (101 & 102)", # Zip tokens and token_ids (excluding the first and last token_ids for [CLS] and [SEP])
list(zip(tokens, token_ids[1:-1])))

print("\n\nDecoded tokens using token IDs:\n", tokenizer.decode(token_ids))

print("\n\nGetting the original sentence:\n", tokenizer.decode(token_ids[1:-1]))

# Tokenize the sentence using the tokenizer's `__call__` method
tokenizer_out = tokenizer(sentence)
tokenizer_out
print("\nTokenizer output structure:\n", tokenizer_out)
print("The 1's in the attention mask show that the token should be paid attention to.")

# Create a new sentence by removing "don't " from the original sentence
sentence2 = sentence.replace("don't ", "")
sentence2

# Tokenize both sentences with padding
tokenizer_out2 = tokenizer([sentence, sentence2], padding=True)
print("\nSentence 1: ", sentence)
print("\nSentence 2: ", sentence2)
print("\nTwo sentences tokenized with padding=True", tokenizer_out2)
print("\nNotice how the attention mask shows which tokens should be paid attention to.")

print("\nThe two sentences decoded:\n", tokenizer.decode(tokenizer_out2["input_ids"][0]), tokenizer.decode(tokenizer_out2["input_ids"][1]))
