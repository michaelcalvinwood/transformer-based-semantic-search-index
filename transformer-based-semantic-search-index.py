# See https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 for more details
# Importing necessary modules
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import torch
import os

# Load a sample from the Multi-News dataset and convert it to a pandas dataframe
dataset = load_dataset("multi_news", split="test", download_mode="force_redownload")
df = dataset.to_pandas().sample(2000, random_state=42)

# Load the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode the summaries and store them as embeddings
df["embedding"] = list(model.encode(df["summary"].to_list(), show_progress_bar=True))
passage_embeddings = df["embedding"].to_list()

def find_relevant_news(query):
    # Encode the query using the same model
    query_embedding = model.encode(query)

    # Calculate the cosine similarity between the query and passage embeddings
    similarities = util.cos_sim(query_embedding, passage_embeddings)

    # Get the indices of the top 3 most similar passages
    top_indices = torch.topk(similarities.flatten(), 3).indices

    # Retrieve the summaries of the top 3 passages and truncate them to 160 characters
    top_relevant_passages = [df.iloc[x.item()]["summary"][:160] + "..." for x in top_indices]

    return top_relevant_passages

def clear_screen():
    os.system("clear")

def interactive_search():
    print("Welcome to the Semantic News Search!\n")
    while True:
        print("Type in a topic you'd like to find articles about, and I'll do the searching! (Type 'exit' to quit)\n> ", end="")

        query = input().strip()

        if query.lower() == "exit":
            print("\nThanks for using the Semantic News Search! Have a great day!")
            break

        print("\n\tHere are 3 articles I found based on your query: \n")

        passages = find_relevant_news(query)
        for passage in passages:
            print("\n\t" + passage)

        input("\nPress Enter to continue searching...")
        clear_screen()

interactive_search()