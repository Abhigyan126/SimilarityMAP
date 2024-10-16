import os
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class DocumentSimilarityApp:
    def __init__(self, folder_path, cache_dir='model_cache'):
        """
        Initialize the application with a folder path and cache directory for BERT model.
        :param folder_path: Path to the folder containing text files
        :param cache_dir: Path to cache BERT model and tokenizer locally
        """
        self.folder_path = folder_path
        self.cache_dir = cache_dir
        self.tokenizer = None
        self.model = None
        self.documents = []
        self.embeddings = []

        # Ensure cache directory exists
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # Load the model and tokenizer from cache
        self.load_model()

    def load_model(self):
        """
        Load the pre-trained BERT model and tokenizer, cache them locally.
        """
        print("Loading BERT model and tokenizer...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=self.cache_dir)
        self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=self.cache_dir)
        self.model.eval()  # Set the model to evaluation mode
        print("Model and tokenizer loaded.")

    def get_bert_embedding(self, text):
        """
        Get the BERT embedding for a given text.
        :param text: The text for which the embedding is calculated
        :return: BERT embedding for the text
        """
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

    def read_text_files(self):
        """
        Read all text files from the given folder path.
        :return: A list of tuples containing filenames and their content
        """
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.txt'):
                with open(os.path.join(self.folder_path, filename), 'r') as file:
                    self.documents.append((filename, file.read()))  # (filename, text content)
        return self.documents

    def calculate_similarity(self):
        """
        Calculate cosine similarity between the BERT embeddings of all documents.
        """
        # Get embeddings for all documents
        for filename, text in self.documents:
            embedding = self.get_bert_embedding(text)
            self.embeddings.append(embedding)

        # Convert embeddings to numpy arrays for similarity computation
        embeddings = torch.cat(self.embeddings).detach().numpy()

        # Calculate pairwise cosine similarity between all document embeddings
        similarity_matrix = cosine_similarity(embeddings)

        # Create a DataFrame for better readability
        document_names = [filename for filename, _ in self.documents]
        similarity_df = pd.DataFrame(similarity_matrix, index=document_names, columns=document_names)

        return similarity_df

    def save_similarity_matrix(self, similarity_df, output_file='similarity_matrix.csv'):
        """
        Save the similarity matrix to a CSV file.
        :param similarity_df: DataFrame containing similarity matrix
        :param output_file: Path to the output CSV file
        """
        similarity_df.to_csv(output_file)
        print(f"Similarity matrix saved to {output_file}")

    def plot_similarity_heatmap(self, similarity_df):
        """
        Plot a heatmap of the similarity matrix.
        :param similarity_df: DataFrame containing similarity matrix
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_df, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
        plt.title("Document Similarity Heatmap")
        plt.show()

    def run(self):
        """
        Run the full process of reading documents, calculating similarity, saving results, and plotting heatmap.
        """
        self.read_text_files()
        similarity_df = self.calculate_similarity()
        self.save_similarity_matrix(similarity_df)
        self.plot_similarity_heatmap(similarity_df)


# Main script execution
if __name__ == '__main__':
    folder_path = 'test'  # Specify the folder path containing text files
    app = DocumentSimilarityApp(folder_path)
    app.run()
