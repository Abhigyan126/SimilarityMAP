import os
import tkinter as tk
from tkinter import filedialog, messagebox
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

class DocumentSimilarityAppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Document Similarity App")
        self.root.geometry("1000x800")  # Default window size
        self.root.resizable(True, True)  # Allow window resizing
        

        # Initialize class variables
        self.folder_path = ''
        self.cache_dir = 'model_cache'
        self.tokenizer = None
        self.model = None
        self.documents = []
        self.embeddings = []
        self.image_label = None  # For displaying the generated image

        # Frame for log and title
        self.top_frame = tk.Frame(root)
        self.top_frame.pack(side="top", fill="x", pady=10)

        # Log label
        self.log_label = tk.Label(self.top_frame, text="Waiting for action...", anchor="center")
        self.log_label.pack(side="top", pady=1, fill="both", expand=True)

        # Title label
        self.title_label = tk.Label(self.top_frame, text="Document Similarity App", font=("Arial", 18))
        self.title_label.pack(side="top", pady=1)

        # Frame for controls
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(side="bottom", pady=10, fill="x")

        # Open Folder Button
        self.open_button = tk.Button(self.control_frame, text="Open Folder", command=self.open_folder)
        self.open_button.pack(side="left", padx=10)

        # Save CSV Button
        self.save_button = tk.Button(self.control_frame, text="Save CSV", command=self.save_csv)
        self.save_button.pack(side="right", padx=10)

        # Frame for image display (dynamically resizing)
        self.image_frame = tk.Frame(root)
        self.image_frame.pack(side="top", fill="both", expand=True)

        # Load model
        self.load_model()

    def load_model(self):
        """Load the pre-trained BERT model and tokenizer, cache them locally."""
        self.log_label.config(text="Loading BERT model and tokenizer...")
        self.root.update()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=self.cache_dir)
        self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=self.cache_dir)
        self.model.eval()  # Set the model to evaluation mode
        self.log_label.config(text="Model and tokenizer loaded.")

    def get_bert_embedding(self, text):
        """Get the BERT embedding for a given text."""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

    def read_text_files(self):
        """Read all text files from the given folder path."""
        self.documents.clear()
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.txt'):
                with open(os.path.join(self.folder_path, filename), 'r') as file:
                    self.documents.append((filename, file.read()))  # (filename, text content)
        return self.documents

    def calculate_similarity(self):
        """Calculate cosine similarity between the BERT embeddings of all documents."""
        self.embeddings.clear()
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
        """Save the similarity matrix to a CSV file."""
        similarity_df.to_csv(output_file)
        messagebox.showinfo("Saved", f"Similarity matrix saved to {output_file}")

    def plot_similarity_heatmap(self, similarity_df):
        """Plot a heatmap of the similarity matrix and display it in the app."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_df, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
        plt.title("Document Similarity Heatmap")

        # Save the plot as a PNG image
        heatmap_image_path = "similarity_heatmap.png"
        plt.savefig(heatmap_image_path, bbox_inches='tight', dpi=300)  # Increase DPI for high-quality image
        plt.close()

        # Update GUI to display the image
        self.display_image(heatmap_image_path)

        # Save the similarity matrix CSV file
        self.save_similarity_matrix(similarity_df, "similarity_matrix.csv")

    def display_image(self, image_path):
        """Display the generated image in the GUI."""
        image = Image.open(image_path)

        # Get the dimensions of the image frame
        frame_width = self.image_frame.winfo_width()
        frame_height = self.image_frame.winfo_height()

        # Resize the image if it exceeds the frame dimensions
        image_width, image_height = image.size
        scale_factor = 1

        if image_width > frame_width:
            scale_factor = frame_width / image_width
        if image_height * scale_factor > frame_height:
            scale_factor = frame_height / image_height

        # Resize the image while maintaining aspect ratio
        if scale_factor < 1:  # Only resize if necessary
            new_size = (int(image_width * scale_factor), int(image_height * scale_factor))
            image = image.resize(new_size, Image.Resampling.LANCZOS)  # Use high-quality resampling

        image = ImageTk.PhotoImage(image)

        if self.image_label:
            self.image_label.destroy()

        # Create a new label to hold the image
        self.image_label = tk.Label(self.image_frame, image=image)
        self.image_label.image = image  # Keep reference to avoid garbage collection
        self.image_label.pack(side="top", fill="both", expand=True)
        self.log_label.config(text="MAP generated")

    def open_folder(self):
        """Open a dialog to select a folder containing text files."""
        self.folder_path = filedialog.askdirectory(title="Select Folder")
        if self.folder_path:
            self.log_label.config(text=f"Folder selected: {self.folder_path}")
            self.run_similarity_calculation()

    def save_csv(self):
        """Save the similarity matrix to CSV after it's calculated."""
        if not self.documents:
            messagebox.showwarning("Warning", "No documents loaded. Please select a folder.")
            return
        similarity_df = self.calculate_similarity()
        self.save_similarity_matrix(similarity_df)

    def run_similarity_calculation(self):
        """Run the full process of reading documents, calculating similarity, and generating a heatmap."""
        if not self.folder_path:
            messagebox.showwarning("Warning", "Please select a folder first.")
            return

        self.log_label.config(text="Reading documents...")
        self.root.update()

        self.read_text_files()

        self.log_label.config(text="Calculating similarity...")
        self.root.update()

        similarity_df = self.calculate_similarity()
        self.plot_similarity_heatmap(similarity_df)


if __name__ == "__main__":
    root = tk.Tk()
    app = DocumentSimilarityAppGUI(root)
    root.mainloop()
