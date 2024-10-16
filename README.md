### **SimilarityMAP**  
This application calculates document similarities using BERT embeddings. It reads text files from a folder, computes pairwise similarities, and generates a heatmap for visualizing the results. The similarity matrix can be saved as a CSV file, and the heatmap as an image.

---

### **Files**:  
- **App.py**: A Tkinter application that requires a folder with text files. It creates a similarity heatmap for all the text files.  
- **get_map.py**: A non-GUI application that performs the same task as `App.py`.

---

### **Screenshots**  
<p align='center'>
<img width="1112" alt="Screenshot 2024-10-16 at 5 02 33 PM" src="https://github.com/user-attachments/assets/6bc8a56b-4667-49bc-a610-92c2a158f094">
</p>

---

### **Usage**  
- Can be used to **automatically grade** documents by identifying the best match and comparing similarity.  
  - **Limitation**: Only provides deviation information, not whether the content is correct.
  
- Can also be used to **detect copied content and plagiarism**.  
  - **Limitation**: Only identifies plagiarism within the scope of the provided text context.
