# 📚 Course Recommendation System

A Machine Learning powered web application that recommends Udemy courses based on user queries using TF-IDF Vectorization and Cosine Similarity.

## 🚀 How to Run

1. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Machine Learning Model**
   Because the pre-trained ML models are too large for GitHub (>100MB), you must generate them locally first. Run the training script:
   ```bash
   python train_model.py
   ```
   *This will process the dataset and generate the `.pkl` files required by the app.*

3. **Start the Web App**
   Once the `.pkl` files are generated, you can start the Streamlit interface:
   ```bash
   streamlit run app.py
   ```
