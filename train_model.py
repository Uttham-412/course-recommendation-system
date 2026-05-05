import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

def train_and_save_model():
    print("Loading dataset...")
    df = pd.read_csv("udemy_courses_dataset.csv")

    print("Cleaning and preparing data...")
    # Keep essential columns including url
    df = df[['course_id', 'course_title', 'url', 'subject', 'level', 'price', 'num_subscribers']]
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    # Create a rich content feature by combining multiple text columns
    # We add level and subject to provide better context
    df['content'] = df['course_title'] + " " + df['subject'] + " " + df['level']

    print("Training TF-IDF Vectorizer...")
    start_time = time.time()
    # Use stop_words='english' to ignore words like 'and', 'the', 'is'
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(df['content'])
    print(f"TF-IDF Vectorization completed in {time.time() - start_time:.2f} seconds.")

    print("Computing Cosine Similarity Matrix...")
    start_time = time.time()
    similarity = cosine_similarity(matrix)
    print(f"Cosine Similarity completed in {time.time() - start_time:.2f} seconds.")

    print("Saving models to disk...")
    # Save the dataframe, vectorizer, and similarity matrix
    with open('dataframe.pkl', 'wb') as f:
        pickle.dump(df, f)
    
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)

    with open('similarity.pkl', 'wb') as f:
        pickle.dump(similarity, f)
        
    print("Models saved successfully!")

def test_model(query):
    print(f"\n--- Testing Query: '{query}' ---")
    
    # Load models
    with open('dataframe.pkl', 'rb') as f:
        df = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('similarity.pkl', 'rb') as f:
        similarity = pickle.load(f)

    # Transform the query using the fitted vectorizer
    query_matrix = tfidf.transform([query])
    
    # Calculate cosine similarity between the query and all courses
    query_similarity = cosine_similarity(query_matrix, tfidf.transform(df['content']))
    
    # Get top 5 recommendations
    distances = list(enumerate(query_similarity[0]))
    courses = sorted(distances, key=lambda x: x[1], reverse=True)[0:5]
    
    for i in courses:
        idx = i[0]
        score = i[1]
        if score > 0: # Only show matches with some relevance
            print(f"[{score:.4f}] {df.iloc[idx]['course_title']}")

if __name__ == "__main__":
    train_and_save_model()
    
    # Run some test queries
    test_model("web development for beginners")
    test_model("machine learning python")
