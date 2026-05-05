import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained ML models and data
try:
    with open('dataframe.pkl', 'rb') as f:
        df = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('similarity.pkl', 'rb') as f:
        similarity = pickle.load(f)
except FileNotFoundError:
    raise Exception("Model files not found. Please run 'train_model.py' first.")

def recommend(user_query):
    # Transform the user's free-text query using the pre-trained TF-IDF vectorizer
    query_matrix = tfidf.transform([user_query])
    
    # Calculate cosine similarity between the query and all courses in the dataset
    query_similarity = cosine_similarity(query_matrix, tfidf.transform(df['content']))
    
    # Get similarity scores and sort them in descending order
    distances = list(enumerate(query_similarity[0]))
    
    # Returning top 25 courses to provide many recommendations
    sorted_courses = sorted(distances, key=lambda x: x[1], reverse=True)[0:25]
    
    recommendations = []
    for i in sorted_courses:
        idx = i[0]
        score = i[1]
        
        # Only include results with a meaningful similarity score
        if score > 0.05:
            course_info = {
                'title': df.iloc[idx]['course_title'],
                'url': df.iloc[idx]['url'],
                'subject': df.iloc[idx]['subject'],
                'level': df.iloc[idx]['level'],
                'price': df.iloc[idx]['price']
            }
            recommendations.append(course_info)
            
    return recommendations