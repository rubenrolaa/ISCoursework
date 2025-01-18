import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from math import sqrt

# --- Step 1: Load and preprocess data ---
def load_and_preprocess_data():
    """
    Load and preprocess the Books, Ratings, and Users datasets.
    """
    # Load datasets
    books = pd.read_csv("Books.csv", encoding="latin-1")
    ratings = pd.read_csv("Ratings.csv", encoding="latin-1")
    users = pd.read_csv("Users.csv", encoding="latin-1")
    
    # Ensure proper columns
    print("Datasets loaded.")
    print("Books:", books.columns)
    print("Ratings:", ratings.columns)
    print("Users:", users.columns)
    
    # Check for missing data and drop unnecessary columns
    ratings = ratings.dropna()
    books = books.dropna()
    
    return books, ratings, users

# --- Step 2: Collaborative Filtering ---
def collaborative_filtering(ratings):
    """
    Implement Collaborative Filtering using User-User similarity.
    """
    print("Running Collaborative Filtering...")
    
    # Create user-item matrix
    user_item_matrix = ratings.pivot(index='UserID', columns='BookID', values='Rating').fillna(0)
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)
    
    # Predict ratings
    def predict_ratings(user_id, book_id):
        if book_id in user_item_matrix.columns:
            user_ratings = user_item_matrix.loc[:, book_id]
            user_similarities = similarity_df.loc[user_id]
            weighted_ratings = np.dot(user_similarities, user_ratings)
            normalization = np.sum(user_similarities)
            return weighted_ratings / normalization if normalization > 0 else 0
        return 0
    
    # Generate predictions for all books for a user
    user_predictions = {}
    for user_id in user_item_matrix.index:
        predictions = []
        for book_id in user_item_matrix.columns:
            predictions.append((book_id, predict_ratings(user_id, book_id)))
        user_predictions[user_id] = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]  # Top 5 recommendations
    
    print("Collaborative Filtering Complete.")
    return user_predictions

# --- Step 3: Content-Based Filtering ---
def content_based_filtering(books, ratings):
    """
    Implement Content-Based Filtering using book features.
    """
    print("Running Content-Based Filtering...")
    
    # Use 'Title' or other metadata as features for recommendation
    books['Title'] = books['Title'].fillna('')
    vectorizer = TfidfVectorizer(stop_words='english')
    book_vectors = vectorizer.fit_transform(books['Title'])
    
    # Compute similarity between books
    similarity_matrix = cosine_similarity(book_vectors)
    
    # Generate recommendations based on similarity
    book_recommendations = {}
    for book_id in books['BookID']:
        book_idx = books.index[books['BookID'] == book_id][0]
        similar_books = similarity_matrix[book_idx]
        similar_books_idx = similar_books.argsort()[-6:-1][::-1]  # Top 5 similar books
        similar_books_ids = books.iloc[similar_books_idx]['BookID'].values
        book_recommendations[book_id] = similar_books_ids
    
    print("Content-Based Filtering Complete.")
    return book_recommendations

# --- Step 4: Evaluation Metrics ---
def evaluate_collaborative(predictions, actual_ratings):
    """
    Evaluate Collaborative Filtering using RMSE.
    """
    y_pred = []
    y_true = []
    
    for user_id, user_predictions in predictions.items():
        for book_id, predicted_rating in user_predictions:
            actual = actual_ratings.loc[(actual_ratings['UserID'] == user_id) & 
                                        (actual_ratings['BookID'] == book_id), 'Rating']
            if not actual.empty:
                y_pred.append(predicted_rating)
                y_true.append(actual.values[0])
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    return rmse

# --- Step 5: Main Function ---
def main():
    # Load data
    books, ratings, users = load_and_preprocess_data()
    
    # Collaborative Filtering
    user_predictions = collaborative_filtering(ratings)
    print("Top 5 Collaborative Recommendations for each user:", user_predictions)
    
    # Content-Based Filtering
    book_recommendations = content_based_filtering(books, ratings)
    print("Content-Based Recommendations:", book_recommendations)
    
    # Evaluate Collaborative Filtering
    rmse = evaluate_collaborative(user_predictions, ratings)
    print("Collaborative Filtering RMSE:", rmse)

# --- Run the program ---
if __name__ == "__main__":
    main()