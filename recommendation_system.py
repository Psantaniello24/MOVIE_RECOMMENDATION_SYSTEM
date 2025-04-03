import numpy as np
import pandas as pd
try:
    # Try importing from scikit-surprise
    from surprise import SVD, Dataset, Reader, accuracy
    from surprise.model_selection import train_test_split
except ImportError:
    # Fallback to separate surprise package
    print("Warning: Using fallback recommendation system without scikit-surprise")
    SVD = None
    Dataset = None
    Reader = None
    accuracy = None
    train_test_split = None

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
import os
import sys
import time
import gc  # Garbage collection

# Configure TensorFlow to use memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")

# Set TensorFlow to use dynamic memory allocation
tf.config.experimental.set_memory_growth = True
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

# Load MovieLens dataset with size option
def load_data(dataset_size='100k', sample_size=None):
    """
    Load MovieLens dataset
    
    Parameters:
    dataset_size -- Size of dataset to load: '100k', '1m', or '25m'
    sample_size -- If provided, randomly sample this many ratings
    
    Returns:
    ratings, movies dataframes
    """
    # Download dataset if not already present
    try:
        # Define paths based on dataset size
        if dataset_size == '100k':
            ratings_path = './ml-100k/u.data'
            movies_path = './ml-100k/u.item'
            
            # Load ratings
            ratings_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
            ratings = pd.read_csv(ratings_path, sep='\t', names=ratings_columns)
            
            # Load movies
            movies_columns = ['movie_id', 'title', 'release_date', 'video_release_date',
                            'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                            'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                            'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                            'Thriller', 'War', 'Western']
            movies = pd.read_csv(movies_path, sep='|', names=movies_columns, encoding='latin-1')
        
        elif dataset_size == '1m':
            ratings_path = './ml-1m/ratings.dat'
            movies_path = './ml-1m/movies.dat'
            
            # Load ratings
            ratings_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
            ratings = pd.read_csv(ratings_path, sep='::', names=ratings_columns, engine='python')
            
            # Load movies
            movies_columns = ['movie_id', 'title', 'genres']
            movies = pd.read_csv(movies_path, sep='::', names=movies_columns, encoding='latin-1', engine='python')
        
        elif dataset_size == '25m':
            ratings_path = './ml-25m/ratings.csv'
            movies_path = './ml-25m/movies.csv'
            
            # For 25M dataset, use chunks to reduce memory usage
            if sample_size is not None:
                # Skip header, read first chunk to get total size
                temp_df = pd.read_csv(ratings_path, nrows=1)
                total_rows = sum(1 for _ in open(ratings_path)) - 1  # Subtract header
                
                # Calculate skip rows to get a random sample
                skip_rows = sorted(np.random.choice(
                    range(1, total_rows + 1),  # Skip header (0)
                    total_rows - min(sample_size, total_rows),
                    replace=False
                ))
                skip_rows = [0] + skip_rows  # Add header back to skip list
                
                # Load just the sampled rows
                ratings = pd.read_csv(ratings_path, skiprows=skip_rows)
            else:
                # If no sample size provided, use chunking for memory efficiency
                print(f"Warning: Loading full 25M dataset. This may require significant memory.")
                chunk_size = 1000000  # Process 1M rows at a time
                ratings_chunks = pd.read_csv(ratings_path, chunksize=chunk_size)
                
                # Take first chunk to initialize
                ratings = next(ratings_chunks)
                
                # If sample size not specified, limit to first 200,000 rows for safety
                if sample_size is None:
                    sample_size = 200000
                    print(f"No sample size specified for 25M dataset. Limiting to {sample_size} ratings.")
                    if len(ratings) > sample_size:
                        ratings = ratings.sample(sample_size, random_state=42)
                    
                del ratings_chunks
                
            # Rename columns to match our expected format
            ratings.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'}, inplace=True)
            
            # Load movies
            movies = pd.read_csv(movies_path)
            # Rename columns to match our expected format
            movies.rename(columns={'movieId': 'movie_id'}, inplace=True)
        
        else:
            raise ValueError(f"Invalid dataset_size: {dataset_size}. Choose '100k', '1m', or '25m'")
        
        # Take a sample if specified
        if sample_size and sample_size < len(ratings) and dataset_size != '25m':  # 25M handled above
            ratings = ratings.sample(sample_size, random_state=42)
        
        # Force garbage collection after loading
        gc.collect()
        
        print(f"Loaded {len(ratings)} ratings and {len(movies)} movies from MovieLens {dataset_size} dataset")
        
        return ratings, movies
        
    except FileNotFoundError:
        print(f"Dataset not found. Downloading MovieLens {dataset_size} dataset...")
        import urllib.request
        import zipfile
        import os
        import socket
        
        # Set URL based on dataset size
        if dataset_size == '100k':
            url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
            zip_path = "./ml-100k.zip"
            extract_path = "./"
            timeout = 60  # 1 minute timeout
        elif dataset_size == '1m':
            url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
            zip_path = "./ml-1m.zip"
            extract_path = "./"
            timeout = 120  # 2 minute timeout
        elif dataset_size == '25m':
            url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
            zip_path = "./ml-25m.zip"
            extract_path = "./"
            timeout = 600  # 10 minute timeout for the large dataset
        else:
            raise ValueError(f"Invalid dataset_size: {dataset_size}. Choose '100k', '1m', or '25m'")
        
        # Create directory if it doesn't exist
        if not os.path.exists(extract_path):
            os.makedirs(extract_path)
        
        # Set socket timeout
        socket.setdefaulttimeout(timeout)
            
        try:
            # Download the file with progress reporting
            print(f"Downloading from {url} (this may take a while for larger datasets)...")
            
            def report_progress(blocknum, blocksize, totalsize):
                """Report download progress"""
                percent = min(int(blocknum * blocksize * 100 / totalsize), 100)
                sys.stdout.write(f"\rDownloading: {percent}% complete")
                sys.stdout.flush()
            
            start_time = time.time()
            urllib.request.urlretrieve(url, zip_path, reporthook=report_progress)
            print(f"\nDownload completed in {time.time() - start_time:.1f} seconds")
            
            # Extract the zip file
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
                
            # Remove the zip file
            os.remove(zip_path)
            
            print("Dataset downloaded and extracted successfully.")
            
            # Force garbage collection
            gc.collect()
            
            # Call load_data again now that the dataset is downloaded
            return load_data(dataset_size, sample_size)
            
        except Exception as e:
            print(f"Error downloading dataset: {str(e)}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            raise RuntimeError(f"Failed to download dataset: {str(e)}")

# Collaborative Filtering with Surprise SVD
class SVDRecommender:
    def __init__(self):
        self.model = SVD()
        self.ratings = None
        self.movies = None
        self.trainset = None
        self.testset = None
        
    def prepare_data(self, ratings, movies):
        self.ratings = ratings
        self.movies = movies
        
        # Create Surprise dataset
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)
        
        # Split into train and test sets
        self.trainset, self.testset = train_test_split(data, test_size=0.2, random_state=42)
        
        # Force garbage collection
        gc.collect()
        
    def train(self):
        # Train the model
        self.model.fit(self.trainset)
        
        # Force garbage collection
        gc.collect()
        
    def evaluate(self):
        # Test the model
        predictions = self.model.test(self.testset)
        # Calculate RMSE
        rmse = accuracy.rmse(predictions)
        print(f"SVD RMSE: {rmse}")
        return rmse
    
    def recommend_movies(self, user_id, n=5):
        # Get all movies the user hasn't rated
        rated_movies = set(self.ratings[self.ratings['user_id'] == user_id]['movie_id'].tolist())
        
        # Use a limited set of candidate movies for very large datasets
        if len(self.movies) > 10000:
            # Use a random sample of movies for prediction to save memory
            candidate_movies = list(set(self.movies.sample(min(5000, len(self.movies)))['movie_id'].tolist()) - rated_movies)
        else:
            candidate_movies = [movie for movie in self.movies['movie_id'].tolist() if movie not in rated_movies]
        
        # Predict ratings for candidate movies
        predictions = []
        for movie_id in candidate_movies:
            predicted_rating = self.model.predict(user_id, movie_id).est
            predictions.append((movie_id, predicted_rating))
        
        # Sort by predicted rating and get top n
        top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
        
        # Get movie details
        recommendations = []
        for movie_id, predicted_rating in top_n:
            movie_title = self.movies[self.movies['movie_id'] == movie_id]['title'].iloc[0]
            recommendations.append({
                'movie_id': movie_id,
                'title': movie_title,
                'predicted_rating': predicted_rating
            })
        
        # Force garbage collection
        gc.collect()
        
        return recommendations
    
    def recommend_from_new_ratings(self, selected_movies, n=5):
        """
        Recommend movies based on a set of 5 movies that a new user has rated 5 stars
        
        Parameters:
        selected_movies -- list of movie_ids that the user has selected (assumed rated 5)
        n -- number of recommendations to provide
        
        Returns:
        List of movie recommendations
        """
        # Create a temporary user ID that is higher than any existing user ID
        temp_user_id = self.ratings['user_id'].max() + 1
        
        # Create new ratings dataframe with the 5 movies the user selected (all rated 5)
        new_ratings = []
        timestamp = int(pd.Timestamp.now().timestamp())
        
        for movie_id in selected_movies:
            new_ratings.append({
                'user_id': temp_user_id,
                'movie_id': movie_id,
                'rating': 5.0,  # High rating to indicate preference
                'timestamp': timestamp
            })
        
        new_ratings_df = pd.DataFrame(new_ratings)
        
        # Combine with existing ratings
        combined_ratings = pd.concat([self.ratings, new_ratings_df])
        
        # Re-train model with the new data
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(combined_ratings[['user_id', 'movie_id', 'rating']], reader)
        trainset = data.build_full_trainset()
        
        # Clone the model to avoid affecting the original
        temp_model = SVD()
        temp_model.fit(trainset)
        
        # Use a limited set of candidate movies for very large datasets
        if len(self.movies) > 10000:
            # Use a random sample of movies for prediction to save memory
            candidate_movies = list(set(self.movies.sample(min(5000, len(self.movies)))['movie_id'].tolist()) - set(selected_movies))
        else:
            candidate_movies = [movie for movie in self.movies['movie_id'].tolist() 
                               if movie not in selected_movies]
        
        # Get predictions
        predictions = []
        for movie_id in candidate_movies:
            predicted_rating = temp_model.predict(temp_user_id, movie_id).est
            predictions.append((movie_id, predicted_rating))
        
        # Sort and get top recommendations
        top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
        
        # Get movie details for the recommendations
        recommendations = []
        for movie_id, predicted_rating in top_n:
            movie_title = self.movies[self.movies['movie_id'] == movie_id]['title'].iloc[0]
            recommendations.append({
                'movie_id': movie_id,
                'title': movie_title,
                'predicted_rating': predicted_rating
            })
        
        # Force garbage collection
        gc.collect()
        
        return recommendations

# Neural Collaborative Filtering with Keras
class NCFRecommender:
    def __init__(self, n_users, n_movies, embedding_size=50):
        self.n_users = n_users
        self.n_movies = n_movies
        self.embedding_size = min(50, embedding_size)  # Limit embedding size for large datasets
        self.model = None  # Lazy initialization
        self.ratings = None
        self.movies = None
        self.train_data = None
        self.test_data = None
        
    def _build_model(self):
        # Clear any existing Keras session
        tf.keras.backend.clear_session()
        
        # Input layers
        user_input = Input(shape=(1,), name='user_input')
        movie_input = Input(shape=(1,), name='movie_input')
        
        # Embedding layers - reduce dimensions for large datasets
        embedding_size = min(self.embedding_size, 30) if self.n_users > 10000 else self.embedding_size
        
        user_embedding = Embedding(self.n_users + 1, embedding_size, name='user_embedding')(user_input)
        movie_embedding = Embedding(self.n_movies + 1, embedding_size, name='movie_embedding')(movie_input)
        
        # Flatten embeddings
        user_vector = Flatten(name='user_flatten')(user_embedding)
        movie_vector = Flatten(name='movie_flatten')(movie_embedding)
        
        # Matrix factorization path
        mf_concat = Dot(axes=1)([user_vector, movie_vector])
        
        # Neural network path - simplify for large datasets
        concat = Concatenate()([user_vector, movie_vector])
        
        # Adjust network size based on dataset size
        if self.n_users * self.n_movies > 10000000:  # Very large dataset
            dense1 = Dense(64, activation='relu')(concat)
            dense2 = Dense(32, activation='relu')(dense1)
            dense3 = Dense(16, activation='relu')(dense2)
        else:
            dense1 = Dense(128, activation='relu')(concat)
            dense2 = Dense(64, activation='relu')(dense1)
            dense3 = Dense(32, activation='relu')(dense2)
        
        # Combine MF and NN paths
        output = Dense(1)(dense3)
        
        # Create and compile model
        model = Model(inputs=[user_input, movie_input], outputs=output)
        
        # Use Adam optimizer with reduced learning rate for stability
        opt = Adam(learning_rate=0.0005)
        model.compile(optimizer=opt, loss='mse')
        
        return model
    
    def prepare_data(self, ratings, movies):
        self.ratings = ratings
        self.movies = movies
        
        # Convert to numpy arrays
        users = ratings['user_id'].values
        movies_arr = ratings['movie_id'].values
        ratings_arr = ratings['rating'].values
        
        # Generate train/test indices
        indices = np.arange(len(ratings))
        np.random.shuffle(indices)
        
        # Use a smaller test set for very large datasets
        test_ratio = 0.1 if len(ratings) > 100000 else 0.2
        train_indices = indices[:int((1-test_ratio) * len(indices))]
        test_indices = indices[int((1-test_ratio) * len(indices)):]
        
        # Create train and test datasets
        self.train_data = {
            'users': users[train_indices],
            'movies': movies_arr[train_indices],
            'ratings': ratings_arr[train_indices]
        }
        
        self.test_data = {
            'users': users[test_indices],
            'movies': movies_arr[test_indices],
            'ratings': ratings_arr[test_indices]
        }
        
        # Build model after data preparation
        self.model = self._build_model()
        
        # Force garbage collection
        gc.collect()
    
    def train(self, epochs=10, batch_size=64):
        # Train the model
        # Adjust batch size for larger datasets
        if len(self.ratings) > 500000:
            batch_size = 512
        elif len(self.ratings) > 100000:
            batch_size = 256
        elif len(self.ratings) > 50000:
            batch_size = 128
        
        # Limit epochs for larger datasets
        if len(self.ratings) > 100000:
            epochs = min(epochs, 3)
        
        # Create callback to reduce memory usage
        class MemoryCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                gc.collect()
                tf.keras.backend.clear_session()
        
        # Train with memory optimization
        history = self.model.fit(
            [self.train_data['users'], self.train_data['movies']], 
            self.train_data['ratings'],
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1,
            callbacks=[MemoryCallback()]
        )
        
        # Force garbage collection
        gc.collect()
        
        return history
    
    def evaluate(self):
        # Evaluate the model
        loss = self.model.evaluate(
            [self.test_data['users'], self.test_data['movies']], 
            self.test_data['ratings'],
            verbose=0,
            batch_size=256  # Use larger batch for evaluation
        )
        rmse = np.sqrt(loss)
        print(f"NCF RMSE: {rmse}")
        return rmse
    
    def recommend_movies(self, user_id, n=5):
        # Get all movies the user hasn't rated
        rated_movies = set(self.ratings[self.ratings['user_id'] == user_id]['movie_id'].tolist())
        
        # Use a limited set of candidate movies for very large datasets
        if len(self.movies) > 10000:
            # Use a random sample of movies for prediction to save memory
            candidate_movies = list(set(self.movies.sample(min(5000, len(self.movies)))['movie_id'].tolist()) - rated_movies)
        else:
            candidate_movies = [movie for movie in self.movies['movie_id'].tolist() if movie not in rated_movies]
        
        # Predict ratings for candidate movies in batches to reduce memory usage
        predictions = []
        batch_size = 1000
        
        for i in range(0, len(candidate_movies), batch_size):
            batch_movies = candidate_movies[i:i+batch_size]
            users = np.array([user_id] * len(batch_movies))
            movies_arr = np.array(batch_movies)
            batch_predictions = self.model.predict([users, movies_arr], verbose=0, batch_size=256).flatten()
            
            predictions.extend(list(zip(batch_movies, batch_predictions)))
            
            # Force garbage collection after each batch
            gc.collect()
        
        # Sort by predicted rating and get top n
        top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
        
        # Get movie details
        recommendations = []
        for movie_id, predicted_rating in top_n:
            movie_title = self.movies[self.movies['movie_id'] == movie_id]['title'].iloc[0]
            recommendations.append({
                'movie_id': movie_id,
                'title': movie_title,
                'predicted_rating': predicted_rating
            })
        
        return recommendations
    
    def recommend_from_new_ratings(self, selected_movies, n=5):
        """
        Recommend movies based on a set of 5 movies that a new user has rated 5 stars
        
        Parameters:
        selected_movies -- list of movie_ids that the user has selected (assumed rated 5)
        n -- number of recommendations to provide
        
        Returns:
        List of movie recommendations
        """
        # Create a temporary user ID that is higher than any existing user ID
        temp_user_id = self.ratings['user_id'].max() + 1
        
        # Create new ratings dataframe with the 5 movies the user selected (all rated 5)
        new_ratings = []
        timestamp = int(pd.Timestamp.now().timestamp())
        
        for movie_id in selected_movies:
            new_ratings.append({
                'user_id': temp_user_id,
                'movie_id': movie_id,
                'rating': 5.0,  # High rating to indicate preference
                'timestamp': timestamp
            })
        
        new_ratings_df = pd.DataFrame(new_ratings)
        
        # Sample the original ratings to reduce memory usage
        if len(self.ratings) > 50000:
            sample_size = min(50000, len(self.ratings))
            sampled_ratings = self.ratings.sample(sample_size, random_state=42)
            combined_ratings = pd.concat([sampled_ratings, new_ratings_df])
        else:
            combined_ratings = pd.concat([self.ratings, new_ratings_df])
        
        # Create a new model for training with the combined data
        n_users = combined_ratings['user_id'].max() + 1
        n_movies = self.n_movies
        temp_recommender = NCFRecommender(n_users, n_movies)
        
        # Convert to numpy arrays
        users = combined_ratings['user_id'].values
        movies_arr = combined_ratings['movie_id'].values
        ratings_arr = combined_ratings['rating'].values
        
        # Create temporary data structures for the new model
        indices = np.arange(len(combined_ratings))
        np.random.shuffle(indices)
        train_indices = indices
        
        temp_train_data = {
            'users': users[train_indices],
            'movies': movies_arr[train_indices],
            'ratings': ratings_arr[train_indices]
        }
        
        # Build a simpler model for recommendation
        tf.keras.backend.clear_session()
        
        # Input layers
        user_input = Input(shape=(1,), name='user_input')
        movie_input = Input(shape=(1,), name='movie_input')
        
        # Simplified embeddings
        embedding_size = 20  # Smaller embedding for faster training
        user_embedding = Embedding(n_users, embedding_size, name='user_embedding')(user_input)
        movie_embedding = Embedding(n_movies + 1, embedding_size, name='movie_embedding')(movie_input)
        
        # Flatten embeddings
        user_vector = Flatten(name='user_flatten')(user_embedding)
        movie_vector = Flatten(name='movie_flatten')(movie_embedding)
        
        # Simple neural network
        concat = Concatenate()([user_vector, movie_vector])
        dense1 = Dense(32, activation='relu')(concat)
        dense2 = Dense(16, activation='relu')(dense1)
        output = Dense(1)(dense2)
        
        # Build and compile model
        temp_model = Model(inputs=[user_input, movie_input], outputs=output)
        temp_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Adjust batch size based on dataset size
        batch_size = 128
        if len(combined_ratings) > 100000:
            batch_size = 256
            
        # Adjust epochs based on dataset size
        epochs = 2
        if len(combined_ratings) > 50000:
            epochs = 1
            
        # Train the model with the combined data
        temp_model.fit(
            [temp_train_data['users'], temp_train_data['movies']],
            temp_train_data['ratings'],
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        # Use a limited set of candidate movies for very large datasets
        if len(self.movies) > 10000:
            # Use a random sample of movies for prediction to save memory
            candidate_movies = list(set(self.movies.sample(min(5000, len(self.movies)))['movie_id'].tolist()) - set(selected_movies))
        else:
            candidate_movies = [movie for movie in self.movies['movie_id'].tolist() 
                               if movie not in selected_movies]
        
        # Predict ratings in batches to reduce memory usage
        predictions = []
        batch_size = 1000
        
        for i in range(0, len(candidate_movies), batch_size):
            batch_movies = candidate_movies[i:i+batch_size]
            users = np.array([temp_user_id] * len(batch_movies))
            movies_arr = np.array(batch_movies)
            batch_predictions = temp_model.predict([users, movies_arr], verbose=0, batch_size=256).flatten()
            
            predictions.extend(list(zip(batch_movies, batch_predictions)))
            
            # Force garbage collection after each batch
            gc.collect()
        
        # Sort by predicted rating and get top n
        top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
        
        # Get movie details
        recommendations = []
        for movie_id, predicted_rating in top_n:
            movie_title = self.movies[self.movies['movie_id'] == movie_id]['title'].iloc[0]
            recommendations.append({
                'movie_id': movie_id,
                'title': movie_title,
                'predicted_rating': predicted_rating
            })
        
        # Clean up and force garbage collection
        del temp_model, temp_train_data
        tf.keras.backend.clear_session()
        gc.collect()
        
        return recommendations 