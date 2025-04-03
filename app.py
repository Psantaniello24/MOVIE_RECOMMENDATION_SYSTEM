import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import gc
import sys
import base64
import warnings

# Suppress numpy warnings about division by zero and invalid values
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in true_divide')

# Try importing TensorFlow and Keras - used for Neural Collaborative Filtering
try:
    import tensorflow as tf
    
    # First, configure TensorFlow for cloud environment
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show errors
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Import specific Keras components
    try:
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
        from tensorflow.keras.optimizers import Adam
    except ImportError:
        # For newer TensorFlow versions
        from keras.models import Model
        from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
        from keras.optimizers import Adam
    
    # Configure TensorFlow to use less memory
    tf.get_logger().setLevel('ERROR')
    
    # Avoid GPU memory issues using compatible API
    try:
        # For TensorFlow 2.x
        tf_config = tf.compat.v1.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=tf_config)
    except:
        # Ignore errors if this doesn't work
        pass
    
    tf_available = True
except ImportError:
    tf_available = False
    st.sidebar.warning("TensorFlow/Keras not available. Neural Collaborative Filtering will not be available.")

# Configure Streamlit page
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Memory optimization settings
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set smaller memory limits for Streamlit Cloud
st.cache_data.clear()
gc.collect()

# Check if psutil is available, but don't require it
try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False

# Function to get memory usage in MB
def get_memory_usage():
    if HAVE_PSUTIL:
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
        return memory_usage
    else:
        return 0  # Return 0 if psutil is not available

# Title and introduction
st.title('Movie Recommendation System')
st.subheader('Collaborative Filtering on MovieLens Dataset')

# Add a sidebar for dataset selection
st.sidebar.header('Dataset Selection')

# Choose dataset size
dataset_size = st.sidebar.selectbox(
    'Select dataset size',
    ('100k', '1m', '25m'),
    index=0,  # Default to 100k
    help='Select the MovieLens dataset size. Note: 25M dataset is very large and may require significant memory and processing time.'
)

# Display memory usage warning for 25M dataset
if dataset_size == '25m':
    st.sidebar.warning('⚠️ The 25M dataset is very large and requires significant memory and processing time. It is recommended to use a small sample size.')

# Sample size selection - make sure this is intuitive for the user
if dataset_size == '100k':
    # 100k has about 100k ratings
    default_sample = 100000
    max_sample = 100000
elif dataset_size == '1m':
    # 1M has about 1M ratings
    default_sample = 100000
    max_sample = 1000000
else:  # 25m
    # Be more conservative with the 25M dataset
    default_sample = 50000
    max_sample = 200000

sample_size = st.sidebar.slider(
    'Sample size (number of ratings)',
    min_value=10000,
    max_value=max_sample,
    value=default_sample,
    step=10000,
    help='Use a smaller sample for faster processing. For the 25M dataset, using more than 100,000 ratings may cause memory issues.'
)

# Check if dataset files exist
dataset_paths = {
    '100k': './ml-100k/u.data',
    '1m': './ml-1m/ratings.dat',
    '25m': './ml-25m/ratings.csv'
}

dataset_exists = os.path.exists(dataset_paths.get(dataset_size, ''))

# Simplified load data function
def load_data(dataset_size='100k', sample_size=None):
    """Load MovieLens dataset"""
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
            
            # For 25M dataset, use sample to reduce memory usage
            if sample_size is not None:
                ratings = pd.read_csv(ratings_path, nrows=sample_size)
            else:
                # If no sample size provided, limit to first 200,000 rows for safety
                sample_size = 200000
                ratings = pd.read_csv(ratings_path, nrows=sample_size)
                
            # Rename columns to match our expected format
            ratings.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'}, inplace=True)
            
            # Load movies
            movies = pd.read_csv(movies_path)
            # Rename columns to match our expected format
            movies.rename(columns={'movieId': 'movie_id'}, inplace=True)
        
        else:
            raise ValueError(f"Invalid dataset_size: {dataset_size}. Choose '100k', '1m', or '25m'")
        
        # Take a sample if specified (except for 25m which is handled above)
        if sample_size and sample_size < len(ratings) and dataset_size != '25m':
            ratings = ratings.sample(sample_size, random_state=42)
        
        # Force garbage collection after loading
        gc.collect()
        
        return ratings, movies
        
    except FileNotFoundError:
        st.error(f"Dataset not found. Please download the MovieLens {dataset_size} dataset and place it in the correct directory.")
        st.info("You can download the dataset from: https://grouplens.org/datasets/movielens/")
        return None, None
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None, None

# Function to load data with caching
@st.cache_data(ttl=3600)
def load_data_cached(dataset_size, sample_size):
    with st.spinner(f'Loading MovieLens {dataset_size} dataset (sample size: {sample_size})...'):
        # Force garbage collection before loading new data
        gc.collect()
        
        # Store starting memory usage
        start_memory = get_memory_usage()
        start_time = time.time()
        
        # For Streamlit Cloud, limit the dataset size
        cloud_mode = os.environ.get('STREAMLIT_SHARING', False)
        if cloud_mode and dataset_size == '25m':
            st.warning("25M dataset may exceed memory limits on Streamlit Cloud. Using 100K instead.")
            dataset_size = '100k'
            sample_size = None
        
        # For 1M dataset in cloud, use smaller sample
        if cloud_mode and dataset_size == '1m' and (sample_size is None or sample_size > 200000):
            st.info("Limiting sample size to 200,000 ratings for Streamlit Cloud.")
            sample_size = 200000
            
        # Load the data
        ratings, movies = load_data(dataset_size, sample_size)
        
        # Calculate memory used and time taken
        memory_used = get_memory_usage()
        time_taken = time.time() - start_time
        
        if ratings is not None:
            # Log metrics
            st.info(f"Dataset loaded: {len(ratings)} ratings, {len(movies)} movies. " +
                   f"Memory used: {memory_used:.1f} MB, Time: {time_taken:.1f} seconds")
            
            # Free memory in cloud environment
            if cloud_mode and memory_used > 500:  # If using more than 500MB
                st.warning("High memory usage detected. Some features may be limited.")
        
        return ratings, movies

# Load data if dataset exists
if dataset_exists or dataset_size in ['100k', '1m', '25m']:
    try:
        ratings, movies = load_data_cached(dataset_size, sample_size)
        
        # Check if data loaded successfully
        if ratings is None or len(ratings) == 0:
            st.error("Failed to load dataset. Please try a smaller sample size.")
            st.stop()
            
        # Force garbage collection after loading data
        gc.collect()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Try refreshing the page or selecting a different dataset/sample size.")
        st.stop()
else:
    st.error(f"Dataset files not found for {dataset_size} dataset.")
    st.info("Please download the dataset from: https://grouplens.org/datasets/movielens/")
    st.stop()

# Create a simple recommendation function (collaborative filtering)
def simple_collaborative_filtering(user_id, ratings_df, n=10):
    """A simple user-based collaborative filtering approach"""
    
    # Get user's ratings
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    
    # Find users with similar ratings (simple approach)
    similar_users = []
    
    # Get all other users
    other_users = ratings_df['user_id'].unique()
    other_users = [u for u in other_users if u != user_id]
    
    # Sample other users for efficiency
    if len(other_users) > 100:
        np.random.seed(42)
        other_users = np.random.choice(other_users, 100, replace=False)
    
    # Find similarity with each user
    for other_user in other_users:
        # Get this user's ratings
        other_ratings = ratings_df[ratings_df['user_id'] == other_user]
        
        # Find movies rated by both users
        common_movies = np.intersect1d(
            user_ratings['movie_id'].values,
            other_ratings['movie_id'].values
        )
        
        # Skip if not enough common movies
        if len(common_movies) < 5:
            continue
        
        # Get ratings for common movies
        user_common = user_ratings[user_ratings['movie_id'].isin(common_movies)]
        other_common = other_ratings[other_ratings['movie_id'].isin(common_movies)]
        
        # Ensure proper ordering
        user_common = user_common.sort_values('movie_id')
        other_common = other_common.sort_values('movie_id')
        
        # Calculate similarity (corrcoef returns a 2x2 matrix, we want [0,1])
        similarity = np.corrcoef(user_common['rating'].values, other_common['rating'].values)[0, 1]
        
        # Only include if similarity is valid (not NaN)
        if not np.isnan(similarity):
            similar_users.append((other_user, similarity))
    
    # Sort by similarity
    similar_users.sort(key=lambda x: x[1], reverse=True)
    
    # Get top similar users
    top_similar = similar_users[:10]
    
    # Find movies that similar users rated highly
    candidate_movies = {}
    
    for sim_user, similarity in top_similar:
        # Get movies rated by this similar user
        sim_user_ratings = ratings_df[ratings_df['user_id'] == sim_user]
        
        # Focus on highly rated movies
        high_ratings = sim_user_ratings[sim_user_ratings['rating'] >= 4]
        
        for _, row in high_ratings.iterrows():
            movie_id = row['movie_id']
            
            # Skip if user already rated this movie
            if movie_id in user_ratings['movie_id'].values:
                continue
            
            # Add to candidates, weighted by similarity
            if movie_id not in candidate_movies:
                candidate_movies[movie_id] = 0
            
            candidate_movies[movie_id] += similarity * (row['rating'] - 2.5)  # Adjust by mean
    
    # Sort candidates by score
    sorted_candidates = sorted(candidate_movies.items(), key=lambda x: x[1], reverse=True)
    
    # Get top recommendations
    top_movie_ids = [movie_id for movie_id, _ in sorted_candidates[:n]]
    
    # Get movie details
    recommendations = []
    for movie_id in top_movie_ids:
        # Find movie in the movies dataframe
        movie_row = movies[movies['movie_id'] == movie_id]
        
        if not movie_row.empty:
            movie_title = movie_row.iloc[0]['title']
            
            # Calculate predicted rating
            pred_rating = 2.5 + candidate_movies[movie_id] / sum([s for _, s in top_similar])
            pred_rating = min(5.0, max(1.0, pred_rating))  # Clamp between 1-5
            
            recommendations.append({
                'movie_id': movie_id,
                'title': movie_title,
                'predicted_rating': pred_rating
            })
    
    return recommendations

# Neural Collaborative Filtering with Keras
class NCFRecommender:
    def __init__(self, n_users, n_movies, embedding_size=50):
        """
        Initialize Neural Collaborative Filtering model
        
        Parameters:
        n_users -- Number of users in the dataset
        n_movies -- Number of movies in the dataset
        embedding_size -- Size of embedding vectors (default 50)
        """
        self.n_users = n_users
        self.n_movies = n_movies
        self.embedding_size = min(50, embedding_size)  # Limit embedding size
        self.model = None
        self.ratings = None
        self.movies = None
        self.user_ratings_dict = None
        
    def _build_model(self):
        """Build the NCF model architecture"""
        # Clear any existing Keras session
        tf.keras.backend.clear_session()
        
        # Input layers
        user_input = Input(shape=(1,), name='user_input')
        movie_input = Input(shape=(1,), name='movie_input')
        
        # Embedding layers
        # Reduce dimensions for large datasets
        embedding_size = min(self.embedding_size, 30) if self.n_users > 10000 else self.embedding_size
        
        user_embedding = Embedding(self.n_users + 1, embedding_size, name='user_embedding')(user_input)
        movie_embedding = Embedding(self.n_movies + 1, embedding_size, name='movie_embedding')(movie_input)
        
        # Flatten embeddings
        user_vector = Flatten(name='user_flatten')(user_embedding)
        movie_vector = Flatten(name='movie_flatten')(movie_embedding)
        
        # Matrix factorization path
        mf_dot_product = Dot(axes=1)([user_vector, movie_vector])
        
        # Neural network path
        concat = Concatenate()([user_vector, movie_vector])
        
        # Adjust network size based on dataset size
        if self.n_users * self.n_movies > 1000000:  # Large dataset
            dense1 = Dense(64, activation='relu')(concat)
            dense2 = Dense(32, activation='relu')(dense1)
            dense3 = Dense(16, activation='relu')(dense2)
        else:
            dense1 = Dense(128, activation='relu')(concat)
            dense2 = Dense(64, activation='relu')(dense1)
            dense3 = Dense(32, activation='relu')(dense2)
        
        # Combine both paths
        combined = Concatenate()([mf_dot_product, dense3])
        
        # Output layer
        output = Dense(1)(combined)
        
        # Create and compile model
        model = Model(inputs=[user_input, movie_input], outputs=output)
        
        # Use Adam optimizer with reduced learning rate for stability
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse')
        
        return model
    
    def prepare_data(self, ratings_df, movies_df, user_ratings_dict=None):
        """
        Prepare data for training the NCF model
        
        Parameters:
        ratings_df -- DataFrame with existing ratings
        movies_df -- DataFrame with movie information
        user_ratings_dict -- Dictionary of user's custom ratings
        """
        self.ratings = ratings_df
        self.movies = movies_df
        self.user_ratings_dict = user_ratings_dict
        
        # Create train data
        train_data = self._create_training_data(user_ratings_dict)
        
        # Build model
        self.model = self._build_model()
        
        return train_data
    
    def _create_training_data(self, user_ratings_dict=None):
        """
        Create training data for the model
        
        Parameters:
        user_ratings_dict -- Dictionary of user's custom ratings
        
        Returns:
        Dictionary with training data
        """
        # Create a temporary user ID for custom ratings
        if user_ratings_dict:
            temp_user_id = self.ratings['user_id'].max() + 1
            
            # Create a list to hold all ratings including custom ones
            all_user_ids = list(self.ratings['user_id'].values)
            all_movie_ids = list(self.ratings['movie_id'].values)
            all_ratings = list(self.ratings['rating'].values)
            
            # Add custom ratings
            for movie_id, rating in user_ratings_dict.items():
                all_user_ids.append(temp_user_id)
                all_movie_ids.append(movie_id)
                all_ratings.append(rating)
            
            # Convert to numpy arrays
            users = np.array(all_user_ids)
            movies = np.array(all_movie_ids)
            ratings_arr = np.array(all_ratings)
        else:
            # Use existing ratings only
            users = self.ratings['user_id'].values
            movies = self.ratings['movie_id'].values
            ratings_arr = self.ratings['rating'].values
            
        # If too many ratings, sample for faster training
        if len(users) > 100000:
            indices = np.random.choice(len(users), 100000, replace=False)
            users = users[indices]
            movies = movies[indices]
            ratings_arr = ratings_arr[indices]
            
        # Create training data
        train_data = {
            'users': users,
            'movies': movies,
            'ratings': ratings_arr
        }
        
        return train_data
    
    def train(self, train_data, epochs=5, batch_size=64, verbose=0):
        """
        Train the NCF model
        
        Parameters:
        train_data -- Dictionary with training data
        epochs -- Number of training epochs
        batch_size -- Batch size for training
        verbose -- Verbosity level (0=silent, 1=progress bar)
        
        Returns:
        Training history
        """
        if self.model is None:
            st.error("Model not initialized. Call prepare_data first.")
            return None
            
        # Adjust batch size for larger datasets
        if len(train_data['users']) > 100000:
            batch_size = 256
        
        # Create callback to reduce memory usage
        class MemoryCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                gc.collect()
                tf.keras.backend.clear_session()
                
        # Create a progress bar for Streamlit if verbose > 0
        if verbose > 0:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            class ProgressCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Training epoch {epoch+1}/{epochs}")
                    
            callbacks = [MemoryCallback(), ProgressCallback()]
        else:
            callbacks = [MemoryCallback()]
            
        # Train the model
        history = self.model.fit(
            [train_data['users'], train_data['movies']], 
            train_data['ratings'],
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0,  # Don't show progress in terminal
            callbacks=callbacks
        )
        
        # Force garbage collection
        gc.collect()
        
        return history
    
    def predict_ratings(self, temp_user_id, candidate_movies):
        """
        Predict ratings for candidate movies
        
        Parameters:
        temp_user_id -- User ID to predict for
        candidate_movies -- List of movie IDs to predict ratings for
        
        Returns:
        List of (movie_id, predicted_rating) tuples
        """
        if self.model is None:
            st.error("Model not trained yet.")
            return []
            
        # Predict in batches to avoid memory issues
        batch_size = 1000
        predictions = []
        
        for i in range(0, len(candidate_movies), batch_size):
            batch_movies = candidate_movies[i:i+batch_size]
            users = np.array([temp_user_id] * len(batch_movies))
            movies_arr = np.array(batch_movies)
            
            # Predict ratings
            batch_preds = self.model.predict([users, movies_arr], verbose=0).flatten()
            
            # Add to predictions
            for j, movie_id in enumerate(batch_movies):
                # Scale predictions to 1-5 range
                pred = batch_preds[j]
                pred = min(5.0, max(1.0, pred))  # Clamp between 1-5
                predictions.append((movie_id, pred))
            
            # Force garbage collection after each batch
            gc.collect()
            
        return predictions
    
    def get_recommendations(self, n=10):
        """
        Get recommendations based on trained model
        
        Parameters:
        n -- Number of recommendations to return
        
        Returns:
        List of recommendation dictionaries
        """
        if self.model is None or self.user_ratings_dict is None:
            st.error("Model not trained with user ratings.")
            return []
            
        # Create a temporary user ID
        temp_user_id = self.ratings['user_id'].max() + 1
        
        # Get all movies except ones the user has already rated
        rated_movies = set(self.user_ratings_dict.keys())
        
        # Use a subset of movies for large catalogs
        if len(self.movies) > 10000:
            # Use most popular movies as candidates
            movie_popularity = self.ratings['movie_id'].value_counts().reset_index()
            movie_popularity.columns = ['movie_id', 'count']
            popular_movies = movie_popularity[~movie_popularity['movie_id'].isin(rated_movies)]
            candidate_movies = popular_movies.head(2000)['movie_id'].values
        else:
            candidate_movies = [m for m in self.movies['movie_id'].unique() if m not in rated_movies]
            
        # Predict ratings for candidate movies
        predictions = self.predict_ratings(temp_user_id, candidate_movies)
        
        # Sort by predicted rating
        sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations
        top_recommendations = sorted_predictions[:n]
        
        # Format recommendations
        recommendations = []
        for movie_id, pred_rating in top_recommendations:
            # Get movie details
            movie_row = self.movies[self.movies['movie_id'] == movie_id]
            if not movie_row.empty:
                movie_title = movie_row.iloc[0]['title']
                recommendations.append({
                    'movie_id': movie_id,
                    'title': movie_title,
                    'predicted_rating': pred_rating
                })
                
        return recommendations

# Function to get recommendations using Neural Collaborative Filtering
def neural_collaborative_filtering(user_ratings_dict, ratings_df, movies_df, n=10):
    """
    Get recommendations using Neural Collaborative Filtering
    
    Parameters:
    user_ratings_dict -- Dictionary mapping movie_id to rating
    ratings_df -- DataFrame with all ratings
    movies_df -- DataFrame with movie information
    n -- Number of recommendations to return
    
    Returns:
    List of recommended movies
    """
    if not tf_available:
        st.error("TensorFlow/Keras not available. Cannot use Neural Collaborative Filtering.")
        return []
        
    if len(user_ratings_dict) < 3:
        st.warning("Please rate at least 3 movies to get personalized recommendations.")
        return []
        
    # Create a progress container
    progress_container = st.empty()
    progress_container.info("Initializing Neural Collaborative Filtering model...")
    
    # Get number of users and movies
    n_users = ratings_df['user_id'].max() + 1
    n_movies = movies_df['movie_id'].max() + 1
    
    # Create model
    recommender = NCFRecommender(n_users, n_movies, embedding_size=30)
    
    # Prepare data
    progress_container.info("Preparing training data...")
    train_data = recommender.prepare_data(ratings_df, movies_df, user_ratings_dict)
    
    # Train model
    progress_container.info("Training model (this may take a minute)...")
    
    # Limit epochs based on dataset size
    if len(ratings_df) > 100000:
        epochs = 3
    else:
        epochs = 5
        
    history = recommender.train(train_data, epochs=epochs, batch_size=64, verbose=1)
    
    # Get recommendations
    progress_container.info("Generating recommendations...")
    recommendations = recommender.get_recommendations(n=n)
    
    # Clear progress container
    progress_container.empty()
    
    return recommendations

# Function to get recommendations based on custom user ratings
def custom_recommendations(user_ratings_dict, ratings_df, movies_df, n=10):
    """
    Get recommendations based on a set of user-provided ratings
    
    Parameters:
    user_ratings_dict -- Dictionary mapping movie_id to rating
    ratings_df -- DataFrame with all ratings
    movies_df -- DataFrame with movie information
    n -- Number of recommendations to return
    
    Returns:
    List of recommended movies
    """
    if len(user_ratings_dict) < 3:
        st.warning("Please rate at least 3 movies to get personalized recommendations.")
        return []
    
    # Create a temporary user ID
    temp_user_id = ratings_df['user_id'].max() + 1
    
    # Print the number of user ratings for debugging
    st.info(f"You've rated {len(user_ratings_dict)} movies. Finding recommendations based on your preferences...")
    
    # Check if rated movies exist in the dataset
    user_movie_ids = list(user_ratings_dict.keys())
    existing_movie_ids = movies_df['movie_id'].unique()
    valid_movie_ids = [mid for mid in user_movie_ids if mid in existing_movie_ids]
    
    if len(valid_movie_ids) < len(user_movie_ids):
        st.warning(f"{len(user_movie_ids) - len(valid_movie_ids)} of your rated movies are not in the dataset and will be ignored.")
        # Update the ratings dict to only include valid movies
        user_ratings_dict = {mid: user_ratings_dict[mid] for mid in valid_movie_ids}
    
    # If no valid movies remain, return empty
    if not valid_movie_ids:
        st.error("None of your rated movies are in the current dataset.")
        return []
    
    # Create a dataframe with the user's ratings
    user_ratings_list = []
    timestamp = int(time.time())
    
    for movie_id, rating in user_ratings_dict.items():
        user_ratings_list.append({
            'user_id': temp_user_id,
            'movie_id': movie_id,
            'rating': rating,
            'timestamp': timestamp
        })
    
    # Create a dataframe from the list
    user_ratings_df = pd.DataFrame(user_ratings_list)
    
    # Get the popularity of rated movies to help in recommendation
    movie_counts = ratings_df['movie_id'].value_counts()
    rated_movie_counts = {mid: movie_counts.get(mid, 0) for mid in user_ratings_dict.keys()}
    
    # Check if any rated movies are popular enough
    popular_threshold = 10  # Minimum number of ratings to consider a movie "popular"
    popular_rated_movies = [mid for mid, count in rated_movie_counts.items() if count >= popular_threshold]
    
    if not popular_rated_movies:
        # Fall back to item-based recommendations instead of user-based
        st.info("Using item-based recommendations since your rated movies are uncommon.")
        return item_based_recommendations(user_ratings_dict, ratings_df, movies_df, n)
    
    # Get all users except the temporary user
    other_users = ratings_df['user_id'].unique()
    
    # Sample users for efficiency
    if len(other_users) > 100:
        np.random.seed(42)
        other_users = np.random.choice(other_users, 100, replace=False)
    
    # Find similarity with each user
    similar_users = []
    
    for other_user in other_users:
        # Get the user's ratings
        other_ratings = ratings_df[ratings_df['user_id'] == other_user]
        
        # Find movies rated by both users
        common_movies = np.intersect1d(list(user_ratings_dict.keys()), other_ratings['movie_id'].values)
        
        # Skip if not enough common movies
        if len(common_movies) < 2:
            continue
        
        # Get ratings for common movies
        user_common_ratings = [user_ratings_dict[movie_id] for movie_id in common_movies]
        other_common = other_ratings[other_ratings['movie_id'].isin(common_movies)]
        other_common_ratings = []
        
        # Ensure proper ordering
        for movie_id in common_movies:
            rating = other_common[other_common['movie_id'] == movie_id]['rating'].iloc[0]
            other_common_ratings.append(rating)
        
        # Calculate similarity (use a try-except to handle correlation issues)
        try:
            # Check if either array has zero variance (all ratings the same)
            user_std = np.std(user_common_ratings)
            other_std = np.std(other_common_ratings)
            
            if user_std == 0 or other_std == 0:
                # Can't compute correlation if no variance, use alternative
                # Use exact match ratio as a substitute measure
                matches = sum(1 for a, b in zip(user_common_ratings, other_common_ratings) if a == b)
                similarity = matches / len(common_movies) * 0.5  # Scale down exact matches
            else:
                # Safe to compute correlation
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    similarity = np.corrcoef(user_common_ratings, other_common_ratings)[0, 1]
            
            # Only include if similarity is valid
            if similarity is not None and not np.isnan(similarity):
                similar_users.append((other_user, similarity))
        except Exception as e:
            # Skip if correlation fails
            continue
    
    # If we couldn't find similar users, try item-based approach
    if not similar_users:
        st.info("Could not find similar users. Using item-based recommendations instead.")
        return item_based_recommendations(user_ratings_dict, ratings_df, movies_df, n)
    
    # Sort by similarity
    similar_users.sort(key=lambda x: x[1], reverse=True)
    
    # Get top similar users
    top_similar = similar_users[:10]
    
    # Find movies that similar users rated highly
    candidate_movies = {}
    
    for sim_user, similarity in top_similar:
        # Get movies rated by this similar user
        sim_user_ratings = ratings_df[ratings_df['user_id'] == sim_user]
        
        # Focus on highly rated movies
        high_ratings = sim_user_ratings[sim_user_ratings['rating'] >= 4]
        
        for _, row in high_ratings.iterrows():
            movie_id = row['movie_id']
            
            # Skip if user already rated this movie
            if movie_id in user_ratings_dict:
                continue
            
            # Add to candidates, weighted by similarity
            if movie_id not in candidate_movies:
                candidate_movies[movie_id] = 0
            
            candidate_movies[movie_id] += similarity * (row['rating'] - 2.5)  # Adjust by mean
    
    # If we couldn't find candidate movies from similar users, fall back to item-based
    if not candidate_movies:
        st.info("Similar users haven't rated enough movies. Using item-based recommendations instead.")
        return item_based_recommendations(user_ratings_dict, ratings_df, movies_df, n)
    
    # Sort candidates by score
    sorted_candidates = sorted(candidate_movies.items(), key=lambda x: x[1], reverse=True)
    
    # Get top recommendations
    top_movie_ids = [movie_id for movie_id, _ in sorted_candidates[:n]]
    
    # Get movie details
    recommendations = []
    for movie_id in top_movie_ids:
        # Find movie in the movies dataframe
        movie_row = movies_df[movies_df['movie_id'] == movie_id]
        
        if not movie_row.empty:
            movie_title = movie_row.iloc[0]['title']
            
            # Calculate predicted rating
            if top_similar:
                pred_rating = 2.5 + candidate_movies[movie_id] / sum([s for _, s in top_similar])
                pred_rating = min(5.0, max(1.0, pred_rating))  # Clamp between 1-5
            else:
                pred_rating = 3.5  # Default if no similar users
            
            recommendations.append({
                'movie_id': movie_id,
                'title': movie_title,
                'predicted_rating': pred_rating
            })
    
    return recommendations

# Add a new item-based recommendation function as fallback
def item_based_recommendations(user_ratings_dict, ratings_df, movies_df, n=10):
    """
    Item-based collaborative filtering for when user-based approach fails
    
    Parameters:
    user_ratings_dict -- Dictionary mapping movie_id to rating
    ratings_df -- DataFrame with all ratings
    movies_df -- DataFrame with movie information
    n -- Number of recommendations to return
    
    Returns:
    List of recommended movies
    """
    # Create a dict to store candidate movies and their scores
    candidate_movies = {}
    
    # For each rated movie, find similar movies
    for rated_movie_id, user_rating in user_ratings_dict.items():
        # Get ratings for this movie
        movie_ratings = ratings_df[ratings_df['movie_id'] == rated_movie_id]
        
        # Get users who rated this movie
        users_who_rated = movie_ratings['user_id'].unique()
        
        # Find other movies rated by these users
        for user_id in users_who_rated:
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            
            # Get only high-rated movies from this user
            high_ratings = user_ratings[user_ratings['rating'] >= user_rating]
            
            for _, row in high_ratings.iterrows():
                other_movie_id = row['movie_id']
                
                # Skip if it's one of the movies the user already rated
                if other_movie_id in user_ratings_dict:
                    continue
                
                # Add to candidates with a score based on rating similarity
                if other_movie_id not in candidate_movies:
                    candidate_movies[other_movie_id] = 0
                
                # Score is based on how similar the ratings are
                rating_similarity = 1 - abs(row['rating'] - user_rating) / 4
                candidate_movies[other_movie_id] += rating_similarity * (user_rating / 5.0)
    
    # If still no candidates, use popular movies
    if not candidate_movies:
        st.warning("Could not find recommendations based on your ratings. Showing popular movies instead.")
        
        # Get most popular movies (ones with most ratings)
        movie_popularity = ratings_df['movie_id'].value_counts().reset_index()
        movie_popularity.columns = ['movie_id', 'count']
        
        # Remove movies user has already rated
        popular_movies = movie_popularity[~movie_popularity['movie_id'].isin(user_ratings_dict.keys())]
        
        # Take top n
        top_movie_ids = popular_movies.head(n)['movie_id'].values
        
        # Create recommendations
        recommendations = []
        for movie_id in top_movie_ids:
            # Find movie in the movies dataframe
            movie_row = movies_df[movies_df['movie_id'] == movie_id]
            
            if not movie_row.empty:
                movie_title = movie_row.iloc[0]['title']
                
                recommendations.append({
                    'movie_id': movie_id,
                    'title': movie_title,
                    'predicted_rating': 3.5  # Default predicted rating for popular movies
                })
        
        return recommendations
    
    # Sort candidates by score
    sorted_candidates = sorted(candidate_movies.items(), key=lambda x: x[1], reverse=True)
    
    # Get top recommendations
    top_movie_ids = [movie_id for movie_id, _ in sorted_candidates[:n]]
    
    # Get movie details
    recommendations = []
    for movie_id in top_movie_ids:
        # Find movie in the movies dataframe
        movie_row = movies_df[movies_df['movie_id'] == movie_id]
        
        if not movie_row.empty:
            movie_title = movie_row.iloc[0]['title']
            
            # Calculate predicted rating (scale the score to a 1-5 range)
            score = candidate_movies[movie_id]
            pred_rating = 2.5 + score
            pred_rating = min(5.0, max(1.0, pred_rating))  # Clamp between 1-5
            
            recommendations.append({
                'movie_id': movie_id,
                'title': movie_title,
                'predicted_rating': pred_rating
            })
    
    return recommendations

# Create tabs for different recommendation approaches
tab1, tab2, tab3 = st.tabs(["User-based Recommendations", "Custom Recommendations", "Dataset Information"])

# Tab 1: User-based Recommendations
with tab1:
    st.header("Get Recommendations for a User")
    
    # Select a user
    user_ids = sorted(ratings['user_id'].unique().tolist())
    selected_user = st.selectbox("Select a user ID", user_ids, index=0)
    
    # Number of recommendations
    num_recommendations = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10,
        step=5
    )
    
    # Get recommendations button
    if st.button("Get Recommendations"):
        try:
            with st.spinner("Calculating recommendations..."):
                start_time = time.time()
                
                # Get recommendations using simple collaborative filtering
                recommendations = simple_collaborative_filtering(
                    selected_user, 
                    ratings, 
                    n=num_recommendations
                )
                
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                
                # Show the time taken
                st.success(f"Recommendations generated in {elapsed_time:.2f} seconds")
                
                # Force garbage collection
                gc.collect()
                
                # Display recommendations in a nice format
                st.subheader(f"Top {num_recommendations} Recommendations for User {selected_user}")
                
                if recommendations:
                    # Create a dataframe for nice display
                    rec_df = pd.DataFrame(recommendations)
                    rec_df.index = np.arange(1, len(rec_df) + 1)  # 1-based indexing for display
                    rec_df['Predicted Rating'] = rec_df['predicted_rating'].round(2)
                    
                    # Display as table
                    st.table(rec_df[['title', 'Predicted Rating']])
                else:
                    st.warning("Couldn't find enough similar users to make recommendations. Try a different user.")
                
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
            st.info("Try a different user or a smaller sample size.")

# Tab 2: Custom Recommendations
with tab2:
    st.header("Get Personalized Recommendations")
    st.write("Rate some movies and get personalized recommendations based on your preferences.")
    
    # Create a session state for storing user ratings
    if 'user_ratings' not in st.session_state:
        st.session_state.user_ratings = {}
    
    # Movie search
    search_term = st.text_input("Search for movies to rate (enter part of the title):")
    
    if search_term:
        # Search for movies
        filtered_movies = movies[movies['title'].str.contains(search_term, case=False)]
        
        if len(filtered_movies) == 0:
            st.info(f"No movies found with '{search_term}' in the title. Try a different search term.")
        else:
            st.write(f"Found {len(filtered_movies)} movies matching '{search_term}'")
            
            # Display movies in a scrollable area with rating options
            with st.container():
                # Limit to top 10 matches for usability
                for _, movie in filtered_movies.head(10).iterrows():
                    movie_id = movie['movie_id']
                    movie_title = movie['title']
                    
                    # Create a unique key for each rating slider
                    slider_key = f"rating_{movie_id}"
                    
                    # Check if this movie is already rated
                    current_rating = st.session_state.user_ratings.get(movie_id, 0)
                    
                    # Create a row with movie title and rating slider
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(movie_title)
                    
                    with col2:
                        # Slider for rating
                        rating = st.select_slider(
                            "Rating", 
                            options=[0, 1, 2, 3, 4, 5],
                            value=current_rating,
                            key=slider_key,
                            label_visibility="collapsed"
                        )
                        
                        # Update session state when rating changes
                        if rating > 0:  # Only store positive ratings
                            st.session_state.user_ratings[movie_id] = rating
                        elif movie_id in st.session_state.user_ratings and rating == 0:
                            # Remove rating if set to 0
                            del st.session_state.user_ratings[movie_id]
                    
                    with col3:
                        # Add a remove button for each rated movie
                        if movie_id in st.session_state.user_ratings:
                            if st.button("Remove", key=f"remove_{movie_id}"):
                                del st.session_state.user_ratings[movie_id]
                                st.rerun()
    
    # Display current ratings
    if st.session_state.user_ratings:
        st.subheader("Your Movie Ratings")
        
        # Create a dataframe from the ratings
        rated_movies = []
        for movie_id, rating in st.session_state.user_ratings.items():
            movie_title = movies[movies['movie_id'] == movie_id]['title'].iloc[0]
            rated_movies.append({
                'Movie ID': movie_id,
                'Title': movie_title,
                'Your Rating': rating
            })
        
        rated_df = pd.DataFrame(rated_movies)
        rated_df = rated_df.sort_values('Your Rating', ascending=False)
        
        # Display the ratings
        st.table(rated_df[['Title', 'Your Rating']])
        
        # Button to clear all ratings
        if st.button("Clear All Ratings"):
            st.session_state.user_ratings = {}
            st.rerun()
        
        # Select recommendation algorithm
        algorithm = st.radio(
            "Select recommendation algorithm",
            ["Collaborative Filtering", "Neural Collaborative Filtering"],
            index=0,
            help="Neural Collaborative Filtering uses deep learning for better recommendations but takes longer to train"
        )
        
        # Number of recommendations
        num_custom_recs = st.slider(
            "Number of recommendations",
            min_value=5,
            max_value=20,
            value=10,
            step=5,
            key="custom_recs_slider"
        )
        
        # Get recommendations button
        if st.button("Get Personalized Recommendations"):
            try:
                with st.spinner("Finding movies based on your ratings..."):
                    start_time = time.time()
                    
                    # Get recommendations based on selected algorithm
                    if algorithm == "Collaborative Filtering":
                        recommendations = custom_recommendations(
                            st.session_state.user_ratings,
                            ratings,
                            movies,
                            n=num_custom_recs
                        )
                    else:  # Neural Collaborative Filtering
                        if not tf_available:
                            st.error("TensorFlow is not available. Using standard collaborative filtering instead.")
                            recommendations = custom_recommendations(
                                st.session_state.user_ratings,
                                ratings,
                                movies,
                                n=num_custom_recs
                            )
                        else:
                            recommendations = neural_collaborative_filtering(
                                st.session_state.user_ratings,
                                ratings,
                                movies,
                                n=num_custom_recs
                            )
                    
                    # Calculate elapsed time
                    elapsed_time = time.time() - start_time
                    
                    if recommendations:
                        # Show success message
                        st.success(f"Recommendations generated in {elapsed_time:.2f} seconds using {algorithm}")
                        
                        # Force garbage collection
                        gc.collect()
                        
                        # Display recommendations
                        st.subheader(f"Top {len(recommendations)} Personalized Recommendations")
                        
                        # Create a dataframe for display
                        rec_df = pd.DataFrame(recommendations)
                        rec_df.index = np.arange(1, len(rec_df) + 1)  # 1-based indexing
                        rec_df['Predicted Rating'] = rec_df['predicted_rating'].round(2)
                        
                        # Display as table
                        st.table(rec_df[['title', 'Predicted Rating']])
                    else:
                        st.warning("Couldn't generate recommendations. Try rating more movies or with different ratings.")
                    
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
                st.info("Try rating more movies or with different ratings.")
    else:
        st.info("Search for movies above and rate them to get personalized recommendations.")

# Tab 3: Dataset Information
with tab3:
    st.header("Dataset Information")
    
    # Dataset statistics
    st.subheader("Dataset Statistics")
    stats_cols = st.columns(2)
    
    with stats_cols[0]:
        st.metric("Number of Ratings", len(ratings))
        st.metric("Number of Users", ratings['user_id'].nunique())
    
    with stats_cols[1]:
        st.metric("Number of Movies", len(movies))
        st.metric("Rating Range", f"{ratings['rating'].min()} - {ratings['rating'].max()}")
    
    # Rating distribution
    st.subheader("Rating Distribution")
    rating_counts = ratings['rating'].value_counts().sort_index()
    
    # Create a dataframe for the rating distribution
    rating_df = pd.DataFrame({
        'Rating': rating_counts.index,
        'Count': rating_counts.values
    })
    
    # Display as a table for better performance
    st.table(rating_df)
    
    # Display memory usage if psutil is available
    current_memory = get_memory_usage()
    if current_memory > 0:
        st.info(f"Current memory usage: {current_memory:.1f} MB")
    
    # Add a button to force garbage collection
    if st.button("Optimize Memory Usage"):
        before_mem = get_memory_usage()
        gc.collect()
        after_mem = get_memory_usage()
        st.success(f"Memory optimization complete. Released {before_mem - after_mem:.1f} MB")


# Add button to download the current dataset
if st.sidebar.button("Download Current Dataset Info"):
    # Create CSV of current dataset
    if 'ratings' in locals() and 'movies' in locals():
        csv = ratings.head(1000).to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="ratings_sample.csv">Download Sample (1000 rows) CSV File</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)

# Try-except block to catch uncaught exceptions and prevent auto refresh
try:
    # The main app content is above
    pass
except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")
    st.info("The application encountered an error. Please try refreshing the page or reducing the sample size.")
    # Log the full error for debugging
    import traceback
    st.text(traceback.format_exc()) 