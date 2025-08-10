import requests
import pandas as pd
import time

API_KEY = '8031c5c2c418cd91876543938f98687a'
BASE_URL = 'https://api.themoviedb.org/3/discover/movie'

# Genre mapping from TMDb
genre_map = {
    "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35,
    "Crime": 80, "Documentary": 99, "Drama": 18, "Family": 10751,
    "Fantasy": 14, "History": 36, "Horror": 27, "Music": 10402,
    "Mystery": 9648, "Romance": 10749, "Science Fiction": 878,
    "TV Movie": 10770, "Thriller": 53, "War": 10752, "Western": 37
}

# Emotion to genres mapping
emotion_genre_map = {
    "hopeful": ["Drama", "Romance", "Family"],
    "depressed": ["Drama", "History"],
    "lonely": ["Drama", "Romance"],
    "sad": ["Drama", "War", "Romance"],
    "angry": ["Action", "Crime", "Thriller"],
    "fear": ["Horror", "Thriller", "Mystery"],
    "regret": ["Drama", "Romance", "History"],
    "excited": ["Action", "Adventure", "Science Fiction", "Comedy"],
    "confused": ["Mystery", "Science Fiction"],
    "grateful": ["Family", "Animation", "Music"],
    "rejected": ["Romance", "Drama"],
    "happy": ["Comedy", "Animation", "Romance"]
}

def get_movies_by_genres(genres, emotion, max_results=50):
    genre_ids = [str(genre_map[g]) for g in genres if g in genre_map]
    query_string = ",".join(genre_ids)
    
    collected = []
    page = 1

    while len(collected) < max_results and page <= 10:
        params = {
            'api_key': API_KEY,
            'with_genres': query_string,
            'page': page,
            'language': 'en-US',
            'sort_by': 'popularity.desc'
        }
        response = requests.get(BASE_URL, params=params).json()
        movies = response.get('results', [])
        for m in movies:
            movie_entry = {
                'title': m.get('title'),
                'overview': m.get('overview'),
                'genres': genres,
                'emotion': emotion,
                'link': f"https://www.themoviedb.org/movie/{m.get('id')}"
            }
            collected.append(movie_entry)
            if len(collected) >= max_results:
                break
        page += 1
        time.sleep(0.5)  # To avoid hitting rate limits

    return collected

# Get movies for all emotions
all_movies = []
for emotion, genres in emotion_genre_map.items():
    print(f"Fetching movies for emotion: {emotion}")
    movies = get_movies_by_genres(genres, emotion)
    all_movies.extend(movies)

# Convert to DataFrame and save
df_all = pd.DataFrame(all_movies)
df_all.to_csv("emotion_based_movies_with_links.csv", index=False)
print("Saved", len(df_all), "movies with links.")

