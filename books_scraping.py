import requests
import pandas as pd
import time

# Each emotion mapped to 2 Google Books search keywords
emotion_genre_map = {
    "hopeful": ["self-help", "motivational"],
    "depressed": ["mental health", "psychology"],
    "lonely": ["poetry", "memoir"],
    "sad": ["drama", "tragedy"],
    "angry": ["politics", "war"],
    "fear": ["horror", "paranormal"],
    "regret": ["biography", "addiction"],
    "excited": ["adventure", "action"],
    "confused": ["mystery", "philosophy"],
    "grateful": ["spirituality", "religion"],
    "rejected": ["teen fiction", "breakup"],
    "happy": ["romance", "comedy"]
}

def fetch_books_by_genres(genre_list, emotion, total=50):
    seen_titles = set()
    books = []
    per_query = total // len(genre_list)

    for genre in genre_list:
        for start_index in range(0, per_query, 40):
            max_results = min(40, per_query - start_index)
            url = f"https://www.googleapis.com/books/v1/volumes?q={genre}&startIndex={start_index}&maxResults={max_results}"
            try:
                response = requests.get(url)
                data = response.json()
            except Exception as e:
                print(f"❌ Request failed for {genre}: {e}")
                break

            if 'items' not in data:
                print(f"⚠️ No items found for {genre}")
                break

            for item in data['items']:
                info = item.get('volumeInfo', {})
                title = info.get("title", "").strip()
                authors = ", ".join(info.get("authors", [])) if "authors" in info else "Unknown"
                key = (title.lower(), authors.lower())

                # Skip duplicates
                if key in seen_titles:
                    continue
                seen_titles.add(key)

                books.append({
                    "title": title,
                    "authors": authors,
                    "category": ", ".join(info.get("categories", [])) if "categories" in info else "Unknown",
                    "description": info.get("description", "No description"),
                    "link": info.get("infoLink", "No link"),
                    "emotion": emotion
                })

            time.sleep(0.2)  # API friendly

    return books

# Fetching books for each emotion
all_books = []

for emotion, genres in emotion_genre_map.items():
    print(f"🔍 Fetching {emotion} books from genres: {genres}")
    books = fetch_books_by_genres(genres, emotion, total=50)
    all_books.extend(books)

# Create DataFrame and remove global duplicates
df_books = pd.DataFrame(all_books)
df_books.drop_duplicates(subset=["title", "authors"], inplace=True)

# Save to CSV
df_books.to_csv("emotion_labeled_books.csv", index=False)
print(f"\n✅ Finished! Saved {len(df_books)} unique books to 'emotion_labeled_books.csv'")
