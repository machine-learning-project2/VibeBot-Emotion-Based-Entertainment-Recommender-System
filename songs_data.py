import pandas as pd

df = pd.read_csv("spotify_tracks.csv")
df.columns

df.drop(['track_id' , 'year', 'popularity',
       'artwork_url',  'acousticness',
       'duration_ms',  'key', 'liveness',
       'loudness', 'mode', 'speechiness',  'time_signature',
       'language','instrumentalness','energy','album_name'],axis=1,inplace=True)

df

# Define the mapping function
def map_to_custom_emotion(valence, energy, danceability):
    if valence > 0.75 and energy > 0.75:
        return 'excited'
    elif valence > 0.6 and energy < 0.4:
        return 'grateful'
    elif valence > 0.65 and energy > 0.5:
        return 'happy'
    elif valence > 0.5 and 0.4 < energy < 0.6:
        return 'hopeful'
    elif valence < 0.35 and energy < 0.35:
        return 'depressed'
    elif valence < 0.3 and energy > 0.6:
        return 'angry'
    elif valence < 0.35 and energy < 0.45:
        return 'lonely'
    elif valence < 0.4 and 0.4 <= energy <= 0.6:
        return 'sad'
    elif 0.35 < valence < 0.55 and energy < 0.4:
        return 'regret'
    elif 0.4 <= valence <= 0.6 and 0.4 <= energy <= 0.6:
        return 'confused'
    elif valence < 0.4 and 0.5 <= energy <= 0.7:
        return 'fear'
    elif valence < 0.4 and 0.4 <= energy <= 0.6:
        return 'rejected'
    else:
        return 'neutral'

# Apply the function row-wise
df['emotion'] = df.apply(lambda row: map_to_custom_emotion(row['valence'], row['danceability'], row['danceability']), axis=1)

# Show the updated DataFrame
print(df[['track_name', 'artist_name', 'valence', 'danceability', 'emotion', 'track_url']].head())

df.columns

# Save the DataFrame to a CSV file
df.to_csv("emotion_labeled_tracks.csv", index=False)

print("CSV file saved as emotion_labeled_tracks.csv")
