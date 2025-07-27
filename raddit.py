import praw #Python Reddit API Wrapper — used to access Reddit's API 
import pandas as pd 
import csv
import time # Time module to handle delays between requests

# Initialize Reddit API
reddit = praw.Reddit(
    client_id="T26tnHJjZTHvutyXP2nneQ",
    client_secret="DcBwJGd4jjKDtaDy9OKR0Mt0me0-lA",
    user_agent="script:EmotionDetection:v1.0 (by u/Emotional-Let-4764)"
)

# Emotion to subreddit mapping
emotion_subreddits = {
    "happy": ["happy", "GetMotivated", "MadeMeSmile", "UpliftingNews"],
    "sad": ["sad", "offmychest", "TrueOffMyChest"],
    "angry": ["angry", "rant", "mildlyinfuriating"],
    "fear": ["anxiety", "HealthAnxiety", "socialanxiety"],
    "depressed": ["depression", "SuicideWatch", "mentalhealth"],
    "excited": ["excited", "UnexpectedlyWholesome", "aww"]
}

dataset = [] # Initialize dataset list
seen_ids = set() # Set to track seen post IDs to avoid duplicates
target_count = 30000 # Target number of records to scrape
listing_types = ['hot', 'new', 'top'] # Types of listings to scrape

# Scraping logic
for emotion, subs in emotion_subreddits.items(): # Iterate over each emotion and its corresponding subreddits
    for sub in subs: # Iterate over each subreddit for the current emotion
        for listing in listing_types: # Iterate over each listing type
            print(f"Scraping {listing} posts from r/{sub} for emotion '{emotion}'...")
            try: # Attempt to scrape posts
                subreddit = reddit.subreddit(sub) #
                if listing == 'hot': # Check listing type and scrape accordingly
                    posts = subreddit.hot(limit=1000)
                elif listing == 'new':
                    posts = subreddit.new(limit=1000)
                elif listing == 'top':
                    posts = subreddit.top(limit=1000)
                else:
                    continue

                for post in posts: # Iterate over each post in the listing
                    if post.id in seen_ids:
                        continue
                    if post.selftext and len(post.selftext.split()) > 5:
                        dataset.append({
                            "text": post.title.strip() + " " + post.selftext.strip(),
                            "emotion": emotion,
                            "subreddit": sub
                        })
                        seen_ids.add(post.id)

                    if len(dataset) >= target_count:
                        break

                if len(dataset) >= target_count:
                    break

                time.sleep(0.05)  # rate limit # Sleep to avoid hitting Reddit's API rate limits

            except Exception as e:
                print(f"Error scraping r/{sub}: {e}")
        if len(dataset) >= target_count:
            break
    if len(dataset) >= target_count:
        break

# Convert to DataFrame
df = pd.DataFrame(dataset)

# Save to CSV with proper quoting
df.to_csv(
    "reddit_emotions.csv",
    index=False,
    quoting=csv.QUOTE_ALL,
    quotechar='"',
    encoding='utf-8'
)

print(f"\n✅ Saved {len(df)} records to reddit_emotions.csv")

#sad
#depressed
#excited
#anger
#regret
#lonely
#fear 
