import praw
import pandas as pd
import time

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

dataset = []
seen_ids = set()
target_count = 30000

# Listing types to use for variety
listing_types = ['hot', 'new', 'top']

# Scrape data
for emotion, subs in emotion_subreddits.items():
    for sub in subs:
        for listing in listing_types:
            print(f"Scraping {listing} posts from r/{sub} for emotion '{emotion}'...")
            try:
                subreddit = reddit.subreddit(sub)
                if listing == 'hot':
                    posts = subreddit.hot(limit=1000)
                elif listing == 'new':
                    posts = subreddit.new(limit=1000)
                elif listing == 'top':
                    posts = subreddit.top(limit=1000)
                else:
                    continue

                for post in posts:
                    if post.id in seen_ids:
                        continue
                    if post.selftext and len(post.selftext.split()) > 5:
                        dataset.append({
                            "text": post.title + " " + post.selftext,
                            "emotion": emotion,
                            "subreddit": sub
                        })
                        seen_ids.add(post.id)

                    if len(dataset) >= target_count:
                        break

                if len(dataset) >= target_count:
                    break

                time.sleep(0.05)  # gentle rate limiting

            except Exception as e:
                print(f"Error scraping r/{sub}: {e}")
        if len(dataset) >= target_count:
            break
    if len(dataset) >= target_count:
        break

# Save to CSV
df = pd.DataFrame(dataset)
df.to_csv("reddit_emotions.csv", index=False)
print(f"\n✅ Saved {len(df)} records to reddit_emotions.csv")
