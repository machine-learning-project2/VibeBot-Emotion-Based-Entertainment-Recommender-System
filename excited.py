import praw
import pandas as pd
import time
import csv

# Initialize Reddit API
reddit = praw.Reddit(
    client_id="T26tnHJjZTHvutyXP2nneQ",
    client_secret="DcBwJGd4jjKDtaDy9OKR0Mt0me0-lA",
    user_agent="script:EmotionDetection:v1.0 (by u/Emotional-Let-4764)"
)

# Emotion and relevant subreddits for "excited"
emotion = "excited"
subreddits = [
    "excited",
    "wholesomememes",
    "wholesome",
    "Happy",
    "wholesomegifs",
    "Happycrowd",
    "Funny",
    "PositiveVibes",
    "Happiness",
    "Smiles"
]

listing_types = ['hot', 'new', 'top']
top_time_filters = ['day', 'week', 'month', 'year', 'all']

dataset = []
seen_ids = set()
target_count = 5000

for sub in subreddits:
    for listing in listing_types:
        if sub == "excited" and listing == "top":
            # 'excited' subreddit top with no time filter (to avoid 404)
            try:
                print(f"Scraping top posts from r/{sub} for emotion '{emotion}' (no time filter)...")
                posts = reddit.subreddit(sub).top(limit=1000)
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
            except Exception as e:
                print(f"Error scraping r/{sub} top: {e}")
            time.sleep(0.1)
            continue

        if listing == "top":
            for tf in top_time_filters:
                try:
                    print(f"Scraping top ({tf}) posts from r/{sub} for emotion '{emotion}'...")
                    posts = reddit.subreddit(sub).top(time_filter=tf, limit=1000)
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
                except Exception as e:
                    print(f"Error scraping r/{sub} top ({tf}): {e}")
                time.sleep(0.1)
            if len(dataset) >= target_count:
                break
        else:
            # For 'hot' or 'new' listings
            try:
                print(f"Scraping {listing} posts from r/{sub} for emotion '{emotion}'...")
                posts = getattr(reddit.subreddit(sub), listing)(limit=1000)
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
            except Exception as e:
                print(f"Error scraping r/{sub} {listing}: {e}")
            time.sleep(0.1)
    if len(dataset) >= target_count:
        break

# Save to CSV with proper quoting
df = pd.DataFrame(dataset)
df.to_csv("reddit_excited_5k.csv", index=False, quoting=csv.QUOTE_ALL)

print(f"\n✅ Saved {len(df)} records to reddit_excited_5k.csv")
