import praw    #PRAW (Python Reddit API Wrapper) library ,Used to access Reddit API.
import pandas as pd
import csv
import time  # Used to pause the loop slightly to avoid hitting rate limits.

# Initialize Reddit API
reddit = praw.Reddit(
    client_id="T26tnHJjZTHvutyXP2nneQ",
    client_secret="DcBwJGd4jjKDtaDy9OKR0Mt0me0-lA",
    user_agent="script:EmotionDetection:v1.0 (by u/Emotional-Let-4764)"
)

# Unique emotion and subreddit mapping
#Maps emotions to relevant subreddits.
unique_emotion_subreddits = {
    "regret": ["confession", "regret", "morality"],
    "hopeful": ["decidingtobebetter", "stopdrinking", "leaves"],
    "rejected": ["JustUnsubbed", "raisedbynarcissists", "exnocontact"],
    "burned_out": ["teachers", "nursing", "cscareerquestions"],
    "embarrassed": ["tifu"],
    "lost": ["nihilism", "existentialcrisis"],
    "peaceful": ["Meditation", "ZenHabits", "slowliving"],
    "curious": ["AskScience", "NoStupidQuestions", "Glitch_in_the_Matrix"]
}

dataset = []  #stores the collected post data.
seen_ids = set()  #ensures no duplicate Reddit posts are added.
target_count = 10000 #stops scraping once 10,000 posts are collected.
listing_types = ['hot', 'new', 'top']  #determines the type of posts to fetch (hot, new, top).

# Scraping loop
for emotion, subs in unique_emotion_subreddits.items(): #Iterates over each emotion and its subreddits.
    for sub in subs:
        for listing in listing_types:
            print(f"Scraping {listing} posts from r/{sub} for emotion '{emotion}'...")
            try:
                subreddit = reddit.subreddit(sub)
                posts = getattr(subreddit, listing)(limit=1000)  #This dynamically gets subreddit.hot(), subreddit.new(), or subreddit.top().

#Skips duplicates.
# Filters out very short posts.
# Combines the post title + body (selftext) as the text.
                for post in posts:
                    if post.id in seen_ids:
                        continue
                    if post.selftext and len(post.selftext.split()) > 5:
                        dataset.append({
                            "text": f"{post.title.strip()} {post.selftext.strip()}",
                            "emotion": emotion,
                            "subreddit": sub
                        })
                        seen_ids.add(post.id)
 
                    if len(dataset) >= target_count: #If total posts collected ≥ target_count, the loop exits early.
                        break

                if len(dataset) >= target_count:
                    break
                time.sleep(0.1)  # safer rate limit ,adds a short delay between subreddit fetches to avoid hitting Reddit’s rate limit.



            except Exception as e:
                print(f"❌ Error scraping r/{sub}: {e}")
        if len(dataset) >= target_count:
            break
    if len(dataset) >= target_count:
        break

# Save to CSV with proper quoting
df = pd.DataFrame(dataset)
df.to_csv(
    "reddit_emotions_10k_unique.csv",
    index=False,
    quoting=csv.QUOTE_ALL,
    quotechar='"',
    encoding='utf-8'
)

print(f"\n✅ Saved {len(df)} unique emotion records to reddit_emotions_10k_unique.csv")
