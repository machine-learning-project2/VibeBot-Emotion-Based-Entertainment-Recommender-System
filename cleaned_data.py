import re  #Regular expressions: Used for pattern matching (e.g., removing URLs, punctuation, hashtags).
import emoji  #Used to detect and remove emojis from text.
import nltk   #The Natural Language Toolkit for tokenization, lemmatization, and stopwords.
import pandas as pd
import csv    ##Handles CSV file operations, especially with quoting.
from nltk.corpus import stopwords  #Imports the list of stopwords (e.g., "and", "the", "is")
from nltk.stem import WordNetLemmatizer  #Imports the WordNet-based lemmatizer for reducing words to their base form (e.g., "running" → "run").
from textblob import TextBlob   #Used here for optional spelling correction.


# Download NLTK data
nltk.download('punkt')     #Sentence and word tokenizer.
nltk.download('wordnet')   #Lexical database used for lemmatization.
nltk.download('stopwords') #Common words (like "the", "is", "in") which are often removed.

lemmatizer = WordNetLemmatizer()  #A lemmatizer (to reduce words to their base form, e.g., "running" → "run").
stop_words = set(stopwords.words('english')) #A set of stopwords to filter out common words.

# Contraction map
contractions_dict = {
    "can't": "cannot", "won't": "will not", "i'm": "i am", "it's": "it is",
    "don't": "do not", "didn't": "did not", "you're": "you are", "i've": "i have",
    "they're": "they are", "that's": "that is", "isn't": "is not", "aren't": "are not",
    "wasn't": "was not", "weren't": "were not", "hasn't": "has not", "haven't": "have not",
    "couldn't": "could not", "shouldn't": "should not", "wouldn't": "would not",
    "there's": "there is", "what's": "what is", "who's": "who is", "let's": "let us",
    "he's": "he is", "she's": "she is", "we're": "we are"
}
contractions_pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in contractions_dict) + r')\b')
#A dictionary to expand contractions (e.g., "can't" → "cannot").
#Compiles a regex pattern to identify them.

def expand_contractions(text):
    return contractions_pattern.sub(lambda x: contractions_dict[x.group()], text)
    #Function to replace contractions in a given text.

def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')
    #Removes emojis using the emoji library.

def clean_text_emotion(text, correct_spelling=False, remove_stopwords=False,
                       min_word_count=3, min_char_length=10):
    if not isinstance(text, str):
        return ""

    text = text.lower()  #Convert to lowercase
    text = expand_contractions(text)  #Expand contractions
    text = remove_emojis(text)    #Remove emojis
    text = re.sub(r"http\S+|www\S+", "", text)  #Remove URLs digits
    text = re.sub(r"@\w+|#\w+", "", text)    #mentions (@user), hashtags
    text = re.sub(r"[^\w\s!?]", "", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)        #remove digits
    text = re.sub(r"\s+", " ", text).strip()

    if correct_spelling:
        text = str(TextBlob(text).correct())
        #Optional spelling correction using TextBlob

    words = nltk.word_tokenize(text)  #Tokenize words
    clean_words = []

    #Remove stopwords (if enabled)
    for word in words:
        if remove_stopwords and word in stop_words:
            continue
        lemma = lemmatizer.lemmatize(word)  #Lemmatize words
        clean_words.append(lemma)

    cleaned_text = " ".join(clean_words)  #Joins words back into a sentence

    if len(clean_words) < min_word_count or len(cleaned_text) < min_char_length:
        return ""  #Returns an empty string if the cleaned text is too short

    return cleaned_text

# Load CSV safely
df = pd.read_csv("merged_data.csv", quoting=csv.QUOTE_ALL, encoding='utf-8', low_memory=False)

# Clean text
df["cleaned_text"] = df["text"].apply(lambda x: clean_text_emotion(x, correct_spelling=False, remove_stopwords=True))

# Drop empty rows
df = df[df["cleaned_text"].str.strip() != ""] #Removes rows where cleaned text is empty

df.drop(['subreddit','text'],inplace=True,axis=1)
# Save cleaned data
df.to_csv("cleaned_data.csv", index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
print("Final cleaned file saved as 'cleaned_data.csv'")

df.head(15)
df.columns
df['emotion'].unique()
df['emotion'].value_counts()