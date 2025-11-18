import re, string, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import emoji

# from textblob import TextBlob
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab") 
nltk.download("wordnet")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

negations = {
    # Core negations
    "not", "no", "nor", "never", "without",

    # Negative pronouns / determiners
    "nobody", "nothing", "none", "nowhere", "neither",
    "nowise", "nought", "nay",

    # Negative adverbs / minimizers
    "hardly", "scarcely", "barely", "seldom", "rarely",

    # Contractions
    "isn't", "aren't", "wasn't", "weren't",
    "don't", "doesn't", "didn't",
    "can't", "couldn't",
    "won't", "wouldn't",
    "shan't", "shouldn't",
    "haven't", "hasn't", "hadn't",
    "ain't"
}

stop_words = stop_words.difference(negations)

punc = {"!", "#", "$", "%", "&", "\'", "\"", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<", "=", ">", "?", "@",
        "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~"}


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()

    # text = re.sub(r"<.*?>", " ", text)

    text = re.sub(r"http\S+|www\S+", "URL", text)

    text = re.sub(r"@\w+", "USER", text)

    text = re.sub(r"#(\w+)", r"\1", text)

    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    text = re.sub(f"[{re.sub(r"\'", "", re.escape(string.punctuation))}0-9]", " ", text)

    text = re.sub(r"\s+", " ", text)

    tokens = nltk.word_tokenize(text)

    tokens = [word for word in tokens if word not in stop_words]

    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


def normalize_contractions(text: str) -> str:
    # remove "'s" (possessives or contractions like "he's")
    text = re.sub(r"'s\b", "", text)

    # expand "n't" to " not"
    text = re.sub(r"n't\b", "not", text)

    return text


# Dictionary for common emoticons
emoticon_dict = {
    ":)": "smile",
    ":-)": "smile",
    ":D": "laugh",
    ":-D": "laugh",
    ":(": "sad",
    ":-(": "sad",
    ":'(": "cry",
    ":-|": "neutral",
    ";)": "wink",
    ";-)": "wink",
    ":P": "playful",
    ":-P": "playful"
}

# Function to replace emojis with words
def replace_emojis(text):
    # Use emoji library to demojize (😊 -> :smiling_face_with_smiling_eyes:)
    text = emoji.demojize(text, language='en')
    # Replace underscores with spaces and remove colons
    text = re.sub(r":", "", text)
    text = text.replace("_", " ")
    return text

# Function to replace emoticons with words
def replace_emoticons(text):
    for emoticon, meaning in emoticon_dict.items():
        text = text.replace(emoticon, meaning)
    return text

# Combined function
def clean_text_with_emojis(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = replace_emoticons(text)
    text = replace_emojis(text)
    return text




