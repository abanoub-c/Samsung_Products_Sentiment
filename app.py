import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import uuid
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from preprocess import preprocess_text  , normalize_contractions , clean_text_with_emojis # your custom preprocessing function
from pymongo import MongoClient
import datetime


tfidf_model = joblib.load("models/tfidf_vectorizer.pkl")
encode_labels = {"0" : "Negative"  , "1" :"Neutral" , "2" :"Positive" }






st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

# -------------------------
# 1. Database connection (MySQL local)
# -------------------------
# st.sidebar.title("Configuration")
#
# server = r'localhost\Ms'
# database = 'samsung_reviews_db'
# username = 'Abanoub01'
# password = 'strong01'

# connection_string = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
# engine = create_engine(connection_string)


# with engine.connect() as conn:
#     print("✅ Connection successful!")


# Replace with your actual connection string
client = MongoClient("mongodb+srv://abanoub01coder_db_user0:9A4qflxHOwIgWeXS@cluster0.vs5oky6.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

# Choose your database and collection
db = client["streamlit_db"]
collection = db["reviews"]

# -------------------------
# 2. Model selection
# -------------------------
# st.sidebar.markdown("---")
st.sidebar.header("Select Model")

model_choices = [
    "Logistic Regression (ML)",
    "Random Forest (ML)",
    "XGBoost (ML)",
    "Transformer (BERT-based)"
]
model_option = st.sidebar.selectbox("Choose Model", model_choices)

# -------------------------
# 3. Load model automatically
# -------------------------
@st.cache_resource
def load_ml_model(path):
    return joblib.load(path)

@st.cache_resource
def load_transformer_model(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    return tokenizer, model

def predict_transformer(texts, tokenizer, model):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=1).numpy()
    return preds, probs.numpy()

# -------------------------
# 4. Load chosen model
# -------------------------
if model_option == "Logistic Regression (ML)":
    model = load_ml_model("models/logistic_model.pkl")
    model_type = "ml"
elif model_option == "Random Forest (ML)":
    model = load_ml_model("models/random_forest.pkl")
    model_type = "ml"
elif model_option == "XGBoost (ML)":
    model = load_ml_model("models/xgboost.pkl")
    model_type = "ml"
else:
    tokenizer, model = load_transformer_model("Abanoub012/sentiment_analysis")
    model_type = "transformer"

# -------------------------
# 5. Predict function (with preprocessing)
# -------------------------
def predict_texts(texts):
    # Step 1: preprocessing
    t0 = [preprocess_text(t) for t in texts]
    t1 = [normalize_contractions(t) for t in t0]
    cleaned_texts = [clean_text_with_emojis(t) for t in t1]

    tfidf_input = tfidf_model.transform(cleaned_texts)

    # Step 2: model prediction
    if model_type == "ml":
        preds = model.predict(tfidf_input)
        probs = model.predict_proba(tfidf_input)[:, 1] if hasattr(model, "predict_proba") else None
    else:
        preds, probs = predict_transformer(cleaned_texts, tokenizer, model)
    return preds, probs , model_type

def display_sentiment(prediction , conf):
    sentiment_colors = {
        "Positive": "#16a34a",  # green
        "Negative": "#dc2626",  # red
        "Neutral": "#2563eb"    # blue
    }
    sentiment_emojis = {
        "Positive": "😄",
        "Negative": "😞",
        "Neutral": "😐"
    }

    color = sentiment_colors.get(prediction, "#6b7280")
    emoji = sentiment_emojis.get(prediction, "")
    if conf:
        st.markdown(
            f"""
            <div style="
                background-color: {color}20;
                border: 2px solid {color};
                border-radius: 20px;
                padding: 20px;
                text-align: center;
                font-size: 28px;
                font-weight: bold;
                color: {color};
                margin-top: 20px; 
                margin-bottom: 20px;">
                {prediction} {emoji} with {conf}% confidence
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
                <div style="
                    background-color: {color}20;
                    border: 2px solid {color};
                    border-radius: 20px;
                    padding: 20px;
                    text-align: center;
                    font-size: 28px;
                    font-weight: bold;
                    color: {color};
                    margin-top: 20px;
                    margin-bottom: 20px;">
                    {prediction} {emoji}
                </div>
                """,
            unsafe_allow_html=True
        )
# -------------------------
# 6. Streamlit UI
# -------------------------
st.title("📊 Sentiment Analysis Dashboard")

st.markdown("""
Upload a CSV/Excel file or enter a single comment to analyze sentiment.
Results will be saved automatically to your local MySQL database.
""")

if "label" not in st.session_state:
    st.session_state.label = None


tab1, tab2 = st.tabs(["Single Comment", "Batch Upload"])

# ---------- Single ----------
with tab1:
    name_input = st.text_input("product name")
    text_input = st.text_area("Enter your comment here")
    if st.button("Analyze Comment"):
        if text_input.strip() == "":
            st.warning("Please enter a valid comment.")
        else:
            preds, probs , type = predict_texts([text_input])
            label = preds[0]
            if probs is not None:
                score = float(np.max(probs[0]))
            else:
                score = None
            sentiment =encode_labels[str(label)]
            st.session_state.label = label
            if type == "ml":
                display_sentiment(sentiment, None)
            else:
                score = int(score*100)
                display_sentiment(sentiment, score)
    # # Save to DB
    if st.session_state.label is not None:

        if st.button("💾 Save Results to Database"):

            if not name_input.strip() or not text_input.strip():
                st.warning("⚠️ There are missing data.")
            else:
                try:
                    data = {
                        "product_name": name_input.strip(),
                        "date": datetime.datetime.now(),
                        "comment": text_input.strip(),
                        "sentiment": str(st.session_state.label)
                    }
                    collection.insert_one(data)
                    st.success("✅ Results saved to the database successfully!")

                except Exception as e:
                    st.error(f"❌ Failed to save results: {e}")


if "DW" not in st.session_state:
    st.session_state.DW = None

if "saved" not in st.session_state:
    st.session_state.saved = None

# ---------- Batch ----------
with tab2:
    uploaded = st.file_uploader("Upload Excel or CSV", type=["csv", "xlsx"])

    if uploaded:
        st.session_state.DW = None
        st.session_state.DW = None
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            text_col = st.selectbox("Select text column", df.columns)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            df = None
        if df is not None:

            if st.button("Analyze File"):
                texts = df[text_col].astype(str).tolist()
                preds, probs , type= predict_texts(texts)
                print(probs)
                df["sentiment"] = preds
                st.session_state.saved = df
                st.session_state.DW = True
                if type == "ml":
                    df["predicted_score"] = None
                else:

                    df["predicted_score"] = [np.max(p)if p is not None else None for p in probs]
        # Download button
        if st.session_state.DW is not None:
            st.download_button("Download Predictions", df.to_csv(index=False).encode("utf-8") , "predictions.csv")




        if st.session_state.saved is not None:

            if st.button("💾 Save Results to Database"):
                dataf = st.session_state.saved
                if 'date'  in dataf.columns:
                    dataf = dataf[["product_name" , "date" , "comment" , "sentiment"]]
                else:

                    dataf['date'] = datetime.datetime.now()
                    dataf = dataf[["product_name" , "date" , "comment" , "sentiment"]]

                data_to_insert = dataf.to_dict(orient="records")

                try:
                    collection.insert_many(data_to_insert)
                    st.success("✅ Results saved to the database successfully!")
                    st.session_state.saved = None
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to save results: {e}")





# ---------- DB Preview ----------
st.markdown("---")
st.subheader("Database Preview")

try:
    df_preview = pd.DataFrame(list(collection.find().sort("date", -1).limit(10)))
    df_preview["sentiment"] = df_preview["sentiment"].apply(lambda x:encode_labels[str(x)])
    st.dataframe(df_preview)
except Exception as e:
    st.info("No data in database yet or connection issue.")
    st.write(e)

