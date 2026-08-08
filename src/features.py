import numpy as np
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

FEATURE_COLS = [
    "colors_count", "power", "toughness",
    "max_number_in_text", "text_length",
    "is_creature", "is_spell", "is_artifact", "is_enchantment"
]

def prepare_features(df, test_sets=["hob", "hoc"], max_features=2000):
    df_train = df[~df["set"].isin(test_sets)]
    df_test = df[df["set"].isin(test_sets)]

    print(f"Train data: {len(df_train)} | Test data ({test_sets}): {len(df_test)}")

    custom_stop_words = list(TfidfVectorizer(stop_words="english").get_stop_words()) + ["tk", "contraption"]
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words=custom_stop_words, min_df=4, ngram_range=(1, 3))
    X_text_train = vectorizer.fit_transform(df_train["oracle_text"]).toarray()
    X_text_test = vectorizer.transform(df_test["oracle_text"]).toarray()

    X_num_train = df_train[FEATURE_COLS].values
    X_num_test = df_test[FEATURE_COLS].values

    X_train = np.hstack((X_text_train, X_num_train))
    X_test = np.hstack((X_text_test, X_num_test))

    y_train = np.log1p(df_train["cmc"].values)
    y_test = np.log1p(df_test["cmc"].values)

    return X_train, X_test, y_train, y_test, vectorizer


def save_processed_features(df, X_train, X_test, y_train, y_test, vectorizer, save_dir="data/processed"):
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "X_test.npy"), X_test)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)
    joblib.dump(vectorizer, os.path.join(save_dir, "vectorizer.pkl"))
    joblib.dump(df, os.path.join(save_dir, "df_processed.pkl"))
    print(f"Processed features and dataframe cached in '{save_dir}/'")


def load_processed_features(save_dir="data/processed"):
    print(f"Loading cached features and dataframe from '{save_dir}/'...")
    X_train = np.load(os.path.join(save_dir, "X_train.npy"))
    X_test = np.load(os.path.join(save_dir, "X_test.npy"))
    y_train = np.load(os.path.join(save_dir, "y_train.npy"))
    y_test = np.load(os.path.join(save_dir, "y_test.npy"))
    vectorizer = joblib.load(os.path.join(save_dir, "vectorizer.pkl"))
    df = joblib.load(os.path.join(save_dir, "df_processed.pkl"))
    return X_train, X_test, y_train, y_test, vectorizer, df