import pandas as pd


def preprocess_data(df, df_original=None):

    df = df.copy()

    df = df.dropna(subset=["Released_Year"])
  
    df["Certificate"] = df["Certificate"].fillna("Unknown")

    df["Primary_Genre"] = df["Genre"].str.split(",").str[0].str.strip()

    if df_original is not None and "Meta_score" in df.columns:
        meta_means = df_original.groupby("Primary_Genre")["Meta_score"].mean()
        df["Meta_score"] = df.apply(
            lambda row: row["Meta_score"]
            if pd.notna(row["Meta_score"])
            else meta_means.get(row["Primary_Genre"], meta_means.mean()),
            axis=1,
        )

    df["Gross"] = df["Gross"].astype(str).str.replace(",", "", regex=False)
    df["Gross"] = pd.to_numeric(df["Gross"], errors="coerce")

    df["has_gross"] = df["Gross"].notna().astype(int)
    df["Gross"] = df["Gross"].fillna(0)

    return df


def create_features(df):

    df_model = df.copy()

    director_counts = df_model["Director"].value_counts()
    df_model["Director_frequency"] = df_model["Director"].map(director_counts).fillna(0)

    star1_counts = df_model["Star1"].value_counts()
    df_model["Star1_frequency"] = df_model["Star1"].map(star1_counts).fillna(0)

    return df_model
