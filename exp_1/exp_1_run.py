"""
Experiment 1 & 2 -- Dataset Detective
Most Streamed Spotify Songs (demo) -- terminal-runnable version
21CSC305P Machine Learning Lab
"""

import pandas as pd

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# Task 2 -- Load the dataset
section("TASK 2 -- Load the dataset into pandas")
df = pd.read_csv("spotify_songs_demo.csv")
print("Shape:", df.shape)

# Task 3 -- df.head()
section("TASK 3 -- df.head()")
print(df.head())

# Task 3 -- df.shape and df.columns
section("TASK 3 -- df.shape and df.columns")
print("Shape:", df.shape)
print("Columns:", list(df.columns))

# Task 3 -- df.info()
section("TASK 3 -- df.info()")
df.info()

# Task 3 -- df.describe()
section("TASK 3 -- df.describe()")
print(df.describe())

# Task 3 -- df.isnull().sum()
section("TASK 3 -- df.isnull().sum()")
print(df.isnull().sum())

# Task 3 -- df.duplicated().sum()
section("TASK 3 -- df.duplicated().sum()")
print("Duplicate rows:", df.duplicated().sum())

# Bonus -- quick insight
section("BONUS -- quick insight")
median_popularity = df["spotify_popularity"].median()
print("Median Spotify popularity:", median_popularity)

top5 = df.sort_values("streams_billions", ascending=False).head(5)
print("\nTop 5 most-streamed tracks:")
print(top5[["track_name", "artist", "streams_billions"]].to_string(index=False))

print("\nDone. Now repeat this on your own team's dataset from the ML Lab site.")
