import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score

os.makedirs('charts', exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("STEAM GAMES ANALYSIS - HELPING INDIE STUDIOS MAKE DATA-DRIVEN DECISIONS")
print("=" * 80)

print("\n[1/9] Loading data...")
df = pd.read_csv('games.csv')


df.columns = ['Name', 'Release date', 'Estimated owners', 'Peak CCU',
       'Required age', 'Price', 'Discount', 'DLC count', 'About the game',
       'Supported languages', 'Full audio languages', 'Reviews',
       'Header image', 'Website', 'Support url', 'Support email', 'Windows',
       'Mac', 'Linux', 'Metacritic score', 'Metacritic url', 'User score',
       'Positive', 'Negative', 'Score rank', 'Achievements', 'Recommendations',
       'Notes', 'Average playtime forever', 'Average playtime two weeks',
       'Median playtime forever', 'Median playtime two weeks', 'Developers',
       'Publishers', 'Categories', 'Genres', 'Tags', 'Screenshots', 'Movies']

print(f"   Dataset loaded: {len(df)} games found")

df_info = df.info()
print(df_info)

missing_values = df.isnull().sum()
print(missing_values)

df = df[df['Estimated owners'] != '0 - 0']
print(f"   After removing 0 owners games: {len(df)} games remaining")

number_to_reduced_number = {
    '0 - 20000': '0 - 20K',
    '20000 - 50000': '20K - 50K',
    '50000 - 100000': '50K - 100K',
    '100000 - 200000': '100K - 200K',
    '200000 - 500000': '200K - 500K',
    '500000 - 1000000': '500K - 1M',
    '1000000 - 2000000': '1M - 2M',
    '2000000 - 5000000': '2M - 5M',
    '5000000 - 10000000': '5M - 10M',
    '10000000 - 20000000': '10M - 20M',
    '20000000 - 50000000': '20M - 50M',
    '50000000 - 100000000': '50M - 100M',
    '100000000 - 200000000': '100M - 200M'}

df['Estimated owners'] = df['Estimated owners'].map(number_to_reduced_number)

estimated_owners_order = ['0 - 20K', '20K - 50K', '50K - 100K', '100K - 200K', '200K - 500K', '500K - 1M',
                          '1M - 2M', '2M - 5M', '5M - 10M', '10M - 20M', '20M - 50M', '50M - 100M', '100M - 200M']

print("\n[2/9] Analyzing game distribution by popularity...")
plt.figure(figsize=(14, 6))
df['Estimated owners'].value_counts().reindex(estimated_owners_order).plot.bar(ylabel='Number of games', color='steelblue')
plt.title('Distribution of Games by Number of Estimated Owners', fontsize=14, fontweight='bold')
plt.xlabel('Estimated Owners')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('charts/01_games_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Chart saved: charts/01_games_distribution.png")

print("\n[3/9] Analyzing price vs popularity...")

df_percent_free = df[df['Price'] == 0]['Estimated owners'].value_counts() / df['Estimated owners'].value_counts() * 100
df_wo_free = df[df['Price'] != 0]

plt.figure(figsize=(14, 6))
df_percent_free.reindex(estimated_owners_order).plot.bar(ylabel='Percentage of free games (%)', color='coral')
plt.title('Percentage of Free Games by Popularity Category', fontsize=14, fontweight='bold')
plt.xlabel('Estimated Owners')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('charts/02_free_games_percentage.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Chart saved: charts/02_free_games_percentage.png")

df_average_price = pd.concat([df.groupby('Estimated owners')['Price'].mean().reindex(estimated_owners_order),
                                     df_wo_free.groupby('Estimated owners')['Price'].mean().reindex(estimated_owners_order)], axis=1)
df_average_price.columns = ['Average price', 'Average price excluding free games']
df_average_price.fillna(0, inplace=True)

plt.figure(figsize=(14, 6))
df_average_price.plot.bar(ylabel='Average price (USD)')
plt.title('Average Game Price by Popularity Category', fontsize=14, fontweight='bold')
plt.xlabel('Estimated Owners')
plt.xticks(rotation=45, ha='right')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('charts/03_average_price.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Chart saved: charts/03_average_price.png")

print("\n[4/9] Analyzing reviews vs popularity...")

df_w_ratings = df[(df['Positive'] > 4) & (df['Negative'] > 4)][['Name', 'Estimated owners', 'Positive', 'Negative']]
df_w_ratings['Fraction positive'] = df_w_ratings['Positive'] / (df_w_ratings['Positive'] + df_w_ratings['Negative'])

df_rating_average = df_w_ratings.groupby('Estimated owners')['Fraction positive'].mean()
df_rating_average = df_rating_average.reindex(estimated_owners_order)

plt.figure(figsize=(14, 6))
(df_rating_average * 100).plot.bar(ylabel='Average fraction of positive reviews (%)', color='mediumseagreen')
plt.title('Average Positive Review Percentage by Popularity Category', fontsize=14, fontweight='bold')
plt.xlabel('Estimated Owners')
plt.xticks(rotation=45, ha='right')
plt.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='70% threshold')
plt.legend()
plt.tight_layout()
plt.savefig('charts/04_positive_reviews.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Chart saved: charts/04_positive_reviews.png")

print("\n[5/9] Analyzing genres vs popularity...")

def get_genre_count(df, to_df=False, df_column_name='Count'):
    """Calculate the count of each genre in a dataframe"""
    genre_dict = {}
    for genres, number in df['Genres'].value_counts().items():
        for genre in genres.split(','):
            if genre not in genre_dict:
                genre_dict[genre] = number
            else:
                genre_dict[genre] += number
    if to_df:
        genre_df = pd.DataFrame.from_dict(genre_dict, orient='index')
        genre_df.columns = [df_column_name]
        return genre_df
    return genre_dict

genre_fraction_per_owner = pd.DataFrame()
for owners_category in estimated_owners_order:
    new_column = get_genre_count(df[df['Estimated owners'] == owners_category], to_df=True, df_column_name=owners_category) / len(df[df['Estimated owners'] == owners_category])
    genre_fraction_per_owner = pd.concat([genre_fraction_per_owner, new_column], axis=1)
genre_fraction_per_owner.fillna(0, inplace=True)

genre_fraction = get_genre_count(df, to_df=True) / len(df)
top_12_genres = genre_fraction.sort_values('Count', ascending=False)[:12].index

plt.figure(figsize=(16, 10))
sns.heatmap((genre_fraction_per_owner.loc[top_12_genres]*100), annot=True, fmt='.1f', cmap='Blues', cbar_kws={'label': 'Percentage (%)'})
plt.title('Top 12 Genres Distribution Across Popularity Categories (%)', fontsize=14, fontweight='bold')
plt.xlabel('Estimated Owners')
plt.ylabel('Genre')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('charts/05_genre_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Chart saved: charts/05_genre_heatmap.png")

print("\n[6/9] Analyzing game descriptions...")

def has_cjk_character(text):
    """Check if a string contains any CJK character"""
    cjk_pattern = re.compile(r'[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]')
    return bool(cjk_pattern.search(text))

df_w_description = df[df['About the game'].notna()]
df_description_length = df_w_description['About the game'].transform(lambda x: len(x.split()))
df_description_length.name = 'Description length'
df_description_length = pd.concat([df_w_description[['Name', 'Estimated owners', 'About the game']], df_description_length], axis=1)

df_description_length_wo_cjk = df_description_length[~df_description_length['About the game'].apply(has_cjk_character)]

df_description_length_average = df_description_length_wo_cjk.groupby('Estimated owners')['Description length'].mean()
df_description_length_average = df_description_length_average.reindex(estimated_owners_order)

plt.figure(figsize=(14, 6))
df_description_length_average.plot.bar(ylabel='Average number of words in game description', color='mediumpurple')
plt.title('Average Game Description Length by Popularity Category', fontsize=14, fontweight='bold')
plt.xlabel('Estimated Owners')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('charts/06_description_length.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Chart saved: charts/06_description_length.png")

print("\n[7/9] Analyzing operating system support...")

systems = ['Windows', 'Mac', 'Linux']

df_system_support = pd.DataFrame()
for system in systems:
    df_system_support[system] = df.groupby('Estimated owners')[system].mean()
    
df_system_support = df_system_support.reindex(estimated_owners_order)

# Plot OS support
plt.figure(figsize=(14, 6))
(df_system_support*100).plot.bar(ylabel='Fraction of games (%)')
plt.title('Operating System Support by Popularity Category', fontsize=14, fontweight='bold')
plt.xlabel('Estimated Owners')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Operating System')
plt.tight_layout()
plt.savefig('charts/07_os_support.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Chart saved: charts/07_os_support.png")

print("\n[8/9] Analyzing genre trends of popular games over the years...")

popular_threshold_categories = ['200K - 500K','500K - 1M', '1M - 2M', '2M - 5M', '5M - 10M', '10M - 20M', '20M - 50M', '50M - 100M', '100M - 200M']
df_popular = df[df['Estimated owners'].isin(popular_threshold_categories)].copy()

df_popular['Release date'] = pd.to_datetime(df_popular['Release date'], errors='coerce')
df_popular['Release year'] = df_popular['Release date'].dt.year

df_popular = df_popular[(df_popular['Release year'] >= 2000) & (df_popular['Release year'] <= 2026)]

print(f"   Analyzing {len(df_popular)} popular games from 2000-2026")

years = sorted(df_popular['Release year'].unique())
genre_trends = pd.DataFrame()

for year in years:
    df_year = df_popular[df_popular['Release year'] == year]
    genre_count = get_genre_count(df_year, to_df=True, df_column_name=str(year))
    genre_trends = pd.concat([genre_trends, genre_count], axis=1)

genre_trends = genre_trends.fillna(0).T

top_genres_popular = genre_trends.sum().sort_values(ascending=False)[:8].index

plt.figure(figsize=(16, 8))
for genre in top_genres_popular:
    plt.plot(genre_trends.index.astype(int), genre_trends[genre], marker='o', label=genre, linewidth=2)

plt.title('Genre Trends of Popular Games Over Years (200K+ Owners)', fontsize=14, fontweight='bold')
plt.xlabel('Release Year')
plt.ylabel('Number of Games')
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('charts/08_genre_trends_over_years.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Chart saved: charts/08_genre_trends_over_years.png")

genre_pct_trends = pd.DataFrame()
for year in years:
    df_year = df_popular[df_popular['Release year'] == year]
    total_games = len(df_year)
    genre_count = get_genre_count(df_year, to_df=True, df_column_name=str(year))
    genre_pct_trends = pd.concat([genre_pct_trends, genre_count / total_games * 100], axis=1)

genre_pct_trends = genre_pct_trends.fillna(0).T

plt.figure(figsize=(16, 8))
for genre in top_genres_popular:
    plt.plot(genre_pct_trends.index.astype(int), genre_pct_trends[genre], marker='o', label=genre, linewidth=2)

plt.title('Genre Percentage Trends of Popular Games Over Years (200K+ Owners)', fontsize=14, fontweight='bold')
plt.xlabel('Release Year')
plt.ylabel('Percentage of Games (%)')
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('charts/09_genre_percentage_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Chart saved: charts/09_genre_percentage_trends.png")

plt.figure(figsize=(20, 10))
sns.heatmap(genre_pct_trends[top_genres_popular].T, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Percentage (%)'})
plt.title('Heatmap: Genre Trends of Popular Games (200K+ Owners) by Year', fontsize=14, fontweight='bold')
plt.xlabel('Release Year')
plt.ylabel('Genre')
plt.tight_layout()
plt.savefig('charts/10_genre_trends_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Chart saved: charts/10_genre_trends_heatmap.png")

print("\n[9/11] Applying K-Means clustering...")

df_ml = df.copy()
df_ml['Release date'] = pd.to_datetime(df_ml['Release date'], errors='coerce')
df_ml['Release year'] = df_ml['Release date'].dt.year.fillna(0)

df_ml['Positive ratio'] = np.where(
    (df_ml['Positive'] + df_ml['Negative']) > 0,
    df_ml['Positive'] / (df_ml['Positive'] + df_ml['Negative']),
    0
)

kmeans_features = [
    'Price',
    'Discount',
    'DLC count',
    'Positive',
    'Negative',
    'Recommendations',
    'Average playtime forever',
    'Achievements',
    'Positive ratio'
]

X_kmeans = df_ml[kmeans_features].fillna(0)
scaler_kmeans = StandardScaler()
X_kmeans_scaled = scaler_kmeans.fit_transform(X_kmeans)

n_clusters = 4
kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
df_ml['Cluster'] = kmeans_model.fit_predict(X_kmeans_scaled)

sil_score = silhouette_score(X_kmeans_scaled, df_ml['Cluster'])
print(f"   K-Means complete: {n_clusters} clusters, silhouette score = {sil_score:.3f}")

cluster_profile = df_ml.groupby('Cluster')[kmeans_features].mean().round(2)
print("\n   Cluster profiles (mean values):")
print(cluster_profile)

plt.figure(figsize=(12, 7))
sns.scatterplot(
    data=df_ml,
    x='Price',
    y='Positive ratio',
    hue='Cluster',
    palette='tab10',
    alpha=0.6,
    s=30
)
plt.title('K-Means Clusters: Price vs Positive Review Ratio', fontsize=14, fontweight='bold')
plt.xlabel('Price (USD)')
plt.ylabel('Positive review ratio')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig('charts/11_kmeans_clusters.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Chart saved: charts/11_kmeans_clusters.png")

print("\n[10/11] Applying Decision Tree classification...")

dt_features = [
    'Price',
    'Discount',
    'DLC count',
    'Positive ratio',
    'Recommendations',
    'Average playtime forever',
    'Achievements',
    'Release year'
]

X_dt = df_ml[dt_features].fillna(0)
y_dt = df_ml['Estimated owners']

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X_dt,
        y_dt,
        test_size=0.2,
        random_state=42,
        stratify=y_dt
    )
except ValueError:
    # Fallback when at least one class has too few samples for stratified split.
    X_train, X_test, y_train, y_test = train_test_split(
        X_dt,
        y_dt,
        test_size=0.2,
        random_state=42
    )

dt_model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=30,
    random_state=42,
    class_weight='balanced'
)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"   Decision Tree accuracy: {acc:.3f}")

print("\n   Classification report:")
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred, labels=estimated_owners_order)
plt.figure(figsize=(14, 9))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='GnBu',
    xticklabels=estimated_owners_order,
    yticklabels=estimated_owners_order
)
plt.title('Decision Tree Confusion Matrix (Estimated Owners)', fontsize=14, fontweight='bold')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('charts/12_decision_tree_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Chart saved: charts/12_decision_tree_confusion_matrix.png")

feature_importance = pd.Series(dt_model.feature_importances_, index=dt_features).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
feature_importance.plot.bar(color='teal')
plt.title('Decision Tree Feature Importance', fontsize=14, fontweight='bold')
plt.xlabel('Feature')
plt.ylabel('Importance score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('charts/13_decision_tree_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Chart saved: charts/13_decision_tree_feature_importance.png")

plt.figure(figsize=(22, 12))
plot_tree(
    dt_model,
    feature_names=dt_features,
    class_names=dt_model.classes_,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=7
)
plt.title('Decision Tree Visualization (Top 3 Levels)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/14_decision_tree_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Chart saved: charts/14_decision_tree_plot.png")

print("\n[11/11] Generating summary statistics and recommendations...")

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(f"Total games analyzed: {len(df):,}")
print(f"Popular games (200K+ owners): {len(df_popular):,}")
print(f"K-Means silhouette score: {sil_score:.3f}")
print(f"Decision Tree accuracy: {acc:.3f}")
print(f"\nMost common genres in popular games:")
for i, genre in enumerate(top_genres_popular, 1):
    count = genre_trends[genre].sum()
    print(f"  {i}. {genre}: {int(count)} games")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - All charts saved to 'charts/' folder")
print("=" * 80)
