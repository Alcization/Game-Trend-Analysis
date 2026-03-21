import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from datetime import datetime

# Create charts directory if it doesn't exist
os.makedirs('charts', exist_ok=True)

# Set style for better-looking charts
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("STEAM GAMES ANALYSIS - HELPING INDIE STUDIOS MAKE DATA-DRIVEN DECISIONS")
print("=" * 80)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

print("\n[1/9] Loading data...")
df = pd.read_csv('games.csv')


# Rename columns to fix the 'DiscountDLC count' error
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

# Remove games with 0 estimated owners (playtests, removed games)
df = df[df['Estimated owners'] != '0 - 0']
print(f"   After removing 0 owners games: {len(df)} games remaining")

# Map estimated owners to K/M notation for better readability
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

# Define the preferred order for estimated owners categories
estimated_owners_order = ['0 - 20K', '20K - 50K', '50K - 100K', '100K - 200K', '200K - 500K', '500K - 1M',
                          '1M - 2M', '2M - 5M', '5M - 10M', '10M - 20M', '20M - 50M', '50M - 100M', '100M - 200M']

# Plot distribution of games by estimated owners
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

# ============================================================================
# PRICE ANALYSIS
# ============================================================================

print("\n[3/9] Analyzing price vs popularity...")

# Calculate fraction of free games
df_percent_free = df[df['Price'] == 0]['Estimated owners'].value_counts() / df['Estimated owners'].value_counts() * 100
df_wo_free = df[df['Price'] != 0]

# Plot fraction of free games
plt.figure(figsize=(14, 6))
df_percent_free.reindex(estimated_owners_order).plot.bar(ylabel='Percentage of free games (%)', color='coral')
plt.title('Percentage of Free Games by Popularity Category', fontsize=14, fontweight='bold')
plt.xlabel('Estimated Owners')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('charts/02_free_games_percentage.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Chart saved: charts/02_free_games_percentage.png")

# Calculate average price
df_average_price = pd.concat([df.groupby('Estimated owners')['Price'].mean().reindex(estimated_owners_order),
                                     df_wo_free.groupby('Estimated owners')['Price'].mean().reindex(estimated_owners_order)], axis=1)
df_average_price.columns = ['Average price', 'Average price excluding free games']
df_average_price.fillna(0, inplace=True)

# Plot average price
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

# ============================================================================
# REVIEWS ANALYSIS
# ============================================================================

print("\n[4/9] Analyzing reviews vs popularity...")

# Filter games with meaningful review counts
df_w_ratings = df[(df['Positive'] > 4) & (df['Negative'] > 4)][['Name', 'Estimated owners', 'Positive', 'Negative']]
df_w_ratings['Fraction positive'] = df_w_ratings['Positive'] / (df_w_ratings['Positive'] + df_w_ratings['Negative'])

# Calculate average positive review fraction
df_rating_average = df_w_ratings.groupby('Estimated owners')['Fraction positive'].mean()
df_rating_average = df_rating_average.reindex(estimated_owners_order)

# Plot average positive reviews
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

# ============================================================================
# GENRES ANALYSIS
# ============================================================================

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

# Create DataFrame with genre fractions for each owner category
genre_fraction_per_owner = pd.DataFrame()
for owners_category in estimated_owners_order:
    new_column = get_genre_count(df[df['Estimated owners'] == owners_category], to_df=True, df_column_name=owners_category) / len(df[df['Estimated owners'] == owners_category])
    genre_fraction_per_owner = pd.concat([genre_fraction_per_owner, new_column], axis=1)
genre_fraction_per_owner.fillna(0, inplace=True)

# Find top 12 genres
genre_fraction = get_genre_count(df, to_df=True) / len(df)
top_12_genres = genre_fraction.sort_values('Count', ascending=False)[:12].index

# Plot genre heatmap
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

# ============================================================================
# GAME DESCRIPTION ANALYSIS
# ============================================================================

print("\n[6/9] Analyzing game descriptions...")

def has_cjk_character(text):
    """Check if a string contains any CJK character"""
    cjk_pattern = re.compile(r'[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]')
    return bool(cjk_pattern.search(text))

# Create dataframe with description length
df_w_description = df[df['About the game'].notna()]
df_description_length = df_w_description['About the game'].transform(lambda x: len(x.split()))
df_description_length.name = 'Description length'
df_description_length = pd.concat([df_w_description[['Name', 'Estimated owners', 'About the game']], df_description_length], axis=1)

# Remove games with CJK characters
df_description_length_wo_cjk = df_description_length[~df_description_length['About the game'].apply(has_cjk_character)]

# Calculate average description length
df_description_length_average = df_description_length_wo_cjk.groupby('Estimated owners')['Description length'].mean()
df_description_length_average = df_description_length_average.reindex(estimated_owners_order)

# Plot description length
plt.figure(figsize=(14, 6))
df_description_length_average.plot.bar(ylabel='Average number of words in game description', color='mediumpurple')
plt.title('Average Game Description Length by Popularity Category', fontsize=14, fontweight='bold')
plt.xlabel('Estimated Owners')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('charts/06_description_length.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Chart saved: charts/06_description_length.png")

# ============================================================================
# OPERATING SYSTEM SUPPORT ANALYSIS
# ============================================================================

print("\n[7/9] Analyzing operating system support...")

systems = ['Windows', 'Mac', 'Linux']

# Create dataframe with OS support fractions
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

# ============================================================================
# NEW ANALYSIS: GENRE TRENDS OVER YEARS FOR POPULAR GAMES
# ============================================================================

print("\n[8/9] Analyzing genre trends of popular games over the years...")

# Define popular games threshold (200K+ owners)
popular_threshold_categories = ['200K - 500K','500K - 1M', '1M - 2M', '2M - 5M', '5M - 10M', '10M - 20M', '20M - 50M', '50M - 100M', '100M - 200M']
df_popular = df[df['Estimated owners'].isin(popular_threshold_categories)].copy()

# Parse release dates and extract year
df_popular['Release date'] = pd.to_datetime(df_popular['Release date'], errors='coerce')
df_popular['Release year'] = df_popular['Release date'].dt.year

# Filter for valid years (2000-2026)
df_popular = df_popular[(df_popular['Release year'] >= 2000) & (df_popular['Release year'] <= 2026)]

print(f"   Analyzing {len(df_popular)} popular games from 2000-2026")

# Calculate genre counts by year
years = sorted(df_popular['Release year'].unique())
genre_trends = pd.DataFrame()

for year in years:
    df_year = df_popular[df_popular['Release year'] == year]
    genre_count = get_genre_count(df_year, to_df=True, df_column_name=str(year))
    genre_trends = pd.concat([genre_trends, genre_count], axis=1)

genre_trends = genre_trends.fillna(0).T

# Get top 8 genres for popular games
top_genres_popular = genre_trends.sum().sort_values(ascending=False)[:8].index

# Plot genre trends over years
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

# Calculate genre percentage trends (normalized by total games per year)
genre_pct_trends = pd.DataFrame()
for year in years:
    df_year = df_popular[df_popular['Release year'] == year]
    total_games = len(df_year)
    genre_count = get_genre_count(df_year, to_df=True, df_column_name=str(year))
    genre_pct_trends = pd.concat([genre_pct_trends, genre_count / total_games * 100], axis=1)

genre_pct_trends = genre_pct_trends.fillna(0).T

# Plot percentage trends
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

# Create a heatmap of genre trends
plt.figure(figsize=(20, 10))
sns.heatmap(genre_pct_trends[top_genres_popular].T, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Percentage (%)'})
plt.title('Heatmap: Genre Trends of Popular Games (200K+ Owners) by Year', fontsize=14, fontweight='bold')
plt.xlabel('Release Year')
plt.ylabel('Genre')
plt.tight_layout()
plt.savefig('charts/10_genre_trends_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Chart saved: charts/10_genre_trends_heatmap.png")

# ============================================================================
# SUMMARY AND RECOMMENDATIONS
# ============================================================================

print("\n[9/9] Generating summary statistics and recommendations...")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(f"Total games analyzed: {len(df):,}")
print(f"Popular games (200K+ owners): {len(df_popular):,}")
print(f"\nMost common genres in popular games:")
for i, genre in enumerate(top_genres_popular, 1):
    count = genre_trends[genre].sum()
    print(f"  {i}. {genre}: {int(count)} games")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - All charts saved to 'charts/' folder")
print("=" * 80)
