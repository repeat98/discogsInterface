#!/usr/bin/env python3
"""
Discogs Database Analysis Script

This script connects to a Discogs SQLite database, loads data from the 'releases' table into a DataFrame,
performs various data preprocessing steps, generates plots, conducts statistical and ML-based analyses,
and saves a comprehensive summary report of the findings.
"""

import argparse
import logging
import os
import sys
import json
import sqlite3
from typing import Any, Dict, List, Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from wordcloud import WordCloud

# Stats/ML libraries
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Set Seaborn style
sns.set(style="whitegrid")

###############################################################################
#                                Helper Functions                             #
###############################################################################
def safe_json_loads(value: Any) -> List[Dict[str, Any]]:
    """
    Safely load a JSON string. Returns an empty list if loading fails.

    :param value: Potential JSON string or other object.
    :return: List of dictionaries (or empty list on failure).
    """
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []

def connect_db(db_path: str) -> sqlite3.Connection:
    """
    Connect to the SQLite database and return the connection object.

    :param db_path: Path to the SQLite database file.
    :return: SQLite connection object.
    """
    if not os.path.exists(db_path):
        logging.error(f"Database file '{db_path}' does not exist.")
        sys.exit(1)
    try:
        conn = sqlite3.connect(db_path)
        logging.info(f"Successfully connected to database: {db_path}")
        return conn
    except sqlite3.Error as e:
        logging.error(f"SQLite error: {e}")
        sys.exit(1)

def load_data(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load data from the 'releases' table into a pandas DataFrame.

    :param conn: SQLite connection object.
    :return: DataFrame containing all rows from 'releases' table.
    """
    query = "SELECT * FROM releases"
    try:
        df = pd.read_sql_query(query, conn)
        logging.info(f"Loaded {len(df)} rows from 'releases' table.")
        return df
    except pd.io.sql.DatabaseError as e:
        logging.error(f"Database error: {e}")
        sys.exit(1)

###############################################################################
#                           Data Preprocessing                                #
###############################################################################
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the DataFrame for analysis. Handles:
      - Converting numeric fields
      - Filling missing values
      - Extracting year from 'released'
      - Safely loading JSON fields
      - Basic standardization for columns like 'genre', 'style', etc.
      - Removing releases with year == 0 to prevent skewing year-based analyses.

    :param df: Input DataFrame.
    :return: Preprocessed DataFrame.
    """
    logging.info("Preprocessing data...")

    numeric_fields = [
        'average_rating', 'rating_count', 'have', 'want',
        'rating_coefficient', 'demand_coefficient', 'gem_value',
        'num_for_sale', 'lowest_price'
    ]

    # Convert numeric fields to appropriate types and fill missing
    for field in numeric_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0)
        else:
            df[field] = 0
            logging.warning(f"Field '{field}' not found in data. Filled with 0.")

    # Handle years: extract from 'released' if missing
    if 'released' in df.columns and 'year' in df.columns:
        df['year'] = df['year'].fillna(df['released'].str[:4])
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')

    # **Remove releases with year == 0**
    if 'year' in df.columns:
        original_length = len(df)
        df = df[df['year'] != 0]
        removed = original_length - len(df)
        logging.info(f"Removed {removed} releases with year 0.")

    # Fill missing genres, styles, country, format using .loc to avoid SettingWithCopyWarning
    for col in ['genre', 'style', 'country', 'format']:
        if col in df.columns:
            df.loc[:, col] = df[col].fillna('Unknown')
        else:
            df[col] = "Unknown"
            logging.warning(f"Field '{col}' not found in data. Filled with 'Unknown'.")

    # Safely load JSON columns if present; otherwise create empty lists
    for col_name in ['tracklist', 'extraartists', 'labels']:
        if col_name in df.columns:
            df[col_name] = df[col_name].apply(safe_json_loads)
        else:
            df[col_name] = [[] for _ in range(len(df))]
            logging.warning(f"Column '{col_name}' not found. Filled with empty lists.")

    # Create track count column if tracklist exists using .loc
    if 'tracklist' in df.columns:
        df.loc[:, 'num_tracks'] = df['tracklist'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    else:
        df['num_tracks'] = 0
        logging.warning("'tracklist' column not found. 'num_tracks' set to 0.")

    logging.info("Data preprocessing completed.")
    return df

###############################################################################
#                                  Plotting                                   #
###############################################################################
def save_figure(filename: str) -> None:
    """
    Helper function to save and close the current matplotlib figure.

    :param filename: Path (including filename) to save the figure.
    """
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_distribution(df: pd.DataFrame, column: str, bins: int, title: str, xlabel: str,
                      output_path: str, color: str = 'skyblue', kde: bool = True) -> None:
    """
    Generic function to plot a distribution (histogram + optional KDE).

    :param df: DataFrame.
    :param column: Column name to plot.
    :param bins: Number of bins for histogram.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param output_path: File path to save figure.
    :param color: Color for the plot.
    :param kde: Whether to include a KDE curve.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=bins, kde=kde, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    save_figure(output_path)

def plot_scatter(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str,
                 output_path: str, color: Optional[str] = None, alpha: float = 0.5) -> None:
    """
    Generic function to plot a scatter plot.

    :param df: DataFrame.
    :param x: Name of the column for x-axis.
    :param y: Name of the column for y-axis.
    :param title: Plot title.
    :param xlabel: X-axis label.
    :param ylabel: Y-axis label.
    :param output_path: File path to save the figure.
    :param color: Color for the points.
    :param alpha: Transparency for the points.
    """
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=x, y=y, data=df, alpha=alpha, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_figure(output_path)

def plot_rating_distribution(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot distribution of average ratings and rating counts.
    """
    logging.info("Plotting rating distributions...")
    plot_distribution(df, 'average_rating', bins=20,
                      title='Distribution of Average Ratings',
                      xlabel='Average Rating',
                      output_path=os.path.join(output_dir, 'average_rating_distribution.png'),
                      color='skyblue', kde=True)

    plot_distribution(df, 'rating_count', bins=30,
                      title='Distribution of Rating Counts',
                      xlabel='Rating Count',
                      output_path=os.path.join(output_dir, 'rating_count_distribution.png'),
                      color='salmon', kde=True)

def plot_demand_gem_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """
    Analyze and plot demand coefficients (rarity) and gem values.
    Also shows correlation with rating_coefficient and average_rating.
    """
    logging.info("Plotting demand coefficient and gem value distributions...")
    plot_distribution(df, 'demand_coefficient', bins=20,
                      title='Distribution of Rarity',
                      xlabel='Rarity',
                      output_path=os.path.join(output_dir, 'demand_coefficient_distribution.png'),
                      color='green', kde=True)

    plot_distribution(df, 'gem_value', bins=20,
                      title='Distribution of Gem Values',
                      xlabel='Gem Value',
                      output_path=os.path.join(output_dir, 'gem_value_distribution.png'),
                      color='purple', kde=True)

    # Correlation between rating_coefficient and demand_coefficient
    logging.info("Plotting scatter: rating_coefficient vs. demand_coefficient...")
    plot_scatter(df, 'rating_coefficient', 'demand_coefficient',
                 title='Correlation between Rating Coefficient and Rarity',
                 xlabel='Rating Coefficient',
                 ylabel='Rarity',
                 output_path=os.path.join(output_dir, 'rating_vs_demand_correlation.png'))

    # Correlation between gem_value and average_rating
    logging.info("Plotting scatter: gem_value vs. average_rating...")
    plot_scatter(df, 'gem_value', 'average_rating',
                 title='Gem Value vs Average Rating',
                 xlabel='Gem Value',
                 ylabel='Average Rating',
                 output_path=os.path.join(output_dir, 'gem_value_vs_average_rating.png'))

def plot_genre_style_insights(df: pd.DataFrame, output_dir: str) -> None:
    """
    Identify and plot top genres and styles.
    """
    logging.info("Plotting top genres and styles...")
    # Top 10 Genres
    top_genres = df['genre'].str.split(', ').explode().value_counts().head(10)
    plt.figure(figsize=(12,8))
    sns.barplot(x=top_genres.values, y=top_genres.index, palette='Blues_d')
    plt.title('Top 10 Genres')
    plt.xlabel('Number of Releases')
    plt.ylabel('Genre')
    save_figure(os.path.join(output_dir, 'top_genres.png'))

    # Top 10 Styles excluding 'Techno'
    top_styles = df['style'].str.split(', ').explode()
    top_styles = top_styles[top_styles != 'Techno']
    top_styles = top_styles.value_counts().head(10)
    plt.figure(figsize=(12,8))
    sns.barplot(x=top_styles.values, y=top_styles.index, palette='Reds_d')
    plt.title('Top 10 Styles (Excluding "Techno")')
    plt.xlabel('Number of Releases')
    plt.ylabel('Style')
    save_figure(os.path.join(output_dir, 'top_styles.png'))

def plot_yearly_trends(df: pd.DataFrame, output_dir: str) -> None:
    """
    Visualize the number of releases over the years.
    """
    logging.info("Plotting yearly release trends...")
    if 'year' not in df.columns or df['year'].isna().all():
        logging.warning("No valid 'year' data found. Skipping yearly trends plot.")
        return

    yearly_counts = df['year'].value_counts().dropna().sort_index()
    plt.figure(figsize=(14,7))
    sns.lineplot(x=yearly_counts.index, y=yearly_counts.values, marker='o', color='blue')
    plt.title('Number of Releases Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Number of Releases')
    plt.xticks(rotation=45)
    save_figure(os.path.join(output_dir, 'releases_over_years.png'))

def plot_price_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """
    Analyze and plot the distribution of lowest prices.
    Also plots relationship between lowest_price and gem_value.
    """
    logging.info("Plotting price analysis...")
    plot_distribution(df, 'lowest_price', bins=30,
                      title='Distribution of Lowest Prices',
                      xlabel='Lowest Price ($)',
                      output_path=os.path.join(output_dir, 'lowest_price_distribution.png'),
                      color='orange', kde=True)

    # Relationship between lowest_price and gem_value
    plot_scatter(df, 'lowest_price', 'gem_value',
                 title='Lowest Price vs Gem Value',
                 xlabel='Lowest Price ($)',
                 ylabel='Gem Value',
                 output_path=os.path.join(output_dir, 'lowest_price_vs_gem_value.png'),
                 alpha=0.5)

def plot_geographical_distribution(df: pd.DataFrame, output_dir: str) -> None:
    """
    Map the number of releases per country.
    """
    logging.info("Plotting geographical distribution...")
    country_counts = df['country'].value_counts().head(20)
    plt.figure(figsize=(14,8))
    sns.barplot(x=country_counts.values, y=country_counts.index, palette='Purples_d')
    plt.title('Top 20 Countries by Number of Releases')
    plt.xlabel('Number of Releases')
    plt.ylabel('Country')
    save_figure(os.path.join(output_dir, 'top_countries.png'))

def plot_label_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """
    Identify and plot the most prolific labels (top 100) based on a popularity score
    (average 'want' + average 'rating_coefficient').
    """
    logging.info("Plotting label analysis (top 100 labels by popularity)...")
    if 'labels' not in df.columns:
        logging.warning("'labels' column not found. Skipping label analysis.")
        return

    df_labels = df.explode('labels')
    df_labels['label_name'] = df_labels['labels'].apply(
        lambda x: x.get('name', 'Unknown') if isinstance(x, dict) else 'Unknown'
    )

    label_stats = df_labels.groupby('label_name').agg(
        average_want=('want', 'mean'),
        average_rating_coeff=('rating_coefficient', 'mean'),
        release_count=('id', 'count')
    ).reset_index()
    label_stats['popularity_score'] = label_stats['average_want'] + label_stats['average_rating_coeff']
    top_100_labels = label_stats.sort_values(by='popularity_score', ascending=False).head(100)

    plt.figure(figsize=(16, 60))
    sns.barplot(x='popularity_score', y='label_name', data=top_100_labels, palette='Blues_d')
    plt.title('Top 100 Labels by Popularity (Average Want + Average Rating Coefficient)')
    plt.xlabel('Popularity Score')
    plt.ylabel('Label')
    save_figure(os.path.join(output_dir, 'top_100_labels.png'))

def plot_artists_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """
    Identify and plot the top 100 artists by popularity based on average 'want' count per release.
    """
    logging.info("Plotting artist analysis (top 100 artists by want)...")
    if 'artists_sort' not in df.columns:
        logging.warning("'artists_sort' column not found. Skipping artist analysis.")
        return

    df_artists = df[['id', 'want', 'artists_sort']].copy()
    df_artists = df_artists.rename(columns={'artists_sort': 'artist'})

    df_artists['artist'] = df_artists['artist'].str.strip().replace('', 'Unknown')
    artist_popularity = df_artists.groupby('artist').agg(
        average_want=('want', 'mean'),
        release_count=('artist', 'count')
    ).reset_index()
    top_100_popular_artists = artist_popularity.sort_values(by='average_want', ascending=False).head(100)

    plt.figure(figsize=(16, 60))
    sns.barplot(x='average_want', y='artist', data=top_100_popular_artists, palette='Greens_d')
    plt.title('Top 100 Artists by Popularity (Average Want Count per Release)')
    plt.xlabel('Average Want Count')
    plt.ylabel('Artist')
    save_figure(os.path.join(output_dir, 'top_100_popular_artists.png'))

def plot_top100_best_rated_artists(df: pd.DataFrame, output_dir: str) -> None:
    """
    Identify and plot the top 100 best rated artists based on average 'rating_coefficient' per release.
    """
    logging.info("Plotting best-rated artists (top 100 by rating_coefficient)...")
    if 'artists_sort' not in df.columns:
        logging.warning("'artists_sort' column not found. Skipping best-rated artist analysis.")
        return

    df_artists = df[['id', 'rating_coefficient', 'artists_sort']].copy()
    df_artists = df_artists.rename(columns={'artists_sort': 'artist'})

    df_artists['artist'] = df_artists['artist'].str.strip().replace('', 'Unknown')
    artist_rating = df_artists.groupby('artist').agg(
        average_rating_coeff=('rating_coefficient', 'mean'),
        release_count=('artist', 'count')
    ).reset_index()

    top_100_best_rated_artists = artist_rating.sort_values(by='average_rating_coeff', ascending=False).head(100)

    plt.figure(figsize=(16, 60))
    sns.barplot(x='average_rating_coeff', y='artist', data=top_100_best_rated_artists, palette='Oranges_d')
    plt.title('Top 100 Best Rated Artists (Avg Rating Coefficient per Release)')
    plt.xlabel('Average Rating Coefficient')
    plt.ylabel('Artist')
    save_figure(os.path.join(output_dir, 'top_100_best_rated_artists.png'))

def plot_top100_best_rated_labels(df: pd.DataFrame, output_dir: str) -> None:
    """
    Identify and plot the top 100 best rated labels based on average 'rating_coefficient' per release.
    """
    logging.info("Plotting best-rated labels (top 100 by rating_coefficient)...")
    if 'labels' not in df.columns:
        logging.warning("'labels' column not found. Skipping best-rated label analysis.")
        return

    df_labels = df.explode('labels')
    df_labels['label_name'] = df_labels['labels'].apply(
        lambda x: x.get('name', 'Unknown') if isinstance(x, dict) else 'Unknown'
    )

    label_rating = df_labels.groupby('label_name').agg(
        average_rating_coeff=('rating_coefficient', 'mean'),
        release_count=('id', 'count')
    ).reset_index()

    top_100_best_rated_labels = label_rating.sort_values(by='average_rating_coeff', ascending=False).head(100)

    plt.figure(figsize=(16, 60))
    sns.barplot(x='average_rating_coeff', y='label_name', data=top_100_best_rated_labels, palette='Purples_d')
    plt.title('Top 100 Best Rated Labels (Avg Rating Coefficient per Release)')
    plt.xlabel('Average Rating Coefficient')
    plt.ylabel('Label')
    save_figure(os.path.join(output_dir, 'top_100_best_rated_labels.png'))

def plot_tracklist_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """
    Analyze the number of tracks per release and compare across top genres.
    """
    logging.info("Plotting tracklist analysis...")
    if 'num_tracks' not in df.columns:
        logging.warning("'num_tracks' column not found. Skipping tracklist analysis.")
        return

    plot_distribution(df, 'num_tracks', bins=range(0, df['num_tracks'].max()+2, 1),
                      title='Distribution of Number of Tracks per Release',
                      xlabel='Number of Tracks',
                      output_path=os.path.join(output_dir, 'num_tracks_distribution.png'),
                      color='teal', kde=False)

    # Average number of tracks per genre (top 5 genres)
    top_genres = df['genre'].str.split(', ').explode().value_counts().head(5).index.tolist()
    df_top_genres = df[df['genre'].str.contains('|'.join(top_genres), na=False)]

    plt.figure(figsize=(12,8))
    sns.boxplot(x='genre', y='num_tracks', data=df_top_genres, palette='Pastel1')
    plt.title('Number of Tracks per Release by Top Genres')
    plt.xlabel('Genre')
    plt.ylabel('Number of Tracks')
    save_figure(os.path.join(output_dir, 'num_tracks_by_genre.png'))

def plot_have_want_correlation(df: pd.DataFrame, output_dir: str) -> None:
    """
    Analyze the correlation between 'have' and 'want'.
    """
    logging.info("Plotting have vs. want correlation...")
    plot_scatter(df, 'have', 'want',
                 title='Correlation between Have and Want Counts',
                 xlabel='Have Count',
                 ylabel='Want Count',
                 output_path=os.path.join(output_dir, 'have_vs_want_correlation.png'),
                 alpha=0.5)

def plot_demand_vs_have_want(df: pd.DataFrame, output_dir: str) -> None:
    """
    Visualize how demand coefficients relate to have and want counts.
    Color-coded by 'want'.
    """
    logging.info("Plotting demand vs. have/want relationship...")
    plt.figure(figsize=(10,6))
    scatter = plt.scatter(df['have'], df['demand_coefficient'], c=df['want'], cmap='viridis', alpha=0.5)
    plt.title('Rarity vs Have Count (Colored by Want Count)')
    plt.xlabel('Have Count')
    plt.ylabel('Rarity')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Want Count')
    save_figure(os.path.join(output_dir, 'demand_vs_have_want.png'))

def plot_price_vs_rating(df: pd.DataFrame, output_dir: str) -> None:
    """
    Investigate if higher-rated releases tend to have higher or lower prices.
    """
    logging.info("Plotting average_rating vs. lowest_price scatter...")
    plot_scatter(df, 'average_rating', 'lowest_price',
                 title='Average Rating vs Lowest Price',
                 xlabel='Average Rating',
                 ylabel='Lowest Price ($)',
                 output_path=os.path.join(output_dir, 'rating_vs_price.png'),
                 alpha=0.5)

def plot_lowest_price_vs_demand_coefficient(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot the lowest price vs. demand_coefficient (rarity).
    """
    logging.info("Plotting lowest_price vs. demand_coefficient scatter...")
    plot_scatter(df, 'demand_coefficient', 'lowest_price',
                 title='Lowest Price vs Rarity',
                 xlabel='Rarity',
                 ylabel='Lowest Price ($)',
                 output_path=os.path.join(output_dir, 'lowest_price_vs_demand_coefficient.png'),
                 color='brown', alpha=0.5)

def plot_gem_value_over_lowest_price(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot gem_value over lowest_price.
    """
    logging.info("Plotting gem_value vs. lowest_price scatter...")
    plot_scatter(df, 'lowest_price', 'gem_value',
                 title='Gem Value over Lowest Price',
                 xlabel='Lowest Price ($)',
                 ylabel='Gem Value',
                 output_path=os.path.join(output_dir, 'gem_value_over_lowest_price.png'),
                 color='darkblue', alpha=0.5)

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    """
    Display a heatmap of correlations between numeric variables.
    """
    logging.info("Plotting correlation heatmap...")
    # Check for required columns
    numeric_cols = [
        'average_rating', 'rating_count', 'have', 'want',
        'rating_coefficient', 'demand_coefficient', 'gem_value',
        'num_for_sale', 'lowest_price', 'num_tracks'
    ]
    if 'total_duration_min' in df.columns:
        numeric_cols.append('total_duration_min')

    valid_cols = [col for col in numeric_cols if col in df.columns]
    if len(valid_cols) < 2:
        logging.warning("Not enough numeric columns found for correlation heatmap. Skipping.")
        return

    corr = df[valid_cols].corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap of Numeric Variables')
    save_figure(os.path.join(output_dir, 'correlation_heatmap.png'))

def plot_label_popularity_over_years(df: pd.DataFrame, output_dir: str) -> None:
    """
    Analyze how the popularity of top 10 labels changes over the years.
    Popularity is based on average 'want' and average 'rating_coefficient'.
    """
    logging.info("Plotting label popularity over years...")
    if 'labels' not in df.columns or 'year' not in df.columns:
        logging.warning("Required columns for label popularity analysis are missing. Skipping.")
        return

    df_labels = df.explode('labels')
    df_labels['label_name'] = df_labels['labels'].apply(
        lambda x: x.get('name', 'Unknown') if isinstance(x, dict) else 'Unknown'
    )
    label_stats = df_labels.groupby('label_name').agg(
        average_want=('want', 'mean'),
        average_rating_coeff=('rating_coefficient', 'mean'),
        release_count=('id', 'count')
    ).reset_index()
    label_stats['popularity_score'] = label_stats['average_want'] + label_stats['average_rating_coeff']
    top_10_labels = label_stats.sort_values(by='popularity_score', ascending=False).head(10)['label_name'].tolist()

    df_top_labels = df_labels[df_labels['label_name'].isin(top_10_labels)].dropna(subset=['year'])
    if df_top_labels.empty:
        logging.warning("No valid data for top labels over years. Skipping.")
        return

    label_trends = df_top_labels.groupby(['year', 'label_name']).size().reset_index(name='count')

    plt.figure(figsize=(14, 10))
    sns.lineplot(data=label_trends, x='year', y='count', hue='label_name', linewidth=2.5, marker='o')
    plt.title('Top 10 Labels Popularity Over Years')
    plt.xlabel('Year')
    plt.ylabel('Number of Releases')
    plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize='small')
    save_figure(os.path.join(output_dir, 'label_popularity_over_years.png'))

def plot_most_popular_styles_over_years(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot the most popular styles (excluding 'Techno') over the years, 
    and save the defining releases for each style's first appearance.
    """
    logging.info("Plotting most popular styles over years...")
    if 'style' not in df.columns or 'year' not in df.columns:
        logging.warning("Required columns for style analysis are missing. Skipping.")
        return

    top_n = 10
    styles_series = df['style'].str.split(', ').explode()
    styles_series = styles_series[styles_series != 'Techno']  # Exclude 'Techno'
    top_styles = styles_series.value_counts().head(top_n).index.tolist()

    df_top_styles = df.assign(style=df['style'].str.split(', ')).explode('style')
    df_top_styles['style'] = df_top_styles['style'].str.strip()
    df_top_styles = df_top_styles[df_top_styles['style'].isin(top_styles)]

    style_trends = df_top_styles.groupby(['year', 'style']).size().reset_index(name='count')
    if style_trends.empty:
        logging.warning("No valid data to plot styles over years. Skipping.")
        return

    pivot_df = style_trends.pivot(index='year', columns='style', values='count').fillna(0).sort_index()

    # Identify the first year each style appears
    first_years = {}
    for style in top_styles:
        years_for_style = style_trends[style_trends['style'] == style]['year'].dropna()
        if not years_for_style.empty:
            first_years[style] = years_for_style.min()

    # Save style-defining releases
    style_defining_releases = []
    for style in top_styles:
        first_year = first_years.get(style, None)
        if first_year is not None:
            defining = df_top_styles[
                (df_top_styles['style'] == style) & (df_top_styles['year'] == first_year)
            ]
            style_defining_releases.append(defining)

    if style_defining_releases:
        df_style_defining = pd.concat(style_defining_releases)
        columns_to_save = ['id', 'title', 'artists_sort', 'year', 'style']
        df_style_defining = df_style_defining[columns_to_save]
        path_csv = os.path.join(output_dir, 'style_defining_releases.csv')
        df_style_defining.to_csv(path_csv, index=False)
        logging.info(f"Saved style defining releases to '{path_csv}'.")

    # Zero counts before the style's first recorded year
    for style, yr in first_years.items():
        if pd.isna(yr):
            continue
        pivot_df.loc[pivot_df.index < yr, style] = 0

    # Plot
    plt.figure(figsize=(14,8))
    for style in top_styles:
        plt.plot(pivot_df.index, pivot_df[style], marker='o', label=style)
    plt.title('Top 10 Styles Popularity Over Years (Excluding "Techno")')
    plt.xlabel('Year')
    plt.ylabel('Number of Releases')
    plt.legend(title='Style', bbox_to_anchor=(1.05, 1), loc='upper left')
    save_figure(os.path.join(output_dir, 'most_popular_styles_over_years.png'))

def plot_average_gem_value_over_years(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot the average gem value over the years.
    """
    logging.info("Plotting average gem value over years...")
    if 'year' not in df.columns or 'gem_value' not in df.columns:
        logging.warning("Required columns for average gem value over years are missing. Skipping.")
        return

    df_gem_year = df.dropna(subset=['year', 'gem_value'])
    if df_gem_year.empty:
        logging.warning("No valid data to plot average gem value over years. Skipping.")
        return

    average_gem_per_year = df_gem_year.groupby('year')['gem_value'].mean().reset_index()

    plt.figure(figsize=(14,7))
    sns.lineplot(data=average_gem_per_year, x='year', y='gem_value', marker='o', color='darkgreen')
    plt.title('Average Gem Value Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Average Gem Value')
    plt.xticks(rotation=45)
    save_figure(os.path.join(output_dir, 'average_gem_value_over_years.png'))

def plot_format_distribution(df: pd.DataFrame, output_dir: str) -> None:
    """
    Visualize the distribution of different release formats.
    """
    logging.info("Plotting format distribution...")
    top_formats = df['format'].str.split(', ').explode().value_counts().head(10)
    plt.figure(figsize=(12,8))
    sns.barplot(x=top_formats.values, y=top_formats.index, palette='Greens_d')
    plt.title('Top 10 Release Formats')
    plt.xlabel('Number of Releases')
    plt.ylabel('Format')
    save_figure(os.path.join(output_dir, 'top_formats.png'))

def plot_wordcloud(text: str, title: str, output_path: str) -> None:
    """
    Generate and save a word cloud from the given text.
    """
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(15,7.5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=20)
    save_figure(output_path)

def plot_wordclouds(df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate word clouds for genres and styles.
    """
    logging.info("Generating word clouds for genres and styles...")
    # Genres Word Cloud
    genres_text = ' '.join(df['genre'].str.split(', ').explode())
    plot_wordcloud(genres_text, 'Genres Word Cloud',
                   os.path.join(output_dir, 'genres_wordcloud.png'))

    # Styles Word Cloud excluding 'Techno'
    styles_excluded = df['style'].str.split(', ').explode()
    styles_excluded = styles_excluded[styles_excluded != 'Techno']
    styles_text = ' '.join(styles_excluded)
    plot_wordcloud(styles_text, 'Styles Word Cloud (Excluding "Techno")',
                   os.path.join(output_dir, 'styles_wordcloud.png'))

def plot_total_duration_per_release(df: pd.DataFrame, output_dir: str) -> None:
    """
    Calculate and analyze the total playtime of each release from track durations.
    """
    logging.info("Plotting total duration per release...")

    def calculate_total_duration(tracklist: List[Dict[str, Any]]) -> float:
        """
        Calculate total duration in minutes from a list of track dictionaries.
        """
        total_seconds = 0
        for track in tracklist:
            duration = track.get('duration', '0:00')
            try:
                mins, secs = map(int, duration.split(':'))
                total_seconds += mins * 60 + secs
            except ValueError:
                # Skip invalid format
                continue
        return total_seconds / 60

    if 'tracklist' not in df.columns:
        logging.warning("'tracklist' column not found. Skipping total duration analysis.")
        return

    df['total_duration_min'] = df['tracklist'].apply(calculate_total_duration)

    plot_distribution(df, 'total_duration_min', bins=30,
                      title='Distribution of Total Duration per Release',
                      xlabel='Total Duration (Minutes)',
                      output_path=os.path.join(output_dir, 'total_duration_distribution.png'),
                      color='cyan', kde=True)

    # Average duration per genre (top 5 genres)
    top_genres = df['genre'].str.split(', ').explode().value_counts().head(5).index.tolist()
    df_top_genres = df[df['genre'].str.contains('|'.join(top_genres), na=False)]

    plt.figure(figsize=(12,8))
    sns.boxplot(x='genre', y='total_duration_min', data=df_top_genres, palette='coolwarm')
    plt.title('Total Duration per Release by Top Genres')
    plt.xlabel('Genre')
    plt.ylabel('Total Duration (Minutes)')
    save_figure(os.path.join(output_dir, 'total_duration_by_genre.png'))

def plot_top_countries_over_years(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot the number of releases for top countries over the years.
    """
    logging.info("Plotting top countries over years...")
    if 'country' not in df.columns or 'year' not in df.columns:
        logging.warning("Required columns for top countries over years are missing. Skipping.")
        return

    top_n = 10
    top_countries = df['country'].value_counts().head(top_n).index.tolist()
    df_top_countries = df[df['country'].isin(top_countries)]

    # Group by year and country
    country_yearly = df_top_countries.groupby(['year', 'country']).size().reset_index(name='count')
    if country_yearly.empty:
        logging.warning("No valid data to plot top countries over years. Skipping.")
        return

    pivot_df = country_yearly.pivot(index='year', columns='country', values='count').fillna(0).sort_index()

    plt.figure(figsize=(14,8))
    sns.lineplot(data=pivot_df, markers=True, dashes=False)
    plt.title('Top 10 Countries by Number of Releases Over Years')
    plt.xlabel('Year')
    plt.ylabel('Number of Releases')
    plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
    save_figure(os.path.join(output_dir, 'top_countries_over_years.png'))

###############################################################################
#                             Statistical Analysis                            #
###############################################################################
def calculate_vif(df: pd.DataFrame, output_dir: str) -> None:
    """
    Calculate Variance Inflation Factor (VIF) for each feature to assess multicollinearity.
    """
    logging.info("Calculating Variance Inflation Factor (VIF)...")
    features = [
        'gem_value', 'demand_coefficient', 'rating_coefficient',
        'average_rating', 'rating_count', 'have', 'want', 'num_for_sale',
        'year', 'num_tracks'
    ]
    if 'total_duration_min' in df.columns:
        features.append('total_duration_min')

    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        logging.warning(f"Missing features for VIF calculation: {missing_features}. Skipping.")
        return

    # Ensure all features are numeric
    vif_df = df[features].dropna().apply(pd.to_numeric, errors='coerce').fillna(0)

    if vif_df.empty:
        logging.warning("No valid data for VIF calculation. Skipping.")
        return

    X = sm.add_constant(vif_df)
    try:
        vif_data = pd.DataFrame()
        vif_data['feature'] = X.columns
        vif_data['VIF'] = [
            variance_inflation_factor(X.values, i) for i in range(X.shape[1])
        ]

        vif_data.to_csv(os.path.join(output_dir, 'vif_data.csv'), index=False)

        plt.figure(figsize=(10,6))
        sns.barplot(x='VIF', y='feature', data=vif_data, palette='coolwarm')
        plt.title('Variance Inflation Factor (VIF) for Features')
        plt.xlabel('VIF')
        plt.ylabel('Feature')
        save_figure(os.path.join(output_dir, 'vif_plot.png'))
        logging.info("VIF calculation completed. Results saved.")
    except Exception as e:
        logging.error(f"Error calculating VIF: {e}")

def plot_correlation_with_lowest_price(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot correlation of all numeric features with 'lowest_price'.
    """
    logging.info("Plotting correlation of features with lowest_price...")
    numeric_cols = [
        'average_rating', 'rating_count', 'have', 'want',
        'rating_coefficient', 'demand_coefficient', 'gem_value',
        'num_for_sale', 'year', 'num_tracks'
    ]
    if 'total_duration_min' in df.columns:
        numeric_cols.append('total_duration_min')

    # Ensure 'lowest_price' is present
    if 'lowest_price' not in df.columns:
        logging.warning("'lowest_price' column missing. Cannot plot correlation.")
        return

    valid_cols = [col for col in numeric_cols if col in df.columns]
    if not valid_cols:
        logging.warning("No valid numeric columns found for correlation analysis. Skipping.")
        return

    corr_df = df[valid_cols + ['lowest_price']].dropna()
    if corr_df.empty:
        logging.warning("No valid data for correlation with lowest_price. Skipping.")
        return

    correlations = corr_df.corr()['lowest_price'].drop('lowest_price')
    plt.figure(figsize=(10, 8))
    sns.barplot(x=correlations.values, y=correlations.index, palette='viridis')
    plt.title('Correlation of Features with Lowest Price')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Features')
    save_figure(os.path.join(output_dir, 'correlation_with_lowest_price.png'))

def plot_pairwise_relationships(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create pairwise scatter plots between 'lowest_price' and other key numeric features.
    """
    logging.info("Plotting pairwise relationships with lowest_price...")
    required_cols = [
        'lowest_price', 'gem_value', 'demand_coefficient',
        'rating_coefficient', 'average_rating', 'rating_count',
        'have', 'want'
    ]
    valid_cols = [col for col in required_cols if col in df.columns]

    if not valid_cols:
        logging.warning("No valid columns found for pairwise relationships. Skipping.")
        return

    pairwise_df = df[valid_cols].dropna()
    if pairwise_df.empty:
        logging.warning("Not enough valid data to create pairwise plots. Skipping.")
        return

    sns.pairplot(pairwise_df, diag_kind='kde')
    plt.suptitle('Pairwise Relationships with Lowest Price', y=1.02)
    save_figure(os.path.join(output_dir, 'pairwise_relationships.png'))

###############################################################################
#                            Summary Generation                               #
###############################################################################
def generate_summary(df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate a basic summary report with key statistics.
    Includes top 100 labels, artists, best-rated labels/artists, and top countries/formats.
    """
    logging.info("Generating summary report...")
    summary_path = os.path.join(output_dir, 'summary_report.txt')
    with open(summary_path, 'w') as f:
        f.write("=== Discogs Database Analysis Summary ===\n\n")
        f.write(f"Total Releases: {len(df)}\n")
        f.write(f"Average Rating: {df['average_rating'].mean():.2f}\n")
        f.write(f"Median Rating: {df['average_rating'].median():.2f}\n")
        f.write(f"Average Rating Count: {df['rating_count'].mean():.2f}\n")
        f.write(f"Average Demand Coefficient: {df['demand_coefficient'].mean():.2f}\n")
        f.write(f"Average Gem Value: {df['gem_value'].mean():.2f}\n")
        f.write(f"Average Lowest Price: ${df['lowest_price'].mean():.2f}\n")
        f.write(f"Average Number of Tracks: {df['num_tracks'].mean():.2f}\n")
        if 'total_duration_min' in df.columns:
            f.write(f"Average Total Duration: {df['total_duration_min'].mean():.2f} minutes\n\n")
        else:
            f.write("Average Total Duration: N/A (no duration data)\n\n")

        f.write("Top 10 Genres:\n")
        top_genres = df['genre'].str.split(', ').explode().value_counts().head(10)
        f.write(top_genres.to_string())
        f.write("\n\n")

        f.write("Top 10 Styles (Excluding 'Techno'):\n")
        styles_series = df['style'].str.split(', ').explode()
        styles_excl = styles_series[styles_series != 'Techno'].value_counts().head(10)
        f.write(styles_excl.to_string())
        f.write("\n\n")

        if 'labels' in df.columns:
            df_labels = df.explode('labels')
            df_labels['label_name'] = df_labels['labels'].apply(
                lambda x: x.get('name', 'Unknown') if isinstance(x, dict) else 'Unknown'
            )
            label_stats = df_labels.groupby('label_name').agg(
                average_want=('want', 'mean'),
                average_rating_coeff=('rating_coefficient', 'mean'),
                release_count=('id', 'count')
            ).reset_index()
            label_stats['popularity_score'] = label_stats['average_want'] + label_stats['average_rating_coeff']
            top_100_labels = label_stats.sort_values(by='popularity_score', ascending=False).head(100)

            f.write("Top 100 Labels by Popularity (Average Want + Average Rating Coefficient):\n")
            f.write(top_100_labels.to_string(index=False))
            f.write("\n\n")

            # Best Rated Labels
            top_100_best_rated_labels = label_stats.sort_values(by='average_rating_coeff', ascending=False).head(100)
            f.write("Top 100 Best Rated Labels (Average Rating Coefficient per Release):\n")
            f.write(top_100_best_rated_labels.to_string(index=False))
            f.write("\n\n")

        if 'artists_sort' in df.columns:
            df_artists_pop = df[['id', 'want', 'artists_sort']].copy()
            df_artists_pop['artist'] = df_artists_pop['artists_sort'].str.strip().replace('', 'Unknown')
            artist_popularity = df_artists_pop.groupby('artist').agg(
                average_want=('want', 'mean'),
                release_count=('artist', 'count')
            ).reset_index()
            top_100_pop_artists = artist_popularity.sort_values(by='average_want', ascending=False).head(100)

            f.write("Top 100 Artists by Popularity (Average Want Count per Release):\n")
            f.write(top_100_pop_artists.to_string(index=False))
            f.write("\n\n")

            df_artists_rate = df[['id', 'rating_coefficient', 'artists_sort']].copy()
            df_artists_rate['artist'] = df_artists_rate['artists_sort'].str.strip().replace('', 'Unknown')
            artist_rating = df_artists_rate.groupby('artist').agg(
                average_rating_coeff=('rating_coefficient', 'mean'),
                release_count=('artist', 'count')
            ).reset_index()
            top_100_best_rated_artists = artist_rating.sort_values(by='average_rating_coeff', ascending=False).head(100)

            f.write("Top 100 Best Rated Artists (Average Rating Coefficient per Release):\n")
            f.write(top_100_best_rated_artists.to_string(index=False))
            f.write("\n\n")

        top_countries = df['country'].value_counts().head(20)
        f.write("Top 20 Countries by Number of Releases:\n")
        f.write(top_countries.to_string())
        f.write("\n\n")

        top_formats = df['format'].str.split(', ').explode().value_counts().head(10)
        f.write("Top 10 Formats:\n")
        f.write(top_formats.to_string())
        f.write("\n\n")

        # Correlation
        numeric_cols = [
            'average_rating', 'rating_count', 'have', 'want',
            'rating_coefficient', 'demand_coefficient', 'gem_value',
            'num_for_sale', 'lowest_price', 'num_tracks'
        ]
        if 'total_duration_min' in df.columns:
            numeric_cols.append('total_duration_min')

        valid_cols = [col for col in numeric_cols if col in df.columns]
        if len(valid_cols) > 1:
            corr_matrix = df[valid_cols].corr()
            f.write("Key Correlations:\n")
            f.write(corr_matrix.to_string())
            f.write("\n\n")
        else:
            f.write("Insufficient numeric columns for correlation analysis.\n\n")

    def generate_extended_summary(df: pd.DataFrame, output_dir: str) -> None:
        """
        Generate an extended summary report with additional analysis (regression results,
        feature importances, and VIF).
        """
        logging.info("Generating extended summary report...")
        summary_path = os.path.join(output_dir, 'extended_summary_report.txt')
        generate_summary(df, output_dir)  # Generate basic summary first

        with open(summary_path, 'w') as f:
            f.write("=== Extended Discogs Database Analysis Summary ===\n\n")
            f.write("See 'summary_report.txt' for base statistics.\n\n")

            # Linear Regression
            reg_summary_file = os.path.join(output_dir, 'linear_regression_summary.txt')
            if os.path.exists(reg_summary_file):
                f.write("=== Multiple Linear Regression Insights ===\n")
                with open(reg_summary_file, 'r') as rf:
                    f.write(rf.read())
                f.write("\n\n")

            # RF Feature Importance
            rf_csv = os.path.join(output_dir, 'feature_importance.csv')
            if os.path.exists(rf_csv):
                f.write("=== Feature Importance (Random Forest) ===\n")
                rf_importance = pd.read_csv(rf_csv)
                f.write(rf_importance.head(20).to_string(index=False))
                f.write("\nSee 'feature_importance.png' for the corresponding plot.\n\n")

            # GBM Feature Importance
            gbm_csv = os.path.join(output_dir, 'gbm_feature_importance.csv')
            if os.path.exists(gbm_csv):
                f.write("=== Feature Importance (Gradient Boosting) ===\n")
                gbm_importance = pd.read_csv(gbm_csv)
                f.write(gbm_importance.head(20).to_string(index=False))
                f.write("\nSee 'gbm_feature_importance.png' for the corresponding plot.\n\n")

            # VIF
            vif_csv = os.path.join(output_dir, 'vif_data.csv')
            if os.path.exists(vif_csv):
                f.write("=== Variance Inflation Factor (VIF) ===\n")
                vif_data = pd.read_csv(vif_csv)
                f.write(vif_data.to_string(index=False))
                f.write("\nSee 'vif_plot.png' for the corresponding plot.\n\n")

        logging.info(f"Extended summary report generated at: {summary_path}")

###############################################################################
#                                    Main                                     #
###############################################################################
def main():
    parser = argparse.ArgumentParser(description='Analyze Discogs SQLite Database and Generate Insights.')
    parser.add_argument('input_db', help='Path to the SQLite database file (e.g., input_db.db)')
    parser.add_argument('--output', '-o', help='Directory to save analysis outputs', default='analysis_output')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Connect to the database
    conn = connect_db(args.input_db)

    # Load data
    df = load_data(conn)

    # Preprocess
    df = preprocess_data(df)

    # Close the DB connection
    conn.close()

    # Plotting and analyses
    plot_rating_distribution(df, args.output)
    plot_demand_gem_analysis(df, args.output)
    plot_genre_style_insights(df, args.output)
    plot_yearly_trends(df, args.output)
    plot_average_gem_value_over_years(df, args.output)
    plot_price_analysis(df, args.output)
    plot_geographical_distribution(df, args.output)
    plot_label_analysis(df, args.output)
    plot_tracklist_analysis(df, args.output)
    plot_have_want_correlation(df, args.output)
    plot_demand_vs_have_want(df, args.output)
    plot_price_vs_rating(df, args.output)
    plot_correlation_heatmap(df, args.output)
    plot_label_popularity_over_years(df, args.output)
    plot_most_popular_styles_over_years(df, args.output)
    plot_format_distribution(df, args.output)
    plot_wordclouds(df, args.output)
    plot_total_duration_per_release(df, args.output)
    plot_top_countries_over_years(df, args.output)
    plot_artists_analysis(df, args.output)
    plot_top100_best_rated_artists(df, args.output)
    plot_top100_best_rated_labels(df, args.output)
    plot_lowest_price_vs_demand_coefficient(df, args.output)
    plot_gem_value_over_lowest_price(df, args.output)
    plot_correlation_with_lowest_price(df, args.output)
    plot_pairwise_relationships(df, args.output)

    # Summaries
    generate_summary(df, args.output)
    calculate_vif(df, args.output)

    logging.info(f"\nAll analysis files have been saved in the '{args.output}' directory.")

if __name__ == "__main__":
    main()