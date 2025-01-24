#!/usr/bin/env python3

import requests
import time
import sys
import os
import json
import sqlite3
from getpass import getpass
from requests_oauthlib import OAuth1Session
from cryptography.fernet import Fernet
from tqdm import tqdm  # for progress bars

# Encryption Utilities
CREDENTIALS_FILE = 'credentials.enc'
KEY_FILE = 'key.key'

def calculate_bayesian_rating(average_rating, rating_count, global_mean=3.0, min_votes=10):
    """
    Returns a Bayesian-weighted rating for a release on a 0–5 scale.
    """
    v = rating_count
    R = average_rating
    m = min_votes
    C = global_mean

    if v == 0:
        return C

    WR = (v / (v + m)) * R + (m / (v + m)) * C
    return WR

def calculate_demand_coefficient(have_count, want_count):
    """
    Calculates the demand coefficient based on 'have' and 'want' counts.
    """
    have_count = max(have_count, 0)
    want_count = max(want_count, 0)

    if (have_count + want_count) == 0:
        return 0.0  # No have and no want

    return want_count / (have_count + 1)

def calculate_gem_value(rating_coefficient, demand_coefficient):
    """
    Combines rating_coefficient and demand_coefficient to compute gem_value.
    """
    return rating_coefficient * demand_coefficient

def load_or_create_key():
    """
    Load the encryption key from a file or create a new one if it doesn't exist.
    """
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, 'rb') as f:
            key = f.read()
    else:
        key = Fernet.generate_key()
        with open(KEY_FILE, 'wb') as f:
            f.write(key)
    return key

fernet = Fernet(load_or_create_key())

def save_credentials(consumer_key, consumer_secret, access_token, access_token_secret):
    """
    Encrypt and save the credentials to a file.
    """
    credentials = {
        'consumer_key': consumer_key,
        'consumer_secret': consumer_secret,
        'access_token': access_token,
        'access_token_secret': access_token_secret
    }
    data = json.dumps(credentials).encode()
    encrypted = fernet.encrypt(data)
    with open(CREDENTIALS_FILE, 'wb') as f:
        f.write(encrypted)
    print("Credentials saved securely.")

def load_credentials():
    """
    Load and decrypt the credentials from the file.
    """
    if not os.path.exists(CREDENTIALS_FILE):
        return None
    with open(CREDENTIALS_FILE, 'rb') as f:
        encrypted = f.read()
    try:
        decrypted = fernet.decrypt(encrypted)
        credentials = json.loads(decrypted.decode())
        return credentials
    except Exception as e:
        print(f"Failed to decrypt credentials: {e}")
        return None

# DiscogsAPI Class
class DiscogsAPI:
    REQUEST_TOKEN_URL = "https://api.discogs.com/oauth/request_token"
    AUTHORIZE_URL = "https://www.discogs.com/oauth/authorize"
    ACCESS_TOKEN_URL = "https://api.discogs.com/oauth/access_token"
    BASE_URL = "https://api.discogs.com"

    def __init__(self, consumer_key, consumer_secret,
                 access_token=None, access_token_secret=None,
                 user_agent="AdvancedSearchApp/1.0"):
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.user_agent = user_agent
        self.session = None

        if access_token and access_token_secret:
            # Use existing tokens
            self.access_token = access_token
            self.access_token_secret = access_token_secret
            self.authenticate()
        else:
            # Launch OAuth flow
            self.authenticate_oauth()

    def authenticate_oauth(self):
        """
        Perform OAuth 1.0a authentication to obtain access tokens.
        """
        oauth = OAuth1Session(
            self.consumer_key,
            client_secret=self.consumer_secret,
            callback_uri='oob'
        )
        try:
            fetch_response = oauth.fetch_request_token(self.REQUEST_TOKEN_URL)
        except Exception as e:
            print(f"Error obtaining request token: {e}")
            sys.exit(1)

        resource_owner_key = fetch_response.get('oauth_token')
        resource_owner_secret = fetch_response.get('oauth_token_secret')

        authorization_url = oauth.authorization_url(self.AUTHORIZE_URL)
        print(f"Please go to the following URL to authorize the application:\n{authorization_url}")
        verifier = input("Enter the verification code provided by Discogs: ")

        oauth = OAuth1Session(
            self.consumer_key,
            client_secret=self.consumer_secret,
            resource_owner_key=resource_owner_key,
            resource_owner_secret=resource_owner_secret,
            verifier=verifier,
        )

        try:
            oauth_tokens = oauth.fetch_access_token(self.ACCESS_TOKEN_URL)
        except Exception as e:
            print(f"Error obtaining access token: {e}")
            sys.exit(1)

        self.access_token = oauth_tokens.get('oauth_token')
        self.access_token_secret = oauth_tokens.get('oauth_token_secret')

        print("Authentication successful.")

    def authenticate(self):
        """
        Authenticate using existing access tokens.
        """
        self.session = OAuth1Session(
            self.consumer_key,
            client_secret=self.consumer_secret,
            resource_owner_key=self.access_token,
            resource_owner_secret=self.access_token_secret,
        )

    def get_headers(self):
        """
        Generate headers for authenticated requests.
        """
        return {
            "User-Agent": self.user_agent
        }

    def _apply_rate_limit(self, response, start_time):
        """
        Dynamically throttle requests based on Discogs Rate-Limit headers
        to avoid large downtimes and use maximum allowed throughput (60/min).
        """
        end_time = time.time()
        limit_header = response.headers.get("X-Discogs-Ratelimit")
        used_header = response.headers.get("X-Discogs-Ratelimit-Used")
        remain_header = response.headers.get("X-Discogs-Ratelimit-Remaining")

        try:
            limit = int(limit_header) if limit_header else 60
            used = int(used_header) if used_header else 0
            remaining = int(remain_header) if remain_header else limit - used
        except:
            limit = 60
            used = 0
            remaining = 60

        # Simple dynamic approach: if we're near the limit, wait longer.
        if remaining <= 1:
            # Sleep for 5 seconds if we're basically at the limit
            time.sleep(5)
        elif remaining < 10:
            # Sleep for 2 seconds if we're under 10 requests left
            time.sleep(2)
        else:
            # Otherwise, just sleep 1 second
            time.sleep(1)

    def search_releases(self, genres=None, styles=None, formats=None,
                        year=None, sort=None, sort_order="desc",
                        per_page=50, page=1):
        """
        Search for releases based on provided parameters (genre, style, format, year, etc.).
        Incorporates dynamic rate-limiting to stay under 60 requests/min.
        """
        search_url = f"{self.BASE_URL}/database/search"
        params = {}

        if genres:
            params['genre'] = ",".join(genres)
        if styles:
            params['style'] = ",".join(styles)
        if formats:
            params['format'] = ",".join(formats)
        if year:
            params['year'] = year

        if sort:
            params['sort'] = sort
            params['sort_order'] = sort_order

        # The user can request any 'per_page' number, though Discogs might cap it at 100.
        params['per_page'] = per_page
        params['page'] = page

        # Ensure we have an authenticated session
        if not self.session:
            self.authenticate()

        start_time = time.time()
        try:
            response = self.session.get(
                search_url,
                headers=self.get_headers(),
                params=params
            )

            if response.status_code == 200:
                data = response.json()
                self._apply_rate_limit(response, start_time)
                return data
            elif response.status_code == 429:
                # Rate-limited by Discogs
                retry_after = int(response.headers.get("Retry-After", 60))
                print(f"Rate limited. Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
                return self.search_releases(genres, styles, formats, year,
                                            sort, sort_order, per_page, page)
            else:
                print(f"Failed to search releases. Status Code: {response.status_code}")
                self._apply_rate_limit(response, start_time)
                return {}
        except requests.RequestException as e:
            print(f"An error occurred during the search: {e}")
            # Even on exception, try to respect rate limit if possible
            dummy_response = requests.Response()
            dummy_response.headers = {}
            self._apply_rate_limit(dummy_response, start_time)
            return {}

    def get_release_details(self, release_id):
        """
        Get detailed information about a specific release (for community rating/stats, plus videos).
        Incorporates dynamic rate-limiting to stay under 60 requests/min.
        """
        release_url = f"{self.BASE_URL}/releases/{release_id}"
        start_time = time.time()
        try:
            response = self.session.get(
                release_url,
                headers=self.get_headers()
            )
            if response.status_code == 200:
                data = response.json()
                self._apply_rate_limit(response, start_time)
                return data
            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                print(f"Rate limited. Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
                return self.get_release_details(release_id)
            else:
                print(f"Failed to get release details for ID {release_id}. Status Code: {response.status_code}")
                self._apply_rate_limit(response, start_time)
                return {}
        except requests.RequestException as e:
            print(f"An error occurred while fetching release details: {e}")
            dummy_response = requests.Response()
            dummy_response.headers = {}
            self._apply_rate_limit(dummy_response, start_time)
            return {}

# ---------- DB Functions ----------

def create_or_open_db(db_path):
    """
    Create or open a SQLite database, ensuring the needed tables exist.
    Added 'label' and 'country' columns.
    Also adds 'demand_coefficient' and 'gem_value' columns.
    Enforces UNIQUE constraint on 'release_id' to avoid duplicates.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create metadata table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY,
            last_page_fetched INTEGER,
            genres TEXT,
            styles TEXT,
            formats TEXT,
            year TEXT,
            sort TEXT,
            sort_order TEXT,
            per_page INTEGER,
            total_items INTEGER
        )
    """)

    # Create releases table with new columns and UNIQUE constraint on 'release_id'
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS releases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            release_id TEXT UNIQUE,
            title TEXT,
            year TEXT,
            genre TEXT,
            style TEXT,
            format TEXT,
            label TEXT,
            country TEXT,
            average_rating REAL,
            rating_count INTEGER,
            have INTEGER,
            want INTEGER,
            rating_coefficient REAL,
            link TEXT,
            youtube_links TEXT,
            demand_coefficient REAL,
            gem_value REAL,
            artists_sort TEXT,
            labels TEXT,
            num_for_sale INTEGER,
            lowest_price REAL,
            released TEXT,
            tracklist TEXT,
            extraartists TEXT,
            images TEXT,
            thumb TEXT
        )
    """)
    conn.commit()
    return conn

def load_metadata(conn):
    """
    Load metadata from the 'metadata' table (we expect only 1 row).
    Returns None if no metadata row is found.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM metadata WHERE id=1")
    row = cursor.fetchone()
    if not row:
        return None
    # row order:
    # (id, last_page_fetched, genres, styles, formats, year, sort, sort_order, per_page, total_items)
    return {
        'last_page_fetched': row[1],
        'genres': json.loads(row[2]) if row[2] else [],
        'styles': json.loads(row[3]) if row[3] else [],
        'formats': json.loads(row[4]) if row[4] else [],
        'year': row[5],
        'sort': row[6],
        'sort_order': row[7],
        'per_page': row[8],
        'total_items': row[9] if row[9] else 0
    }

def save_metadata(conn, metadata_dict):
    """
    Save/update metadata in the 'metadata' table, storing it in row with id=1.
    """
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO metadata (id, last_page_fetched,
            genres, styles, formats, year, sort, sort_order, per_page, total_items)
        VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        metadata_dict['last_page_fetched'],
        json.dumps(metadata_dict['genres']),
        json.dumps(metadata_dict['styles']),
        json.dumps(metadata_dict['formats']),
        metadata_dict['year'],
        metadata_dict['sort'],
        metadata_dict['sort_order'],
        metadata_dict['per_page'],
        metadata_dict['total_items']
    ))
    conn.commit()

def insert_release(conn, release_dict):
    """
    Insert a single release record into the 'releases' table.
    Includes label, country, demand_coefficient, and gem_value fields.
    Uses INSERT OR IGNORE to avoid duplicate entries based on 'release_id'.
    """
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO releases (
            release_id, title, year, genre, style, format,
            label, country,
            average_rating, rating_count, have, want,
            rating_coefficient, link, youtube_links,
            demand_coefficient, gem_value,
            artists_sort, labels, num_for_sale, lowest_price,
            released, tracklist, extraartists, images, thumb
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        release_dict.get("release_id", ""),
        release_dict.get("title", ""),
        release_dict.get("year", ""),
        release_dict.get("genre", ""),
        release_dict.get("style", ""),
        release_dict.get("format", ""),
        release_dict.get("label", ""),
        release_dict.get("country", ""),
        float(release_dict.get("average_rating", 0)),
        int(release_dict.get("rating_count", 0)),
        int(release_dict.get("have", 0)),
        int(release_dict.get("want", 0)),
        float(release_dict.get("rating_coefficient", 0)),
        release_dict.get("link", ""),
        release_dict.get("youtube_links", ""),
        float(release_dict.get("demand_coefficient", 0)),
        float(release_dict.get("gem_value", 0)),
        release_dict.get("artists_sort", ""),
        release_dict.get("labels", ""),
        int(release_dict.get("num_for_sale", 0)),
        float(release_dict.get("lowest_price", 0.0)),
        release_dict.get("released", ""),
        release_dict.get("tracklist", ""),
        release_dict.get("extraartists", ""),
        release_dict.get("images", ""),
        release_dict.get("thumb", "")
    ))
    conn.commit()

def load_all_releases(conn):
    """
    Return a list of all releases from the DB as dictionaries (for sorting, etc.).
    """
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM releases")
    rows = cursor.fetchall()
    releases = []
    for row in rows:
        # columns in 'releases' table:
        # 0 id, 1 release_id, 2 title, 3 year, 4 genre, 5 style, 6 format,
        # 7 label, 8 country, 9 avg_rating, 10 rating_count, 11 have,
        # 12 want, 13 rating_coeff, 14 link, 15 youtube_links,
        # 16 demand_coefficient, 17 gem_value,
        # 18 artists_sort, 19 labels, 20 num_for_sale, 21 lowest_price,
        # 22 released, 23 tracklist, 24 extraartists, 25 images, 26 thumb
        releases.append({
            "id": row[0],
            "release_id": row[1],
            "title": row[2],
            "year": row[3],
            "genre": row[4],
            "style": row[5],
            "format": row[6],
            "label": row[7],
            "country": row[8],
            "average_rating": str(row[9]),
            "rating_count": str(row[10]),
            "have": str(row[11]),
            "want": str(row[12]),
            "rating_coefficient": str(row[13]),
            "link": row[14],
            "youtube_links": row[15],
            "demand_coefficient": str(row[16]),
            "gem_value": str(row[17]),
            "artists_sort": row[18],
            "labels": row[19],
            "num_for_sale": str(row[20]),
            "lowest_price": str(row[21]),
            "released": row[22],
            "tracklist": row[23],
            "extraartists": row[24],
            "images": row[25],
            "thumb": row[26]
        })
    return releases

def delete_all_releases(conn):
    """
    Wipe out all release records in the DB (useful if user restarts a new search).
    """
    conn.execute("DELETE FROM releases")
    conn.commit()

def sort_and_rewrite_releases(conn, release_list):
    """
    Sort the 'release_list' by gem_value (descending),
    then rewrite the entire 'releases' table to maintain a consistent state.
    """
    delete_all_releases(conn)

    def sort_key(r):
        try:
            gv = float(r.get('gem_value', 0))
        except ValueError:
            gv = 0
        return gv

    release_list.sort(key=sort_key, reverse=True)

    for r in release_list:
        insert_release(conn, r)

# ---------- Prompt Functions ----------

def prompt_genres():
    """
    Instead of calling an API endpoint for genres, use a local list of common Discogs genres.
    """
    local_genres = [
        "Rock", "Electronic", "Hip Hop", "Jazz",
        "Funk / Soul", "Pop", "Classical", "Reggae",
        "Latin", "Non-Music", "Country"
    ]
    print("\nAvailable Genres:")
    for idx, genre in enumerate(local_genres, start=1):
        print(f"{idx}. {genre}")
    selected = input("Enter genre numbers separated by commas (e.g., 1,3,5) or press Enter to skip: ")
    if not selected.strip():
        return []
    try:
        indices = [int(num.strip()) for num in selected.split(',')]
        selected_genres = [local_genres[i-1] for i in indices if 0 < i <= len(local_genres)]
        return selected_genres
    except (ValueError, IndexError):
        print("Invalid input. No genres will be applied.")
        return []

def prompt_styles():
    """
    Discogs also has 'styles' that are sub-categories under genres.
    Allows selecting either by numeric index or typing in a custom style string.
    """
    local_styles = [
        "Alternative Rock", "House", "Techno", "Soul",
        "Funk", "Psychedelic Rock", "Dub", "Ska",
        "Ambient", "Minimal", "Swing"
    ]
    print("\nAvailable Styles:")
    for idx, style in enumerate(local_styles, start=1):
        print(f"{idx}. {style}")

    print("You can enter a mix of numeric choices and/or custom style names, separated by commas.")
    print("Examples: '2,3', or 'Ambient,Minimal,MyCustomStyle', or '2, MyCustomStyle'")

    selected = input("Enter style selections, or press Enter to skip: ")
    if not selected.strip():
        return []

    items = [item.strip() for item in selected.split(',')]
    selected_styles = []
    for item in items:
        # If it's purely a digit, treat it as an index
        if item.isdigit():
            index = int(item)
            if 1 <= index <= len(local_styles):
                selected_styles.append(local_styles[index - 1])
            else:
                print(f"Index {index} is out of range. Ignoring.")
        else:
            # Otherwise, treat it as a custom style string
            selected_styles.append(item)

    return selected_styles

def prompt_formats():
    """
    Instead of calling an API endpoint for formats, use a local list of common Discogs formats.
    """
    local_formats = [
        "Vinyl", "CD", "Album", "EP",
        "Compilation", "Single", "Cassette",
        "DVD", "File"
    ]
    print("\nAvailable Formats:")
    for idx, fmt in enumerate(local_formats, start=1):
        print(f"{idx}. {fmt}")
    selected = input("Enter format numbers separated by commas (e.g., 2,4) or press Enter to skip: ")
    if not selected.strip():
        return []
    try:
        indices = [int(num.strip()) for num in selected.split(',')]
        selected_formats = [local_formats[i-1] for i in indices if 0 < i <= len(local_formats)]
        return selected_formats
    except (ValueError, IndexError):
        print("Invalid input. No formats will be applied.")
        return []

def prompt_year():
    """
    Prompt the user to enter a specific year or a range (e.g. 1990-2000).
    """
    year_input = input("\nEnter release year or range (e.g., 1990 or 1990-2000) or press Enter to skip: ")
    if not year_input.strip():
        return None
    if '-' in year_input:
        parts = year_input.split('-')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return year_input
    elif year_input.isdigit():
        return year_input
    print("Invalid year format. No year filter will be applied.")
    return None

def prompt_sort():
    """
    Prompt the user to choose a sort field (rating, year, title) and order (asc, desc).
    """
    sort_fields = ["rating", "year", "title"]
    print("\nSort Options (Discogs side; final local sort is by gem_value):")
    for idx, field in enumerate(sort_fields, start=1):
        print(f"{idx}. {field.capitalize()}")
    selected = input("Choose a sort field by number or press Enter to skip sorting: ")
    if not selected.strip():
        return None, "desc"
    try:
        index = int(selected.strip())
        if 1 <= index <= len(sort_fields):
            sort_field = sort_fields[index - 1]
            order = input("Choose sort order: 1. Ascending 2. Descending (default is Descending): ")
            if order.strip() == '1':
                sort_order = "asc"
            else:
                sort_order = "desc"
            return sort_field, sort_order
    except (ValueError, IndexError):
        print("Invalid selection. No sorting will be applied.")
    return None, "desc"

def prompt_per_page():
    """
    Prompt the user to choose how many results per page.
    (No artificial max restriction. Discogs might still cap at 100.)
    """
    per_page_input = input("\nEnter number of results per page (default 50): ")
    if not per_page_input.strip():
        return 50
    try:
        per_page = int(per_page_input)
        if per_page < 1:
            print("Invalid number. Using default value of 50.")
            return 50
        return per_page
    except ValueError:
        print("Invalid input. Using default value of 50.")
        return 50

# ---------- Main Program ----------

def main():
    print("=== Discogs Advanced Search (No 100-Page Limit, DB Resume) ===")

    # 1) Check if user passed an existing DB file to resume
    db_file = None
    if len(sys.argv) > 1:
        db_file = sys.argv[1].strip()
    else:
        db_file = input("Enter path to SQLite DB file (or press Enter to use 'search_results.db'): ")
        if not db_file.strip():
            db_file = "search_results.db"

    # 2) Open (or create) the DB, read metadata if any
    conn = create_or_open_db(db_file)

    metadata = load_metadata(conn)
    if metadata:
        print(f"Found existing metadata in '{db_file}'. Resuming previous search parameters.")
    else:
        print(f"No metadata found in '{db_file}'. Will start a new search.")

    # 3) Load or prompt for Discogs credentials
    credentials = load_credentials()

    if credentials:
        consumer_key = credentials.get('consumer_key')
        consumer_secret = credentials.get('consumer_secret')
        access_token = credentials.get('access_token')
        access_token_secret = credentials.get('access_token_secret')
        print("Loaded encrypted credentials.")
        discogs = DiscogsAPI(
            consumer_key,
            consumer_secret,
            access_token=access_token,
            access_token_secret=access_token_secret,
            user_agent="AdvancedSearchApp/1.0 (your_email@example.com)"  # Update with your email
        )
    else:
        print("No saved credentials found. Please enter your Discogs API credentials.")
        consumer_key = input("Enter your Discogs Consumer Key: ").strip()
        consumer_secret = getpass("Enter your Discogs Consumer Secret: ").strip()
        # Trigger OAuth flow
        discogs = DiscogsAPI(consumer_key, consumer_secret, user_agent="AdvancedSearchApp/1.0 (your_email@example.com)")  # Update with your email
        access_token = discogs.access_token
        access_token_secret = discogs.access_token_secret
        # Save credentials
        save_credentials(consumer_key, consumer_secret, access_token, access_token_secret)
        print("Credentials saved. You won't need to enter them again.")
        # Reinstantiate to confirm
        discogs = DiscogsAPI(
            consumer_key,
            consumer_secret,
            access_token=access_token,
            access_token_secret=access_token_secret,
            user_agent="AdvancedSearchApp/1.0 (your_email@example.com)"  # Update with your email
        )

    # 4) If no metadata, prompt for new search criteria
    if not metadata:
        genres = prompt_genres()
        styles = prompt_styles()
        formats = prompt_formats()
        year = prompt_year()
        sort, sort_order = prompt_sort()
        per_page = prompt_per_page()

        print("\nStarting search (fetching page 1 to check if results exist and total count)...")
        first_page_results = discogs.search_releases(
            genres=genres,
            styles=styles,
            formats=formats,
            year=year,
            sort=sort,
            sort_order=sort_order,
            per_page=per_page,
            page=1
        )
        releases_page_1 = first_page_results.get('results', [])
        if not releases_page_1:
            print("No releases found for the given criteria.")
            next_action = input("\nEnter 'q' to quit or any other key to start a new search: ").strip().lower()
            if next_action == 'q':
                print("Exiting without saving (no data).")
                sys.exit(0)
            else:
                # Restart
                return

        # Get total items from pagination data (for the big progress bar)
        pagination_data = first_page_results.get('pagination', {})
        total_items = pagination_data.get('items', 0)

        # Clear any old releases in DB
        delete_all_releases(conn)

        # Create fresh metadata
        metadata = {
            "genres": genres,
            "styles": styles,
            "formats": formats,
            "year": year,
            "sort": sort,
            "sort_order": sort_order,
            "per_page": per_page,
            "last_page_fetched": 0,
            "total_items": total_items
        }
        save_metadata(conn, metadata)

    else:
        # If we have metadata, re-use it
        genres = metadata['genres']
        styles = metadata['styles']
        formats = metadata['formats']
        year = metadata['year']
        sort = metadata['sort']
        sort_order = metadata['sort_order']
        per_page = metadata['per_page']
        # total_items might be 0 if the code was older before we added it; handle that
        total_items = metadata.get('total_items', 0)

        # If total_items is still 0, let's do a quick call on page=1 to get an updated total.
        if total_items == 0:
            print("\nPreviously no total count was stored. Fetching page=1 to get total count...")
            first_page_results = discogs.search_releases(
                genres=genres,
                styles=styles,
                formats=formats,
                year=year,
                sort=sort,
                sort_order=sort_order,
                per_page=per_page,
                page=1
            )
            pagination_data = first_page_results.get('pagination', {})
            total_items = pagination_data.get('items', 0)
            metadata['total_items'] = total_items
            save_metadata(conn, metadata)

    # Current page to start from
    last_page_fetched = metadata['last_page_fetched']
    start_page = last_page_fetched + 1

    # 5) Load existing release data (for in-memory sorting, etc.)
    all_releases_data = load_all_releases(conn)
    already_fetched_count = len(all_releases_data)

    if total_items > 0:
        print(f"\nTotal results for this query: {total_items}")
    else:
        print("\nNote: Discogs did not provide a total count. Overall progress bar might not be accurate.")

    print(f"Resuming from page {start_page}... (Will continue until empty results)")

    # Create the overall progress bar if total_items > 0
    # Initialize at 'already_fetched_count' so the bar reflects progress so far.
    overall_bar = None
    if total_items > 0:
        overall_bar = tqdm(
            total=total_items,
            initial=already_fetched_count,
            desc="Overall Progress",
            unit="release"
        )

    page_counter = start_page
    while True:
        print(f"\nFetching page {page_counter} ...")
        try:
            search_results = discogs.search_releases(
                genres=genres,
                styles=styles,
                formats=formats,
                year=year,
                sort=sort,
                sort_order=sort_order,
                per_page=per_page,
                page=page_counter
            )
        except Exception as e:
            print(f"Error when requesting page {page_counter}: {e}")
            break

        releases = search_results.get('results', [])
        if not releases:
            print(f"No results found for page {page_counter}. Stopping.")
            # This means we've likely hit the end of available pages
            break

        # Page-level progress bar
        page_start_time = time.time()
        with tqdm(total=len(releases), desc=f"Processing page {page_counter}", unit="release") as release_bar:
            page_total_time = 0.0
            for i, release in enumerate(releases, start=1):
                release_start_time = time.time()

                release_id = release.get('id')
                title = release.get('title', 'N/A')
                release_year = release.get('year', 'Unknown')
                genre = ", ".join(release.get('genre', []))
                style = ", ".join(release.get('style', []))
                format_release = ", ".join(release.get('format', []))

                details = discogs.get_release_details(release_id)
                community_data = details.get('community', {})
                rating_data = community_data.get('rating', {})
                average_rating = rating_data.get('average', 0) or 0
                rating_count = rating_data.get('count', 0) or 0
                have = community_data.get('have', 0) or 0
                want = community_data.get('want', 0) or 0
                rating_coefficient = calculate_bayesian_rating(
                    float(average_rating),
                    int(rating_count),
                    global_mean=3.0,   # or any other value you prefer
                    min_votes=10
                )

                # Demand Coefficient: how "sought after" a release is
                # More want, fewer have => higher demand
                demand_coefficient = calculate_demand_coefficient(
                    float(have),
                    float(want)
                )

                # Gem Value: Combines rating_coefficient and demand_coefficient
                gem_value = calculate_gem_value(
                    rating_coefficient,
                    demand_coefficient
                )

                # Fetch YouTube video links
                videos = details.get('videos', [])
                youtube_urls = []
                for vid in videos:
                    uri = vid.get('uri', '')
                    if 'youtube.com' in uri.lower():
                        youtube_urls.append(uri)

                # Fetch label & country from release details
                country = details.get('country', '')
                labels_data = details.get('labels', [])
                label_names = [lbl.get('name', '') for lbl in labels_data]
                label_string = ", ".join(label_names)

                # Extract additional fields
                artists_sort = details.get('artists_sort', '')
                labels_full = json.dumps(labels_data)  # Store as JSON string
                num_for_sale = community_data.get('num_for_sale', 0) or 0

                # Handle 'lowest_price' which can be a float or dict
                lowest_price_data = details.get('lowest_price', 0.0)
                if isinstance(lowest_price_data, dict):
                    lowest_price = float(lowest_price_data.get('value', 0.0))
                elif isinstance(lowest_price_data, (int, float)):
                    lowest_price = float(lowest_price_data)
                else:
                    lowest_price = 0.0  # Default value if type is unexpected

                released = details.get('released', '')
                tracklist = json.dumps(details.get('tracklist', []))  # Store as JSON string
                extraartists = json.dumps(details.get('extraartists', []))  # Store as JSON string
                images = json.dumps(details.get('images', []))  # Store as JSON string
                thumb = details.get('thumb', '')

                record = {
                    "release_id": str(release_id),
                    "title": title,
                    "year": str(release_year),
                    "genre": genre,
                    "style": style,
                    "format": format_release,
                    "label": label_string,
                    "country": country,
                    "average_rating": str(average_rating),
                    "rating_count": str(rating_count),
                    "have": str(have),
                    "want": str(want),
                    "rating_coefficient": str(rating_coefficient),
                    "link": f"https://www.discogs.com/release/{release_id}",
                    "youtube_links": ", ".join(youtube_urls),
                    "demand_coefficient": str(demand_coefficient),
                    "gem_value": str(gem_value),
                    "artists_sort": artists_sort,
                    "labels": labels_full,
                    "num_for_sale": str(num_for_sale),
                    "lowest_price": str(lowest_price),
                    "released": released,
                    "tracklist": tracklist,
                    "extraartists": extraartists,
                    "images": images,
                    "thumb": thumb
                }

                # Append to the in-memory list
                all_releases_data.append(record)
                # Insert immediately into the DB
                insert_release(conn, record)

                # End of release processing
                release_end_time = time.time()
                release_time = release_end_time - release_start_time
                page_total_time += release_time

                # Update both progress bars
                release_bar.update(1)
                # Show average processing time per release in postfix:
                release_bar.set_postfix(avg_time=f"{page_total_time / i:.2f}s")

                if overall_bar:
                    overall_bar.update(1)

        page_end_time = time.time()
        elapsed_for_page = page_end_time - page_start_time
        print(f"Page {page_counter} completed in {elapsed_for_page:.2f}s total.")

        # Update metadata (no local sort step here)
        metadata['last_page_fetched'] = page_counter
        save_metadata(conn, metadata)

        page_counter += 1

    # Close the overall progress bar if created
    if overall_bar:
        overall_bar.close()

    if not all_releases_data:
        print("\nNo data collected. Exiting.")
        sys.exit(0)

    print(f"\nAll results have been written to '{db_file}' (SQLite database).")
    next_action = input("\nEnter 'q' to quit or any other key to start a new search (this will delete old data): ").strip().lower()
    if next_action == 'q':
        print("Goodbye!")
    else:
        delete_all_releases(conn)
        conn.execute("DELETE FROM metadata")
        conn.commit()
        main()

if __name__ == "__main__":
    main()