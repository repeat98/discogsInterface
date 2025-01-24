#!/usr/bin/env python3

import argparse
import csv
import json
import os
import sqlite3
import sys
import time
from getpass import getpass
from tqdm import tqdm
import requests
from requests_oauthlib import OAuth1Session
from cryptography.fernet import Fernet

# ------------------------------------------------------------------------------------
# 1) Encryption / Credentials Logic
# ------------------------------------------------------------------------------------

CREDENTIALS_FILE = 'credentials.enc'
KEY_FILE = 'key.key'

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
    Encrypt and save Discogs OAuth credentials to disk.
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
    Load and decrypt Discogs OAuth credentials from disk.
    Returns None if the file doesn't exist or decryption fails.
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

# ------------------------------------------------------------------------------------
# 2) Discogs API Logic
# ------------------------------------------------------------------------------------

class DiscogsAPI:
    REQUEST_TOKEN_URL = "https://api.discogs.com/oauth/request_token"
    AUTHORIZE_URL = "https://www.discogs.com/oauth/authorize"
    ACCESS_TOKEN_URL = "https://api.discogs.com/oauth/access_token"
    BASE_URL = "https://api.discogs.com"

    def __init__(self, consumer_key, consumer_secret,
                 access_token=None, access_token_secret=None,
                 user_agent="AdvancedSearchApp/1.0 (example@example.com)"):
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.user_agent = user_agent
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        self.session = None

        # If access tokens exist, skip the OAuth flow and just authenticate.
        if self.access_token and self.access_token_secret:
            self.authenticate()
        else:
            self.authenticate_oauth()

    def authenticate_oauth(self):
        """
        Perform OAuth 1.0a authentication to obtain access tokens.
        """
        print("Starting OAuth authentication flow...")
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
        print(f"\nPlease go to the following URL to authorize this application:\n{authorization_url}")
        verifier = input("Enter the verification code provided by Discogs: ")

        oauth = OAuth1Session(
            self.consumer_key,
            client_secret=self.consumer_secret,
            resource_owner_key=resource_owner_key,
            resource_owner_secret=resource_owner_secret,
            verifier=verifier
        )

        try:
            oauth_tokens = oauth.fetch_access_token(self.ACCESS_TOKEN_URL)
        except Exception as e:
            print(f"Error obtaining access token: {e}")
            sys.exit(1)

        self.access_token = oauth_tokens.get('oauth_token')
        self.access_token_secret = oauth_tokens.get('oauth_token_secret')

        print("Authentication successful. Access tokens obtained.")
        self.authenticate()

    def authenticate(self):
        """
        Initialize the session with our known OAuth tokens.
        """
        self.session = OAuth1Session(
            self.consumer_key,
            client_secret=self.consumer_secret,
            resource_owner_key=self.access_token,
            resource_owner_secret=self.access_token_secret
        )

    def get_headers(self):
        return {
            "User-Agent": self.user_agent
        }

    def _apply_rate_limit(self, response, start_time):
        """
        Throttle requests based on Discogs Rate-Limit headers (60 requests/min).
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

        # Simple dynamic approach
        if remaining <= 1:
            time.sleep(5)  # near limit, sleep more
        elif remaining < 10:
            time.sleep(2)
        else:
            time.sleep(1)

    def get_release_details(self, release_id):
        """
        Fetch detailed release info from Discogs by release ID.
        """
        url = f"{self.BASE_URL}/releases/{release_id}"
        start_time = time.time()

        try:
            response = self.session.get(url, headers=self.get_headers())
            if response.status_code == 200:
                data = response.json()
                self._apply_rate_limit(response, start_time)
                return data
            elif response.status_code == 429:
                # Rate-limited
                retry_after = int(response.headers.get("Retry-After", 60))
                print(f"Rate limited. Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
                return self.get_release_details(release_id)
            else:
                print(f"Failed to get release {release_id} details. Status {response.status_code}.")
                self._apply_rate_limit(response, start_time)
                return {}
        except requests.RequestException as e:
            print(f"Error getting release details for {release_id}: {e}")
            # Attempt to respect rate limit
            dummy_response = requests.Response()
            dummy_response.headers = {}
            self._apply_rate_limit(dummy_response, start_time)
            return {}

# ------------------------------------------------------------------------------------
# 3) Rating / Calculation Helpers
# ------------------------------------------------------------------------------------

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
    want / (have+1) as a simple measure of "demand".
    """
    have_count = max(have_count, 0)
    want_count = max(want_count, 0)
    if (have_count + want_count) == 0:
        return 0.0
    return want_count / (have_count + 1)

def calculate_gem_value(rating_coefficient, demand_coefficient):
    """
    Combine rating and demand to get a "gem value".
    """
    return rating_coefficient * demand_coefficient

# ------------------------------------------------------------------------------------
# 4) SQLite DB Functions
# ------------------------------------------------------------------------------------

def create_or_open_db(db_path):
    """
    Create (or open) the SQLite DB with a 'releases' table.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
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

def insert_release(conn, release_dict):
    """
    Insert a single release record into the 'releases' table, ignoring duplicates.
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

def load_existing_release_ids(db_path):
    """
    Return a set of all release_ids in the 'releases' table of the given db_path.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT release_id FROM releases")
    rows = cursor.fetchall()
    conn.close()

    # Convert to a set of strings
    return {str(row[0]) for row in rows}

# ------------------------------------------------------------------------------------
# 5) Main Logic
# ------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch missing release IDs from a CSV and add them to a single Discogs DB."
    )
    parser.add_argument("-i", "--input_csv", required=True,
                        help="Path to input CSV (must have a 'release_id' column).")
    parser.add_argument("db_path", help="Path to the SQLite DB (created if it doesn't exist).")
    args = parser.parse_args()

    # 1) Read release IDs from the CSV
    csv_release_ids = []
    with open(args.input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get('release_id', '').strip()
            # Only consider numeric IDs
            if rid.isdigit():
                csv_release_ids.append(rid)

    if not csv_release_ids:
        print("No valid release_ids found in the CSV. Exiting.")
        sys.exit(0)

    # 2) Create or open the DB
    conn = create_or_open_db(args.db_path)

    # 3) Determine which release IDs from CSV are NOT in the DB
    existing_ids = load_existing_release_ids(args.db_path)
    missing_ids = [rid for rid in csv_release_ids if rid not in existing_ids]

    if not missing_ids:
        print("All CSV releases are already present in the DB. No new data to fetch.")
        conn.close()
        sys.exit(0)

    print(f"Found {len(missing_ids)} missing release IDs. Will fetch from Discogs...")

    # 4) Discogs Authentication
    credentials = load_credentials()
    if credentials:
        consumer_key = credentials['consumer_key']
        consumer_secret = credentials['consumer_secret']
        access_token = credentials['access_token']
        access_token_secret = credentials['access_token_secret']
        discogs = DiscogsAPI(
            consumer_key, consumer_secret,
            access_token=access_token,
            access_token_secret=access_token_secret,
            user_agent="AdvancedSearchApp/1.0 (example@example.com)"
        )
    else:
        print("No saved Discogs credentials found. Please enter your Consumer Key/Secret.")
        consumer_key = input("Consumer Key: ").strip()
        consumer_secret = getpass("Consumer Secret: ").strip()
        discogs = DiscogsAPI(
            consumer_key,
            consumer_secret,
            user_agent="AdvancedSearchApp/1.0 (example@example.com)"
        )
        # Save newly obtained tokens so we won't need to re-enter them next time
        save_credentials(
            consumer_key,
            consumer_secret,
            discogs.access_token,
            discogs.access_token_secret
        )

    # 5) For each missing release ID, fetch details from Discogs and insert into the DB
    print("Fetching missing releases from Discogs:")
    for rid in tqdm(missing_ids, desc="Missing Releases", unit="release"):
        details = discogs.get_release_details(rid)
        if not details:
            # If the API returned nothing, skip
            continue

        # Basic fields
        title = details.get('title', 'N/A')
        release_year = details.get('year', 'Unknown')
        genre_list = details.get('genres', [])
        style_list = details.get('styles', [])
        format_info = details.get('formats', [])
        if isinstance(format_info, list):
            format_names = []
            for fmt_item in format_info:
                fmt_name = fmt_item.get('name', '')
                if fmt_name:
                    format_names.append(fmt_name)
            format_str = ", ".join(format_names)
        else:
            format_str = ""

        genre_str = ", ".join(genre_list) if isinstance(genre_list, list) else ""
        style_str = ", ".join(style_list) if isinstance(style_list, list) else ""

        # Community data
        community_data = details.get('community', {})
        rating_data = community_data.get('rating', {})
        average_rating = rating_data.get('average', 0) or 0
        rating_count = rating_data.get('count', 0) or 0
        have = community_data.get('have', 0) or 0
        want = community_data.get('want', 0) or 0

        # Bayesian rating
        rating_coefficient = calculate_bayesian_rating(
            float(average_rating),
            int(rating_count),
            global_mean=3.0,
            min_votes=10
        )

        # Demand
        demand_coefficient = calculate_demand_coefficient(have, want)

        # Gem value
        gem_value = calculate_gem_value(rating_coefficient, demand_coefficient)

        # Videos
        videos = details.get('videos', [])
        youtube_urls = []
        for vid in videos:
            uri = vid.get('uri', '')
            if 'youtube.com' in uri.lower():
                youtube_urls.append(uri)

        # Label & country
        country = details.get('country', '')
        labels_data = details.get('labels', [])
        label_names = [lbl.get('name', '') for lbl in labels_data]
        label_string = ", ".join(label_names)

        # Additional info
        artists_sort = details.get('artists_sort', '')
        labels_full = json.dumps(labels_data)
        num_for_sale = community_data.get('num_for_sale', 0) or 0

        # Handle lowest_price
        lowest_price_data = details.get('lowest_price', 0.0)
        if isinstance(lowest_price_data, dict):
            lowest_price = float(lowest_price_data.get('value', 0.0) or 0.0)
        else:
            lowest_price = float(lowest_price_data or 0.0)

        released = details.get('released', '')
        tracklist = json.dumps(details.get('tracklist', []))
        extraartists = json.dumps(details.get('extraartists', []))
        images = json.dumps(details.get('images', []))
        thumb = details.get('thumb', '')

        # Build a record dict
        record = {
            "release_id": str(rid),
            "title": title,
            "year": str(release_year),
            "genre": genre_str,
            "style": style_str,
            "format": format_str,
            "label": label_string,
            "country": country,
            "average_rating": str(average_rating),
            "rating_count": str(rating_count),
            "have": str(have),
            "want": str(want),
            "rating_coefficient": str(rating_coefficient),
            "link": f"https://www.discogs.com/release/{rid}",
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

        insert_release(conn, record)

    conn.close()

    print("\nAll done!")
    print(f"The DB at '{args.db_path}' now contains all CSV releases plus {len(missing_ids)} newly-fetched releases.")
    
if __name__ == "__main__":
    main()