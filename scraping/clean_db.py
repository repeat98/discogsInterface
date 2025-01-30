#!/usr/bin/env python3

import sqlite3
import argparse
import sys
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Clean releases database.')
    parser.add_argument('input_db', help='Path to input SQLite database')
    parser.add_argument('output_db', help='Path to output SQLite database')
    return parser.parse_args()

def create_output_table(cursor):
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS releases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            year TEXT,
            genre TEXT,
            style TEXT,
            label TEXT,
            country TEXT,
            average_rating REAL,
            rating_count INTEGER,
            have INTEGER,
            want INTEGER,
            rating_coeff REAL,
            link TEXT,
            youtube_links TEXT,
            demand_coeff REAL,
            gem_value REAL,
            lowest_price REAL,
            format TEXT
        )
    ''')

def fetch_rows(input_cursor):
    try:
        input_cursor.execute('''
            SELECT 
                id, 
                title, 
                year, 
                genre, 
                style, 
                label, 
                country, 
                average_rating, 
                rating_count, 
                have, 
                want, 
                rating_coefficient, 
                link, 
                youtube_links, 
                demand_coefficient, 
                gem_value, 
                lowest_price, 
                format, 
                artists_sort 
            FROM releases
        ''')
        return input_cursor.fetchall()
    except sqlite3.Error as e:
        print(f"Error fetching data from input database: {e}")
        sys.exit(1)

def process_youtube_links(youtube_links):
    if not youtube_links:
        return None
    # Assuming youtube_links are separated by commas, spaces, or another delimiter
    # Adjust the delimiter as per actual data format
    delimiters = [',', ' ', ';']
    for delimiter in delimiters:
        if delimiter in youtube_links:
            links = youtube_links.split(delimiter)
            return links[0].strip()
    return youtube_links.strip()

def process_title(title, artist):
    if not artist:
        return title
    artist = artist.strip()
    title = title.strip()
    if artist.lower() in title.lower():
        return title
    else:
        return f"{artist} - {title}"

def clean_and_insert_data(input_rows, output_cursor):
    for row in input_rows:
        (
            id_,
            title,
            year,
            genre,
            style,
            label,
            country,
            average_rating,
            rating_count,
            have,
            want,
            rating_coefficient,
            link,
            youtube_links,
            demand_coefficient,
            gem_value,
            lowest_price,
            format_,
            artists_sort
        ) = row

        # Process YouTube links
        first_youtube_link = process_youtube_links(youtube_links)

        # Process title
        cleaned_title = process_title(title, artists_sort)

        # Prepare data for insertion
        cleaned_row = (
            id_,                # Retain the same ID
            cleaned_title,
            year,
            genre,
            style,
            label,
            country,
            average_rating,
            rating_count,
            have,
            want,
            rating_coefficient,
            link,
            first_youtube_link,
            demand_coefficient,
            gem_value,
            lowest_price,
            format_
        )

        try:
            output_cursor.execute('''
                INSERT INTO releases (
                    id,
                    title,
                    year,
                    genre,
                    style,
                    label,
                    country,
                    average_rating,
                    rating_count,
                    have,
                    want,
                    rating_coeff,
                    link,
                    youtube_links,
                    demand_coeff,
                    gem_value,
                    lowest_price,
                    format
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', cleaned_row)
        except sqlite3.IntegrityError as e:
            print(f"Skipping row with id {id_} due to integrity error: {e}")
        except sqlite3.Error as e:
            print(f"Error inserting row with id {id_}: {e}")

def main():
    args = parse_arguments()

    # Check if input_db exists
    if not os.path.isfile(args.input_db):
        print(f"Input database '{args.input_db}' does not exist.")
        sys.exit(1)

    # Connect to input and output databases
    try:
        input_conn = sqlite3.connect(args.input_db)
        input_cursor = input_conn.cursor()
    except sqlite3.Error as e:
        print(f"Error connecting to input database: {e}")
        sys.exit(1)

    try:
        output_conn = sqlite3.connect(args.output_db)
        output_cursor = output_conn.cursor()
    except sqlite3.Error as e:
        print(f"Error connecting to output database: {e}")
        input_conn.close()
        sys.exit(1)

    # Create output table
    create_output_table(output_cursor)

    # Fetch rows from input database
    input_rows = fetch_rows(input_cursor)

    # Clean and insert data into output database
    clean_and_insert_data(input_rows, output_cursor)

    # Commit and close connections
    output_conn.commit()
    input_conn.close()
    output_conn.close()

    print(f"Database cleaning completed. Cleaned data saved to '{args.output_db}'.")

if __name__ == "__main__":
    main()