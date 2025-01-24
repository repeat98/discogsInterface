# Discogs Data Manager


## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Clean Releases Database](#1-clean-releases-database)
  - [2. Fetch Releases from Discogs](#2-fetch-releases-from-discogs)
  - [3. Scrape Data Without a Database](#3-scrape-data-without-a-database)
  - [4. Analyze Data](#4-analyze-data)
  - [5. Web Interface](#5-web-interface)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Access the Database

You can download the pre-scraped Discogs database from the following Google Drive folder:

🔗 [**Download Discogs Scraping Database**](https://drive.google.com/drive/folders/1qSxAppODHQZaoWAQWcRIxZ-z46sc_56E?usp=sharing)

## Overview

**Discogs Data Manager** is a comprehensive toolset designed to interact with the Discogs API, manage release data within a SQLite database, analyze the collected data, and present insights through a user-friendly web interface. Whether you're a music enthusiast, collector, or developer, this repository provides the necessary tools to efficiently handle and analyze your Discogs data.

## Features

- **Data Cleaning**: Clean and sanitize release data from the Discogs database.
- **Data Fetching**: Fetch missing release information from Discogs using input CSV files.
- **Data Scraping Without a Database**: Scrape release data directly to JSON or CSV files without the need for a SQLite database.
- **Data Analysis**: Analyze your collection to identify top artists, labels, and other valuable metrics.
- **Web Interface**: Visualize your data through an intuitive web interface with embedded JSON data.
- **Secure Credential Management**: Encrypt and securely store your Discogs API credentials.
- **Rate Limiting Compliance**: Automatically manages API rate limits to ensure seamless data fetching.

## Directory Structure

```
├── analysis
│   └── analyze.py
├── html_templates
│   ├── main_interface_template.html
│   ├── template_top_artists.html
│   └── template_top_labels.html
└── scraping
    ├── clean_db.py
    ├── discogs_with_input_db.py
    └── discogs_without_input_db.py
```

- **analysis/**: Contains scripts for analyzing the data stored in the SQLite database.
- **html_templates/**: Houses HTML templates for the web interface, including the main interface and specific sections like top artists and labels.
- **scraping/**: Includes scripts for cleaning the database, fetching data from Discogs, and scraping data without using a database.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/repeat98/discogsInterface.git
   cd discogsInterface
   ```

2. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   *If `requirements.txt` is not provided, install the following packages:*
   ```bash
   pip install sqlite3 argparse requests requests_oauthlib cryptography tqdm
   ```

## Usage

### 1. Clean Releases Database

The `clean_db.py` script cleans and sanitizes the releases database, ensuring consistency and removing duplicates or unwanted entries.

**Usage:**
```bash
python3 scraping/clean_db.py input_database.db output_cleaned_database.db
```

**Arguments:**
- `input_database.db`: Path to your existing SQLite database containing release data.
- `output_cleaned_database.db`: Path where the cleaned database will be saved.

**Example:**
```bash
python3 scraping/clean_db.py data/releases.db data/releases_cleaned.db
```

### 2. Fetch Releases from Discogs

The `discogs_with_input_db.py` script fetches missing release details from Discogs based on a provided CSV file containing release IDs and updates the SQLite database accordingly.

**Usage:**
```bash
python3 scraping/discogs_with_input_db.py -i path_to_input.csv path_to_database.db
```

**Arguments:**
- `-i`, `--input_csv`: Path to the input CSV file containing a `release_id` column.
- `path_to_database.db`: Path to the SQLite database where release details will be stored or updated.

**Example:**
```bash
python3 scraping/discogs_with_input_db.py -i data/missing_releases.csv data/releases.db
```

**Note:** The first time you run this script, you'll be prompted to enter your Discogs API credentials. These credentials are securely encrypted and stored for future use.

### 3. Scrape Data Without a Database

If you prefer to scrape and manage data without using a SQLite database, the `discogs_without_input_db.py` script allows you to scrape release data directly into JSON or CSV files. This approach is beneficial for simpler workflows or when you intend to process data immediately without persistent storage.

**Usage:**
```bash
python3 scraping/discogs_without_input_db.py -i path_to_input.csv -o output_data.json
```

**Arguments:**
- `-i`, `--input_csv`: Path to the input CSV file containing a `release_id` column.
- `-o`, `--output_json`: Path where the scraped JSON data will be saved.

**Example:**
```bash
python3 scraping/discogs_without_input_db.py -i data/missing_releases.csv -o data/scraped_releases.json
```

**Steps to Use the Script:**

1. **Prepare Input CSV:**
   
   Ensure your input CSV (`path_to_input.csv`) contains a column named `release_id` with the Discogs release IDs you wish to scrape.

   **Example CSV (`missing_releases.csv`):**
   ```csv
   release_id
   123456
   234567
   345678
   ```

2. **Run the Scraping Script:**
   
   Execute the script with the appropriate arguments.

   ```bash
   python3 scraping/discogs_without_input_db.py -i data/missing_releases.csv -o data/scraped_releases.json
   ```

3. **Provide Discogs API Credentials:**
   
   - If running for the first time, you'll be prompted to enter your Discogs `Consumer Key` and `Consumer Secret`.
   - Follow the OAuth flow by visiting the provided authorization URL and entering the verification code.
   - Credentials are encrypted and stored for future use.

4. **Access Scraped Data:**
   
   - The scraped data will be saved in the specified output file (`output_data.json`).
   - You can open this JSON file directly or convert it to CSV as needed.

**Sample Output (`scraped_releases.json`):**
```json
[
    {
        "release_id": "123456",
        "title": "Album Title",
        "year": "2020",
        "genre": "Rock, Alternative",
        "style": "Indie Rock",
        "format": "Vinyl, LP",
        "label": "Example Label",
        "country": "USA",
        "average_rating": "4.5",
        "rating_count": "10",
        "have": "5",
        "want": "15",
        "rating_coefficient": "3.75",
        "link": "https://www.discogs.com/release/123456",
        "youtube_links": "https://www.youtube.com/watch?v=example",
        "demand_coefficient": "3.0",
        "gem_value": "11.25",
        "artists_sort": "Artist Name",
        "labels": "[{\"name\": \"Example Label\", \"catno\": \"EX123\"}]",
        "num_for_sale": "2",
        "lowest_price": "25.00",
        "released": "2020-01-01",
        "tracklist": "[]",
        "extraartists": "[]",
        "images": "[]",
        "thumb": "https://img.discogs.com/example.jpg"
    },
    ...
]
```

**Benefits of Scraping Without a Database:**
- **Simplicity**: Directly obtaining JSON or CSV files can simplify workflows, especially for smaller datasets.
- **Portability**: JSON and CSV files are easily shareable and can be used across different platforms and applications.
- **Flexibility**: Allows for immediate data manipulation and integration with various data processing tools.

**Considerations:**
- **Data Persistence**: Unlike databases, JSON and CSV files may not handle large datasets as efficiently.
- **Concurrency**: Managing concurrent data access and modifications is more challenging without a database.
- **Scalability**: For extensive scraping tasks, databases offer better performance and organization.

### 4. Analyze Data

The `analyze.py` script in the `analysis/` directory provides tools to analyze the data within your SQLite database or JSON files, such as identifying top artists, labels, and other metrics.

**Usage with SQLite Database:**
```bash
python3 analysis/analyze.py path_to_database.db
```

**Example:**
```bash
python3 analysis/analyze.py data/releases.db
```

**Usage with JSON Data:**
If you've scraped data without a database, you can modify `analyze.py` to accept JSON input or create a separate analysis script tailored to JSON.

### 5. Web Interface

The `html_templates/` directory contains HTML templates for the web interface. These HTML files are designed to be opened directly in your web browser with embedded JSON data, eliminating the need for a server-side framework like Flask.

**Steps to Use the Web Interface:**

1. **Generate JSON Data:**
   
   - Use the `analyze.py` script or any other script to generate the necessary JSON data from your SQLite database or scraped JSON files.
   - Ensure that the JSON data is properly formatted and embedded into the HTML templates.

2. **Embed JSON Data into HTML Templates:**
   
   - Open the desired HTML template located in the `html_templates/` directory (e.g., `main_interface_template.html`).
   - Insert your JSON data directly into the HTML file, either by embedding it within `<script>` tags or by using inline JavaScript.

   **Example:**
   ```html
   <!DOCTYPE html>
   <html lang="en">
   <head>
       <meta charset="UTF-8">
       <title>Discogs Data Visualization</title>
       <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
   </head>
   <body>
       <h1>Top Artists</h1>
       <canvas id="topArtistsChart"></canvas>
       
       <script>
           const topArtistsData = {
               "labels": ["Artist 1", "Artist 2", "Artist 3"],
               "datasets": [{
                   "label": "Number of Releases",
                   "data": [50, 30, 20],
                   "backgroundColor": ["#FF6384", "#36A2EB", "#FFCE56"]
               }]
           };
           
           const ctx = document.getElementById('topArtistsChart').getContext('2d');
           const topArtistsChart = new Chart(ctx, {
               type: 'bar',
               data: topArtistsData,
               options: {
                   responsive: true,
                   scales: {
                       y: {
                           beginAtZero: true
                       }
                   }
               }
           });
       </script>
   </body>
   </html>
   ```

3. **Open HTML Files Directly:**
   
   - Navigate to the `html_templates/` directory.
   - Open the desired HTML file (e.g., `main_interface_template.html`) in your web browser.
   
   **Example:**
   - Double-click on `main_interface_template.html` or right-click and choose "Open with" followed by your preferred browser.

4. **View and Interact:**
   
   - The embedded JSON data will render visualizations and insights directly within the HTML page.
   - Ensure that your JSON data is kept up-to-date by re-running the analysis or scraping scripts as needed.

**Note:** Since the HTML files are opened directly, ensure that any embedded JavaScript does not rely on server-side processing. All data manipulations and visualizations should be handled client-side.

## Dependencies

The project relies on the following Python packages:

- `sqlite3`: For interacting with SQLite databases.
- `argparse`: For parsing command-line arguments.
- `requests`: For making HTTP requests to the Discogs API.
- `requests_oauthlib`: For handling OAuth authentication with Discogs.
- `cryptography`: For encrypting and decrypting credentials.
- `tqdm`: For displaying progress bars in the terminal.

*Ensure all dependencies are installed as per the [Installation](#installation) section.*

## Configuration

### Discogs API Credentials

To interact with the Discogs API, you'll need to obtain API credentials:

1. **Register an Application:**
   - Visit the [Discogs Developer Page](https://www.discogs.com/settings/developers).
   - Register a new application to obtain your `Consumer Key` and `Consumer Secret`.

2. **Provide Credentials:**
   - The first time you run the fetching or scraping script (`discogs_with_input_db.py` or `discogs_without_input_db.py`), you'll be prompted to enter your Discogs `Consumer Key` and `Consumer Secret`.
   - Follow the OAuth flow by visiting the provided authorization URL and entering the verification code.

3. **Secure Storage:**
   - Your credentials are encrypted and stored in `credentials.enc` using a symmetric key stored in `key.key`.
   - **Keep these files secure and do not share them.**

### Database

- The SQLite database will store all fetched release data (if using scripts that interact with the database).
- Ensure you have write permissions to the directory where the database is stored.

### JSON Output

- When using the `discogs_without_input_db.py` script, ensure that the output JSON file path provided has write permissions.
- The JSON structure can be customized within the scraping script as needed for your web interface.

## Contributing

Contributions are welcome! If you'd like to improve the project, please follow these steps:

1. **Fork the Repository**
2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/YourFeature
   ```
3. **Commit Your Changes**
   ```bash
   git commit -m "Add some feature"
   ```
4. **Push to the Branch**
   ```bash
   git push origin feature/YourFeature
   ```
5. **Open a Pull Request**

Please ensure that your code adheres to the project's coding standards and that you've included appropriate documentation and comments.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions, issues, or suggestions, please open an issue on the [GitHub repository](https://github.com/yourusername/discogs-data-manager/issues) or contact [jannik.assfalg@gmail.com](mailto:jannik.assfalg@gmail.com).

---

*Happy Collecting! 🎶*