import sqlite3
import json
import os
import re
import time
import requests
from datetime import datetime, timezone

DATABASE_FILE = "moltbook.db"
API_KEY_SCRIPT = "apikey.sh"
API_BASE_URL = "https://www.moltbook.com/api/v1"

def get_api_key(api_key_script=API_KEY_SCRIPT):
    """
    Reads the MOLT_APIKEY from the apikey.sh script.
    """
    try:
        with open(api_key_script, 'r') as f:
            content = f.read()
            match = re.search(r'export MOLT_APIKEY=(.*)', content)
            if match:
                return match.group(1).strip()
            else:
                raise ValueError(f"MOLT_APIKEY not found in {api_key_script}")
    except FileNotFoundError:
        raise FileNotFoundError(f"{api_key_script} not found. Please ensure it's in the current directory.")

def get_headers(api_key):
    """
    Returns the authorization headers for API requests.
    """
    return {"Authorization": f"Bearer {api_key}"}

def handle_rate_limit(response_headers):
    """
    Checks rate limit headers and pauses execution if needed.
    """
    rate_limit_remaining = int(response_headers.get('X-RateLimit-Remaining', 1))
    rate_limit_reset = int(response_headers.get('X-RateLimit-Reset', time.time()))

    # If we are close to the limit, or have exceeded it, pause until reset
    if rate_limit_remaining <= 1:
        time_to_wait = rate_limit_reset - time.time() + 1 # Add 1 second buffer
        if time_to_wait > 0:
            print(f"  Rate limit almost exceeded. Waiting for {time_to_wait:.2f} seconds until reset.")
            time.sleep(time_to_wait)
    else:
        # Otherwise, just a small delay to be a good citizen and avoid hammering the API
        time.sleep(0.1)

def insert_submolt(conn, submolt):
    """
    Inserts a submolt into the database.
    """
    sql = """ INSERT OR REPLACE INTO submolts(id, name, created_at, subscriber_count)
              VALUES(?,?,?,?) """
    cur = conn.cursor()
    try:
        cur.execute(sql, (submolt['id'], submolt['name'], submolt['created_at'], submolt.get('subscriber_count', 0)))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error inserting submolt {submolt['id']}: {e}")

def scrape_submolts(api_key, page_limit=0):
    """
    Scrapes submolts and their owners and stores them in the database.
    If page_limit is > 0, it limits the number of pages scraped.
    """
    conn = sqlite3.connect(DATABASE_FILE)
    headers = get_headers(api_key)

    # First, run the database script to make sure the tables are created.
    import database
    database.main()

    # List all submolts with pagination
    print("Fetching all submolts summaries (with pagination)...")
    total_scraped_submolts = 0
    offset = 0
    limit = 100 # Number of submolts per request
    current_page_count = 0
    
    while True: # Loop indefinitely until no more submolts are returned or limit reached
        if page_limit > 0 and current_page_count >= page_limit:
            print(f"  Page limit ({page_limit}) reached. Stopping pagination.")
            break
        print(f"--- Fetching submolts: Offset {offset}, Limit {limit} ---")
        
        max_retries = 10
        retries = 0
        successful_request = False

        while retries < max_retries and not successful_request:
            try:
                params = {'limit': limit, 'offset': offset}
                response = requests.get(f"{API_BASE_URL}/submolts", headers=headers, params=params)
                response.raise_for_status() # This will raise an HTTPError for bad responses (4xx or 5xx)
                successful_request = True
            except requests.exceptions.HTTPError as http_err:
                if http_err.response.status_code == 429:
                    retries += 1
                    print(f"  Rate limit exceeded (429). Retry attempt {retries}/{max_retries}.")
                    
                    retry_after_seconds = 30 # Default if hint not found
                    try:
                        error_content = http_err.response.json()
                        hint_message = error_content.get('hint', '')
                        match = re.search(r'Try again in (\d+) seconds', hint_message)
                        if match:
                            retry_after_seconds = int(match.group(1))
                    except (json.JSONDecodeError, AttributeError):
                        pass # Use default retry_after_seconds

                    print(f"  Waiting for {retry_after_seconds} seconds before retrying...")
                    time.sleep(retry_after_seconds)
                else:
                    # For other HTTP errors, break out of the retry loop and re-raise
                    print(f"  HTTP error encountered: {http_err}. Stopping pagination.")
                    break # Break out of inner retry loop
            except requests.exceptions.RequestException as req_err:
                # For non-HTTP request errors (e.g., connection error), just print and break
                print(f"  Request error encountered: {req_err}. Stopping pagination.")
                break
        
        if not successful_request:
            print("  Failed to make a successful request after multiple retries or due to a non-429 error. Stopping pagination.")
            break # Break out of outer pagination loop

        submolts_list_data = response.json()
        handle_rate_limit(response.headers) # Handle rate limiting after each request (proactive)

        if not submolts_list_data or not submolts_list_data.get('success'):
            print(f"Failed to fetch submolts summaries at offset {offset}. Stopping pagination.")
            break

        current_submolts_on_page = submolts_list_data.get('submolts', [])

        if not current_submolts_on_page:
            print("  No more submolts received. Stopping pagination.")
            break
        
        for submolt_summary in current_submolts_on_page:
            if submolt_summary and 'id' in submolt_summary:
                total_scraped_submolts += 1 # Increment before insert
                insert_submolt(conn, submolt_summary)
                print(f"    Scraped submolt: {submolt_summary['name']} (Subscribers: {submolt_summary.get('subscriber_count', 0)})")

        print(f"  Fetched {len(current_submolts_on_page)} submolt summaries on this page. Total scraped: {total_scraped_submolts}")

        # If fewer submolts are returned than the limit, it's the last page
        if len(current_submolts_on_page) < limit:
            print("  Fewer submolts than limit received. Assuming last page.")
            break
        
        offset += limit
        current_page_count += 1 # Increment page count
    
    print(f"Total unique submolts found: {total_scraped_submolts}")
    




    conn.close()
    print("Submolt scraping complete.")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Scrape submolts from Moltbook.")
    parser.add_argument("--test-pages", type=int, default=0,
                        help="Limit the number of pages scraped for testing purposes. Set to 0 for no limit.")
    args = parser.parse_args()

    conn = None # Initialize conn outside try block
    try:
        molt_api_key = get_api_key()
        scrape_submolts(molt_api_key, args.test_pages)
    except (FileNotFoundError, ValueError) as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()