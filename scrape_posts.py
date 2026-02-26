
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

def insert_post(conn, post):
    """
    Inserts a post into the database.
    """
    sql = """ INSERT OR REPLACE INTO posts(id,title,content,upvotes,downvotes,comment_count,created_at,submolt_id,submolt_name,author_id,author_name)
              VALUES(?,?,?,?,?,?,?,?,?,?,?) """
    cur = conn.cursor()
    try:
        cur.execute(sql, (post['id'], post['title'], post.get('content'), post['upvotes'], post['downvotes'],
                          post['comment_count'], post['created_at'], post['submolt']['id'],
                          post['submolt']['name'], post['author']['id'], post['author']['name']))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error inserting post {post['id']}: {e}")

def get_submolts(conn):
    """
    Retrieves all submolts from the database, ordered by subscriber_count descending.
    """
    cur = conn.cursor()
    cur.execute("SELECT name, subscriber_count FROM submolts")
    rows = cur.fetchall()
    
    # Convert to list of dicts for easier sorting and access
    submolts_data = [{'name': row[0], 'subscriber_count': row[1]} for row in rows]
    
    # Sort by subscriber_count in descending order
    submolts_data.sort(key=lambda x: x['subscriber_count'], reverse=True)
    
    return submolts_data

def scrape_posts(api_key, submolt_limit=0, submolt_start_offset=0):
    """
    Scrapes posts from all submolts and stores them in the database.
    If submolt_limit is > 0, it limits the number of submolts processed.
    submolt_start_offset allows starting from a specific index in the submolt list.
    """
    conn = sqlite3.connect(DATABASE_FILE)
    headers = get_headers(api_key)
    
    all_submolts_data = get_submolts(conn)
    print(f"Found {len(all_submolts_data)} submolts in total.")

    submolts_to_process = all_submolts_data[submolt_start_offset:]
    print(f"Starting post scraping from submolt offset {submolt_start_offset}. Processing {len(submolts_to_process)} submolts.")

    processed_submolts_count = 0
    fatal_error_occurred = False # Flag to indicate if a non-429 error occurred

    for submolt_info in submolts_to_process:
        submolt_name = submolt_info['name'] # Correctly extract submolt_name
        if submolt_limit > 0 and processed_submolts_count >= submolt_limit:
            print(f"  Submolt processing limit ({submolt_limit}) reached. Stopping.")
            break
        
        
        print(f"Fetching posts for submolt: {submolt_name}")
        offset = 0
        limit = 100
        total_scraped_posts_for_submolt = 0 # Track posts scraped for current submolt

        non_429_retries = 0
        max_non_429_retries = 2
        non_429_retry_delays = [60, 300] # 1 minute, 5 minutes

        # Outer loop for retrying non-429 errors
        while non_429_retries <= max_non_429_retries:
            max_429_retries = 10
            retries = 0
            successful_request_for_page = False # Flag for successful request for the current page

            while retries < max_429_retries and not successful_request_for_page:
                try:
                    params = {'submolt': submolt_name, 'sort': 'new', 'limit': limit, 'offset': offset}
                    response = requests.get(f"{API_BASE_URL}/posts", headers=headers, params=params)
                    response.raise_for_status() # This will raise an HTTPError for bad responses (4xx or 5xx)
                    successful_request_for_page = True
                except requests.exceptions.HTTPError as http_err:
                    if http_err.response.status_code == 429:
                        retries += 1
                        print(f"  Rate limit exceeded (429) for posts in {submolt_name}, offset {offset}. Retry attempt {retries}/{max_429_retries}.")
                        retry_after_seconds = 30
                        try:
                            error_content = http_err.response.json()
                            hint_message = error_content.get('hint', '')
                            match = re.search(r'Try again in (\d+) seconds', hint_message)
                            if match:
                                retry_after_seconds = int(match.group(1))
                        except (json.JSONDecodeError, AttributeError):
                            pass
                        print(f"  Waiting for {retry_after_seconds} seconds before retrying...")
                        time.sleep(retry_after_seconds)
                    else: # Non-429 HTTP error
                        print(f"  HTTP error encountered for posts in {submolt_name} from URL {response.url}: {http_err}. Retrying non-429 error...")
                        if hasattr(http_err, 'response') and http_err.response is not None:
                            print(f"  Error Response Content: {http_err.response.text}")
                        
                        non_429_retries += 1
                        if non_429_retries <= max_non_429_retries:
                            wait_time = non_429_retry_delays[non_429_retries - 1]
                            print(f"  Waiting for {wait_time} seconds before retrying page {offset} due to non-429 error.")
                            time.sleep(wait_time)
                            continue # Continue the outer non-429 retry loop
                        else:
                            print(f"  Max non-429 retries ({max_non_429_retries}) exhausted for page {offset}. Fatal error, stopping all processing.")
                            fatal_error_occurred = True
                            break # Break out of inner 429 retry loop
                except requests.exceptions.RequestException as req_err: # Non-HTTP error
                    print(f"  Request error encountered for posts in {submolt_name}, offset {offset}: {req_err}. Retrying non-429 error...")
                    
                    non_429_retries += 1
                    if non_429_retries <= max_non_429_retries:
                        wait_time = non_429_retry_delays[non_429_retries - 1]
                        print(f"  Waiting for {wait_time} seconds before retrying page {offset} due to non-429 error.")
                        time.sleep(wait_time)
                        continue # Continue the outer non-429 retry loop
                    else:
                        print(f"  Max non-429 retries ({max_non_429_retries}) exhausted for page {offset}. Fatal error, stopping all processing.")
                        fatal_error_occurred = True
                        break # Break out of inner 429 retry loop
            
            if fatal_error_occurred:
                break # Break out of outer non-429 retry loop (propagates fatal error)
            
            if not successful_request_for_page:
                print(f"  Failed to make a successful request for posts in {submolt_name} at offset {offset} after multiple 429 retries. Moving to next page/submolt.")
                break # Break out of outer non-429 retry loop

            # If we reached here, the request was successful for the current page
            posts_data = response.json()
            handle_rate_limit(response.headers)

            if not posts_data or not posts_data.get('success'):
                print(f"  Failed to fetch posts for submolt {submolt_name} at offset {offset} or no more posts available. Stopping for this submolt.")
                break # Break out of inner page loop (move to next submolt)

            posts = posts_data.get('posts', [])
            if not posts:
                print(f"  No posts received for submolt {submolt_name} at offset {offset}. Stopping for this submolt.")
                break

            for post in posts:
                insert_post(conn, post)
                total_scraped_posts_for_submolt += 1
                print(f"    Scraped post: {post['title']} ({post['id']})")
            
            print(f"  Fetched {len(posts)} posts on this page for {submolt_name}. Total scraped for submolt: {total_scraped_posts_for_submolt}")

            # If fewer posts are returned than the limit, it's the last page
            if len(posts) < limit:
                print(f"  Fewer posts ({len(posts)}) than limit ({limit}) received for {submolt_name}. Assuming last page.")
                break
            
            offset += limit
            # Reset non_429_retries if a page request was successful
            non_429_retries = 0 
        
        if fatal_error_occurred: # If a fatal error occurred during page processing, break submolt loop
            break # Break out of submolt processing loop

        processed_submolts_count += 1 # Increment processed submolts count
        if fatal_error_occurred: # Check again after processing a submolt fully
            print(f"Fatal error detected. Processed {processed_submolts_count} submolts before stopping.")
            break # Break out of the submolt processing loop

    conn.close()
    if not fatal_error_occurred:
        print("Post scraping complete.")
    else:
        print("Post scraping terminated due to fatal error.")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Scrape posts from Moltbook submolts.")
    parser.add_argument("--test-submolts", type=int, default=0,
                        help="Limit the number of submolts to process for testing purposes. Set to 0 for no limit.")
    parser.add_argument("--submolt-start-offset", type=int, default=0,
                        help="Start processing submolts from this offset in the retrieved list. Set to 0 to start from the beginning.")
    args = parser.parse_args()

    conn = None # Initialize conn outside try block
    try:
        molt_api_key = get_api_key()
        scrape_posts(molt_api_key, args.test_submolts, args.submolt_start_offset)
    except (FileNotFoundError, ValueError) as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
