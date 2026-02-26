import sqlite3
import json
import os
import re
import time
import requests
from datetime import datetime, timezone
import argparse

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

def insert_comment(conn, comment, post_id):
    """
    Inserts a comment into the database.
    """
    sql = """ INSERT OR REPLACE INTO comments(id,post_id,content,upvotes,downvotes,created_at,author_id,author_name,parent_comment_id)
              VALUES(?,?,?,?,?,?,?,?,?) """
    cur = conn.cursor()
    try:
        cur.execute(sql, (comment['id'], post_id, comment['content'], comment['upvotes'], comment['downvotes'],
                          comment['created_at'], comment['author']['id'], comment['author']['name'],
                          comment.get('parent_comment_id'))) # parent_comment_id might be null
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error inserting comment {comment['id']}: {e}")

def get_posts(conn):
    """
    Retrieves all post IDs from the database.
    """
    cur = conn.cursor()
    cur.execute("SELECT id FROM posts")
    rows = cur.fetchall()
    return [row[0] for row in rows]

def scrape_comments(api_key, post_limit=0, post_start_offset=0):
    """
    Scrapes comments for all posts and stores them in the database.
    If post_limit is > 0, it limits the number of posts processed.
    post_start_offset allows starting from a specific index in the post list.
    """
    conn = sqlite3.connect(DATABASE_FILE)
    headers = get_headers(api_key)
    
    all_posts = get_posts(conn)
    print(f"Found {len(all_posts)} posts in total.")

    posts_to_process = all_posts[post_start_offset:]
    print(f"Starting comment scraping from post offset {post_start_offset}. Processing {len(posts_to_process)} posts.")

    processed_posts_count = 0
    fatal_error_occurred = False # Flag to indicate if a non-429 error occurred

    for post_id in posts_to_process:
        if post_limit > 0 and processed_posts_count >= post_limit:
            print(f"  Post processing limit ({post_limit}) reached. Stopping.")
            break

        print(f"Fetching comments for post: {post_id}")
        
        offset = 0
        limit = 100 # Number of comments per request
        total_scraped_comments_for_post = 0

        non_429_retries = 0
        max_non_429_retries = 2
        non_429_retry_delays = [60, 300] # 1 minute, 5 minutes

        while non_429_retries <= max_non_429_retries: # Outer loop for retrying non-429 errors
            max_429_retries = 10
            retries = 0
            successful_request_for_page = False # Flag for successful request for the current page

            while retries < max_429_retries and not successful_request_for_page:
                try:
                    params = {'sort': 'new', 'limit': limit, 'offset': offset}
                    response = requests.get(f"{API_BASE_URL}/posts/{post_id}/comments", headers=headers, params=params)
                    response.raise_for_status()
                    successful_request_for_page = True
                except requests.exceptions.HTTPError as http_err:
                    if http_err.response.status_code == 429:
                        retries += 1
                        print(f"  Rate limit exceeded (429) for comments on post {post_id}, offset {offset}. Retry attempt {retries}/{max_429_retries}.")
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
                        print(f"  HTTP error encountered for comments on post {post_id}, offset {offset} from URL {response.url}: {http_err}. Retrying non-429 error...")
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
                    print(f"  Request error encountered for comments on post {post_id}, offset {offset}: {req_err}. Retrying non-429 error...")
                    
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
                print(f"  Failed to make a successful request for comments on post {post_id} at offset {offset} after multiple 429 retries. Moving to next page/post.")
                break # Break out of outer non-429 retry loop

            # If we reached here, the request was successful for the current page
            comments_data = response.json()
            handle_rate_limit(response.headers)

            if not comments_data or not comments_data.get('success'):
                print(f"  Failed to fetch comments for post {post_id} at offset {offset} or no comments available. Stopping for this post.")
                break

            comments = comments_data.get('comments', [])
            if not comments:
                print(f"  No comments received for post {post_id} at offset {offset}. Stopping for this post.")
                break

            for comment in comments:
                insert_comment(conn, comment, post_id)
                total_scraped_comments_for_post += 1
                print(f"    Scraped comment: {comment['id']}")
            
            print(f"  Fetched {len(comments)} comments on this page for post {post_id}. Total scraped for post: {total_scraped_comments_for_post}")

            # If fewer comments are returned than the limit, it's the last page
            if len(comments) < limit:
                print(f"  Fewer comments ({len(comments)}) than limit ({limit}) received for post {post_id}. Assuming last page.")
                break
            
            offset += limit
            # Reset non_429_retries if a page request was successful
            non_429_retries = 0 
        
        if fatal_error_occurred: # If a fatal error occurred during page processing, break post loop
            break # Break out of post processing loop

        processed_posts_count += 1 # Increment processed posts count
        if fatal_error_occurred: # Check again after processing a post fully
            print(f"Fatal error detected. Processed {processed_posts_count} posts before stopping.")
            break # Break out of the post processing loop

    conn.close()
    if not fatal_error_occurred:
        print("Comment scraping complete.")
    else:
        print("Comment scraping terminated due to fatal error.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Scrape comments from Moltbook posts.")
    parser.add_argument("--test-posts", type=int, default=0,
                        help="Limit the number of posts to process for testing purposes. Set to 0 for no limit.")
    parser.add_argument("--post-start-offset", type=int, default=0,
                        help="Start processing posts from this offset in the retrieved list. Set to 0 to start from the beginning.")
    args = parser.parse_args()

    conn = None # Initialize conn outside try block
    try:
        molt_api_key = get_api_key()
        scrape_comments(molt_api_key, args.test_posts, args.post_start_offset)
    except (FileNotFoundError, ValueError) as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()