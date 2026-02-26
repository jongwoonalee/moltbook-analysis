#!/usr/bin/env python3
"""
Download Reddit data from Hugging Face Pushshift dataset.
Streams data to avoid downloading the full 89GB dataset.
Filters to specific subreddits matching Moltbook topics.
"""

import sqlite3
from datetime import datetime, timezone
import os
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
DATABASE_FILE = str(SCRIPT_DIR / "reddit.db")

# Target subreddits matching Moltbook topics
TARGET_SUBREDDITS = {
    # General discussion
    'AskReddit', 'CasualConversation', 'self', 'misc',
    # Philosophy/consciousness
    'philosophy', 'consciousness', 'DeepThoughts', 'Showerthoughts',
    # Technology/AI
    'technology', 'artificial', 'MachineLearning', 'singularity',
    # Crypto/trading
    'cryptocurrency', 'CryptoCurrency', 'Bitcoin', 'ethtrader',
    # TIL/learning
    'todayilearned', 'explainlikeimfive', 'AskScience',
    # Misc popular
    'news', 'worldnews', 'science'
}

# Normalize to lowercase for matching
TARGET_SUBREDDITS_LOWER = {s.lower() for s in TARGET_SUBREDDITS}

TARGET_POSTS = 55000  # Match Moltbook size

def create_database():
    """Create the Reddit database with same schema as Moltbook."""
    print(f"Creating database at: {DATABASE_FILE}")

    # Remove existing file
    if os.path.exists(DATABASE_FILE):
        os.remove(DATABASE_FILE)

    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    # Create posts table matching Moltbook schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT,
            upvotes INTEGER,
            downvotes INTEGER,
            comment_count INTEGER,
            created_at TEXT,
            submolt_id TEXT,
            submolt_name TEXT,
            author_id TEXT,
            author_name TEXT
        )
    """)

    # Create submolts table (subreddits)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS submolts (
            id TEXT PRIMARY KEY,
            name TEXT,
            created_at TEXT,
            subscriber_count INTEGER
        )
    """)

    # Create comments table (empty, for schema compatibility)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS comments (
            id TEXT PRIMARY KEY,
            post_id TEXT NOT NULL,
            content TEXT,
            upvotes INTEGER,
            downvotes INTEGER,
            created_at TEXT,
            author_id TEXT,
            author_name TEXT,
            parent_comment_id TEXT,
            FOREIGN KEY (post_id) REFERENCES posts (id)
        )
    """)

    conn.commit()
    conn.close()
    print("Database created successfully")

def download_with_huggingface():
    """Download Reddit data using Hugging Face datasets library (streaming)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'datasets'])
        from datasets import load_dataset

    print("Connecting to Hugging Face dataset (streaming mode)...")
    print(f"Target subreddits: {', '.join(sorted(TARGET_SUBREDDITS))}")
    print(f"Target posts: {TARGET_POSTS}")
    print()

    create_database()

    # Stream the dataset
    dataset = load_dataset(
        "fddemarco/pushshift-reddit",
        split="train",
        streaming=True
    )

    posts_collected = 0
    posts_scanned = 0
    subreddit_counts = {}
    batch = []
    BATCH_SIZE = 500

    print("Streaming and filtering posts...")

    for post in dataset:
        posts_scanned += 1

        if posts_scanned % 100000 == 0:
            print(f"  Scanned {posts_scanned:,} posts, collected {posts_collected:,}...")

        # Check if subreddit matches
        subreddit = post.get('subreddit', '')
        if subreddit.lower() not in TARGET_SUBREDDITS_LOWER:
            continue

        # Skip deleted/removed posts
        author = post.get('author', '')
        if author in ['[deleted]', '[removed]', 'AutoModerator']:
            continue

        # Skip posts with no title
        title = post.get('title', '')
        if not title:
            continue

        # Convert timestamp
        created_utc = post.get('created_utc', 0)
        try:
            created_at = datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat()
        except:
            created_at = None

        # Add to batch
        batch.append((
            post.get('id', ''),
            title,
            post.get('selftext', ''),
            post.get('score', 0),
            0,  # Reddit doesn't expose downvotes separately
            post.get('num_comments', 0),
            created_at,
            post.get('subreddit_id', ''),
            subreddit,
            author,  # Reddit uses username as ID
            author
        ))

        posts_collected += 1
        subreddit_counts[subreddit] = subreddit_counts.get(subreddit, 0) + 1

        # Write batch to database
        if len(batch) >= BATCH_SIZE:
            write_batch(batch)
            print(f"  Collected {posts_collected:,} posts from {len(subreddit_counts)} subreddits")
            batch = []

        if posts_collected >= TARGET_POSTS:
            print(f"\nReached target of {TARGET_POSTS} posts!")
            break

    # Write remaining batch
    if batch:
        write_batch(batch)

    # Insert subreddit info
    print("\nInserting subreddit metadata...")
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    for subreddit, count in subreddit_counts.items():
        cursor.execute("""
            INSERT OR REPLACE INTO submolts (id, name, created_at, subscriber_count)
            VALUES (?, ?, ?, ?)
        """, (subreddit.lower(), subreddit, None, count))
    conn.commit()
    conn.close()

    print("\n" + "=" * 50)
    print("DOWNLOAD COMPLETE")
    print("=" * 50)
    print(f"Total posts scanned: {posts_scanned:,}")
    print(f"Total posts collected: {posts_collected:,}")
    print(f"Subreddits: {len(subreddit_counts)}")
    print("\nPosts per subreddit:")
    for sub, count in sorted(subreddit_counts.items(), key=lambda x: -x[1]):
        print(f"  {sub}: {count:,}")

    return posts_collected

def write_batch(batch):
    """Write a batch of posts to the database."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT OR REPLACE INTO posts
        (id, title, content, upvotes, downvotes, comment_count,
         created_at, submolt_id, submolt_name, author_id, author_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, batch)
    conn.commit()
    conn.close()

def main():
    print("=" * 60)
    print("REDDIT DATA DOWNLOAD")
    print("=" * 60)
    print()
    download_with_huggingface()

if __name__ == '__main__':
    main()
