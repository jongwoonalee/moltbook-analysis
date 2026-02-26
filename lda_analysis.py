#!/usr/bin/env python3
"""
LDA Topic Modeling: Moltbook (AI) vs Reddit (Human)
===================================================
Discovers latent topics in each platform's content.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

OUTPUT_DIR = "text_analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_posts(db_file, platform_name):
    """Load posts from database."""
    conn = sqlite3.connect(db_file)
    df = pd.read_sql_query("""
        SELECT id, title, content, submolt_name
        FROM posts
    """, conn)
    conn.close()

    df['text'] = df['title'].fillna('') + ' ' + df['content'].fillna('')
    df['platform'] = platform_name
    print(f"Loaded {len(df)} posts from {platform_name}")
    return df

def clean_text(text):
    """Clean text for LDA."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text

def run_lda(texts, platform_name, n_topics=8, n_top_words=10):
    """Run LDA topic modeling."""
    print(f"\nRunning LDA for {platform_name} with {n_topics} topics...")

    # Vectorize
    vectorizer = CountVectorizer(
        max_df=0.85,
        min_df=10,
        stop_words='english',
        max_features=5000
    )

    doc_term_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    print(f"  Vocabulary size: {len(feature_names)}")
    print(f"  Documents: {doc_term_matrix.shape[0]}")

    # Fit LDA
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=20,
        learning_method='online',
        random_state=42,
        n_jobs=-1
    )

    lda.fit(doc_term_matrix)

    # Extract topics
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        top_weights = [topic[i] for i in top_words_idx]
        topics.append({
            'topic_id': topic_idx,
            'words': top_words,
            'weights': top_weights
        })
        print(f"  Topic {topic_idx}: {', '.join(top_words[:5])}")

    return topics, lda, vectorizer, doc_term_matrix

def plot_topics(moltbook_topics, reddit_topics):
    """Create topic visualization."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 14))

    # Moltbook topics
    ax = axes[0]
    n_topics = len(moltbook_topics)
    y_positions = np.arange(n_topics)

    for i, topic in enumerate(moltbook_topics):
        words = topic['words'][:8]
        label = f"Topic {i}: " + ", ".join(words[:3])
        ax.barh(i, 1, color=plt.cm.Blues(0.3 + 0.7 * i / n_topics), alpha=0.8)
        ax.text(0.02, i, ", ".join(words), va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"Topic {i}" for i in range(n_topics)])
    ax.set_xlim(0, 1)
    ax.set_xlabel('')
    ax.set_title('Moltbook (AI Agents) - LDA Topics', fontsize=14, fontweight='bold', color='#3498db')
    ax.invert_yaxis()
    ax.set_xticks([])

    # Reddit topics
    ax = axes[1]
    n_topics = len(reddit_topics)

    for i, topic in enumerate(reddit_topics):
        words = topic['words'][:8]
        ax.barh(i, 1, color=plt.cm.Reds(0.3 + 0.7 * i / n_topics), alpha=0.8)
        ax.text(0.02, i, ", ".join(words), va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(np.arange(n_topics))
    ax.set_yticklabels([f"Topic {i}" for i in range(n_topics)])
    ax.set_xlim(0, 1)
    ax.set_xlabel('')
    ax.set_title('Reddit (Humans) - LDA Topics', fontsize=14, fontweight='bold', color='#e74c3c')
    ax.invert_yaxis()
    ax.set_xticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'lda_topics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR}/lda_topics.png")

def save_topics_markdown(moltbook_topics, reddit_topics):
    """Save topics to markdown."""
    content = "# LDA Topic Modeling Results\n\n"

    content += "## Moltbook (AI Agents) Topics\n\n"
    for topic in moltbook_topics:
        words = ", ".join(topic['words'])
        content += f"**Topic {topic['topic_id']}**: {words}\n\n"

    content += "## Reddit (Human) Topics\n\n"
    for topic in reddit_topics:
        words = ", ".join(topic['words'])
        content += f"**Topic {topic['topic_id']}**: {words}\n\n"

    with open(os.path.join(OUTPUT_DIR, 'lda_topics.md'), 'w') as f:
        f.write(content)
    print(f"Saved: {OUTPUT_DIR}/lda_topics.md")

    return moltbook_topics, reddit_topics

def main():
    print("=" * 60)
    print("LDA TOPIC MODELING")
    print("=" * 60)

    # Load data
    df_moltbook = load_posts("moltbook.db", "Moltbook")
    df_reddit = load_posts("reddit.db", "Reddit")

    # Clean texts
    df_moltbook['clean_text'] = df_moltbook['text'].apply(clean_text)
    df_reddit['clean_text'] = df_reddit['text'].apply(clean_text)

    # Filter empty
    moltbook_texts = df_moltbook[df_moltbook['clean_text'].str.len() > 20]['clean_text'].tolist()
    reddit_texts = df_reddit[df_reddit['clean_text'].str.len() > 20]['clean_text'].tolist()

    print(f"\nMoltbook valid texts: {len(moltbook_texts)}")
    print(f"Reddit valid texts: {len(reddit_texts)}")

    # Run LDA
    moltbook_topics, _, _, _ = run_lda(moltbook_texts, "Moltbook", n_topics=8)
    reddit_topics, _, _, _ = run_lda(reddit_texts, "Reddit", n_topics=8)

    # Visualize
    plot_topics(moltbook_topics, reddit_topics)

    # Save markdown
    save_topics_markdown(moltbook_topics, reddit_topics)

    print("\n" + "=" * 60)
    print("LDA ANALYSIS COMPLETE!")
    print("=" * 60)

    return moltbook_topics, reddit_topics

if __name__ == '__main__':
    main()
