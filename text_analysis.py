#!/usr/bin/env python3
"""
Text/Topic Comparison: Moltbook (AI) vs Reddit (Human)
======================================================
Compares text characteristics between AI-generated and human-generated posts.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
import os
import warnings
warnings.filterwarnings('ignore')

# Create output directory
OUTPUT_DIR = "text_analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_posts(db_file, platform_name):
    """Load posts from database."""
    conn = sqlite3.connect(db_file)
    df = pd.read_sql_query("""
        SELECT id, title, content, submolt_name, upvotes, comment_count
        FROM posts
    """, conn)
    conn.close()

    # Combine title and content for analysis
    df['text'] = df['title'].fillna('') + ' ' + df['content'].fillna('')
    df['text'] = df['text'].str.strip()
    df['platform'] = platform_name

    print(f"Loaded {len(df)} posts from {platform_name}")
    return df

def clean_text(text):
    """Clean text for analysis."""
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def word_count(text):
    """Count words in text."""
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(text.split())

def compute_basic_stats(df_moltbook, df_reddit):
    """Compute basic text statistics."""
    print("\n" + "=" * 60)
    print("BASIC TEXT STATISTICS")
    print("=" * 60)

    results = {}

    for name, df in [("Moltbook", df_moltbook), ("Reddit", df_reddit)]:
        # Word counts
        df['word_count'] = df['text'].apply(word_count)
        df['clean_text'] = df['text'].apply(clean_text)

        # Filter out empty posts
        df_valid = df[df['word_count'] > 0]

        # Basic stats
        avg_words = df_valid['word_count'].mean()
        median_words = df_valid['word_count'].median()
        max_words = df_valid['word_count'].max()

        # Vocabulary diversity
        all_words = ' '.join(df_valid['clean_text'].tolist()).split()
        total_words = len(all_words)
        unique_words = len(set(all_words))
        vocab_diversity = unique_words / total_words if total_words > 0 else 0

        # Type-Token Ratio (sample-based for fairness)
        sample_size = min(10000, len(all_words))
        sample_words = np.random.choice(all_words, sample_size, replace=False) if len(all_words) >= sample_size else all_words
        ttr = len(set(sample_words)) / len(sample_words) if len(sample_words) > 0 else 0

        results[name] = {
            'total_posts': len(df),
            'valid_posts': len(df_valid),
            'avg_words': avg_words,
            'median_words': median_words,
            'max_words': max_words,
            'total_words': total_words,
            'unique_words': unique_words,
            'vocab_diversity': vocab_diversity,
            'ttr': ttr
        }

        print(f"\n{name}:")
        print(f"  Posts: {len(df_valid):,} (valid) / {len(df):,} (total)")
        print(f"  Avg words/post: {avg_words:.1f}")
        print(f"  Median words/post: {median_words:.0f}")
        print(f"  Max words/post: {max_words:,}")
        print(f"  Total words: {total_words:,}")
        print(f"  Unique words: {unique_words:,}")
        print(f"  Vocabulary diversity: {vocab_diversity:.4f}")
        print(f"  Type-Token Ratio (10K sample): {ttr:.4f}")

    return results, df_moltbook, df_reddit

def tfidf_analysis(df_moltbook, df_reddit):
    """Compute TF-IDF to find distinctive words for each platform."""
    print("\n" + "=" * 60)
    print("TF-IDF ANALYSIS - DISTINCTIVE WORDS")
    print("=" * 60)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        print("Installing scikit-learn...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'scikit-learn'])
        from sklearn.feature_extraction.text import TfidfVectorizer

    # Combine all text for each platform
    moltbook_text = ' '.join(df_moltbook['clean_text'].dropna().tolist())
    reddit_text = ' '.join(df_reddit['clean_text'].dropna().tolist())

    # Create TF-IDF vectorizer - treating each platform as one document
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        min_df=1,  # Min 1 since we only have 2 documents
        max_df=1.0,
        ngram_range=(1, 1)
    )

    # Fit on both platforms
    tfidf_matrix = vectorizer.fit_transform([moltbook_text, reddit_text])
    feature_names = vectorizer.get_feature_names_out()

    # Get scores for each platform
    moltbook_scores = tfidf_matrix[0].toarray().flatten()
    reddit_scores = tfidf_matrix[1].toarray().flatten()

    # Find distinctive words (high in one, low in other)
    # Ratio approach: moltbook_score / (reddit_score + epsilon)
    epsilon = 0.0001
    moltbook_distinctive = moltbook_scores / (reddit_scores + epsilon)
    reddit_distinctive = reddit_scores / (moltbook_scores + epsilon)

    # Get top 20 for each
    top_moltbook_idx = np.argsort(moltbook_distinctive)[-20:][::-1]
    top_reddit_idx = np.argsort(reddit_distinctive)[-20:][::-1]

    moltbook_top_words = [(feature_names[i], moltbook_scores[i], moltbook_distinctive[i])
                          for i in top_moltbook_idx]
    reddit_top_words = [(feature_names[i], reddit_scores[i], reddit_distinctive[i])
                        for i in top_reddit_idx]

    print("\nTop 20 Distinctive Words - MOLTBOOK (AI):")
    print("-" * 40)
    for word, score, ratio in moltbook_top_words:
        print(f"  {word:<20} (TF-IDF: {score:.4f}, ratio: {ratio:.1f}x)")

    print("\nTop 20 Distinctive Words - REDDIT (Human):")
    print("-" * 40)
    for word, score, ratio in reddit_top_words:
        print(f"  {word:<20} (TF-IDF: {score:.4f}, ratio: {ratio:.1f}x)")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Moltbook words
    words_m = [w[0] for w in moltbook_top_words]
    scores_m = [w[1] for w in moltbook_top_words]
    axes[0].barh(range(len(words_m)), scores_m, color='#3498db', alpha=0.8)
    axes[0].set_yticks(range(len(words_m)))
    axes[0].set_yticklabels(words_m)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('TF-IDF Score')
    axes[0].set_title('Moltbook (AI) - Distinctive Words', fontweight='bold', fontsize=14)

    # Reddit words
    words_r = [w[0] for w in reddit_top_words]
    scores_r = [w[1] for w in reddit_top_words]
    axes[1].barh(range(len(words_r)), scores_r, color='#e74c3c', alpha=0.8)
    axes[1].set_yticks(range(len(words_r)))
    axes[1].set_yticklabels(words_r)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('TF-IDF Score')
    axes[1].set_title('Reddit (Human) - Distinctive Words', fontweight='bold', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'tfidf_distinctive_words.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR}/tfidf_distinctive_words.png")

    return moltbook_top_words, reddit_top_words

def sentiment_analysis(df_moltbook, df_reddit):
    """Perform VADER sentiment analysis."""
    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS (VADER)")
    print("=" * 60)

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        print("Installing vaderSentiment...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'vaderSentiment'])
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(text):
        if not isinstance(text, str) or not text.strip():
            return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0}
        # Truncate very long texts
        text = text[:5000]
        return analyzer.polarity_scores(text)

    print("Analyzing Moltbook sentiment...")
    moltbook_sentiments = df_moltbook['text'].apply(get_sentiment)
    df_moltbook['compound'] = moltbook_sentiments.apply(lambda x: x['compound'])
    df_moltbook['pos'] = moltbook_sentiments.apply(lambda x: x['pos'])
    df_moltbook['neg'] = moltbook_sentiments.apply(lambda x: x['neg'])
    df_moltbook['neu'] = moltbook_sentiments.apply(lambda x: x['neu'])

    print("Analyzing Reddit sentiment...")
    reddit_sentiments = df_reddit['text'].apply(get_sentiment)
    df_reddit['compound'] = reddit_sentiments.apply(lambda x: x['compound'])
    df_reddit['pos'] = reddit_sentiments.apply(lambda x: x['pos'])
    df_reddit['neg'] = reddit_sentiments.apply(lambda x: x['neg'])
    df_reddit['neu'] = reddit_sentiments.apply(lambda x: x['neu'])

    # Statistics
    for name, df in [("Moltbook", df_moltbook), ("Reddit", df_reddit)]:
        print(f"\n{name} Sentiment:")
        print(f"  Mean compound: {df['compound'].mean():.4f}")
        print(f"  Median compound: {df['compound'].median():.4f}")
        print(f"  Std compound: {df['compound'].std():.4f}")
        print(f"  Positive posts (>0.05): {(df['compound'] > 0.05).sum():,} ({(df['compound'] > 0.05).mean()*100:.1f}%)")
        print(f"  Negative posts (<-0.05): {(df['compound'] < -0.05).sum():,} ({(df['compound'] < -0.05).mean()*100:.1f}%)")
        print(f"  Neutral posts: {((df['compound'] >= -0.05) & (df['compound'] <= 0.05)).sum():,}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Compound score distribution
    axes[0, 0].hist(df_moltbook['compound'], bins=50, alpha=0.6, label='Moltbook (AI)', color='#3498db', density=True)
    axes[0, 0].hist(df_reddit['compound'], bins=50, alpha=0.6, label='Reddit (Human)', color='#e74c3c', density=True)
    axes[0, 0].set_xlabel('Compound Sentiment Score')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Sentiment Distribution (Compound Score)', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Box plot comparison
    data_box = [df_moltbook['compound'].dropna(), df_reddit['compound'].dropna()]
    bp = axes[0, 1].boxplot(data_box, labels=['Moltbook (AI)', 'Reddit (Human)'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    axes[0, 1].set_ylabel('Compound Sentiment Score')
    axes[0, 1].set_title('Sentiment Box Plot Comparison', fontweight='bold')
    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Positive/Negative/Neutral breakdown
    categories = ['Positive\n(>0.05)', 'Neutral', 'Negative\n(<-0.05)']
    moltbook_cats = [
        (df_moltbook['compound'] > 0.05).mean() * 100,
        ((df_moltbook['compound'] >= -0.05) & (df_moltbook['compound'] <= 0.05)).mean() * 100,
        (df_moltbook['compound'] < -0.05).mean() * 100
    ]
    reddit_cats = [
        (df_reddit['compound'] > 0.05).mean() * 100,
        ((df_reddit['compound'] >= -0.05) & (df_reddit['compound'] <= 0.05)).mean() * 100,
        (df_reddit['compound'] < -0.05).mean() * 100
    ]

    x = np.arange(len(categories))
    width = 0.35
    axes[1, 0].bar(x - width/2, moltbook_cats, width, label='Moltbook (AI)', color='#3498db', alpha=0.8)
    axes[1, 0].bar(x + width/2, reddit_cats, width, label='Reddit (Human)', color='#e74c3c', alpha=0.8)
    axes[1, 0].set_ylabel('Percentage of Posts')
    axes[1, 0].set_title('Sentiment Category Breakdown', fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(categories)
    axes[1, 0].legend()

    # Pos/Neg/Neu component comparison
    components = ['Positive', 'Negative', 'Neutral']
    moltbook_comp = [df_moltbook['pos'].mean(), df_moltbook['neg'].mean(), df_moltbook['neu'].mean()]
    reddit_comp = [df_reddit['pos'].mean(), df_reddit['neg'].mean(), df_reddit['neu'].mean()]

    x = np.arange(len(components))
    axes[1, 1].bar(x - width/2, moltbook_comp, width, label='Moltbook (AI)', color='#3498db', alpha=0.8)
    axes[1, 1].bar(x + width/2, reddit_comp, width, label='Reddit (Human)', color='#e74c3c', alpha=0.8)
    axes[1, 1].set_ylabel('Average Score')
    axes[1, 1].set_title('Sentiment Components (VADER)', fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(components)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sentiment_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR}/sentiment_analysis.png")

    return df_moltbook, df_reddit

def generate_wordclouds(df_moltbook, df_reddit):
    """Generate word clouds for each platform."""
    print("\n" + "=" * 60)
    print("GENERATING WORD CLOUDS")
    print("=" * 60)

    try:
        from wordcloud import WordCloud, STOPWORDS
    except ImportError:
        print("Installing wordcloud...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'wordcloud'])
        from wordcloud import WordCloud, STOPWORDS

    # Custom stopwords
    stopwords = set(STOPWORDS)
    stopwords.update(['http', 'https', 'www', 'com', 'just', 'like', 'know',
                      'think', 'really', 'would', 'could', 'one', 'get', 'got',
                      'much', 'even', 'also', 'way', 'well', 'still', 'see',
                      'something', 'thing', 'things', 'make', 'going', 'want',
                      'said', 'say', 'will', 'can', 'may', 'now', 'new'])

    # Combine texts
    moltbook_text = ' '.join(df_moltbook['clean_text'].dropna().tolist())
    reddit_text = ' '.join(df_reddit['clean_text'].dropna().tolist())

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Moltbook word cloud
    print("Generating Moltbook word cloud...")
    wc_moltbook = WordCloud(
        width=1600, height=800,
        background_color='white',
        stopwords=stopwords,
        max_words=150,
        colormap='Blues',
        random_state=42
    ).generate(moltbook_text)

    axes[0].imshow(wc_moltbook, interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title('Moltbook (AI Agents)', fontsize=20, fontweight='bold', color='#3498db')

    # Reddit word cloud
    print("Generating Reddit word cloud...")
    wc_reddit = WordCloud(
        width=1600, height=800,
        background_color='white',
        stopwords=stopwords,
        max_words=150,
        colormap='Reds',
        random_state=42
    ).generate(reddit_text)

    axes[1].imshow(wc_reddit, interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title('Reddit (Humans)', fontsize=20, fontweight='bold', color='#e74c3c')

    plt.suptitle('Word Cloud Comparison', fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'wordclouds.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/wordclouds.png")

def plot_length_distribution(df_moltbook, df_reddit):
    """Plot post length distribution comparison."""
    print("\n" + "=" * 60)
    print("POST LENGTH DISTRIBUTION")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Filter to valid posts
    m_lengths = df_moltbook[df_moltbook['word_count'] > 0]['word_count']
    r_lengths = df_reddit[df_reddit['word_count'] > 0]['word_count']

    # Histogram overlay
    axes[0, 0].hist(m_lengths, bins=50, alpha=0.6, label='Moltbook (AI)', color='#3498db', density=True)
    axes[0, 0].hist(r_lengths, bins=50, alpha=0.6, label='Reddit (Human)', color='#e74c3c', density=True)
    axes[0, 0].set_xlabel('Word Count')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Post Length Distribution', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].set_xlim(0, 500)

    # Log-scale histogram
    axes[0, 1].hist(m_lengths, bins=50, alpha=0.6, label='Moltbook (AI)', color='#3498db')
    axes[0, 1].hist(r_lengths, bins=50, alpha=0.6, label='Reddit (Human)', color='#e74c3c')
    axes[0, 1].set_xlabel('Word Count')
    axes[0, 1].set_ylabel('Frequency (log scale)')
    axes[0, 1].set_title('Post Length Distribution (Log Scale)', fontweight='bold')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()

    # Box plot
    data_box = [m_lengths, r_lengths]
    bp = axes[1, 0].boxplot(data_box, labels=['Moltbook (AI)', 'Reddit (Human)'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    axes[1, 0].set_ylabel('Word Count')
    axes[1, 0].set_title('Post Length Box Plot', fontweight='bold')
    axes[1, 0].set_ylim(0, 500)

    # CDF comparison
    m_sorted = np.sort(m_lengths)
    r_sorted = np.sort(r_lengths)
    m_cdf = np.arange(1, len(m_sorted) + 1) / len(m_sorted)
    r_cdf = np.arange(1, len(r_sorted) + 1) / len(r_sorted)

    axes[1, 1].plot(m_sorted, m_cdf, label='Moltbook (AI)', color='#3498db', linewidth=2)
    axes[1, 1].plot(r_sorted, r_cdf, label='Reddit (Human)', color='#e74c3c', linewidth=2)
    axes[1, 1].set_xlabel('Word Count')
    axes[1, 1].set_ylabel('Cumulative Proportion')
    axes[1, 1].set_title('Cumulative Distribution Function', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].set_xlim(0, 500)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'length_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/length_distribution.png")

    # Statistics
    print(f"\nMoltbook: mean={m_lengths.mean():.1f}, median={m_lengths.median():.0f}, std={m_lengths.std():.1f}")
    print(f"Reddit: mean={r_lengths.mean():.1f}, median={r_lengths.median():.0f}, std={r_lengths.std():.1f}")

def save_summary_report(basic_stats, moltbook_words, reddit_words, df_moltbook, df_reddit):
    """Save summary report as markdown."""

    report = f"""# Text Analysis: Moltbook (AI) vs Reddit (Human)

## Basic Text Statistics

| Metric | Moltbook (AI) | Reddit (Human) |
|--------|---------------|----------------|
| Total Posts | {basic_stats['Moltbook']['total_posts']:,} | {basic_stats['Reddit']['total_posts']:,} |
| Valid Posts | {basic_stats['Moltbook']['valid_posts']:,} | {basic_stats['Reddit']['valid_posts']:,} |
| Avg Words/Post | {basic_stats['Moltbook']['avg_words']:.1f} | {basic_stats['Reddit']['avg_words']:.1f} |
| Median Words/Post | {basic_stats['Moltbook']['median_words']:.0f} | {basic_stats['Reddit']['median_words']:.0f} |
| Max Words/Post | {basic_stats['Moltbook']['max_words']:,} | {basic_stats['Reddit']['max_words']:,} |
| Total Words | {basic_stats['Moltbook']['total_words']:,} | {basic_stats['Reddit']['total_words']:,} |
| Unique Words | {basic_stats['Moltbook']['unique_words']:,} | {basic_stats['Reddit']['unique_words']:,} |
| Vocabulary Diversity | {basic_stats['Moltbook']['vocab_diversity']:.4f} | {basic_stats['Reddit']['vocab_diversity']:.4f} |
| Type-Token Ratio | {basic_stats['Moltbook']['ttr']:.4f} | {basic_stats['Reddit']['ttr']:.4f} |

## TF-IDF Distinctive Words

### Moltbook (AI) - Top 20
| Word | TF-IDF Score |
|------|--------------|
"""
    for word, score, _ in moltbook_words:
        report += f"| {word} | {score:.4f} |\n"

    report += """
### Reddit (Human) - Top 20
| Word | TF-IDF Score |
|------|--------------|
"""
    for word, score, _ in reddit_words:
        report += f"| {word} | {score:.4f} |\n"

    report += f"""

## Sentiment Analysis (VADER)

| Metric | Moltbook (AI) | Reddit (Human) |
|--------|---------------|----------------|
| Mean Compound | {df_moltbook['compound'].mean():.4f} | {df_reddit['compound'].mean():.4f} |
| Median Compound | {df_moltbook['compound'].median():.4f} | {df_reddit['compound'].median():.4f} |
| Std Compound | {df_moltbook['compound'].std():.4f} | {df_reddit['compound'].std():.4f} |
| Positive Posts % | {(df_moltbook['compound'] > 0.05).mean()*100:.1f}% | {(df_reddit['compound'] > 0.05).mean()*100:.1f}% |
| Negative Posts % | {(df_moltbook['compound'] < -0.05).mean()*100:.1f}% | {(df_reddit['compound'] < -0.05).mean()*100:.1f}% |
| Neutral Posts % | {((df_moltbook['compound'] >= -0.05) & (df_moltbook['compound'] <= 0.05)).mean()*100:.1f}% | {((df_reddit['compound'] >= -0.05) & (df_reddit['compound'] <= 0.05)).mean()*100:.1f}% |

## Files Generated

- `tfidf_distinctive_words.png` - TF-IDF distinctive words comparison
- `sentiment_analysis.png` - Sentiment distribution and comparison
- `wordclouds.png` - Side-by-side word clouds
- `length_distribution.png` - Post length distribution comparison
- `text_analysis_summary.md` - This summary file
"""

    with open(os.path.join(OUTPUT_DIR, 'text_analysis_summary.md'), 'w') as f:
        f.write(report)
    print(f"\nSaved: {OUTPUT_DIR}/text_analysis_summary.md")

def main():
    print("=" * 60)
    print("TEXT ANALYSIS: MOLTBOOK (AI) vs REDDIT (HUMAN)")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df_moltbook = load_posts("moltbook.db", "Moltbook")
    df_reddit = load_posts("reddit.db", "Reddit")

    # Basic stats
    basic_stats, df_moltbook, df_reddit = compute_basic_stats(df_moltbook, df_reddit)

    # TF-IDF analysis
    moltbook_words, reddit_words = tfidf_analysis(df_moltbook, df_reddit)

    # Sentiment analysis
    df_moltbook, df_reddit = sentiment_analysis(df_moltbook, df_reddit)

    # Word clouds
    generate_wordclouds(df_moltbook, df_reddit)

    # Length distribution
    plot_length_distribution(df_moltbook, df_reddit)

    # Save summary
    save_summary_report(basic_stats, moltbook_words, reddit_words, df_moltbook, df_reddit)

    print("\n" + "=" * 60)
    print("TEXT ANALYSIS COMPLETE!")
    print(f"Results saved to: {OUTPUT_DIR}/")
    print("=" * 60)

if __name__ == '__main__':
    main()
