#!/usr/bin/env python3
"""Moltbook agent feature extraction, anomaly detection, and analysis."""

import json
import os
import pickle
import warnings
from collections import defaultdict

import pandas as pd
import numpy as np
import networkx as nx
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

POSTS_DIR = "/Users/jongwonlee/moltbook_data/data/posts/"
PICKLE_PATH = "/Users/jongwonlee/moltbook_results/reply_network.pickle"
OUT_DIR = "/Users/jongwonlee/moltbook_results"

# ═══════════════════════════════════════════════════════════════
# STEP 1a: Load reply network
# ═══════════════════════════════════════════════════════════════
print("Loading reply network...")
with open(PICKLE_PATH, "rb") as f:
    net_data = pickle.load(f)

G_dir = net_data["directed"]       # 24,173 nodes
G_undir = net_data["undirected"]
partition = net_data["partition"]
network_nodes = set(G_dir.nodes())
print(f"  Network nodes: {len(network_nodes):,}")

# ═══════════════════════════════════════════════════════════════
# STEP 1b: Compute network features
# ═══════════════════════════════════════════════════════════════
print("Computing network features...")
print("  - degree stats...")
in_deg = dict(G_dir.in_degree())
out_deg = dict(G_dir.out_degree())

print("  - clustering coefficients...")
clust = nx.clustering(G_undir)

print("  - approximate betweenness centrality (k=1000)...")
between = nx.betweenness_centrality(G_dir, k=1000, seed=42)

print("  - pagerank...")
pr = nx.pagerank(G_dir, max_iter=200)

# ═══════════════════════════════════════════════════════════════
# STEP 1c: Parse all posts for behavioral + text + temporal features
# ═══════════════════════════════════════════════════════════════
print("Parsing all post JSONs for behavioral/text/temporal features...")

# Per-author accumulators
author_posts = defaultdict(list)        # author → [(timestamp, content, submolt)]
author_comments = defaultdict(list)     # author → [(timestamp, content)]
author_submolts = defaultdict(set)      # author → {submolt_names}

files = sorted(os.listdir(POSTS_DIR))
analyzer = SentimentIntensityAnalyzer()

for i, fn in enumerate(files):
    if not fn.endswith(".json"):
        continue
    try:
        with open(os.path.join(POSTS_DIR, fn)) as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        continue

    if not data.get("success", False):
        continue

    post = data.get("post", {})
    post_author = (post.get("author") or {}).get("name")
    post_content = post.get("content", "")
    post_ts = post.get("created_at", "")
    submolt_info = post.get("submolt") or {}
    submolt_name = submolt_info.get("name", "unknown")

    if post_author and post_author in network_nodes:
        author_posts[post_author].append((post_ts, post_content or "", submolt_name))
        author_submolts[post_author].add(submolt_name)

    # Process comments (and nested replies) for comment counts + text
    def walk_comments(comments):
        for c in comments:
            ca = (c.get("author") or {}).get("name")
            cc = c.get("content", "") or ""
            cts = c.get("created_at", "")
            if ca and ca in network_nodes:
                author_comments[ca].append((cts, cc))
            for r in c.get("replies") or []:
                ra = (r.get("author") or {}).get("name")
                rc = r.get("content", "") or ""
                rts = r.get("created_at", "")
                if ra and ra in network_nodes:
                    author_comments[ra].append((rts, rc))
                for rr in r.get("replies") or []:
                    rra = (rr.get("author") or {}).get("name")
                    rrc = rr.get("content", "") or ""
                    rrts = rr.get("created_at", "")
                    if rra and rra in network_nodes:
                        author_comments[rra].append((rrts, rrc))

    walk_comments(data.get("comments", []))

    if (i + 1) % 25000 == 0:
        print(f"  ...processed {i+1}/{len(files)} files")

print(f"  Authors with posts: {len(author_posts):,}")
print(f"  Authors with comments: {len(author_comments):,}")

# ═══════════════════════════════════════════════════════════════
# STEP 1d: Build feature DataFrame
# ═══════════════════════════════════════════════════════════════
print("Building feature DataFrame...")

def parse_ts(s):
    """Parse ISO timestamp string to datetime."""
    if not s:
        return None
    try:
        return pd.Timestamp(s)
    except Exception:
        return None

rows = []
for node in sorted(network_nodes):
    # ── Network features ──
    feat = {
        "author": node,
        "in_degree": in_deg.get(node, 0),
        "out_degree": out_deg.get(node, 0),
        "total_degree": in_deg.get(node, 0) + out_deg.get(node, 0),
        "clustering_coefficient": clust.get(node, 0.0),
        "betweenness_centrality": between.get(node, 0.0),
        "pagerank": pr.get(node, 0.0),
    }

    # ── Behavioral features ──
    posts = author_posts.get(node, [])
    comments = author_comments.get(node, [])
    submolts = author_submolts.get(node, set())

    feat["num_unique_submolts"] = len(submolts)
    feat["total_post_count"] = len(posts)
    feat["total_comment_count"] = len(comments)
    feat["posts_per_submolt"] = len(posts) / max(len(submolts), 1)

    # Gather all timestamps (posts + comments) for temporal features
    all_ts_raw = [t for t, _, _ in posts] + [t for t, _ in comments]
    all_ts = sorted([parse_ts(t) for t in all_ts_raw if parse_ts(t) is not None])

    if len(all_ts) >= 2:
        span_days = (all_ts[-1] - all_ts[0]).total_seconds() / 86400.0
        feat["posting_frequency"] = len(all_ts) / max(span_days, 1e-6)
    else:
        feat["posting_frequency"] = 0.0

    # ── Text features (from post content only — more substantive than comments) ──
    all_texts = [c for _, c, _ in posts] + [c for _, c in comments]
    all_texts = [t for t in all_texts if t and len(t.strip()) > 0]

    if all_texts:
        word_counts = [len(t.split()) for t in all_texts]
        feat["avg_post_word_count"] = np.mean(word_counts)

        sentiments = [analyzer.polarity_scores(t)["compound"] for t in all_texts]
        feat["sentiment_mean"] = np.mean(sentiments)
        feat["sentiment_variance"] = np.var(sentiments)

        # Type-token ratio across all texts
        all_words = []
        for t in all_texts:
            all_words.extend(t.lower().split())
        if all_words:
            feat["type_token_ratio"] = len(set(all_words)) / len(all_words)
        else:
            feat["type_token_ratio"] = 0.0
    else:
        feat["avg_post_word_count"] = 0.0
        feat["sentiment_mean"] = 0.0
        feat["sentiment_variance"] = 0.0
        feat["type_token_ratio"] = 0.0

    # ── Temporal features ──
    if len(all_ts) >= 2:
        intervals = np.array([(all_ts[j] - all_ts[j-1]).total_seconds()
                              for j in range(1, len(all_ts))])
        intervals = intervals[intervals >= 0]  # drop negatives from bad data
        if len(intervals) > 0:
            mean_int = intervals.mean()
            std_int = intervals.std()
            feat["inter_post_interval_mean"] = mean_int
            feat["inter_post_interval_std"] = std_int
            feat["inter_post_interval_cov"] = std_int / mean_int if mean_int > 0 else 0.0
        else:
            feat["inter_post_interval_mean"] = 0.0
            feat["inter_post_interval_std"] = 0.0
            feat["inter_post_interval_cov"] = 0.0
    else:
        feat["inter_post_interval_mean"] = 0.0
        feat["inter_post_interval_std"] = 0.0
        feat["inter_post_interval_cov"] = 0.0

    rows.append(feat)

df = pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════
# STEP 2: Save agent_features.csv
# ═══════════════════════════════════════════════════════════════
feature_path = os.path.join(OUT_DIR, "agent_features.csv")
df.to_csv(feature_path, index=False)
print(f"\nSaved: {feature_path}")
print(f"  Shape: {df.shape}")
print(f"\nFeature summary:")
print(df.describe().to_string())

# ═══════════════════════════════════════════════════════════════
# STEP 3: Isolation Forest
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Isolation Forest (contamination=0.15)")
print("=" * 70)

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

feature_cols = [c for c in df.columns if c != "author"]
X = df[feature_cols].fillna(0).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso = IsolationForest(contamination=0.15, random_state=42, n_estimators=200)
labels = iso.fit_predict(X_scaled)  # 1 = inlier, -1 = outlier
df["outlier"] = (labels == -1).astype(int)

n_outliers = df["outlier"].sum()
print(f"  Outliers:  {n_outliers:,} ({n_outliers/len(df)*100:.1f}%)")
print(f"  Inliers:   {len(df) - n_outliers:,}")

# Show some outlier stats
print("\n  Feature means by group:")
for col in feature_cols:
    inlier_mean = df.loc[df["outlier"]==0, col].mean()
    outlier_mean = df.loc[df["outlier"]==1, col].mean()
    ratio = outlier_mean / inlier_mean if inlier_mean != 0 else float("inf")
    if abs(ratio - 1) > 0.5 or ratio > 2 or ratio < 0.5:
        print(f"    {col:35s}  inlier={inlier_mean:12.2f}  outlier={outlier_mean:12.2f}  ratio={ratio:.2f}")

# ═══════════════════════════════════════════════════════════════
# STEP 4: PCA 2D visualization
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: PCA 2D Visualization")
print("=" * 70)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(12, 8))
inlier_mask = df["outlier"] == 0
outlier_mask = df["outlier"] == 1

ax.scatter(X_pca[inlier_mask, 0], X_pca[inlier_mask, 1],
           c="steelblue", alpha=0.3, s=8, label=f"Inlier (n={inlier_mask.sum():,})")
ax.scatter(X_pca[outlier_mask, 0], X_pca[outlier_mask, 1],
           c="crimson", alpha=0.6, s=18, label=f"Outlier (n={outlier_mask.sum():,})", marker="x")

# Label extreme outliers
extreme_idx = np.argsort(np.linalg.norm(X_pca, axis=1))[-8:]
for idx in extreme_idx:
    ax.annotate(df.iloc[idx]["author"], (X_pca[idx, 0], X_pca[idx, 1]),
                fontsize=7, alpha=0.8)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax.set_title("Moltbook Agents — PCA (Isolation Forest Outliers in Red)")
ax.legend()
plt.tight_layout()
pca_path = os.path.join(OUT_DIR, "pca_outlier_visualization.png")
fig.savefig(pca_path, dpi=150)
plt.close()
print(f"  Saved: {pca_path}")
print(f"  PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
print(f"  PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% of variance")

# ═══════════════════════════════════════════════════════════════
# STEP 5: Feature importance via Random Forest on IF labels
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: Feature Importance (Random Forest on IF labels)")
print("=" * 70)

rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
rf.fit(X_scaled, df["outlier"].values)

importances = pd.Series(rf.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=False)

print("\n  Top 10 most important features for outlier classification:")
for rank, (feat, imp) in enumerate(importances.head(10).items(), 1):
    bar = "█" * int(imp * 100)
    print(f"    {rank:2d}. {feat:35s}  {imp:.4f}  {bar}")

# Bar chart
fig, ax = plt.subplots(figsize=(10, 6))
top10 = importances.head(10)
ax.barh(range(len(top10)), top10.values[::-1], color="steelblue")
ax.set_yticks(range(len(top10)))
ax.set_yticklabels(top10.index[::-1])
ax.set_xlabel("Feature Importance (Gini)")
ax.set_title("Top 10 Features Distinguishing Outlier Agents")
plt.tight_layout()
imp_path = os.path.join(OUT_DIR, "feature_importance.png")
fig.savefig(imp_path, dpi=150)
plt.close()
print(f"  Saved: {imp_path}")

# ═══════════════════════════════════════════════════════════════
# STEP 6: Cross-tabulate CoV > 1.0 vs Isolation Forest outliers
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6: Cross-Tabulation — High CoV (>1.0) vs IF Outliers")
print("=" * 70)

df["high_cov"] = (df["inter_post_interval_cov"] > 1.0).astype(int)

ct = pd.crosstab(
    df["high_cov"].map({0: "CoV ≤ 1.0", 1: "CoV > 1.0"}),
    df["outlier"].map({0: "Inlier", 1: "Outlier"}),
    margins=True,
)
print("\n  Contingency Table:")
print(ct.to_string())

# Percentages
print("\n  Outlier rate by CoV group:")
for cov_label, group in df.groupby("high_cov"):
    label = "CoV > 1.0" if cov_label == 1 else "CoV ≤ 1.0"
    rate = group["outlier"].mean() * 100
    print(f"    {label}: {rate:.1f}% outlier  (n={len(group):,})")

# Overlap stats
both = ((df["high_cov"] == 1) & (df["outlier"] == 1)).sum()
either = ((df["high_cov"] == 1) | (df["outlier"] == 1)).sum()
jaccard = both / either if either > 0 else 0
print(f"\n  Overlap:")
print(f"    Both high-CoV AND outlier: {both:,}")
print(f"    Either:                    {either:,}")
print(f"    Jaccard index:             {jaccard:.4f}")

# Chi-squared test
from scipy.stats import chi2_contingency
chi2, p, dof, expected = chi2_contingency(
    pd.crosstab(df["high_cov"], df["outlier"])
)
print(f"\n  Chi-squared test:")
print(f"    χ² = {chi2:.2f},  df = {dof},  p = {p:.2e}")
print(f"    {'Significant' if p < 0.05 else 'Not significant'} association (α=0.05)")

# Save updated features with labels
df.to_csv(os.path.join(OUT_DIR, "agent_features.csv"), index=False)
print(f"\n  Updated agent_features.csv with outlier + high_cov columns.")

print("\nDone.")
