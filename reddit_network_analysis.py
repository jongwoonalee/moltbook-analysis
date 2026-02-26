#!/usr/bin/env python3
"""
Reddit Author-Subreddit Interaction Network Analysis
=====================================================
Builds a co-posting network where authors are connected if they post in the same subreddit.
Mirrors the Moltbook analysis for comparison.
"""

import sqlite3
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False

DATABASE_FILE = "reddit.db"
OUTPUT_DIR = "reddit_network_output"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_posts_data():
    """Load posts data from SQLite database."""
    conn = sqlite3.connect(DATABASE_FILE)

    # No need to exclude "introductions" - Reddit data is already filtered
    query = """
    SELECT author_id, author_name, submolt_id, submolt_name, id as post_id,
           title, upvotes, downvotes, comment_count, created_at
    FROM posts
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"Loaded {len(df)} posts")
    return df

def filter_active_authors(df, min_posts=3):
    """Filter to only include authors with >= min_posts."""
    author_counts = df.groupby('author_id').size()
    active_authors = author_counts[author_counts >= min_posts].index
    df_filtered = df[df['author_id'].isin(active_authors)]

    print(f"Authors with >= {min_posts} posts: {len(active_authors)}")
    print(f"Posts from active authors: {len(df_filtered)}")

    return df_filtered

def build_author_submolt_mapping(df):
    """Build mapping of author -> set of subreddits they've posted in."""
    author_submolts = defaultdict(set)
    author_names = {}

    for _, row in df.iterrows():
        author_submolts[row['author_id']].add(row['submolt_name'])
        author_names[row['author_id']] = row['author_name']

    return author_submolts, author_names

def build_coposting_network(author_submolts, author_names):
    """
    Build network where two authors are connected if they posted in the same subreddit.
    Edge weight = number of shared subreddits.
    """
    G = nx.Graph()

    authors = list(author_submolts.keys())
    print(f"Building network for {len(authors)} authors...")

    # Add nodes with attributes
    for author_id in authors:
        G.add_node(author_id,
                   name=author_names.get(author_id, 'unknown'),
                   submolt_count=len(author_submolts[author_id]))

    # Build inverted index: subreddit -> set of authors
    submolt_authors = defaultdict(set)
    for author_id, submolts in author_submolts.items():
        for submolt in submolts:
            submolt_authors[submolt].add(author_id)

    # Count shared subreddits for each pair
    edge_weights = defaultdict(int)

    for submolt, authors_in_submolt in submolt_authors.items():
        authors_list = list(authors_in_submolt)
        for i in range(len(authors_list)):
            for j in range(i + 1, len(authors_list)):
                a1, a2 = authors_list[i], authors_list[j]
                if a1 > a2:
                    a1, a2 = a2, a1
                edge_weights[(a1, a2)] += 1

    # Add edges
    for (a1, a2), weight in edge_weights.items():
        G.add_edge(a1, a2, weight=weight)

    print(f"Network built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G

def compute_network_stats(G, df, author_names):
    """Compute comprehensive network statistics."""
    stats = {}

    # Basic stats
    stats['num_nodes'] = G.number_of_nodes()
    stats['num_edges'] = G.number_of_edges()
    stats['density'] = nx.density(G)

    # Degree statistics
    degrees = dict(G.degree())
    degree_values = list(degrees.values())
    stats['avg_degree'] = np.mean(degree_values)
    stats['median_degree'] = np.median(degree_values)
    stats['max_degree'] = max(degree_values)
    stats['min_degree'] = min(degree_values)

    # Connected components
    components = list(nx.connected_components(G))
    stats['num_components'] = len(components)
    stats['largest_component_size'] = len(max(components, key=len))
    stats['largest_component_pct'] = stats['largest_component_size'] / stats['num_nodes'] * 100

    # Clustering coefficient
    if G.number_of_nodes() > 5000:
        sample_nodes = np.random.choice(list(G.nodes()), size=min(5000, G.number_of_nodes()), replace=False)
        clustering_values = [nx.clustering(G, n) for n in sample_nodes]
        stats['avg_clustering'] = np.mean(clustering_values)
        stats['clustering_note'] = 'sampled'
    else:
        stats['avg_clustering'] = nx.average_clustering(G)
        stats['clustering_note'] = 'full'

    # Top 20 most active authors
    author_post_counts = df.groupby(['author_id', 'author_name']).size().reset_index(name='post_count')
    author_post_counts = author_post_counts.sort_values('post_count', ascending=False).head(20)
    stats['top_active_authors'] = author_post_counts.to_dict('records')

    # Top 20 most connected authors
    top_by_degree = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:20]
    stats['top_connected_authors'] = [
        {'author_id': aid, 'author_name': author_names.get(aid, 'unknown'), 'degree': deg}
        for aid, deg in top_by_degree
    ]

    return stats, degrees

def run_community_detection(G, df):
    """Run Louvain community detection."""
    print("Running community detection...")

    largest_cc = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(largest_cc).copy()

    if HAS_LOUVAIN:
        partition = community_louvain.best_partition(G_lcc, weight='weight', random_state=42)
        modularity = community_louvain.modularity(partition, G_lcc, weight='weight')
    else:
        communities = nx.community.greedy_modularity_communities(G_lcc, weight='weight')
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
        modularity = nx.community.modularity(G_lcc, communities, weight='weight')

    community_counts = Counter(partition.values())
    num_communities = len(community_counts)

    print(f"Found {num_communities} communities")
    print(f"Modularity: {modularity:.4f}")

    # Analyze dominant subreddits in each community
    author_submolts = defaultdict(list)
    for _, row in df.iterrows():
        author_submolts[row['author_id']].append(row['submolt_name'])

    community_submolts = defaultdict(list)
    for author_id, comm_id in partition.items():
        community_submolts[comm_id].extend(author_submolts.get(author_id, []))

    community_info = []
    for comm_id in sorted(community_counts.keys(), key=lambda x: community_counts[x], reverse=True)[:15]:
        submolt_counter = Counter(community_submolts[comm_id])
        top_submolts = submolt_counter.most_common(5)
        community_info.append({
            'community_id': comm_id,
            'size': community_counts[comm_id],
            'top_submolts': top_submolts
        })

    return partition, modularity, num_communities, community_info, G_lcc

def plot_degree_distribution(degrees, output_dir):
    """Plot degree distribution."""
    degree_values = list(degrees.values())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(degree_values, bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[0].set_xlabel('Degree', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Reddit: Degree Distribution (Histogram)', fontsize=14)

    degree_count = Counter(degree_values)
    degrees_sorted = sorted(degree_count.keys())
    counts_sorted = [degree_count[d] for d in degrees_sorted]

    axes[1].loglog(degrees_sorted, counts_sorted, 'o', markersize=4, alpha=0.7, color='coral')
    axes[1].set_xlabel('Degree (log scale)', fontsize=12)
    axes[1].set_ylabel('Frequency (log scale)', fontsize=12)
    axes[1].set_title('Reddit: Degree Distribution (Log-Log Scale)', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'degree_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved degree distribution plot")

def plot_network(G, partition, degrees, output_dir, max_nodes=500):
    """Create network visualization."""
    print(f"Creating network visualization...")

    if G.number_of_nodes() > max_nodes:
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_node_ids = [n[0] for n in top_nodes]
        G_sub = G.subgraph(top_node_ids).copy()
        print(f"Subsampled to top {max_nodes} authors by degree")
    else:
        G_sub = G.copy()

    print("Computing layout...")
    pos = nx.spring_layout(G_sub, k=1/np.sqrt(G_sub.number_of_nodes()), iterations=50, seed=42)

    node_colors = []
    for node in G_sub.nodes():
        if node in partition:
            node_colors.append(partition[node])
        else:
            node_colors.append(-1)

    node_sizes = [min(300, 20 + degrees.get(n, 0) * 0.5) for n in G_sub.nodes()]

    fig, ax = plt.subplots(figsize=(16, 16))

    nx.draw_networkx_edges(G_sub, pos, alpha=0.1, edge_color='gray', ax=ax)
    nodes = nx.draw_networkx_nodes(G_sub, pos,
                                    node_color=node_colors,
                                    node_size=node_sizes,
                                    cmap=plt.cm.Set3,
                                    alpha=0.8,
                                    ax=ax)

    top_10_nodes = sorted([(n, degrees.get(n, 0)) for n in G_sub.nodes()],
                          key=lambda x: x[1], reverse=True)[:10]
    labels = {n[0]: G.nodes[n[0]].get('name', '')[:15] for n in top_10_nodes}
    nx.draw_networkx_labels(G_sub, pos, labels, font_size=8, ax=ax)

    ax.set_title(f'Reddit Author Co-Posting Network\n(Top {G_sub.number_of_nodes()} authors by degree, colored by community)',
                 fontsize=14)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'network_visualization.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved network visualization")

def plot_community_sizes(community_info, output_dir):
    """Plot community size distribution."""
    sizes = [c['size'] for c in community_info]
    labels = [f"C{c['community_id']}" for c in community_info]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(sizes)), sizes, color=plt.cm.Set3(np.linspace(0, 1, len(sizes))))
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_xlabel('Community', fontsize=12)
    ax.set_ylabel('Number of Authors', fontsize=12)
    ax.set_title('Reddit: Community Sizes (Top 15)', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'community_sizes.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved community sizes plot")

def save_results(G, stats, partition, modularity, num_communities, community_info, output_dir):
    """Save all results to files."""

    graphml_path = os.path.join(output_dir, 'author_network.graphml')

    for node in G.nodes():
        G.nodes[node]['community'] = partition.get(node, -1)

    nx.write_graphml(G, graphml_path)
    print(f"Saved graph to {graphml_path}")

    md_content = f"""# Reddit Author Co-Posting Network Analysis

## Overview
This analysis builds a network where **human Reddit authors** are connected if they posted in the same subreddit.
- Edge weight = number of shared subreddits
- Filtered to authors with >= 3 posts
- Data from Pushshift Reddit archive (historical data ~2012)

## Basic Network Statistics

| Metric | Value |
|--------|-------|
| **Nodes (Authors)** | {stats['num_nodes']:,} |
| **Edges (Connections)** | {stats['num_edges']:,} |
| **Density** | {stats['density']:.6f} |
| **Average Degree** | {stats['avg_degree']:.2f} |
| **Median Degree** | {stats['median_degree']:.0f} |
| **Max Degree** | {stats['max_degree']:,} |
| **Min Degree** | {stats['min_degree']} |
| **Avg Clustering Coefficient** | {stats['avg_clustering']:.4f} ({stats['clustering_note']}) |

## Connected Components

| Metric | Value |
|--------|-------|
| **Number of Components** | {stats['num_components']:,} |
| **Largest Component Size** | {stats['largest_component_size']:,} |
| **Largest Component %** | {stats['largest_component_pct']:.1f}% |

## Community Detection (Louvain)

| Metric | Value |
|--------|-------|
| **Number of Communities** | {num_communities} |
| **Modularity Score** | {modularity:.4f} |

### Top Communities and Their Dominant Subreddits

"""

    for comm in community_info:
        submolts_str = ', '.join([f"{s[0]} ({s[1]})" for s in comm['top_submolts']])
        md_content += f"**Community {comm['community_id']}** ({comm['size']} authors)\n"
        md_content += f"- Top subreddits: {submolts_str}\n\n"

    md_content += """## Top 20 Most Active Authors (by Post Count)

| Rank | Author | Post Count |
|------|--------|------------|
"""

    for i, author in enumerate(stats['top_active_authors'], 1):
        md_content += f"| {i} | {author['author_name']} | {author['post_count']} |\n"

    md_content += """

## Top 20 Most Connected Authors (by Degree)

| Rank | Author | Degree (Connections) |
|------|--------|---------------------|
"""

    for i, author in enumerate(stats['top_connected_authors'], 1):
        md_content += f"| {i} | {author['author_name']} | {author['degree']:,} |\n"

    md_content += """

## Files Generated

- `author_network.graphml` - Full network graph
- `degree_distribution.png` - Histogram and log-log plot
- `network_visualization.png` - Network plot colored by community
- `community_sizes.png` - Community size bar chart
- `network_summary.md` - This summary file

## Notes

- This is a **human social network** from Reddit
- Used for comparison against Moltbook AI agent network
"""

    md_path = os.path.join(output_dir, 'network_summary.md')
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"Saved summary to {md_path}")

def main():
    print("=" * 60)
    print("REDDIT AUTHOR CO-POSTING NETWORK ANALYSIS")
    print("=" * 60)
    print()

    print("STEP 1: Loading and filtering data...")
    print("-" * 40)
    df = load_posts_data()
    df = filter_active_authors(df, min_posts=3)

    author_submolts, author_names = build_author_submolt_mapping(df)
    print()

    print("STEP 2: Building co-posting network...")
    print("-" * 40)
    G = build_coposting_network(author_submolts, author_names)
    print()

    print("STEP 3: Computing network statistics...")
    print("-" * 40)
    stats, degrees = compute_network_stats(G, df, author_names)

    print(f"  Nodes: {stats['num_nodes']:,}")
    print(f"  Edges: {stats['num_edges']:,}")
    print(f"  Density: {stats['density']:.6f}")
    print(f"  Avg Degree: {stats['avg_degree']:.2f}")
    print(f"  Clustering Coefficient: {stats['avg_clustering']:.4f}")
    print(f"  Connected Components: {stats['num_components']}")
    print(f"  Largest Component: {stats['largest_component_size']:,} ({stats['largest_component_pct']:.1f}%)")
    print()

    print("STEP 4: Community detection...")
    print("-" * 40)
    partition, modularity, num_communities, community_info, G_lcc = run_community_detection(G, df)
    print()

    print("STEP 5: Creating visualizations...")
    print("-" * 40)
    plot_degree_distribution(degrees, OUTPUT_DIR)
    plot_network(G_lcc, partition, degrees, OUTPUT_DIR, max_nodes=500)
    plot_community_sizes(community_info, OUTPUT_DIR)
    print()

    print("STEP 6: Saving results...")
    print("-" * 40)
    save_results(G, stats, partition, modularity, num_communities, community_info, OUTPUT_DIR)
    print()

    print("=" * 60)
    print("ANALYSIS COMPLETE!")
    print(f"Results saved to: {OUTPUT_DIR}/")
    print("=" * 60)

if __name__ == '__main__':
    main()
