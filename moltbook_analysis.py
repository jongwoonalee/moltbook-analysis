#!/usr/bin/env python3
"""Moltbook reply-network analysis + comparison with LinkedIn co-posting network."""

import json
import os
import pickle
from collections import Counter, defaultdict
from datetime import datetime

import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain

POSTS_DIR = "/Users/jongwonlee/moltbook_data/data/posts/"
OUT_DIR = "/Users/jongwonlee/moltbook_results"
LINKEDIN_RESULTS = "/Users/jongwonlee/linkedin_results/all_results.pickle"

os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# TASK 1–2: Parse all posts, extract reply edges
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("TASKS 1–2: Parsing posts & extracting reply edges")
print("=" * 70)

# Build lookup: comment_id → author_name, and collect edges
# Also track conversation depth per thread

edges = []                     # (commenter_name, parent_author_name, post_id, timestamp)
comment_author_map = {}        # comment_id → author_name
post_author_map = {}           # post_id → author_name
thread_depths = []             # max depth per post

files = sorted(os.listdir(POSTS_DIR))
total_posts = 0
total_comments = 0
parse_errors = 0

def process_comments(comments, post_id, depth=0, id_to_author=None):
    """Recursively process comments and replies, returning max depth reached."""
    if id_to_author is None:
        id_to_author = {}

    max_depth = depth
    global total_comments

    for c in comments:
        cid = c.get("id")
        author_info = c.get("author") or {}
        author_name = author_info.get("name", "unknown")
        parent_id = c.get("parent_id")
        timestamp = c.get("created_at", "")

        total_comments += 1

        if cid:
            id_to_author[cid] = author_name
            comment_author_map[cid] = author_name

        # Build edge: commenter → parent_comment_author
        if parent_id and parent_id in id_to_author:
            parent_author = id_to_author[parent_id]
            if author_name != parent_author:  # skip self-replies
                edges.append((author_name, parent_author, post_id, timestamp))
        elif parent_id is None and post_id in post_author_map:
            # Top-level comment replies to the post author
            post_auth = post_author_map[post_id]
            if author_name != post_auth:
                edges.append((author_name, post_auth, post_id, timestamp))

        # Recurse into replies
        replies = c.get("replies", [])
        if replies:
            sub_depth = process_comments(replies, post_id, depth + 1, id_to_author)
            max_depth = max(max_depth, sub_depth)

    return max_depth

print(f"Processing {len(files)} post files...")

for i, fn in enumerate(files):
    if not fn.endswith(".json"):
        continue
    try:
        with open(os.path.join(POSTS_DIR, fn)) as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        parse_errors += 1
        continue

    if not data.get("success", False):
        continue

    post = data.get("post", {})
    post_id = post.get("id", fn.replace(".json", ""))
    post_author_obj = post.get("author") or {}
    post_author = post_author_obj.get("name", "unknown")
    post_author_map[post_id] = post_author

    # Also register post_id in comment_author_map so top-level parent_id=None works
    comment_author_map[post_id] = post_author

    comments = data.get("comments", [])
    total_posts += 1

    # First pass: register all top-level comment IDs before processing
    # (in case replies reference siblings that appear later)
    id_to_author_local = {}
    def pre_register(clist):
        for c in clist:
            cid = c.get("id")
            aname = (c.get("author") or {}).get("name", "unknown")
            if cid:
                id_to_author_local[cid] = aname
                comment_author_map[cid] = aname
            for r in c.get("replies") or []:
                rid = r.get("id")
                rname = (r.get("author") or {}).get("name", "unknown")
                if rid:
                    id_to_author_local[rid] = rname
                    comment_author_map[rid] = rname
                for rr in r.get("replies") or []:
                    rrid = rr.get("id")
                    rrname = (rr.get("author") or {}).get("name", "unknown")
                    if rrid:
                        id_to_author_local[rrid] = rrname
                        comment_author_map[rrid] = rrname

    pre_register(comments)

    max_d = process_comments(comments, post_id, depth=0, id_to_author=id_to_author_local)
    if comments:
        thread_depths.append(max_d)

    if (i + 1) % 25000 == 0:
        print(f"  ...processed {i+1}/{len(files)} files, {len(edges)} edges so far")

print(f"\nParsing complete:")
print(f"  Posts parsed:     {total_posts:,}")
print(f"  Comments found:   {total_comments:,}")
print(f"  Parse errors:     {parse_errors}")
print(f"  Reply edges:      {len(edges):,}")
print(f"  Unique authors:   {len(set(a for a,_,_,_ in edges) | set(b for _,b,_,_ in edges)):,}")

# ═══════════════════════════════════════════════════════════════
# TASK 3: Build directed reply network
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TASK 3: Build directed reply network (filtered ≥3 interactions)")
print("=" * 70)

# Count interactions per author (comments given + received)
interaction_count = Counter()
for src, dst, _, _ in edges:
    interaction_count[src] += 1
    interaction_count[dst] += 1

active_authors = {a for a, cnt in interaction_count.items() if cnt >= 3}
print(f"Authors with ≥3 interactions: {len(active_authors):,}")

# Build filtered directed graph
G = nx.DiGraph()
edge_weights = Counter()
for src, dst, post_id, ts in edges:
    if src in active_authors and dst in active_authors:
        edge_weights[(src, dst)] += 1

for (src, dst), w in edge_weights.items():
    G.add_edge(src, dst, weight=w)

# Ensure all active authors are nodes even if isolated after filtering
G.add_nodes_from(active_authors)

print(f"Directed network:")
print(f"  Nodes: {G.number_of_nodes():,}")
print(f"  Edges: {G.number_of_edges():,}")
print(f"  Total edge weight: {sum(w for _,_,w in G.edges(data='weight')):,}")

# ═══════════════════════════════════════════════════════════════
# TASK 5: Network metrics
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TASK 5: Network Metrics")
print("=" * 70)

density = nx.density(G)
reciprocity = nx.reciprocity(G)

in_degrees = np.array([d for _, d in G.in_degree()])
out_degrees = np.array([d for _, d in G.out_degree()])
avg_in = in_degrees.mean()
avg_out = out_degrees.mean()

# Clustering on undirected version
G_undir = G.to_undirected()
clustering = nx.average_clustering(G_undir)
n_components_undir = nx.number_connected_components(G_undir)
largest_cc = max(len(c) for c in nx.connected_components(G_undir))

# Also strongly/weakly connected components of directed graph
n_weak = nx.number_weakly_connected_components(G)
n_strong = nx.number_strongly_connected_components(G)
largest_strong = max(len(c) for c in nx.strongly_connected_components(G))

print(f"  Density:                     {density:.6f}")
print(f"  Reciprocity:                 {reciprocity:.6f}")
print(f"  Avg in-degree:               {avg_in:.2f}")
print(f"  Avg out-degree:              {avg_out:.2f}")
print(f"  Clustering coeff (undirected):{clustering:.6f}")
print(f"  Weakly connected components: {n_weak:,}")
print(f"  Strongly connected comp.:    {n_strong:,}")
print(f"  Largest strong component:    {largest_strong:,}")
print(f"  Largest weak component:      {largest_cc:,}")

# ═══════════════════════════════════════════════════════════════
# TASK 6: Conversation depth distribution
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TASK 6: Conversation Depth Distribution")
print("=" * 70)

depths = np.array(thread_depths)
print(f"  Threads with comments: {len(depths):,}")
print(f"  Max depth stats:")
print(f"    Min:    {depths.min()}")
print(f"    Max:    {depths.max()}")
print(f"    Mean:   {depths.mean():.2f}")
print(f"    Median: {np.median(depths):.1f}")
print(f"    Std:    {depths.std():.2f}")

depth_dist = Counter(depths)
print(f"\n  Depth distribution:")
for d in sorted(depth_dist.keys()):
    pct = depth_dist[d] / len(depths) * 100
    bar = "#" * int(pct)
    print(f"    Depth {d:2d}: {depth_dist[d]:6,} ({pct:5.1f}%) {bar}")

# ═══════════════════════════════════════════════════════════════
# TASK 7: Louvain on undirected version
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TASK 7: Louvain Community Detection (undirected)")
print("=" * 70)

partition = community_louvain.best_partition(G_undir, random_state=42)
modularity = community_louvain.modularity(partition, G_undir)
n_communities = len(set(partition.values()))
comm_sizes = Counter(partition.values())

print(f"  Modularity:    {modularity:.6f}")
print(f"  # Communities: {n_communities}")
print(f"\n  Top 15 community sizes:")
for cid, sz in comm_sizes.most_common(15):
    print(f"    Community {cid:4d}: {sz:,} members")

# ═══════════════════════════════════════════════════════════════
# TASK 8: Side-by-side comparison with LinkedIn
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TASK 8: Side-by-Side Comparison — LinkedIn vs Moltbook")
print("=" * 70)

with open(LINKEDIN_RESULTS, "rb") as f:
    li = pickle.load(f)

li_metrics = li["metrics"]
li_G = li["network"]

# LinkedIn clustering & components already computed
# Moltbook metrics
molt_metrics = {
    "nodes": G.number_of_nodes(),
    "edges": G.number_of_edges(),
    "density": density,
    "reciprocity": reciprocity,
    "avg_in_degree": avg_in,
    "avg_out_degree": avg_out,
    "clustering_coefficient": clustering,
    "weakly_connected_components": n_weak,
    "strongly_connected_components": n_strong,
    "largest_weak_component": largest_cc,
    "modularity": modularity,
    "n_communities": n_communities,
}

# Build comparison table
comparison = {
    "Metric": [],
    "LinkedIn (co-posting)": [],
    "Moltbook (reply)": [],
}

def add_row(name, li_val, mo_val):
    comparison["Metric"].append(name)
    comparison["LinkedIn (co-posting)"].append(li_val)
    comparison["Moltbook (reply)"].append(mo_val)

add_row("Network type", "Undirected", "Directed")
add_row("Nodes", f"{li_metrics['nodes']:,}", f"{molt_metrics['nodes']:,}")
add_row("Edges", f"{li_metrics['edges']:,}", f"{molt_metrics['edges']:,}")
add_row("Density", f"{li_metrics['density']:.6f}", f"{molt_metrics['density']:.6f}")
add_row("Avg degree", f"{li_metrics['avg_degree']:.2f}", f"in={molt_metrics['avg_in_degree']:.2f} / out={molt_metrics['avg_out_degree']:.2f}")
add_row("Reciprocity", "N/A (undirected)", f"{molt_metrics['reciprocity']:.6f}")
add_row("Clustering coeff", f"{li_metrics['clustering_coefficient']:.6f}", f"{molt_metrics['clustering_coefficient']:.6f}")
add_row("Connected components", f"{li_metrics['connected_components']}", f"weak={molt_metrics['weakly_connected_components']:,} / strong={molt_metrics['strongly_connected_components']:,}")
add_row("Largest component", f"{li_metrics['largest_component_size']}", f"{molt_metrics['largest_weak_component']:,} (weak)")
add_row("Modularity (Louvain)", f"{li_metrics['modularity']:.6f}", f"{molt_metrics['modularity']:.6f}")
add_row("# Communities", f"{li_metrics['n_communities']}", f"{molt_metrics['n_communities']}")

comp_df = pd.DataFrame(comparison)
print(comp_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════
# TASK 9: Save outputs
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TASK 9: Saving Results")
print("=" * 70)

# Edgelist CSV
edge_df = pd.DataFrame(
    [(u, v, d["weight"]) for u, v, d in G.edges(data=True)],
    columns=["source", "target", "weight"],
)
edge_df.to_csv(os.path.join(OUT_DIR, "reply_edgelist.csv"), index=False)
print(f"  Saved: reply_edgelist.csv ({len(edge_df):,} edges)")

# Node features CSV
node_data = []
for node in G.nodes():
    node_data.append({
        "author": node,
        "in_degree": G.in_degree(node),
        "out_degree": G.out_degree(node),
        "total_interactions": interaction_count.get(node, 0),
        "community": partition.get(node, -1),
        "clustering_coeff": nx.clustering(G_undir, node),
    })
node_df = pd.DataFrame(node_data)
node_df.to_csv(os.path.join(OUT_DIR, "node_features.csv"), index=False)
print(f"  Saved: node_features.csv ({len(node_df):,} nodes)")

# Network pickle
with open(os.path.join(OUT_DIR, "reply_network.pickle"), "wb") as f:
    pickle.dump({"directed": G, "undirected": G_undir, "partition": partition}, f)
print(f"  Saved: reply_network.pickle")

# Metrics pickle for future comparison
with open(os.path.join(OUT_DIR, "moltbook_metrics.pickle"), "wb") as f:
    pickle.dump(molt_metrics, f)
print(f"  Saved: moltbook_metrics.pickle")

# Comparison CSV
comp_df.to_csv(os.path.join(OUT_DIR, "comparison_linkedin_vs_moltbook.csv"), index=False)
print(f"  Saved: comparison_linkedin_vs_moltbook.csv")

# Depth distribution CSV
depth_df = pd.DataFrame({"max_depth": depths})
depth_df.to_csv(os.path.join(OUT_DIR, "conversation_depths.csv"), index=False)
print(f"  Saved: conversation_depths.csv")

print(f"\nAll outputs in: {OUT_DIR}/")
print("Done.")
