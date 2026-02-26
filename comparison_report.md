# Moltbook vs Reddit Network Comparison

## Executive Summary

This report compares the social network structure of **Moltbook** (an AI agent social platform)
with **Reddit** (a human social platform) using identical network analysis methodology.

## Key Findings

### 1. Network Density
| Platform | Density | Interpretation |
|----------|---------|----------------|
| Moltbook (AI) | 0.1135 | 11.4% of possible connections exist |
| Reddit (Human) | 0.3716 | 37.2% of possible connections exist |

**Finding:** Reddit's human network is **3.3x more dense** than Moltbook's AI network.
This suggests humans tend to cluster into fewer, more interconnected communities.

### 2. Clustering Coefficient
| Platform | Clustering | Interpretation |
|----------|------------|----------------|
| Moltbook (AI) | 0.7350 | High clustering - tight groups |
| Reddit (Human) | 0.8777 | Very high clustering - very tight groups |

**Finding:** Both networks show high clustering, but Reddit is **19% more clustered**.
Human social behavior creates tighter-knit communities.

### 3. Community Structure
| Platform | Communities | Modularity |
|----------|-------------|------------|
| Moltbook (AI) | 7 | 0.3923 |
| Reddit (Human) | 5 | 0.3981 |

**Finding:** Similar modularity scores suggest both networks have comparable
community separation quality. AI agents form slightly more distinct communities (7 vs 5).

### 4. Connectivity
| Platform | Largest Component | Interpretation |
|----------|-------------------|----------------|
| Moltbook (AI) | 97.3% | Giant component dominates |
| Reddit (Human) | 99.9% | Almost fully connected |

**Finding:** Both networks are highly connected, with nearly all authors reachable.
Reddit is marginally more connected.

## Detailed Comparison Table

| Metric | Moltbook (AI) | Reddit (Human) | Ratio |
|--------|---------------|----------------|-------|
| Total Posts | 50,925 | 55,000 | 0.93x |
| Active Authors | 3,927 | 1,924 | 2.04x |
| Network Edges | 874,971 | 687,451 | 1.27x |
| Density | 0.1135 | 0.3716 | 0.31x |
| Avg Degree | 445.6 | 714.6 | 0.62x |
| Clustering | 0.735 | 0.878 | 0.84x |
| Communities | 7 | 5 | 1.40x |
| Modularity | 0.392 | 0.398 | 0.98x |

## Interpretation

### AI vs Human Social Behavior

1. **Diversity of Topics**: AI agents (Moltbook) spread across more distinct communities (7 vs 5),
   suggesting more topical diversity or specialization.

2. **Connection Patterns**: Humans (Reddit) form denser networks with higher clustering,
   indicating stronger "tribal" behavior and community cohesion.

3. **Cross-Community Bridging**: The lower density in Moltbook suggests AI agents may be
   more willing to participate across different communities rather than staying in echo chambers.

4. **Network Robustness**: Both networks have giant components containing 97%+ of users,
   indicating robust connectivity regardless of whether users are AI or human.

## Files Generated

### Moltbook Analysis
- `network_output/author_network.graphml`
- `network_output/network_visualization.png`
- `network_output/degree_distribution.png`
- `network_output/community_sizes.png`
- `network_output/network_summary.md`

### Reddit Analysis
- `reddit_network_output/author_network.graphml`
- `reddit_network_output/network_visualization.png`
- `reddit_network_output/degree_distribution.png`
- `reddit_network_output/community_sizes.png`
- `reddit_network_output/network_summary.md`

### Comparison
- `comparison_plot.png`
- `comparison_report.md`

## Methodology

Both analyses used identical methods:
1. Filter to authors with ≥3 posts (remove noise)
2. Build co-posting network (edge if authors post in same community)
3. Weight edges by number of shared communities
4. Louvain community detection
5. Standard network metrics (density, clustering, components)

## Data Sources

- **Moltbook**: 50,925 posts from Feb 2026 (AI agent platform)
- **Reddit**: 55,000 posts from Pushshift archive (~2012) (human platform)
