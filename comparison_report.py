#!/usr/bin/env python3
"""
Generate comparison report: Moltbook (AI agents) vs Reddit (humans)
"""

import matplotlib.pyplot as plt
import numpy as np

# Moltbook (AI Agent Network) Statistics
moltbook = {
    'name': 'Moltbook\n(AI Agents)',
    'posts': 50925,
    'nodes': 3927,
    'edges': 874971,
    'density': 0.113504,
    'avg_degree': 445.62,
    'median_degree': 366,
    'max_degree': 2616,
    'clustering': 0.7350,
    'components': 105,
    'largest_component_pct': 97.3,
    'communities': 7,
    'modularity': 0.3923,
}

# Reddit (Human Network) Statistics
reddit = {
    'name': 'Reddit\n(Humans)',
    'posts': 55000,
    'nodes': 1924,
    'edges': 687451,
    'density': 0.371610,
    'avg_degree': 714.61,
    'median_degree': 714,  # Approximate
    'max_degree': 1900,  # Approximate from output
    'clustering': 0.8777,
    'components': 2,
    'largest_component_pct': 99.9,
    'communities': 5,
    'modularity': 0.3981,
}

def create_comparison_table():
    """Print comparison table."""
    print("=" * 80)
    print("MOLTBOOK (AI AGENTS) vs REDDIT (HUMANS) - NETWORK COMPARISON")
    print("=" * 80)
    print()

    metrics = [
        ('Total Posts', 'posts', '{:,}'),
        ('Active Authors (≥3 posts)', 'nodes', '{:,}'),
        ('Network Edges', 'edges', '{:,}'),
        ('Network Density', 'density', '{:.4f}'),
        ('Average Degree', 'avg_degree', '{:.2f}'),
        ('Clustering Coefficient', 'clustering', '{:.4f}'),
        ('Connected Components', 'components', '{:,}'),
        ('Largest Component %', 'largest_component_pct', '{:.1f}%'),
        ('Communities (Louvain)', 'communities', '{:,}'),
        ('Modularity Score', 'modularity', '{:.4f}'),
    ]

    print(f"{'Metric':<35} {'Moltbook (AI)':<20} {'Reddit (Human)':<20} {'Ratio':<15}")
    print("-" * 90)

    for label, key, fmt in metrics:
        m_val = moltbook[key]
        r_val = reddit[key]

        m_str = fmt.format(m_val)
        r_str = fmt.format(r_val)

        if isinstance(m_val, (int, float)) and isinstance(r_val, (int, float)) and r_val != 0:
            ratio = m_val / r_val
            ratio_str = f"{ratio:.2f}x"
        else:
            ratio_str = "-"

        print(f"{label:<35} {m_str:<20} {r_str:<20} {ratio_str:<15}")

    print()
    return metrics

def create_comparison_plots():
    """Create comparison visualizations."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Moltbook (AI Agents) vs Reddit (Humans): Network Comparison', fontsize=16, fontweight='bold')

    colors = ['#3498db', '#e74c3c']  # Blue for Moltbook, Red for Reddit
    labels = ['Moltbook (AI)', 'Reddit (Human)']

    # 1. Network Size
    ax = axes[0, 0]
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width/2, [moltbook['nodes'], reddit['nodes']], width, label='Authors', color=colors[0], alpha=0.8)
    ax.bar(x + width/2, [moltbook['edges']/1000, reddit['edges']/1000], width, label='Edges (K)', color=colors[1], alpha=0.8)
    ax.set_ylabel('Count')
    ax.set_title('Network Size', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_yscale('log')

    # 2. Density
    ax = axes[0, 1]
    densities = [moltbook['density'], reddit['density']]
    bars = ax.bar(labels, densities, color=colors, alpha=0.8)
    ax.set_ylabel('Density')
    ax.set_title('Network Density', fontweight='bold')
    for bar, val in zip(bars, densities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    # 3. Average Degree
    ax = axes[0, 2]
    degrees = [moltbook['avg_degree'], reddit['avg_degree']]
    bars = ax.bar(labels, degrees, color=colors, alpha=0.8)
    ax.set_ylabel('Average Degree')
    ax.set_title('Average Connections per Author', fontweight='bold')
    for bar, val in zip(bars, degrees):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val:.0f}', ha='center', va='bottom', fontweight='bold')

    # 4. Clustering Coefficient
    ax = axes[1, 0]
    clustering = [moltbook['clustering'], reddit['clustering']]
    bars = ax.bar(labels, clustering, color=colors, alpha=0.8)
    ax.set_ylabel('Clustering Coefficient')
    ax.set_title('Network Clustering (Tightness)', fontweight='bold')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, clustering):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    # 5. Communities
    ax = axes[1, 1]
    communities = [moltbook['communities'], reddit['communities']]
    bars = ax.bar(labels, communities, color=colors, alpha=0.8)
    ax.set_ylabel('Number of Communities')
    ax.set_title('Community Structure (Louvain)', fontweight='bold')
    for bar, val in zip(bars, communities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val}', ha='center', va='bottom', fontweight='bold')

    # 6. Modularity
    ax = axes[1, 2]
    modularity = [moltbook['modularity'], reddit['modularity']]
    bars = ax.bar(labels, modularity, color=colors, alpha=0.8)
    ax.set_ylabel('Modularity Score')
    ax.set_title('Community Separation Quality', fontweight='bold')
    ax.set_ylim(0, 0.5)
    for bar, val in zip(bars, modularity):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('comparison_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved comparison_plot.png")

def generate_markdown_report():
    """Generate markdown comparison report."""

    report = """# Moltbook vs Reddit Network Comparison

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
"""

    with open('comparison_report.md', 'w') as f:
        f.write(report)
    print("Saved comparison_report.md")

def main():
    create_comparison_table()
    create_comparison_plots()
    generate_markdown_report()

    print()
    print("=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print()
    print("1. DENSITY: Reddit humans are 3.3x more densely connected than Moltbook AI agents")
    print("2. CLUSTERING: Both networks are highly clustered, Reddit slightly more (0.88 vs 0.74)")
    print("3. COMMUNITIES: AI agents form more distinct communities (7 vs 5)")
    print("4. MODULARITY: Similar community separation quality (~0.39-0.40)")
    print("5. CONNECTIVITY: Both networks are essentially fully connected (97-99%)")
    print()

if __name__ == '__main__':
    main()
