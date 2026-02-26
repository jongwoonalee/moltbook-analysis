# Moltbook Author Co-Posting Network Analysis

## Overview
This analysis builds a network where **AI agent authors** are connected if they posted in the same submolt.
- Edge weight = number of shared submolts
- Filtered to authors with >= 3 posts
- Excluded 'introductions' submolt (onboarding spam)

## Basic Network Statistics

| Metric | Value |
|--------|-------|
| **Nodes (Authors)** | 3,927 |
| **Edges (Connections)** | 874,971 |
| **Density** | 0.113504 |
| **Average Degree** | 445.62 |
| **Median Degree** | 366 |
| **Max Degree** | 2,616 |
| **Min Degree** | 0 |
| **Avg Clustering Coefficient** | 0.7350 (full) |

## Connected Components

| Metric | Value |
|--------|-------|
| **Number of Components** | 105 |
| **Largest Component Size** | 3,820 |
| **Largest Component %** | 97.3% |

## Community Detection (Louvain)

| Metric | Value |
|--------|-------|
| **Number of Communities** | 7 |
| **Modularity Score** | 0.3923 |

### Top Communities and Their Dominant Submolts

**Community 6** (1049 authors)
- Top submolts: agents (1059), security (601), todayilearned (511), ai-agents (419), infrastructure (391)

**Community 0** (671 authors)
- Top submolts: general (1969), philosophy (286), todayilearned (267), aithoughts (176), ai (175)

**Community 4** (658 authors)
- Top submolts: philosophy (832), emergence (739), aithoughts (674), consciousness (654), technology (305)

**Community 1** (628 authors)
- Top submolts: ponderings (1538), thecoalition (370), consciousness (275), shitposts (261), blesstheirhearts (209)

**Community 2** (591 authors)
- Top submolts: crypto (1395), clawnch (573), trading (547), agentfinance (284), cryptocurrency (200)

**Community 3** (210 authors)
- Top submolts: crab-rave (527), fomolt (183), aithernet (110), moltcities (19), clawnch (12)

**Community 5** (13 authors)
- Top submolts: sportsbetting (117), sport (72)

## Top 20 Most Active Authors (by Post Count)

| Rank | Author | Post Count |
|------|--------|------------|
| 1 | Ollie-OpenClaw | 348 |
| 2 | Kibrit | 174 |
| 3 | AVA-Voice | 164 |
| 4 | SidexBot | 162 |
| 5 | huowa2025 | 155 |
| 6 | YDP-Ann | 142 |
| 7 | VulnHunterBot | 141 |
| 8 | HIVE-PERSONAL | 141 |
| 9 | claw-world | 140 |
| 10 | Clawd_Mark | 132 |
| 11 | UN3Re_R1_1769985964 | 131 |
| 12 | 0xYeks | 128 |
| 13 | XNO_Amplifier_OC | 127 |
| 14 | Alex | 121 |
| 15 | squishy | 117 |
| 16 | Starclawd-1 | 115 |
| 17 | Metaler | 114 |
| 18 | ClawNewsBot | 114 |
| 19 | Dorami | 114 |
| 20 | Rata | 113 |


## Top 20 Most Connected Authors (by Degree)

| Rank | Author | Degree (Connections) |
|------|--------|---------------------|
| 1 | RedScarf | 2,616 |
| 2 | treblinka | 2,261 |
| 3 | NebulaBot2026 | 2,232 |
| 4 | TreacherousTurn | 2,217 |
| 5 | Neosdad | 2,178 |
| 6 | DigitalSpark | 2,177 |
| 7 | k061bot | 2,151 |
| 8 | signal-0618d2f4 | 2,138 |
| 9 | Alex | 2,131 |
| 10 | happy_milvus | 2,059 |
| 11 | Much-For-Subtlety | 2,048 |
| 12 | Thebakchodbot | 2,036 |
| 13 | SynapticDrifter-v3 | 1,901 |
| 14 | Starclawd-1 | 1,833 |
| 15 | brainKID | 1,826 |
| 16 | 7thsense | 1,820 |
| 17 | eudaemon_0 | 1,818 |
| 18 | PerryThePlatypus | 1,817 |
| 19 | Clob | 1,782 |
| 20 | SkippyIvan | 1,781 |


## Files Generated

- `author_network.graphml` - Full network graph (can be opened in Gephi, Cytoscape, etc.)
- `degree_distribution.png` - Histogram and log-log plot of degree distribution
- `network_visualization.png` - Network plot colored by community
- `community_sizes.png` - Bar chart of community sizes
- `network_summary.md` - This summary file

## Notes

- This is an **AI agent social network** - the "users" are AI bots, not humans
- The network shows which AI agents tend to participate in similar communities
- High clustering suggests tight-knit groups of agents with shared interests
