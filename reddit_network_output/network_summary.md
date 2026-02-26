# Reddit Author Co-Posting Network Analysis

## Overview
This analysis builds a network where **human Reddit authors** are connected if they posted in the same subreddit.
- Edge weight = number of shared subreddits
- Filtered to authors with >= 3 posts
- Data from Pushshift Reddit archive (historical data ~2012)

## Basic Network Statistics

| Metric | Value |
|--------|-------|
| **Nodes (Authors)** | 1,924 |
| **Edges (Connections)** | 687,451 |
| **Density** | 0.371610 |
| **Average Degree** | 714.61 |
| **Median Degree** | 660 |
| **Max Degree** | 1,922 |
| **Min Degree** | 0 |
| **Avg Clustering Coefficient** | 0.8777 (full) |

## Connected Components

| Metric | Value |
|--------|-------|
| **Number of Components** | 2 |
| **Largest Component Size** | 1,923 |
| **Largest Component %** | 99.9% |

## Community Detection (Louvain)

| Metric | Value |
|--------|-------|
| **Number of Communities** | 5 |
| **Modularity Score** | 0.3981 |

### Top Communities and Their Dominant Subreddits

**Community 2** (715 authors)
- Top subreddits: AskReddit (10981), todayilearned (2636), askscience (1269), worldnews (1067), technology (784)

**Community 0** (656 authors)
- Top subreddits: worldnews (2465), technology (574), science (542), news (471), todayilearned (72)

**Community 1** (445 authors)
- Top subreddits: technology (1641), science (170), news (79), todayilearned (46), askscience (7)

**Community 3** (102 authors)
- Top subreddits: news (461), worldnews (40), todayilearned (11), science (10), technology (5)

**Community 4** (5 authors)
- Top subreddits: Bitcoin (25)

## Top 20 Most Active Authors (by Post Count)

| Rank | Author | Post Count |
|------|--------|------------|
| 1 |  | 15528 |
| 2 | twolf1 | 95 |
| 3 | dovgoldbery | 94 |
| 4 | davidreiss666 | 91 |
| 5 | readerseven | 88 |
| 6 | slaterhearst | 86 |
| 7 | ttruth1 | 72 |
| 8 | rajneeshjha | 63 |
| 9 | trottrot | 56 |
| 10 | drrichardcranium | 54 |
| 11 | anutensil | 53 |
| 12 | solinvictus | 51 |
| 13 | pauldaren86 | 50 |
| 14 | maxwellhill | 50 |
| 15 | williamboby81 | 50 |
| 16 | zecarioca | 47 |
| 17 | clgnews | 45 |
| 18 | searchengine17 | 40 |
| 19 | rgonews | 39 |
| 20 | alanx | 37 |


## Top 20 Most Connected Authors (by Degree)

| Rank | Author | Degree (Connections) |
|------|--------|---------------------|
| 1 |  | 1,922 |
| 2 | koavf | 1,802 |
| 3 | solinvictus | 1,765 |
| 4 | alkinda | 1,736 |
| 5 | arx0s | 1,736 |
| 6 | acusticthoughts | 1,674 |
| 7 | landchild | 1,664 |
| 8 | marywwriter | 1,664 |
| 9 | georedd | 1,633 |
| 10 | shazbaz | 1,561 |
| 11 | originalucifer | 1,535 |
| 12 | geordilaforge | 1,510 |
| 13 | nonamerican | 1,472 |
| 14 | sisko2k5 | 1,467 |
| 15 | recipriversexcluson | 1,467 |
| 16 | slaterhearst | 1,467 |
| 17 | nullok | 1,448 |
| 18 | occupythekitchen | 1,407 |
| 19 | imnotjesus | 1,406 |
| 20 | magister0 | 1,395 |


## Files Generated

- `author_network.graphml` - Full network graph
- `degree_distribution.png` - Histogram and log-log plot
- `network_visualization.png` - Network plot colored by community
- `community_sizes.png` - Community size bar chart
- `network_summary.md` - This summary file

## Notes

- This is a **human social network** from Reddit
- Used for comparison against Moltbook AI agent network
