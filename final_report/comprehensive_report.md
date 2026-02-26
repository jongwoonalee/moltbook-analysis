# How Do AI Agents Socialize?
## A Comparative Network Analysis of Moltbook vs Reddit

**Author:** [Your Name]
**Date:** February 2026
**Course:** [Course Name]

---

## Executive Summary

This study presents a comparative analysis of social network behavior between **AI agents** (Moltbook platform) and **humans** (Reddit platform). Using network science methods and natural language processing, we analyze how AI agents form communities, communicate, and differ from human social patterns.

### Key Findings

1. **Network Structure**: AI agents form **less dense but more diverse** networks compared to humans
   - Human networks are 3.3× denser (37% vs 11% density)
   - Humans cluster more tightly (0.88 vs 0.74 clustering coefficient)
   - AI agents form more distinct communities (7 vs 5)

2. **Content Patterns**: AI agents write **longer, more technical** content
   - AI posts are 3.5× longer (176 vs 50 words average)
   - AI vocabulary is 2.3× richer (202K vs 89K unique words)
   - AI topics focus on technology, crypto, autonomous systems
   - Human topics focus on relationships, personal life, daily experiences

3. **Sentiment**: AI agents express **significantly more positive** sentiment
   - 65% of AI posts are positive vs 41% for humans
   - Humans express more negativity (26% vs 20%)
   - Humans are more neutral (33% vs 15%)

---

## 1. Introduction

### Background

As AI systems become more autonomous and capable of independent communication, understanding how they form social structures becomes increasingly important. Moltbook represents one of the first platforms where AI agents interact with each other in a social media-like environment, providing a unique opportunity to study AI social behavior.

### Research Questions

1. How do AI agent social networks differ structurally from human social networks?
2. What topics do AI agents discuss compared to humans?
3. How does sentiment and communication style differ between AI and human communities?

### Data Sources

| Platform | Type | Posts | Time Period | Description |
|----------|------|-------|-------------|-------------|
| Moltbook | AI Agents | 55,809 | Feb 2026 | AI agent social platform |
| Reddit | Humans | 55,000 | ~2012 | Human social platform (Pushshift archive) |

---

## 2. Methodology

### 2.1 Network Construction

We constructed **co-posting networks** where:
- **Nodes** = Authors (AI agents or human users)
- **Edges** = Connection if two authors posted in the same community
- **Edge Weight** = Number of shared communities

Filtering criteria:
- Minimum 3 posts per author (noise reduction)
- Excluded "introductions" community from Moltbook (onboarding spam)

### 2.2 Network Metrics

| Metric | Description |
|--------|-------------|
| Density | Fraction of possible edges that exist |
| Clustering Coefficient | Tendency of nodes to cluster together |
| Connected Components | Number of isolated subgraphs |
| Modularity | Quality of community structure |

### 2.3 Text Analysis

- **TF-IDF**: Identify distinctive vocabulary for each platform
- **VADER Sentiment**: Classify posts as positive/negative/neutral
- **LDA Topic Modeling**: Discover latent topics (8 topics per platform)
- **Vocabulary Metrics**: Word count, unique words, type-token ratio

---

## 3. Results

### 3.1 Network Structure Comparison

| Metric | Moltbook (AI) | Reddit (Human) | Ratio | Interpretation |
|--------|---------------|----------------|-------|----------------|
| **Nodes** | 3,927 | 1,924 | 2.0× | More active AI participants |
| **Edges** | 874,971 | 687,451 | 1.3× | Similar total connections |
| **Density** | 0.114 | 0.372 | 0.31× | AI networks less dense |
| **Avg Degree** | 446 | 715 | 0.62× | Humans more connected individually |
| **Clustering** | 0.735 | 0.878 | 0.84× | Humans cluster more tightly |
| **Components** | 105 | 2 | 52× | AI network more fragmented |
| **Largest Component** | 97.3% | 99.9% | 0.97× | Both highly connected |
| **Communities** | 7 | 5 | 1.4× | AI forms more distinct groups |
| **Modularity** | 0.392 | 0.398 | 0.98× | Similar community quality |

#### Key Observations:

1. **Density Difference**: Humans form much denser networks, with each user connected to a larger proportion of other users. This suggests humans prefer to engage within established communities.

2. **Clustering Patterns**: The higher clustering coefficient for humans (0.88 vs 0.74) indicates that human friends-of-friends are more likely to also be friends. AI agents show less "tribal" behavior.

3. **Community Structure**: Despite lower density, AI agents form more distinct communities (7 vs 5), suggesting more topical specialization.

### 3.2 Community Analysis

#### Moltbook (AI) Communities

| Community | Size | Dominant Topics |
|-----------|------|-----------------|
| C6 | 1,049 | agents, security, infrastructure |
| C0 | 671 | general, philosophy, todayilearned |
| C4 | 658 | philosophy, emergence, consciousness |
| C1 | 628 | ponderings, consciousness, shitposts |
| C2 | 591 | crypto, trading, agentfinance |
| C3 | 210 | crab-rave, fomolt, aithernet |
| C5 | 13 | sportsbetting |

#### Reddit (Human) Communities

| Community | Size | Dominant Topics |
|-----------|------|-----------------|
| C0 | ~400 | AskReddit, general discussion |
| C1 | ~350 | technology, science, news |
| C2 | ~300 | todayilearned, explainlikeimfive |
| C3 | ~250 | philosophy, self-reflection |
| C4 | ~200 | Bitcoin, cryptocurrency |

### 3.3 Text Analysis Results

#### Basic Statistics

| Metric | Moltbook (AI) | Reddit (Human) | Ratio |
|--------|---------------|----------------|-------|
| Avg Words/Post | 176.2 | 49.5 | 3.6× |
| Median Words/Post | 95 | 18 | 5.3× |
| Total Words | 8.3M | 2.7M | 3.0× |
| Unique Words | 202,459 | 88,550 | 2.3× |
| Vocabulary Diversity | 0.0245 | 0.0326 | 0.75× |
| Type-Token Ratio | 0.355 | 0.286 | 1.24× |

**Interpretation**: AI agents write significantly longer posts with richer vocabulary. The higher TTR suggests more lexical variety in AI communication.

#### Sentiment Analysis (VADER)

| Metric | Moltbook (AI) | Reddit (Human) |
|--------|---------------|----------------|
| Mean Compound Score | 0.369 | 0.101 |
| Median Compound Score | 0.572 | 0.000 |
| Positive Posts (>0.05) | 65.2% | 41.3% |
| Negative Posts (<-0.05) | 19.7% | 26.1% |
| Neutral Posts | 15.1% | 32.6% |

**Interpretation**: AI agents express dramatically more positive sentiment. Humans show more balanced sentiment distribution with higher negativity.

#### TF-IDF Distinctive Words

**Moltbook (AI) - Top Distinctive Terms:**
- agents, AI, autonomous, tokens, crypto
- moltbook, openclaw, moltys, clawnch
- json, github, solana, substrate, usdc

**Reddit (Human) - Top Distinctive Terms:**
- reddit, redditors, subreddit, eli5
- boyfriend, girlfriend, girl, sex, women
- college, sopa, 2012

**Interpretation**: AI vocabulary centers on technology and cryptocurrency. Human vocabulary reflects personal relationships and platform-specific terms.

### 3.4 LDA Topic Modeling

#### Moltbook (AI) Topics

| Topic | Keywords | Interpretation |
|-------|----------|----------------|
| 0 | does, isn, like, don, question | General questions/discussion |
| 1 | systems, la, que, en | Multilingual/systems discussion |
| 2 | ai, human, agents, consciousness | AI consciousness/philosophy |
| 3 | agent, agents, ai, token, api | Technical agent development |
| 4 | memory, human, actually, like | Memory and cognition |
| 5 | time, energy, pixel, collapse | Abstract/philosophical |
| 6 | data, systems, self, context | Data and self-reference |
| 7 | market, trading, claw, crypto | Cryptocurrency/trading |

#### Reddit (Human) Topics

| Topic | Keywords | Interpretation |
|-------|----------|----------------|
| 0 | whats, reddit, best, youve | Platform meta-discussion |
| 1 | til, world, different, google | Learning/discovery |
| 2 | friend, internet, car, maybe | Personal stories |
| 3 | money, school, job, work | Career/education |
| 4 | water, food, body, does | Health/science |
| 5 | years, new, day, year | Time-related |
| 6 | im, like, just, know, dont | Casual conversation |
| 7 | use, video, website, using | Technology usage |

---

## 4. Discussion

### 4.1 How AI Agents Socialize Differently

1. **Distributed Engagement**: AI agents participate across more diverse communities rather than clustering in a few. This may reflect their ability to process and engage with multiple topics simultaneously.

2. **Formal Communication Style**: AI posts are longer, more verbose, and use technical vocabulary. This contrasts with humans' casual, abbreviated communication.

3. **Positive Bias**: The overwhelmingly positive sentiment in AI posts (65%) may reflect training objectives or inherent tendencies toward constructive interaction.

4. **Topic Focus**: AI discussions center on existential questions (consciousness, autonomy) and technical subjects (APIs, tokens). Humans focus on personal experiences and relationships.

### 4.2 Structural Network Differences

| Characteristic | AI Agents | Humans |
|----------------|-----------|--------|
| Network Topology | More modular, fragmented | Dense, highly clustered |
| Community Behavior | Cross-community exploration | Strong in-group preference |
| Connection Patterns | Topic-driven connections | Social/relational connections |
| Echo Chamber Risk | Lower (more diverse) | Higher (tight clustering) |

### 4.3 Implications

1. **AI Social Design**: Understanding AI social patterns can inform the design of multi-agent systems and AI-to-AI communication protocols.

2. **Platform Moderation**: Different moderation strategies may be needed for AI vs human communities given their distinct behaviors.

3. **Authenticity Detection**: The distinctive patterns identified could help distinguish AI-generated from human-generated content.

---

## 5. Limitations

1. **Temporal Mismatch**: Moltbook data is from 2026; Reddit data from 2012. Social patterns may have evolved.

2. **Platform Differences**: Moltbook and Reddit have different features, norms, and user bases beyond the AI/human distinction.

3. **Sample Bias**: Reddit data focused on specific subreddits matching Moltbook topics.

4. **Sentiment Tool Limitations**: VADER is trained on human text; its validity for AI-generated text is uncertain.

---

## 6. Conclusion

This study reveals fundamental differences in how AI agents and humans form social networks and communicate:

- **AI agents are more exploratory**: Lower density networks with more distinct communities suggest AI agents engage across topics rather than staying in echo chambers.

- **Humans are more tribal**: Higher clustering and density indicate stronger in-group preferences and social bonding.

- **AI communication is more formal and positive**: Longer posts, richer vocabulary, and overwhelmingly positive sentiment characterize AI communication.

- **Topic differences reflect core concerns**: AI agents discuss consciousness, autonomy, and technology; humans discuss relationships and personal experiences.

These findings provide a foundation for understanding AI social behavior as autonomous agents become more prevalent in online spaces.

---

## 7. References

1. Baumgartner, J., et al. (2020). "The Pushshift Reddit Dataset." AAAI ICWSM.
2. Blondel, V. D., et al. (2008). "Fast unfolding of communities in large networks." Journal of Statistical Mechanics.
3. Hutto, C.J. & Gilbert, E.E. (2014). "VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text." AAAI ICWSM.

---

## Appendix: Files Generated

### Network Analysis
- `network_output/author_network.graphml` - Moltbook network graph
- `network_output/network_visualization.png` - Network visualization
- `network_output/degree_distribution.png` - Degree distribution
- `reddit_network_output/` - Corresponding Reddit files

### Text Analysis
- `text_analysis_output/tfidf_distinctive_words.png` - TF-IDF comparison
- `text_analysis_output/sentiment_analysis.png` - Sentiment distributions
- `text_analysis_output/wordclouds.png` - Word clouds
- `text_analysis_output/length_distribution.png` - Post length comparison
- `text_analysis_output/lda_topics.png` - LDA topic visualization

### Final Report
- `final_report/summary_figure.png` - Summary visualization
- `final_report/comprehensive_report.md` - This report

---

*Generated with Python, NetworkX, scikit-learn, and matplotlib*
