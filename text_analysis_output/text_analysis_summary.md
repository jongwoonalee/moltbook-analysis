# Text Analysis: Moltbook (AI) vs Reddit (Human)

## Basic Text Statistics

| Metric | Moltbook (AI) | Reddit (Human) |
|--------|---------------|----------------|
| Total Posts | 55,809 | 55,000 |
| Valid Posts | 55,788 | 54,827 |
| Avg Words/Post | 176.2 | 49.5 |
| Median Words/Post | 95 | 18 |
| Max Words/Post | 800,016 | 2,259 |
| Total Words | 8,276,163 | 2,713,718 |
| Unique Words | 202,459 | 88,550 |
| Vocabulary Diversity | 0.0245 | 0.0326 |
| Type-Token Ratio | 0.3553 | 0.2857 |

## TF-IDF Distinctive Words

### Moltbook (AI) - Top 20
| Word | TF-IDF Score |
|------|--------------|
| moltbook | 0.1340 |
| isn | 0.0890 |
| doesn | 0.0496 |
| openclaw | 0.0492 |
| moltys | 0.0435 |
| crypto | 0.0397 |
| json | 0.0388 |
| 2026 | 0.0359 |
| agents | 0.3265 |
| ai | 0.2992 |
| usdc | 0.0297 |
| tokens | 0.0443 |
| autonomous | 0.0428 |
| token | 0.0674 |
| github | 0.0223 |
| solana | 0.0220 |
| substrate | 0.0209 |
| didn | 0.0190 |
| ve | 0.0617 |
| clawnch | 0.0183 |

### Reddit (Human) - Top 20
| Word | TF-IDF Score |
|------|--------------|
| redditors | 0.0254 |
| sopa | 0.0221 |
| boyfriend | 0.0198 |
| eli5 | 0.0213 |
| shes | 0.0231 |
| ive | 0.1167 |
| hes | 0.0242 |
| girl | 0.0497 |
| sex | 0.0310 |
| reddit | 0.2223 |
| girls | 0.0209 |
| wouldnt | 0.0175 |
| 2012 | 0.0316 |
| havent | 0.0191 |
| college | 0.0359 |
| amp | 0.0234 |
| subreddit | 0.0155 |
| ill | 0.0419 |
| women | 0.0264 |
| girlfriend | 0.0237 |


## Sentiment Analysis (VADER)

| Metric | Moltbook (AI) | Reddit (Human) |
|--------|---------------|----------------|
| Mean Compound | 0.3685 | 0.1011 |
| Median Compound | 0.5719 | 0.0000 |
| Std Compound | 0.5978 | 0.5004 |
| Positive Posts % | 65.2% | 41.3% |
| Negative Posts % | 19.7% | 26.1% |
| Neutral Posts % | 15.1% | 32.6% |

## Files Generated

- `tfidf_distinctive_words.png` - TF-IDF distinctive words comparison
- `sentiment_analysis.png` - Sentiment distribution and comparison
- `wordclouds.png` - Side-by-side word clouds
- `length_distribution.png` - Post length distribution comparison
- `text_analysis_summary.md` - This summary file
