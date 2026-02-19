# Astronaut Log Sentiment Analyzer — System Design Document

**Author:** [Your Name]
**Date:** February 2026
**Status:** Pre-Implementation

---

## 1. Problem Statement

Long-duration spaceflight is one of humanity's hardest engineering problems — and one of its hardest *human* problems. NASA invests heavily in crew behavioral health research, but most of that data is locked behind medical privacy barriers. What *is* publicly available are two underutilized text corpora:

1. **The ISS Blog** — ~4,000 posts spanning 15+ years of continuous habitation, written during active missions
2. **NASA Oral History Transcripts** — Long-form interviews with 21 ISS program participants, recorded after their missions

This project asks: **Can we detect sentiment drift and emotional patterns across ISS mission timelines using NLP?** And more subtly: **Do we see different emotional signatures in real-time operational communications vs. retrospective personal narratives?**

The answer builds a complete data pipeline — scraping, storage, multi-model NLP analysis, and interactive visualization — that demonstrates applied ML engineering on a novel, domain-specific dataset.

---

## 2. System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                 │
│                                                                     │
│   NASA ISS Blog Archive          NASA Oral History Program          │
│   (396 pages, ~4K posts)         (21 ISS transcripts, PDF)         │
│   blogs.nasa.gov/spacestation    nasa.gov/history/...              │
└──────────────┬───────────────────────────┬──────────────────────────┘
               │                           │
               ▼                           ▼
┌──────────────────────────┐  ┌──────────────────────────────┐
│   ISS Blog Scraper       │  │   Oral History Scraper        │
│   - Paginated crawl      │  │   - PDF download              │
│   - HTML → text          │  │   - PyMuPDF extraction        │
│   - Rate-limited (1.5s)  │  │   - Q&A segment splitting     │
│   - Checkpoint/resume    │  │   - Header/footer cleanup     │
└──────────┬───────────────┘  └──────────────┬───────────────┘
           │                                  │
           ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        SQLite Database                              │
│                                                                     │
│   blog_posts │ oral_histories │ expeditions │ sentiment_results     │
│              │                │             │ emotion_results       │
│              │                │             │ linguistic_features   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Analysis Pipeline                              │
│                                                                     │
│   ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐       │
│   │ Sentiment    │  │ Emotion      │  │ Linguistic         │       │
│   │ (RoBERTa)   │  │ (DistilRoBE) │  │ (Rule-based)       │       │
│   │             │  │              │  │                    │       │
│   │ pos/neg/neu │  │ 7 emotions   │  │ readability,       │       │
│   │ scores      │  │ scores       │  │ pronouns, lexical  │       │
│   └─────────────┘  └──────────────┘  └────────────────────┘       │
│                                                                     │
│   Token Chunker: sliding window (400 tok, 100 overlap)             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                               │
│                                                                     │
│   ┌──────────────┐ ┌──────────────┐ ┌───────────┐ ┌────────────┐  │
│   │ Mission      │ │ Emotion      │ │ Expedition│ │ Oral       │  │
│   │ Timeline     │ │ Breakdown    │ │ Compare   │ │ Histories  │  │
│   │              │ │              │ │           │ │ Deep Dive  │  │
│   │ Line charts  │ │ Bar + radar  │ │ Side-by-  │ │ Per-astro  │  │
│   │ + heatmap    │ │ by expedition│ │ side cards│ │ profiles   │  │
│   └──────────────┘ └──────────────┘ └───────────┘ └────────────┘  │
│                                                                     │
│   Plotly charts · NASA dark theme · Interactive filters             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Sources

### 3.1 ISS Blog Archive

**What it is:** NASA's official Space Station blog at `blogs.nasa.gov/spacestation`. A chronological record of ISS operations — spacewalks, science experiments, crew arrivals/departures, vehicle traffic. Updated multiple times per week during active expeditions.

**Critical insight for the project:** These posts are written by NASA communications staff on the ground, not by the astronauts themselves. The writing is professional, factual, and emotionally restrained. A typical post reads like:

> *"Expedition 68 Flight Engineers Josh Cassada and Frank Rubio completed a 7-hour, 11-minute spacewalk today to install a new solar array..."*

This means raw sentiment scores will cluster around "neutral." Rather than treating this as a failure, we reframe it: **this is operational tone analysis.** We're measuring *how* NASA communicates about missions — does language become more urgent during emergencies? More celebratory during milestones? Do linguistic features (sentence complexity, lexical diversity) shift across expedition phases?

**Scraping strategy:**
- Archive is paginated: `blogs.nasa.gov/spacestation/page/{1..396}/`
- Each listing page contains ~10 post summaries with links to full articles
- Full articles contain the complete post text, author, date, and sometimes images
- We cache raw HTML locally to avoid re-fetching
- 1.5-second delay between requests (respectful rate limiting)
- Checkpoint file tracks which pages have been indexed, enabling resume after interruption
- **MVP mode:** Scrape only the last ~3 years (~100 pages, ~1K posts) to get results faster

### 3.2 NASA Oral History Transcripts

**What it is:** The Johnson Space Center History Portal maintains long-form interview transcripts with ISS program participants. These are 30-90 page PDFs of conversations between a NASA historian and an astronaut/engineer, conducted *after* their missions.

**Why this is the emotional counterweight:** Unlike the blog posts, oral histories are deeply personal. Astronauts describe fear during close calls, awe at seeing Earth from space, frustration with equipment failures, and joy at scientific discoveries. This gives us a genuine emotional signal to analyze.

**21 ISS-related participants:** Michael Barratt, Randy Brinkley, Robert Cabana, John Charles, Kevin Chilton, Laurie Hansen, Albert Holland, Gregory Johnson, Charles Lundquist, Jeffrey Manber, Hans Mark, Donald Pettit, Michael Read, Julie Robinson, Melanie Saunders, Michael Suffredini, Suzan Voss, Peggy Whitson, Jeffrey Williams, Sunita Williams, and one additional participant.

**Extraction strategy:**
- PDFs hosted at `nasa.gov/wp-content/uploads/` paths
- Extract text with PyMuPDF (fast, reliable, handles multi-column layouts)
- Clean artifacts: page numbers, headers ("NASA Johnson Space Center Oral History Project"), footers
- Split on interviewer name patterns to create Q&A segments (each question + answer = one analyzable unit)
- Temporal span: interviews conducted 1998–2017

---

## 4. Data Model

The core entities and their relationships:

```
expeditions (static reference data)
    │
    │  blog_posts.expedition_id → expeditions.number
    │
    ▼
blog_posts ──────────► sentiment_results
    │                  emotion_results
    │                  linguistic_features
    │
oral_histories ──────► sentiment_results
    │                  emotion_results
    │                  linguistic_features
```

### Tables

| Table | Purpose | Key Fields |
|-------|---------|------------|
| `expeditions` | Static ISS expedition metadata | number, name, start_date, end_date, crew (JSON) |
| `blog_posts` | Scraped blog content | url, title, author, date, text, expedition_id, word_count |
| `oral_histories` | Extracted interview segments | astronaut_name, segment_index, speaker (interviewer/astronaut), text, interview_date |
| `sentiment_results` | Sentiment model output | source_type, source_id, label (pos/neg/neu), score, model_name |
| `emotion_results` | Emotion model output | source_type, source_id, emotion, score, model_name |
| `linguistic_features` | Rule-based text features | source_type, source_id, flesch_score, avg_sentence_len, lexical_diversity, first_person_ratio, exclamation_count, question_count |

**Why SQLite?** This is a single-user analytics project with a bounded dataset (~4K blog posts + ~21 transcripts). SQLite is zero-config, works everywhere, and the entire database fits in a single file. No reason to introduce Postgres/Docker complexity for a dataset that fits in RAM.

---

## 5. Analysis Pipeline

Three analysis layers run on every text segment:

### 5.1 Sentiment Analysis

**Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`

This is a RoBERTa model fine-tuned on ~124M tweets for 3-class sentiment (positive, negative, neutral). Despite being trained on tweets, it generalizes well to both the formal blog register and the conversational oral history register. It outputs probability scores for each class, not just a label — important for tracking subtle shifts.

**Why this model over alternatives:**
- VADER (rule-based) is too crude for this domain — it would flag "explosion" as negative even in "a successful explosion of the booster stage"
- GPT-based approaches are expensive and overkill for classification
- This model is ~500MB, runs on CPU in seconds per batch, and is well-benchmarked

### 5.2 Emotion Detection

**Model:** `j-hartmann/emotion-english-distilroberta-base`

Outputs probability scores across 7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise. This gives us a much richer picture than positive/negative alone. We can ask questions like: "Is the dominant emotion during spacewalk posts *joy* or *surprise*?" or "Do oral history accounts of emergencies show more *fear* or *anger*?"

### 5.3 Linguistic Features (No Model Required)

Custom rule-based metrics computed with `textstat` and simple regex:

| Feature | What it tells us | Computation |
|---------|-----------------|-------------|
| Flesch readability | Text complexity / accessibility | `textstat.flesch_reading_ease()` |
| Avg sentence length | Writing density | Word count / sentence count |
| Lexical diversity | Vocabulary richness | Unique words / total words (type-token ratio) |
| First-person pronoun ratio | Personal vs. institutional voice | Count of I/me/my/we/our / total words |
| Exclamation count | Emphasis / excitement | Simple regex |
| Question count | Interrogative density | Simple regex |

These features are cheap to compute and add analytical depth that the neural models miss. For the emotionally flat blog posts, linguistic features may actually be more informative than sentiment scores.

### 5.4 Token Chunking

Both HuggingFace models have a 512-token context window. Many blog posts and most oral history segments exceed this. We handle it with a **sliding window chunker:**

1. Tokenize the text using the model's tokenizer
2. If token count ≤ 512: analyze directly
3. If token count > 512: create overlapping windows of 400 tokens, stepping by 300 (100-token overlap)
4. Run each window through the model
5. Aggregate: mean of all window scores per class/emotion

The overlap ensures no sentence is split mid-context at a boundary.

### 5.5 Expedition Mapping

Blog posts are dated but don't explicitly mention which ISS expedition they belong to. We maintain a static `expeditions.json` file mapping expedition numbers to date ranges. During analysis, each blog post is assigned an expedition number based on its publication date falling within an expedition's date range. This enables the "by expedition" groupings in the dashboard.

---

## 6. Dashboard Design

### Design Philosophy

The dashboard uses a NASA-inspired dark theme (dark navy background, white text, blue/cyan accents) that both looks professional and provides good contrast for data visualization. All charts use Plotly for hover-based interactivity. The multi-page Streamlit layout keeps each analytical perspective focused.

### Page 1: Mission Timeline

**Core question:** "How does operational tone evolve over the ISS program's lifetime?"

- **Main visualization:** Rolling-average sentiment line chart (7-day or 30-day window, user-selectable) with vertical lines marking expedition boundaries
- **Supporting visualization:** Emotion heatmap — time on X axis, 7 emotions on Y axis, color intensity = score
- **Filters:** Date range slider, expedition number selector

### Page 2: Emotion Breakdown

**Core question:** "What's the emotional fingerprint of each expedition?"

- **Main visualization:** Grouped bar chart — expeditions on X, emotion scores on Y, one bar per emotion
- **Supporting visualization:** Radar/spider chart comparing the emotion profile of 2-3 user-selected expeditions
- **Insight:** Expeditions with EVA emergencies or crew rotations may show distinct emotional signatures

### Page 3: Expedition Compare

**Core question:** "How do two expeditions differ in communication style?"

- **Layout:** Side-by-side cards (user picks two expeditions)
- **Per card:** Summary stats (post count, avg sentiment, dominant emotion), word cloud, linguistic feature bar chart
- **Insight:** Comparison reveals whether communication style varies by crew commander, mission phase, or era

### Page 4: Oral Histories Deep Dive

**Core question:** "What emotional arc does an astronaut describe when recounting their mission?"

- **Main visualization:** Per-astronaut sentiment line chart showing how sentiment moves across the interview (segment by segment)
- **Supporting visualization:** Emotion progression within a single interview
- **Key feature:** Quotes table — shows the actual text segments alongside their detected emotions, sortable by emotion type or score intensity
- **This is the "wow" page** — connecting raw human language to model predictions

---

## 7. Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Database | SQLite | Single-user, bounded dataset, zero-config, portable |
| Sentiment model | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Good generalization across registers, probability outputs, CPU-friendly |
| Emotion model | `j-hartmann/emotion-english-distilroberta-base` | 7-class granularity, well-benchmarked, small footprint |
| PDF extraction | PyMuPDF | Fast, reliable, handles NASA's PDF formatting |
| Dashboard | Streamlit | Rapid prototyping, native Python, easy deployment |
| Charts | Plotly | Interactive hover/zoom, good dark theme support |
| Data validation | Pydantic | Type safety at ingestion boundary, clear error messages |
| Blog ↔ expedition mapping | Static JSON | Expedition dates change infrequently; avoids external API dependency |
| Operational vs. personal framing | Two-corpus design | Turns a weakness (flat blog tone) into a strength (comparative analysis) |

---

## 8. Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Blog posts are emotionally flat | Sentiment charts look boring | High | Reframe as "operational tone"; lean on linguistic features; use blogs as *baseline* against which oral histories contrast |
| Scraping ~4K pages takes hours | Slow dev iteration | Medium | Checkpoint/resume, raw HTML cache, MVP mode (recent posts only) |
| RoBERTa 512-token limit | Long texts get truncated | High | Sliding window chunker with overlap + score aggregation |
| PDF extraction artifacts | Dirty text → bad analysis | Medium | Regex cleaning for headers/footers/page numbers; manual spot-checks |
| Model disagreement | Sentiment says "positive" but emotion says "fear" | Low | These measure different things — document this in dashboard tooltips |
| NASA changes site structure | Scrapers break | Low | Raw HTML cache means we only need to scrape once; selectors are defensive |
| WSL2 `/mnt/c/` slow I/O | Database operations are sluggish | Medium | Store SQLite + raw data on Linux filesystem (`~/astro-data/`), code stays on `/mnt/c/` for IDE access |

---

## 9. Project Structure

```
astro-sentiment/
├── pyproject.toml              # Project metadata + dependencies
├── requirements.txt            # Pip-installable dependency list
├── .gitignore                  # Ignore data/, raw HTML, .db files
├── docs/
│   ├── DESIGN_DOC.md           # This document
│   └── DESIGN_DOC_AI.md       # AI/LLM implementation reference
├── src/
│   ├── scraping/
│   │   ├── __init__.py
│   │   ├── iss_blog_scraper.py     # ISS blog paginated scraper
│   │   └── oral_history_scraper.py # PDF download + text extraction
│   ├── data/
│   │   ├── __init__.py
│   │   ├── models.py               # Pydantic data models
│   │   └── db.py                   # SQLite schema + CRUD helpers
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── sentiment.py            # RoBERTa sentiment pipeline
│   │   ├── emotion.py              # Emotion detection pipeline
│   │   ├── linguistic.py           # Rule-based text features
│   │   └── runner.py               # Orchestrate all analysis
│   └── dashboard/
│       ├── __init__.py
│       ├── app.py                  # Streamlit entry point
│       └── pages/
│           ├── 1_Mission_Timeline.py
│           ├── 2_Emotion_Breakdown.py
│           ├── 3_Expedition_Compare.py
│           └── 4_Oral_Histories.py
├── data/
│   ├── astro_sentiment.db          # SQLite database (gitignored)
│   ├── raw/                        # Cached HTML + PDFs (gitignored)
│   └── expeditions.json            # Static expedition metadata
├── notebooks/                      # Jupyter exploration notebooks
├── scripts/
│   ├── scrape.py                   # CLI: run scrapers
│   └── analyze.py                  # CLI: run analysis pipeline
└── tests/
    ├── test_models.py
    ├── test_chunking.py
    └── test_expedition_mapping.py
```

---

## 10. Open Questions

1. **Scope for MVP:** Should we scrape all 396 pages (~4K posts) or start with last 3 years (~1K posts)? The full scrape takes ~1.5 hours.
2. **Deployment:** Is Streamlit Community Cloud sufficient, or do we want Docker + a VM?
3. **Additional data sources:** Reddit AMAs by astronauts could add a third register (casual/public), but adds scraping complexity.
