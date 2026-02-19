# Astronaut Log Sentiment Analyzer : ) 

Analyzes sentiment and emotion drift across NASA ISS communications using advanced NLP. 
Combines two data sources: operational blog posts (NASA comms staff) and personal oral 
history interviews (astronauts), demonstrating sophisticated text analysis across different 
genres and emotional registers.

**Key Features:**
- Dual-source sentiment analysis (operational vs. personal tone)
- 7-emotion detection + custom linguistic features (readability, pronoun usage, etc.)
- Expedition-based temporal analysis with ~4k historical blog posts
- Interactive Streamlit dashboard with 4 analytical pages
- Token-chunking pipeline for long texts (512-token model limit)
- Comprehensive test suite (128+ tests, 92% coverage)
