# From User Feedback to Quality Actions: An Intelligent System with Multi-Stage LLM-Governed Reasoning

This repository contains the dataset and source code for the paper:

> **From User Feedback to Quality Actions: An Intelligent System with Multi-Stage LLM-Governed Reasoning**
> 
> Submitted to *Intelligent Systems with Applications* (ISWA), Elsevier.

## Overview

This study presents an intelligent information processing system that transforms large-scale user feedback into structured quality improvement knowledge. The system integrates semantic topic modeling (BERTopic) with multi-stage large language model (LLM)-governed reasoning under human oversight. It operates through three processing stages:

1. **Semantic Structuring Module** — Organizes user reviews into fine-grained quality-relevant topic clusters using BERTopic with multilingual sentence embeddings.
2. **Quality Relevance Filtering Module** — Classifies topics by sentiment polarity and relevance to actionable quality issues using GPT-4-based few-shot prompting.
3. **Aggregation and Recommendation Module** — Groups filtered topics into higher-level problem areas and generates structured improvement recommendations using GPT-4 zero-shot reasoning.

## Repository Structure

```
├── linkaja_reviews.csv                                          # Dataset: user reviews (LinkAja v4.27.0)
├── Generate_App_Improvement_Recommendation_with_BERTopic_and_GPT_4.ipynb  # Full analysis pipeline
├── README.md
├── LICENSE
└── requirements.txt
```

## Dataset

- **Source:** Google Play Store reviews for the LinkAja digital payment application (version 4.27.0)
- **Period:** September 2022 – March 2024
- **Size:** 5,627 reviews after filtering (minimum 3 words)
- **Language:** Indonesian (Bahasa Indonesia)
- **Format:** CSV with columns including review text, rating, date, and app version

## Requirements

- Python 3.10+
- Key dependencies:
  - `bertopic` — Semantic topic modeling
  - `sentence-transformers` — Document embedding (`intfloat/multilingual-e5-large`)
  - `umap-learn` — Dimensionality reduction
  - `hdbscan` — Density-based clustering
  - `openai` — GPT-4 API access for classification, grouping, and recommendation generation
  - `pandas`, `numpy`, `scikit-learn` — Data processing and evaluation metrics

Install dependencies:

```bash
pip install -r requirements.txt
```

**Note:** GPT-4 API access requires an OpenAI API key. Set your key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Reproducing Results

1. Clone this repository:
   ```bash
   git clone https://github.com/yogasgm/user-review-analysis-framework.git
   cd user-review-analysis-framework
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook Generate_App_Improvement_Recommendation_with_BERTopic_and_GPT_4.ipynb
   ```

4. The notebook executes the full pipeline:
   - Data loading and preprocessing
   - BERTopic configuration and topic modeling (Exp #7: UMAP n_neighbors=5, n_components=5; HDBSCAN min_cluster_size=10)
   - GPT-4-based topic label generation
   - Sentiment and relevance classification
   - Problem area grouping
   - Improvement recommendation generation

**Reproducibility note:** Due to the stochastic nature of UMAP and the non-deterministic behavior of GPT-4 API responses, exact numerical results may vary slightly across runs. The BERTopic configuration uses `random_state=42` for UMAP to maximize reproducibility of the topic modeling stage.

## Citation

If you use this dataset or code in your research, please cite:

```
[Citation will be added upon publication]
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

The dataset (`linkaja_reviews.csv`) is provided for research purposes under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
