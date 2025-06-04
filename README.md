# BBC News Topic Modeling and Clustering

This project compares topic modeling and text clustering techniques on BBC news articles. We use both KNIME and Python-based approaches to assess model quality and coherence in unsupervised text analysis.

## Objective

To determine which models best discover underlying news topics:
- Topic modeling: LDA, BTM, GSDMM, LSA
- Clustering: KMeans, HDBSCAN, Agglomerative (BERT embeddings)
  
This extends prior research from short-text domains (like tweets) into longer-form text.

## Tools and Techniques

- **Python** (pandas, sklearn, seaborn, matplotlib)
- **KNIME** (LDA, preprocessing workflows)
- **Vectorization**: CountVectorizer, TF-IDF, Doc2Vec, BERT (semantic)
- **Models**: LDA, BTM, GSDMM, KMeans, HDBSCAN, Agglomerative
- **Evaluation**: Accuracy, Precision, Recall, F1-score, Category alignment

## Methodology

### Preprocessing:
- Text normalization (contractions, punctuation, stopwords)
- Lemmatization
- Vectorization (TF-IDF and CountVectorizer)

### Topic Modeling (KNIME):
- Latent Dirichlet Allocation
- Grid search on topic count
- Topic-to-category mapping

### Clustering (Python):
- Sentence embedding with BERT
- Dimensionality reduction (UMAP)
- Clustering with KMeans, HDBSCAN, Agglomerative

### Evaluation:
- Topic alignment via labeled categories
- Internal metrics (Silhouette score, coherence)
- Visual inspection (WordClouds, TF-IDF bar plots)

## Results Summary

- **Best Topic Model**: LDA with grid search, followed closely by BTM
- **Best Clustering Model**: Agglomerative with BERT
- **Weak Models**: GSDMM and HDBSCAN underperformed due to high-dimensional sparsity
- **Insight**: Semantic clustering can match topic models in long-form corpora

## Visual Outputs

- WordClouds per topic
- TF-IDF distinctiveness plots
- UMAP projections with cluster coloring
- Category-to-topic ID mapping
