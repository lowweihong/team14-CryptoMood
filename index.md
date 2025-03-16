---
layout: default
---

## Team 14  
Ke Xin Chong, Joel J Jude, Shinhaeng Lee, Wei Hong Low, Abhijith Sreeraj

---

## Introduction/Background

CryptoSentiment Pulse aims to develop an advanced NLP-based model for classifying cryptocurrency-related social media text into bearish, bullish, or neutral sentiments. Given the high volatility of cryptocurrency markets, sentiment analysis is crucial for understanding public perception and improving trading models (Kulakowski & Frasincar, 2023)[[1]](#1). However, the unique nature of cryptocurrency markets requires specialized sentiment analysis models that can accurately capture the nuances of this domain.

Despite the existence of general sentiment models, there is a scarcity of models specifically tailored for cryptocurrency-related texts. This scarcity leads to suboptimal classification accuracy when applying these models to financial contexts, where precise sentiment analysis is crucial for predicting market trends and making informed investment decisions. This research aims to bridge this gap by developing an advanced NLP-based model, CryptoSentiment Pulse, which integrates domain-specific features to improve sentiment classification in cryptocurrency-related social media posts.

---

## Problem Definition

Current sentiment analysis models often struggle to accurately capture the unique and rapidly evolving terminology used in cryptocurrency discussions. This limitation is exacerbated by the high volatility and specialized nature of cryptocurrency markets, leading to suboptimal sentiment classification and potentially impacting trading decisions.

**Motivation:**  
There is a pressing need for a tailored model that incorporates domain-specific features from the financial sector, particularly those relevant to cryptocurrency markets. Such a model would provide more accurate and actionable sentiment analysis, enabling investors to make better-informed decisions in these volatile markets.

---

## Methods

The approach involves two primary components: data preprocessing and machine learning techniques. Data preprocessing is crucial for ensuring that the dataset is clean and representative of the cryptocurrency domain, while the machine learning techniques are designed to effectively classify sentiments in social media posts.

### Data Preprocessing
1) **Cleaning:** Remove unnecessary elements such as crypto wallet addresses, URLs, and fix encoding errors. Filter noisy data to improve the quality of the dataset, similar to the preprocessing steps taken in cryptocurrency sentiment analysis research.
2) **Augmentation:** Utilize large language models (LLMs) to generate synthetic data and rephrase tweets while preserving sentiment. This technique can help increase the size and diversity of the dataset, which is crucial for improving model performance.
3) **Handling Imbalance:** Apply techniques such as upsampling or downsampling to balance the sentiment categories, ensuring that the model is not biased towards any particular class.
4) **Word Embedding:** Transform text into numerical representations using Word2Vec or BERT embeddings. These embeddings capture contextual information and are effective for sentiment analysis tasks.

### ML Approaches
- **Unsupervised:**  
  - **DBSCAN Clustering**: Apply DBSCAN to detect sentiment clusters in the data. This algorithm is useful for identifying patterns and outliers in high-density regions, which can help in understanding the distribution of sentiments
- **Supervised:**  
  - **Fine-Tuning Pre-Trained Transformers**: Fine-tune pre-trained transformer models like RoBERTa and XLM-RoBERTa by adding a dense layer for sentiment classification. These models are known for their ability to capture complex linguistic patterns and can be effectively adapted for domain-specific tasks. (Roumeliotis, Tselikas, & Nasiopoulos, 2024)[[4]](#4)
  - **BiLSTM Network**: Implement a Bidirectional Long Short-Term Memory (BiLSTM) network to capture sequential context in text data. This architecture is particularly useful for modeling temporal relationships and has been applied in sentiment analysis for cryptocurrency markets

### Libraries and Tools
These methods utilize libraries such as scikit-learn for clustering, Hugging Face transformers for pre-trained models, and PyTorch for implementing neural networks.

### Dataset:
- **Source:** Financial Tweets on Cryptocurrency  
- **Volume:** ~57.9k tweets  
- **Labels:** Bearish (0), Neutral (1), Bullish (2)  
- **Link:** [Hugging Face Dataset](https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto)

---

## Potential Results and Discussion

**Evaluation Metrics:**  
- **Accuracy:** This metric will assess the overall correctness of the model across all sentiment categories (bearish, neutral, bullish). It is particularly useful for balanced datasets.
- **Precision and Recall:** These metrics will measure the true positive rate for each sentiment category. Precision will help evaluate how often the model correctly identifies a sentiment when it predicts it, while recall will assess how well the model captures all instances of a particular sentiment.

**Expected Outcomes:**  
We expect that fine-tuned transformer models, such as RoBERTa and XLM-RoBERTa, will outperform unsupervised methods like DBSCAN clustering and supervised BiLSTM networks. This is because transformer models are adept at capturing complex linguistic patterns and domain-specific terminology, which is crucial for accurately classifying sentiments in cryptocurrency-related texts.

The use of pre-trained models like CryptoBERT, as discussed by Kulakowski and Frasincar (2023), demonstrates the effectiveness of fine-tuning BERT-based models for cryptocurrency sentiment analysis. These models can better handle the nuances of cryptocurrency terminology and the dynamic nature of market sentiments compared to more general models.

Moreover, the performance of our models will be evaluated using metrics such as the F1 score, which provides a balanced measure of precision and recall, especially useful in imbalanced datasets. Additionally, metrics like Cohen's Kappa can be employed to assess the agreement between model predictions and human annotations, accounting for chance agreement

## Result and Discussion (Midterm Checkpoint)
### Data Preprocessing Method Implemented

### CryptoBERT Performance 

---

## References
<a id="1">[1]</a> M. Kulakowski and F. Frasincar, "Sentiment Classification of Cryptocurrency-Related Social Media Posts," *IEEE Intelligent Systems*, vol. 38, no. 4, pp. 5-9, July-Aug. 2023, https://doi.org/10.1109/MIS.2023.3283170. 

<a id="2">[2]</a> K. Qureshi and T. Zaman, "Social media engagement and cryptocurrency performance," *PLOS ONE*, vol. 18, no. 5, p. e0284501, May 2023, https://doi.org/10.1371/journal.pone.0284501.  

<a id="3">[3]</a> M. Wilksch and O. Abramova, "PyFin-sentiment: Towards a machine-learning-based model for deriving sentiment from financial tweets," *Int. J. Inf. Manag. Data Insights*, vol. 3, no. 1, p. 100171, 2023, https://doi.org/10.1016/j.jjimei.2023.100171.

<a id="4">[4]</a> Roumeliotis, K. I., Tselikas, N. D., & Nasiopoulos, D. K. (2024). LLMs and NLP Models in Cryptocurrency Sentiment Analysis: A Comparative Classification Study. Big Data and Cognitive Computing, 8(6), 63. https://doi.org/10.3390/bdcc8060063

---

## Gantt Chart

![Gantt Chart](assets/css/gantt-chart-midterm.png)  

---

## Contribution Table

| **Team Member**    | **Contribution(Midterm)**                            | **Contribution(Proposal)**                            |
|--------------------|------------------------------------------------------|-------------------------------------------------------|
| Ke Xin Chong       | 1. Data Analysis<br> 2. Visualisation<br> 3. Performance Comparison<br> 4. Report Writing | 1. Project management<br> 2. methodology design                |
| Joel J Jude        |                                                      | 1. Data processing <br> 2. ML model implementation              |
| Shinhaeng Lee      |                                                      | 1. Data augmentation <br>2. preprocessing                      |
| Wei Hong Low       |                                                      | 1. Literature review <br> 2. model evaluation                   | 
| Abhijith Sreeraj   |                                                      | 1. Report writing <br> 2. visualization                         |

## Presentation Youtube Link

Proposal: https://youtu.be/cke5F-7VIsE
