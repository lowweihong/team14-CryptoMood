---
layout: default
---

## Team 14  
Ke Xin Chong, Joel J Jude, Shinhaeng Lee, Wei Hong Low, Abhijith Sreeraj

---

## Introduction/Background

CryptoSentiment Pulse aims to develop an advanced NLP-based model for classifying cryptocurrency-related social media text into bearish, bullish, or neutral sentiments. Given the high volatility of cryptocurrency markets, sentiment analysis is crucial for understanding public perception and improving trading models (Kulakowski & Frasincar, 2023)[[1]](#1). However, the unique nature of cryptocurrency markets requires specialized sentiment analysis models that can accurately capture the nuances of this domain.

Despite the existence of general sentiment models, there is a scarcity of models specifically tailored for cryptocurrency-related texts. This scarcity leads to suboptimal classification accuracy when applying these models to financial contexts, where precise sentiment analysis is crucial for predicting market trends and making informed investment decisions. This research aims to bridge this gap by developing an advanced NLP-based model, CryptoSentiment Pulse, which integrates domain-specific features to improve sentiment classification in cryptocurrency-related social media posts.

Recent advancements in natural language processing have shifted sentiment trend analysis toward transformer-based models, such as BERT, due to their ability to capture contextual nuances in text. For instance, (Nguyen, Vu, and Nguyen, 2020)) [[5]](#5) introduced BERTweet, a pre-trained language model specifically fine-tuned on English tweets, demonstrating its effectiveness in processing the unique linguistic patterns.

---

## Problem Definition

Current sentiment analysis models often struggle to accurately capture the unique and rapidly evolving terminology used in cryptocurrency discussions. This limitation is exacerbated by the high volatility and specialized nature of cryptocurrency markets, leading to suboptimal sentiment classification and potentially impacting trading decisions. Currently, there are no open-source models specifically designed for analyzing crypto sentiment in tweets. The only existing model is limited to two predicted classes—bullish and negative—which fails to account for the prevalence of neutral sentiment in tweet text. To address this gap, there is a clear need for a three-class model that includes bullish, bearish, and neutral categories to better capture the full range of sentiments expressed on X platform.






---

**Motivation:**  
The highly volatile nature of cryptocurrency markets demands sentiment analysis models that go beyond generic NLP techniques. Traditional models often fail to capture the nuances of financial discourse, market sentiment shifts, and domain-specific jargon unique to crypto. By incorporating domain-specific features tailored to the cryptocurrency sector, a specialized model can provide deeper insights, more accurate sentiment predictions, and ultimately empower investors with data-driven strategies to navigate market fluctuations effectively.



---

## Methods

The approach involves two primary components: data preprocessing and machine learning techniques. Data preprocessing is crucial for ensuring that the dataset is clean and representative of the cryptocurrency domain, while the machine learning techniques are designed to effectively classify sentiments in social media posts.

### Data Preprocessing
1) **Cleaning (Completed):**  Remove unnecessary elements such as crypto wallet addresses, URLs, and fix encoding errors. Filter noisy data to improve the quality of the dataset, similar to the preprocessing steps taken in cryptocurrency sentiment analysis research. Additionally, exclude short texts (less than 4 words) and remove quote tweets (which combine original and quoted text) to avoid confusion in sentiment scoring, as the model may struggle to determine whether to provide sentiment score to the original, quoted, or combined text.

2) **Text Embedding (Completed):** Transform text into numerical representations using BERT embeddings. These embeddings capture contextual information and are effective for sentiment analysis tasks.

3) **Augmentation:** Utilize large language models (LLMs) to generate synthetic data and rephrase tweets while preserving sentiment. This technique can help increase the size and diversity of the dataset, which is crucial for improving model performance.

4) **Handling Imbalance:** Apply techniques such as upsampling or downsampling to balance the sentiment categories, ensuring that the model is not biased towards any particular class.

### Data Split
For the train-validation-test split we used a randomly split according to a fix random seed on the dataset (88647 data points) as follows:

- 80% - training set (36453 data points)
- 10% - validation set (4556 data points)
- 10% - test set (4558 data points)

All of the splits have the similar distribution across each different classes.


### ML Approaches
- **Unsupervised:**  
  - **DBSCAN Clustering**: Apply DBSCAN to detect sentiment clusters in the data. This algorithm is useful for identifying patterns and outliers in high-density regions, which can help in understanding the distribution of sentiments
- **Supervised:**  
  - Bert Models
    - Finetune model using pretrain model ElKulako/stocktwits-crypto, the pretrain model originally has bullish, bearish and neutral labels. (Kulakowski & Frasincar, 2023)[[1]](#1)
    - Finetune model using pretrain model kk08/CryptoBERT, the pretrain model originally has only bullish and bearish labels. Thus, will discard the pre-trained weights of the original 2-label classification head and initialize a new classification head with random weights for 3 labels. These models are known for their ability to capture complex linguistic patterns and can be effectively adapted for domain-specific tasks. (Roumeliotis, Tselikas, & Nasiopoulos, 2024)[[4]](#4)
    - Both of these models train for 5 epoch, and select the best models in according to the least validation loss.
  - BiLSTM model
    - Implement a Bidirectional Long Short-Term Memory (BiLSTM) network to capture sequential context in text data. This architecture is particularly useful for modeling temporal relationships and has been applied in sentiment analysis for cryptocurrency markets. (Chen, Zhang, & Ye, 2019)[[6]](#6)

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
- **Precision and F1-score:** These metrics will measure the true positive rate for each sentiment category. Precision will help evaluate how often the model correctly identifies a sentiment when it predicts it. The F1-score will provide a balanced evaluation by considering both precision and recall.

**Expected Outcomes:**  
We expect that fine-tuned Bert models, will outperform unsupervised methods like DBSCAN clustering and supervised BiLSTM networks. This is because transformer models are adept at capturing complex linguistic patterns and domain-specific terminology, which is crucial for accurately classifying sentiments in cryptocurrency-related texts.

The use of pre-trained models like CryptoBERT, as discussed by Kulakowski and Frasincar (2023), demonstrates the effectiveness of fine-tuning BERT-based models for cryptocurrency sentiment analysis. These models can better handle the nuances of cryptocurrency terminology and the dynamic nature of market sentiments compared to more general models.

Moreover, the performance of our models will be evaluated using metrics such as the F1 score, which provides a balanced measure of precision and recall, especially useful in imbalanced datasets. Additionally, metrics like Cohen's Kappa can be employed to assess the agreement between model predictions and human annotations, accounting for chance agreement

## Results and Discussion (Midterm Checkpoint)

In our midterm checkpoint, we implemented fine-tuned versions of pre-trained BERT models on the `StephanAkkerman/financial-tweets-crypto` dataset, evaluating performance on the same train and test splits.

The reproducible notebooks used can be found here:
1. [EDA.ipynb](./src/data_cleaning/eda.ipynb) 
2. [kk08_cryptobert_finetune.ipynb](./src/models/kk08_cryptobert_finetune.ipynb)
3. [ElKulako_stocktwits_crypto_finetune.ipynb](./src/models/ElKulako_stocktwits_crypto_finetune.ipynb)
4. [ElKulako_stocktwits_crypto_baseline.ipynb](./src/models/ElKulako_stocktwits_crypto_baseline.ipynb)

### Data Preprocessing Implementation
For our initial models, we implemented a comprehensive preprocessing pipeline that included:

1. **Text Cleaning:** We removed unnecessary elements such as URLs, wallet addresses, and special characters using regex patterns in our `preprocessing.py` module.
   - These components do not carry useful semantic meaning for sentiment analysis and could interfere with model performance. Cleaning the text ensures cleaner, more consistent input data for the models.

2. **Tokenization:** We used the tokenizers from pre-trained models (BERT-based) to convert text into token IDs suitable for deep learning models.
   - This step preserves the contextual relationships between words, which is crucial for understanding sentiment in financial and crypto-related language.

3. **Data Filtering:** We excluded short texts (fewer than 4 words) and removed quote tweets to avoid confusion in sentiment scoring.
   - Short texts often lack sufficient context for reliable sentiment classification, and quote tweets can introduce conflicting sentiment signals. This filtering step helped improve data quality and model accuracy.

### Chosen Models
1. **ElKulako/stocktwits-crypto baseline**
   - It provides a quick, interpretable starting point for evaluating the effectiveness of more complex models while it lacks the depth of contextual understanding.
2. **ElKulako/stocktwits-crypto finetune**
   - We selected this model because it is already fine-tuned on financial social media text from StockTwits, a platform known for investment-related sentiment. It was specifically trained with bullish, bearish, and neutral sentiment labels, which perfectly align with our classification goals. This domain alignment reduces the amount of task-specific fine-tuning required and helps the model generalize well to our crypto-related sentiment data. By leveraging its pre-learned understanding of financial terminology and sentiment, we gain a strong starting point for accurate predictions with minimal retraining effort.
3. **kk08/CryptoBERT finetune**
   - This model was chosen due to its pre-training on cryptocurrency-specific text, giving it an inherent understanding of the vocabulary, slang, and market dynamics unique to the crypto domain. Although it was originally trained for binary classification (bullish vs. bearish), we modified it by replacing the classification head to support three sentiment classes, allowing it to fit our task. This adaptation retained the model’s domain-specific strengths while making it flexible for multi-class classification. Using CryptoBERT ensures that the model can better interpret the nuanced language used in crypto discussions, which traditional general-purpose models might miss.


### Model Implementation and Evaluation

We implemented several models to establish benchmarks and assess performance on our cryptocurrency sentiment classification task:

#### Supervised Learning Models
  
1. **Fine-tuned BERT Models:**
   - **ElKulako/stocktwits-crypto:** Fine-tuned from a model already trained on financial text with existing bullish, bearish, and neutral labels.
   - **kk08/CryptoBERT:** Adapted from a model pre-trained on cryptocurrency text but with only bullish and bearish labels originally. We discarded the pre-trained weights of the original 2-label classification head and initialized a new 3-label classification head.

#### Performance Metrics

We evaluated our models using several key metrics:

| Model                               | Precision (Bullish) | Precision (Neutral) | Precision (Bearish) | Accuracy | F1-Score |
|-------------------------------------|---------------------|---------------------|---------------------|----------|----------|
| kk08/CryptoBERT finetune            | 81.79               | 65.55               | 62.19               | 75.01    | 69.79    |
| ElKulako/stocktwits-crypto finetune | 80.01               | 63.33               | 64.42               | 74.10    | 68.39    |
| ElKulako/stocktwits-crypto baseline | 70.30               | 17.00               | 25.34               | 38.99    | 33.11    |

![Model Performance Comparison](./assets/css/model_performance_comparison.png)
*Figure 1: Performance metrics across different models showing precision for each sentiment class, overall accuracy, and F1-score.*

### Analysis of Model Performance

Our analysis of the model performance revealed several key insights:

1. **Domain-Specific Pre-training Advantage:** The kk08/CryptoBERT model, which was pre-trained specifically on cryptocurrency text, outperformed the more general financial text model (ElKulako/stocktwits-crypto).

2. **Class Imbalance Impact:** All models showed better performance on the majority class (Bullish) compared to minority classes (Neutral and Bearish).

3. **Fine-tuning Benefits:** The significant performance gap between the fine-tuned models and the baseline demonstrates the effectiveness of transfer learning for this task.

<!-- Suggestion: Add confusion matrix visualization to show classification patterns -->

### Summary and Next Steps

#### Summary
Our initial results demonstrate that fine-tuned transformer-based models can achieve promising performance on cryptocurrency sentiment analysis, with the best model (kk08/CryptoBERT) achieving 75.01% accuracy and a macro F1-score of 69.79%.

#### Next Steps

1. **Unsupervised Learning Exploration:** For the next phase of our research, we plan to implement several unsupervised learning approaches to complement our supervised models. We will explore DBSCAN (Density-Based Spatial Clustering of Applications with Noise) as our primary clustering algorithm due to its ability to discover clusters of arbitrary shapes without requiring a predetermined number of clusters—a valuable feature when dealing with the nuanced language patterns in cryptocurrency discussions. Additionally, we intend to compare DBSCAN's performance with other clustering techniques such as K-means (for its simplicity and efficiency with large datasets) and BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) for its memory-efficient handling of large datasets. In theory, these unsupervised approaches will provide valuable insights into the natural groupings within cryptocurrency tweets that may not be captured by supervised models. By mapping these clusters to sentiment categories, we hope to identify linguistic patterns and topic-based sentiments that could enhance our overall classification framework. The implementation will involve converting tweets to embeddings using Sentence Transformers, applying dimensionality reduction techniques, and carefully tuning clustering parameters to achieve optimal results.

2. **Supervised Learning Exploration:** We would also plan to finalize the training of BiLSTM by the final report. In theory, BiLSTM should be well-suited for capturing contextual dependencies in sequential text data, allowing it to effectively classify sentiment in tweets by considering both past and future word contexts. While we have already implemented the BERT model, we aim to explore BiLSTM as a comparison to evaluate the performance trade-offs between traditional RNN-based approaches and transformer-based models. BiLSTM offers a simpler architecture with fewer parameters, making it more computationally efficient and potentially beneficial in scenarios with limited resources. Additionally, understanding how BiLSTM performs relative to BERT on our dataset provides insights into whether complex pre-trained models are necessary or if a more lightweight approach can achieve comparable results.

---

## References
<a id="1">[1]</a> M. Kulakowski and F. Frasincar, "Sentiment Classification of Cryptocurrency-Related Social Media Posts," *IEEE Intelligent Systems*, vol. 38, no. 4, pp. 5-9, July-Aug. 2023, https://doi.org/10.1109/MIS.2023.3283170. 

<a id="2">[2]</a> K. Qureshi and T. Zaman, "Social media engagement and cryptocurrency performance," *PLOS ONE*, vol. 18, no. 5, p. e0284501, May 2023, https://doi.org/10.1371/journal.pone.0284501.  

<a id="3">[3]</a> M. Wilksch and O. Abramova, "PyFin-sentiment: Towards a machine-learning-based model for deriving sentiment from financial tweets," *Int. J. Inf. Manag. Data Insights*, vol. 3, no. 1, p. 100171, 2023, https://doi.org/10.1016/j.jjimei.2023.100171.

<a id="4">[4]</a> Roumeliotis, K. I., Tselikas, N. D., & Nasiopoulos, D. K. (2024). LLMs and NLP Models in Cryptocurrency Sentiment Analysis: A Comparative Classification Study. Big Data and Cognitive Computing, 8(6), 63. https://doi.org/10.3390/bdcc8060063

<a id='5'>[5]</a> Nguyen, D. Q., Vu, T., & Nguyen, A. T. (2020). BERTweet: A pre-trained language model for English Tweets. In Q. Liu & D. Schlangen (Eds.), Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations (pp. 9–14). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.emnlp-demos.2

<a id='6'>[6]</a> Chen, X., Zhang, T., & Ye, Y. (2019). Sentiment analysis of comment texts based on BiLSTM. IEEE Access, 7, 51176–51185. https://doi.org/10.1109/ACCESS.2019.2909918
---

## Gantt Chart

![Gantt Chart](assets/css/gantt-chart-midterm.png)  

---

## Contribution Table

| **Team Member**    | **Contribution(Midterm)**                            | **Contribution(Proposal)**                            |
|--------------------|------------------------------------------------------|-------------------------------------------------------|
| Ke Xin Chong       | 1. Data Analysis<br> 3. Implementaton of `kk08/CryptoBERT finetune` <br> 4. Report Writing | 1. Project management<br> 2. methodology design                |
| Joel J Jude        | 1. Report Writing<br> 2. Visualisation  <br> 3. Unsupervised models implementation (WIP)                                                  | 1. Data processing <br> 2. ML model implementation              |
| Shinhaeng Lee      | 1. Report Writing<br> 2. BiLSTM (WIP) <br> 3. Performance summarisation                                                   | 1. Data augmentation <br>2. preprocessing                      |
| Wei Hong Low       | 1. Implementaton of `ElKulako/stocktwits-crypto finetune`<br> 2. Performance evaluation on `ElKulako/stocktwits-crypto finetune` baseline model <br> 4. Report Writing                                                    | 1. Literature review <br> 2. model evaluation                   | 
| Abhijith Sreeraj   | 1. Report Writing<br> 2. Unsupervised models implementation (WIP) <br> 3. Visualisation                                                      | 1. Report writing <br> 2. visualization                         |

## Presentation Youtube Link

Proposal: https://youtu.be/cke5F-7VIsE
