# team14-CryptoMood

For complete project report, please see [here](https://github.gatech.edu/pages/asreeraj3/team14-CryptoMood/)

## Project Structure

### Directories and Files
- `/src/`: Main source code directory containing all project implementation
- `/src/data_cleaning/`: Contains data processing notebooks and scripts
- `/src/data_cleaning/eda.ipynb`: Exploratory data analysis notebook with visualizations and statistical analysis
- `/src/data_cleaning/template.ipynb`: Template for experiments
- `/src/models/`: Contains model training and inference scripts
- `/src/models/kmeans`: (WIP)

Tree structure:
```
src/
├── data_cleaning/
│   ├── eda.ipynb                                  # Data Exploration
|   ├── template.ipynb                             # Template for experiments
└── models/
   └── kmeans/
      └── kmeans_model.pkl                         # Trained model
      └── kmeans.ipynb                             # Unsupervised Experiments (WIP)
      └── tfidf_vectorizer.pkl                     # Trained model
   └── ElKulako_stocktwits_crypto_finetune.ipynb   # Generate baseline performance with ElKulako_stocktwits model
   └── ElKulako_stocktwits_crypto_finetune.ipynb   # Experiments with pretrain Roberta model 
   └── kk08_cryptobert_finetune.ipynb              # Experiments with pretrained Bert base model
   └── (wip) crypto_BiLSTM.ipynb                   # BiLSTM Experiments (WIP)
   └── (wip) crypto_birch.ipynb                    # Unsupervised Experiments (WIP)
   └── (wip) crypto_hdbscan.ipynb                  # Unsupervised Experiments (WIP)
   └── (wip) crypto_kmeans.ipynb                   # Unsupervised Experiments (WIP)
└── Visualizations/
   └── Visualizations.ipynb                        # Notebook to generate graph for performance comparison
   

```

