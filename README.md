# team14-CryptoMood

For the complete project report, please see [here](https://github.gatech.edu/pages/asreeraj3/team14-CryptoMood/).

## Setup
To run the notebooks, install the required dependencies:
- Python 3.8+
- Libraries: `transformers`, `scikit-learn`, `pytorch`, `pandas`, `matplotlib`
Run `pip install -r requirements.txt` to install dependencies.

## Running the Notebooks
1. Clone the repository: `git clone <repo-url>`
2. Navigate to the `src/` directory.
3. Open and run the notebooks using Jupyter Notebook or JupyterLab.

## Dataset
The project uses the `StephanAkkerman/financial-tweets-crypto` dataset from Hugging Face, accessible at [https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto](https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto).

## Project Structure

### Directories and Files
- `/src/`: Main source code directory containing all project implementation
- `/src/data_cleaning/`: Contains data processing notebooks and scripts
- `/src/data_cleaning/eda.ipynb`: Exploratory data analysis notebook with visualizations and statistical analysis
- `/src/data_cleaning/template.ipynb`: Template notebook for standardizing data preprocessing and model experimentation workflows
- `/src/models/`: Contains model training and inference scripts
- `/src/models/kmeans/`: Directory for K-means clustering experiments (WIP)
- `/src/models/kmeans/kmeans_model.pkl`: Trained K-means model (WIP)
- `/src/models/kmeans/kmeans.ipynb`: Notebook for K-means clustering experiments (WIP)
- `/src/models/kmeans/tfidf_vectorizer.pkl`: Trained TF-IDF vectorizer for K-means (WIP)
- `/src/models/ElKulako_stocktwits_crypto_finetune.ipynb`: Experiments with pre-trained RoBERTa model for fine-tuning
- `/src/models/ElKulako_stocktwits_crypto_baseline.ipynb`: Generates baseline performance with ElKulako/stocktwits-crypto model
- `/src/models/kk08_cryptobert_finetune.ipynb`: Experiments with pre-trained BERT base model for fine-tuning
- `/src/models/crypto_BiLSTM.ipynb`: Notebook for BiLSTM model experiments (WIP)
- `/src/models/crypto_birch.ipynb`: Notebook for BIRCH clustering experiments (WIP)
- `/src/models/crypto_hdbscan.ipynb`: Notebook for HDBSCAN clustering experiments (WIP)
- `/src/models/crypto_kmeans.ipynb`: Notebook for K-means clustering experiments (WIP)
- `/Visualizations/`: Contains notebooks for generating visualizations used in the project report
- `/Visualizations/Visualizations.ipynb`: Notebook to generate graphs for performance comparison


Tree structure:
```
src/
├── data_cleaning/
│   ├── eda.ipynb                                  # Data Exploration
|   ├── template.ipynb                             # Template notebook for standardizing data preprocessing and model experimentation workflows
└── models/
   └── kmeans/
      └── kmeans_model.pkl                         # Trained model (WIP)
      └── kmeans.ipynb                             # Unsupervised Experiments (WIP)
      └── tfidf_vectorizer.pkl                     # Trained model (WIP)
   └── ElKulako_stocktwits_crypto_finetune.ipynb   # Generate baseline performance with ElKulako_stocktwits model
   └── ElKulako_stocktwits_crypto_finetune.ipynb   # Experiments with pretrain Roberta model 
   └── kk08_cryptobert_finetune.ipynb              # Experiments with pretrained Bert base model
   └── (wip) crypto_BiLSTM.ipynb                   # BiLSTM Experiments (WIP)
   └── (wip) crypto_birch.ipynb                    # Unsupervised Experiments (WIP)
   └── (wip) crypto_hdbscan.ipynb                  # Unsupervised Experiments (WIP)
   └── (wip) crypto_kmeans.ipynb                   # Unsupervised Experiments (WIP)
└── Visualizations/                                # Contains notebooks for generating visualizations
   └── Visualizations.ipynb                        # Notebook to generate graph for performance comparison
   

```

