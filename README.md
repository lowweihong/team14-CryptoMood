# team14-CryptoMood

For the complete project report, please see [here](https://lowweihong.github.io/team14-CryptoMood/).

*Original publish page on gatech can be found [here](https://github.gatech.edu/pages/wlow7/team14-CryptoMood/).

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
- `/src/models/experimental/kmeans/`: Directory for K-means clustering with dimensionality reduction (research exploration only)
- `/src/models/experimental/kmeans/kmeans_model.pkl`: Pickled K-means model trained on tweet embeddings
- `/src/models/experimental/kmeans/kmeans.ipynb`: Jupyter notebook containing K-means + t-SNE visualization experimentsExperiments with TSNE
- `/src/models/experimental/kmeans/tfidf_vectorizer.pkl`: Trained TF-IDF vectorizer for K-means
- `/src/models/crypto_BiLSTM.ipynb`: Bidirectional LSTM implementation for sentiment analysis
- `/src/models/crypto_birch.ipynb`: Notebook for BIRCH clustering experiments
- `/src/models/crypto_hdbscan.ipynb`: Notebook for HDBSCAN clustering experiments
- `/src/models/crypto_kmeans.ipynb`: Notebook for K-means clustering experiments
- `/src/models/ElKulako_stocktwits_crypto_baseline.ipynb`: Generates baseline performance with ElKulako/stocktwits-crypto model
- `/src/models/ElKulako_stocktwits_crypto_finetune.ipynb`: Experiments with pre-trained RoBERTa model for fine-tuning
- `/src/models/kk08_cryptobert_finetune.ipynb`: Experiments with pre-trained BERT base model for fine-tuning
- `/Visualizations/`: Contains notebooks for generating visualizations used in the project report
- `/Visualizations/preminilary_visualizations.ipynb`: Notebook for Initial exploratory data and results visualization
- `/Visualizations/bar_chart.ipynb`: Performance metrics visualization and model comparison


Tree structure:
```
src/
├── data_cleaning/
│   ├── eda.ipynb                                  # Exploratory data analysis notebook with visualizations and statistical analysis
|   ├── template.ipynb                             # Template notebook for standardizing data preprocessing and model experimentation workflows
└── models/
   └── experimental/                               # Research exploration and prototype implementations
      └── kmeans/                                  # K-means clustering with dimensionality reduction
         └── kmeans_model.pkl                      # Pickled K-means model trained on tweet embeddings
         └── kmeans.ipynb                          # Jupyter notebook containing K-means + t-SNE visualization experimentsExperiments with TSNE
         └── tfidf_vectorizer.pkl                  # Pickled TF-IDF vectorizer for text feature extraction
   └── crypto_BiLSTM.ipynb                         # Bidirectional LSTM implementation for sentiment analysis
   └── crypto_birch.ipynb                          # Notebook for BIRCH clustering experiments
   └── crypto_hdbscan.ipynb                        # Notebook for HDBSCAN clustering experiments
   └── crypto_kmeans.ipynb                         # Notebook for K-means clustering experiments
   └── ElKulako_stocktwits_crypto_baseline.ipynb   # Generate baseline performance with ElKulako_stocktwits model
   └── ElKulako_stocktwits_crypto_finetune.ipynb   # Experiments with pre-trained RoBERTa model for fine-tuning
   └── kk08_cryptobert_finetune.ipynb              # Experiments with pre-trained BERT base model for fine-tuning
└── Visualizations/                                
   └── bar_chart.ipynb                             # Performance metrics visualization and model comparison
   └── preminilary_visualizations.ipynb            # Notebook for initial exploratory data and results visualization
```

