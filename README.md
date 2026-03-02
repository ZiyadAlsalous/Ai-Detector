# AI Essay Detector

A machine learning model that classifies essays as **human-written** or **AI-generated** using TF-IDF and traditional ML classifiers.

Built as part of exploring NLP classification techniques — the model compares Logistic Regression, Linear SVC, and Random Forest on TF-IDF features extracted from essay text.

## How it works

1. Downloads and merges two Kaggle datasets for a larger training corpus
2. Cleans and normalizes the text (lowercase, remove URLs/emails, collapse whitespace)
3. Converts text to numerical features using TF-IDF (unigrams + bigrams, 10k features)
4. Trains and compares three classifiers
5. Tunes the best one with GridSearchCV
6. Evaluates on a held-out test set

## Dataset

Uses two public Kaggle datasets:
- [LLM Detect AI Generated Text](https://www.kaggle.com/datasets/sunilthite/llm-detect-ai-generated-text-dataset) by sunilthite
- [Augmented Data for LLM Detect AI Generated Text](https://www.kaggle.com/datasets/jdragonxherrera/augmented-data-for-llm-detect-ai-generated-text) by jdragonxherrera

You'll need a Kaggle account and API key set up to download them automatically via `kagglehub`.

## Setup

```bash
pip install -r requirements.txt
```

Make sure your Kaggle API credentials are configured (`~/.kaggle/kaggle.json`). Then just run the notebook top to bottom.

## Project Structure

```
├── updated_model.ipynb      # main notebook with the full pipeline
├── requirements.txt         # python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

After running the notebook, it will also generate:
- `essay_detector_model.pkl` — trained model
- `tfidf_vectorizer.pkl` — fitted TF-IDF vectorizer

## Usage

Once you've run the notebook and have the pickle files, you can classify new essays:

```python
import pickle, re

with open('essay_detector_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def predict(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z\s.,!?;:\'\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    x = vectorizer.transform([text])
    pred = model.predict(x)[0]
    return 'AI-Generated' if pred == 1 else 'Human'

print(predict("Your essay text here..."))
```

## Tools Used

- Python, Pandas, NumPy
- Scikit-learn (TF-IDF, Logistic Regression, SVM, Random Forest, GridSearchCV)
- Matplotlib, Seaborn
- kagglehub

## License

MIT
