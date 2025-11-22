# Persian Text Spell Checker and Classification System

A Natural Language Processing (NLP) project for Persian/Farsi text that combines spell checking, text normalization, and text classification using machine learning algorithms.

## Overview

This project demonstrates a complete NLP pipeline for Persian text processing, including:
- Automatic spell correction and normalization of Persian text
- Visual highlighting of spelling errors and corrections
- Text classification using multiple machine learning models
- Performance evaluation and visualization

## Features

### 1. **Persian Text Processing**
- Text normalization using Parsivar library
- Pinglish (Persian written in English letters) to Persian conversion
- Automatic spell checking and correction
- Visual highlighting of errors (blue) and corrections (red) in HTML format

### 2. **Text Preprocessing**
- Custom Persian stopword removal
- Duplicate text detection and removal
- Text tokenization and cleaning

### 3. **Machine Learning Classification**
- **Baseline Model**: Dummy Classifier for comparison
- **Naive Bayes**: MultinomialNB with TF-IDF features
- **Decision Tree**: DecisionTreeClassifier
- Model evaluation with accuracy metrics
- Confusion matrix visualization

### 4. **Feature Extraction**
- TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- Binary count vectorization
- Configurable min/max document frequency filtering

## Requirements

### Dependencies
```
numpy
pandas
parsivar
scikit-learn
matplotlib
IPython
openpyxl  # For Excel file handling
```

### Python Version
- Python 3.6 or higher

## Installation

1. Clone the repository:
```bash
git clone https://github.com/akhavansafaei/spell_checker_NLP.git
cd spell_checker_NLP
```

2. Install required packages:
```bash
pip install numpy pandas parsivar scikit-learn matplotlib ipython openpyxl
```

## Project Structure

```
spell_checker_NLP/
├── nlp.py              # Main Python script
├── nlp.ipynb           # Jupyter notebook with interactive examples
├── nlp.xlsx            # Dataset (Excel format with Persian text)
├── Report.docx         # Project report document
└── README.md           # This file
```

## Usage

### Running the Python Script

1. Update the Excel file path in the code (line 172 in nlp.py):
```python
df=pd.read_excel(r'/path/to/your/nlp.xlsx', sheet_name='Sheet2')
```

2. Run the script:
```bash
python nlp.py
```

### Using the Jupyter Notebook

1. Launch Jupyter Notebook:
```bash
jupyter notebook nlp.ipynb
```

2. Run cells sequentially to see:
   - Spell correction examples with highlighting
   - Model training progress
   - Accuracy comparisons
   - Confusion matrix visualization

## How It Works

### 1. Spell Checking Pipeline
```
Raw Persian Text → Normalization → Spell Correction → Highlighting → Display
```

- **Normalization**: Converts various Persian character encodings to standard form
- **Spell Correction**: Uses Parsivar's spell checker to identify and correct errors
- **Highlighting**: Visual feedback with HTML formatting (incorrect in blue, corrected in red)

### 2. Classification Pipeline
```
Text Data → Stopword Removal → TF-IDF Vectorization → Train/Test Split → Model Training → Evaluation
```

**Steps:**
1. Load data from Excel file (Sheet2)
2. Remove duplicates based on text content
3. Apply custom Persian stopword removal
4. Create TF-IDF feature vectors (min_df=5, max_df=0.5)
5. Split data (75% training, 25% testing)
6. Train multiple classifiers
7. Compare performance metrics

### 3. Models Used

| Model | Purpose | Configuration |
|-------|---------|---------------|
| **DummyClassifier** | Baseline comparison | strategy='uniform' |
| **MultinomialNB** | Main classifier | alpha=0.2 |
| **DecisionTreeClassifier** | Alternative classifier | Default parameters |

## Output Examples

### Spell Correction Table
The system displays two tables:
1. **Word-level corrections**: Individual incorrect words and their corrections
2. **Sentence-level corrections**: Full sentences with highlighted errors

### Model Performance
```
| base line Accuracy | model Accuracy |
|--------------------|----------------|
| 0.XX               | 0.YY           |
```

### Confusion Matrix
A visual heatmap showing the classification performance across different classes.

## Dataset

- **Format**: Excel (.xlsx)
- **Sheet**: Sheet2
- **Required Columns**:
  - `text`: Persian text samples
  - `class`: Classification labels

## Key Functions

### `highlight1_specific_text(text)`
Highlights specific incorrect words in blue using HTML span tags.

### `highlight1_specific_text1(text)`
Highlights corrected words in red using HTML span tags.

### `delet_stopword(text)`
Removes Persian stopwords and short words (length ≤ 3) from text.

### `return_spli(x1, y1, h)`
Converts sparse matrices to DataFrames for train/test data visualization.

## Customization

### Adding Custom Stopwords
Edit the `mylist` in the `delet_stopword()` function:
```python
mylist = ['کردن', 'شوید', 'دارای', ...]  # Add your stopwords here
```

### Adjusting TF-IDF Parameters
Modify the vectorizer settings:
```python
vec = TfidfVectorizer(min_df=5, max_df=0.5)  # Adjust min_df and max_df
```

### Changing Sample Size
Modify the `n` variable to process more/fewer samples:
```python
n = 5  # Number of samples for spell checking demo
```

## Visualization

The project generates:
- **HTML Tables**: For spell correction comparisons
- **Confusion Matrix**: Matplotlib heatmap for classification results
- **Performance Metrics**: Accuracy comparison table

## Notes

- The project is specifically designed for Persian/Farsi text
- Parsivar library handles Persian-specific NLP tasks
- HTML output is best viewed in Jupyter Notebook or browser
- The spell checker uses a predefined Persian dictionary

## Limitations

- Spell checker accuracy depends on Parsivar's dictionary coverage
- Hard-coded word replacements in highlighting functions
- File path needs manual configuration
- Limited to Persian language only

## Future Improvements

- [ ] Support for batch processing of large datasets
- [ ] Web interface for interactive spell checking
- [ ] Additional ML models (SVM, Random Forest, Neural Networks)
- [ ] Cross-validation for more robust evaluation
- [ ] API endpoint for spell checking service
- [ ] Configurable file paths through command-line arguments
- [ ] Support for multiple Persian dialects

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is available for educational and research purposes.

## Acknowledgments

- **Parsivar**: Persian NLP library for tokenization, normalization, and spell checking
- **scikit-learn**: Machine learning framework
- **pandas**: Data manipulation and analysis

## Contact

For questions or feedback, please open an issue in the repository.
