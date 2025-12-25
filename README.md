# CSE-440 Natural Language Processing - Multi-Class Text Classification Project

A comprehensive comparison of word representation techniques and machine learning/neural network models for multi-class text classification. This project systematically evaluates 22 model-representation combinations on a Q&A dataset with 10 classes.

## 📋 Project Overview

This project presents an extensive empirical comparison of different word representation techniques paired with various machine learning and neural network models. The study evaluates four word representation methods (Bag of Words, TF-IDF, GloVe, Skip-gram) combined with ten different classification models, including traditional ML algorithms and advanced recurrent neural networks.

### Key Highlights

- **Dataset**: Q&A multi-class text classification dataset (10 classes)
- **Total Models Evaluated**: 22 combinations
  - 8 Traditional ML models (with BoW, TF-IDF)
  - 14 Neural Network models (with GloVe, Skip-gram)
- **Word Representations**: BoW, TF-IDF, GloVe, Skip-gram
- **Best Performance**: Bidirectional LSTM with GloVe embeddings (85.3% accuracy, 0.850 macro F1-score)

## 🎯 Objectives

1. Systematic evaluation of 22 model-representation combinations on a multi-class text classification dataset
2. Detailed analysis of the impact of different word representation techniques on model performance
3. Comparison between traditional ML and neural network approaches
4. Identification of optimal model-representation pairs for different performance criteria

## 📊 Dataset

- **Source**: Question-Answer pairs from an online Q&A platform
- **Structure**: Each sample contains Question Title, Question Content, and Best Answer
- **Classes**: 10 distinct categories
- **Split**: Pre-provided 80-20 train-test split
  - Training set: ~24,000 samples
  - Test set: ~6,000 samples
- **Characteristics**: Well-balanced dataset with approximately 2,800 samples per class

## 🔬 Methodology

### Word Representation Techniques

1. **Bag of Words (BoW)**: Count-based representation creating vocabulary of unique words
2. **TF-IDF**: Term Frequency-Inverse Document Frequency weighting scheme
3. **GloVe**: Pre-trained embeddings (glove.6B.100d) trained on 6 billion tokens
4. **Skip-gram**: Custom-trained Word2Vec embeddings on the dataset (context window: 5, dimensions: 100)

### Model Architectures

#### Traditional Machine Learning Models
- **Logistic Regression**: Multi-class linear classifier with L2 regularization
- **Naive Bayes**: Multinomial Naive Bayes with Laplace smoothing
- **Random Forest**: Ensemble method with 100 decision trees
- **Deep Neural Network**: Feedforward architecture with two hidden layers

#### Recurrent Neural Network Models
- **SimpleRNN**: Basic recurrent architecture with 32 hidden units
- **GRU**: Gated Recurrent Unit with reset and update gates
- **LSTM**: Long Short-Term Memory with input, forget, and output gates
- **Bidirectional Variants**: Bidirectional versions of SimpleRNN, GRU, and LSTM

### Data Preprocessing Pipeline

1. Text parsing and extraction (Question Title, Content, Answer)
2. Text normalization (lowercase conversion)
3. Cleaning (removal of non-alphabetic characters, punctuation)
4. Stopword removal using NLTK
5. Tokenization and sequence preparation

## 📁 Project Structure

```
CSE-440-Natural-Language-Processing/
├── README.md                                    # Project documentation
├── comprehensive_analysis.ipynb                 # Main comprehensive analysis notebook
├── cse440project-Glove-with-7-models.ipynb     # Neural networks with GloVe embeddings
├── cse440project-skipgram-with-7-models.ipynb  # Neural networks with Skip-gram embeddings
├── ML_models_with_TF_IDF_and_SkipGram.ipynb    # Traditional ML models with TF-IDF/BoW
├── merged_comprehensive_analysis.ipynb          # Merged analysis notebook
├── merge_notebooks.py                          # Utility script to merge notebooks
├── project_report.tex                          # LaTeX project report (IEEE format)
├── additional_latex_content.tex                # Additional LaTeX content
├── CHART_PLACEMENT_INSTRUCTIONS.md             # Instructions for chart placement
├── *.png                                       # Performance visualization charts
└── ollama_ubuntu_installation_guide.txt        # Installation guide (reference)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- GPU support (recommended for neural network training)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CSE-440-Natural-Language-Processing
```

2. Install required packages:
```bash
pip install pandas numpy scikit-learn tensorflow keras nltk gensim matplotlib seaborn wordcloud gdown tqdm
```

3. Download NLTK data:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

4. Download GloVe embeddings (if not using pre-downloaded):
   - Download `glove.6B.100d.txt` from [GloVe website](https://nlp.stanford.edu/projects/glove/)
   - Place in the project directory

### Dataset Setup

The notebooks include code to download the dataset from Google Drive. The dataset links are embedded in the notebooks. Alternatively, you can:

1. Download `train.csv` and `test.csv` manually
2. Place them in the project root directory
3. Update the file paths in the notebooks if needed

### Running the Notebooks

1. **Comprehensive Analysis**: Start with `comprehensive_analysis.ipynb` for the complete workflow
2. **Individual Analyses**:
   - `ML_models_with_TF_IDF_and_SkipGram.ipynb` - Traditional ML models
   - `cse440project-Glove-with-7-models.ipynb` - Neural networks with GloVe
   - `cse440project-skipgram-with-7-models.ipynb` - Neural networks with Skip-gram

## 📈 Results Summary

### Best Performing Models

1. **Bidirectional LSTM + GloVe**: 85.3% accuracy, 0.850 macro F1-score
2. **Random Forest + TF-IDF**: 82.1% accuracy (best traditional ML)
3. **Bidirectional GRU + GloVe**: 84.2% accuracy

### Key Findings

- **Pre-trained embeddings outperform count-based representations**: GloVe and Skip-gram show significant improvements over BoW and TF-IDF
- **Bidirectional architectures improve performance**: Average 4.4% improvement over unidirectional counterparts
- **Neural networks vs Traditional ML**: Neural networks achieve 3.2% average performance advantage at the cost of increased computational complexity
- **GloVe vs Skip-gram**: Pre-trained GloVe embeddings slightly outperform custom-trained Skip-gram embeddings

## 📊 Visualizations

The project includes several performance comparison charts:
- `comprehensive_model_comparison.png` - Overall model performance comparison
- `gLOVE MODEL COMPARISON.png` - GloVe-based model comparisons
- `SKIPGRAM MODEL COMPARISON.png` - Skip-gram-based model comparisons
- `bi LSTM +GLOVE.png` - Best model detailed analysis
- `RF+TF-IDF.png` - Best traditional ML model analysis
- `summary_performance_insights.png` - Summary insights
- `eda1.png`, `eda2.png` - Exploratory data analysis visualizations

## 📝 Report

The project includes a comprehensive LaTeX report (`project_report.tex`) in IEEE conference format covering:
- Abstract and Introduction
- Detailed Methodology
- Experimental Results
- Performance Analysis
- Conclusions and Future Work

## 🛠️ Utilities

- **merge_notebooks.py**: Utility script to merge multiple Jupyter notebooks into one
  ```bash
  python merge_notebooks.py notebook1.ipynb notebook2.ipynb -o merged.ipynb
  ```

## 📚 Technologies Used

- **Python**: Core programming language
- **TensorFlow/Keras**: Deep learning framework
- **scikit-learn**: Traditional machine learning algorithms
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **NLTK**: Natural language processing utilities
- **Gensim**: Word2Vec implementation
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development environment

## 📄 License

This project is part of a CSE-440 Natural Language Processing course assignment.

## 👥 Authors

Course project by CSE-440 students.

## 🙏 Acknowledgments

- Stanford NLP Group for GloVe embeddings
- Google for TensorFlow framework
- NLTK and Gensim communities for NLP tools

---

**Note**: This project is for educational purposes as part of the CSE-440 Natural Language Processing course.
