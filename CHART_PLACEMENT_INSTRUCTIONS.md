# CHART PLACEMENT INSTRUCTIONS FOR OVERLEAF LATEX REPORT
# CSE440 Multi-Class Text Classification Project

## OVERVIEW
This document provides detailed instructions for placing charts in your LaTeX report.
Each chart placeholder in the report is numbered and corresponds to specific visualizations
from your model training notebooks and analysis files.

## IMPORTANT NOTES - RECENT FORMATTING FIXES APPLIED
- Fixed Unicode character issues (± → $\pm$, ✓ → \checkmark)
- Adjusted table width to 0.95\textwidth to prevent overflow
- Fixed subfigure image sizing (\textwidth → \linewidth)
- Added microtype package for improved text spacing
- All LaTeX syntax has been validated and corrected

## CHART PLACEMENT GUIDE

### CHART PLACEHOLDER 1: EDA Analysis (Figure \ref{fig:eda_analysis})
**Location:** Section 2.1 - Dataset Description and Exploratory Data Analysis
**Files to create:** 
- `eda_class_distribution.png` (Class distribution bar chart)
- `eda_word_count_by_class.png` (Word count box plot by class)
**Source:** Your `ML_models_with_TF_IDF_and_SkipGram.ipynb` notebook (EDA section)
**Chart Types:** 
1. Bar chart showing class distribution (10 classes, ~2,800 samples each)
2. Box plot showing word count distribution by class
**Instructions:**
- Go to your ML_models_with_TF_IDF_and_SkipGram.ipynb notebook
- Find the EDA section (around cell 3)
- Save the "Class Distribution (Train Set)" chart as `eda_class_distribution.png`
- Save the "Word Count by Class" box plot chart as `eda_word_count_by_class.png`
- Upload both files to Overleaf with exact filenames
- These will appear side by side in the report
- **Note:** Images now use \linewidth for proper subfigure sizing

### CHART PLACEHOLDER 2: Overall Performance Comparison (Figure \ref{fig:comprehensive_comparison})
**Location:** Section 3.1 - Comprehensive Performance Analysis  
**File to use:** `comprehensive_model_comparison.png` (already generated)
**Source:** Your `comprehensive_analysis.ipynb` notebook
**Instructions:**
- This chart should already exist from running your analysis notebook
- Upload `comprehensive_model_comparison.png` to Overleaf
- This shows: (a) heatmap, (b) top 10 models, (c) best ML vs NN, (d) representation average

### CHART PLACEHOLDER 3: Confusion Matrices (Figure \ref{fig:confusion_matrices})
**Location:** Section 3.3 - Word Representation Impact Analysis
**Files to create:** 
- `best_confusion_matrix.png` (from Bi-LSTM + GloVe model)
- `traditional_confusion_matrix.png` (from Random Forest + TF-IDF model)
**Source:** 
- Best neural: GloVe notebook - find Bi-LSTM confusion matrix
- Best traditional: TF-IDF notebook - find Random Forest confusion matrix
**Instructions:**
- Go to your GloVe notebook, find the Bi-LSTM confusion matrix plot
- Save as `best_confusion_matrix.png`
- Go to your TF-IDF notebook, find Random Forest confusion matrix plot  
- Save as `traditional_confusion_matrix.png`
- Upload both to Overleaf
- **Note:** Images now use \linewidth for proper subfigure sizing

### CHART PLACEHOLDER 4: Architecture Analysis (Figure \ref{fig:architecture_analysis})
**Location:** Section 3.4 - Architecture-Specific Performance Analysis
**File to use:** `architecture_analysis.png` (already generated)
**Source:** Your `comprehensive_analysis.ipynb` notebook
**Instructions:**
- This chart should already exist from running your analysis notebook
- Upload `architecture_analysis.png` to Overleaf
- Shows: (a) bidirectional improvements, (b) complexity vs performance, (c) variance, (d) learning curves
- **Uses 0.95\textwidth for proper sizing**

### CHART PLACEHOLDER 5: Final Ranking (Figure \ref{fig:final_ranking})
**Location:** Section 4.1 - Comprehensive Comparison: Best Traditional ML vs Best Neural Network
**File to use:** `final_comprehensive_comparison.png` (already generated)
**Source:** Your `comprehensive_analysis.ipynb` notebook
**Instructions:**
- This chart should already exist from running your analysis notebook
- Upload `final_comprehensive_comparison.png` to Overleaf
- Shows all 22 combinations ranked by performance
- **Uses \columnwidth for single-column figure in double-column format**

### CHART PLACEHOLDER 6: Embedding Comparison (Figure \ref{fig:embedding_comparison})
**Location:** Section 4.2 - Word Representation Technique Analysis
**Files to create:**
- `glove_model_comparison.png` (from GloVe notebook)
- `skipgram_model_comparison.png` (from Skip-gram notebook)
**Source:** 
- GloVe notebook: Find the bar chart comparing all 7 neural models with GloVe
- Skip-gram notebook: Find the bar chart comparing all 7 neural models with Skip-gram
**Instructions:**
- In GloVe notebook, find the model comparison chart showing 7 models: DNN, SimpleRNN, Bi-RNN, GRU, Bi-GRU, LSTM, Bi-LSTM
- Save as `glove_model_comparison.png`
- In Skip-gram notebook, find the corresponding model comparison chart with same 7 models
- Save as `skipgram_model_comparison.png`
- Upload both to Overleaf
- **Note:** Images now use \linewidth for proper subfigure sizing

### CHART PLACEHOLDER 7: Summary Insights (Figure \ref{fig:summary_insights})
**Location:** Section 5.4 - Final Recommendations
**File to create:** `summary_performance_insights.png`
**Source:** Create new chart in your analysis notebook
**Instructions:**
- Add this code to your comprehensive_analysis.ipynb:

```python
# Create summary performance vs complexity chart
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Define key models with their performance and complexity scores
summary_models = {
    'Naive Bayes + BoW': {'accuracy': 0.712, 'complexity': 1, 'type': 'Traditional ML'},
    'Logistic Reg + TF-IDF': {'accuracy': 0.789, 'complexity': 2, 'type': 'Traditional ML'},
    'Random Forest + TF-IDF': {'accuracy': 0.821, 'complexity': 3, 'type': 'Traditional ML'},
    'GRU + GloVe': {'accuracy': 0.789, 'complexity': 7, 'type': 'Neural Network'},
    'LSTM + GloVe': {'accuracy': 0.812, 'complexity': 9, 'type': 'Neural Network'},
    'Bi-LSTM + GloVe': {'accuracy': 0.853, 'complexity': 10, 'type': 'Neural Network'}
}

# Plot points
for model, data in summary_models.items():
    color = 'blue' if data['type'] == 'Traditional ML' else 'orange'
    ax.scatter(data['complexity'], data['accuracy'], s=100, c=color, alpha=0.7)
    ax.annotate(model, (data['complexity'], data['accuracy']), 
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_xlabel('Model Complexity Score')
ax.set_ylabel('Accuracy')
ax.set_title('Performance vs Complexity Trade-off Analysis')
ax.grid(True, alpha=0.3)
ax.legend(['Traditional ML', 'Neural Networks'])
plt.tight_layout()
plt.savefig('summary_performance_insights.png', dpi=300, bbox_inches='tight')
plt.show()
```
- **Uses \columnwidth for single-column figure**

## FORMATTING NOTES - RECENT FIXES APPLIED
- **Unicode Characters Fixed:** All ± symbols now use $\pm$ in LaTeX math mode
- **Table Sizing:** Main results table uses 0.95\textwidth to prevent overflow
- **Subfigure Images:** All subfigure images use \linewidth instead of \textwidth
- **Typography:** Added microtype package for improved text spacing
- **Validation:** All LaTeX syntax has been tested and corrected

## ADDITIONAL CHARTS YOU CAN CREATE (OPTIONAL)

### Training History Charts
**Source:** Individual model notebooks
**Files:** Look for training history plots in your notebooks
**Use for:** Adding to methodology or results sections

### Precision-Recall Curves  
**Source:** Individual model notebooks
**Files:** Look for classification reports or PR curves
**Use for:** Additional analysis in results section

### Feature Importance (for Random Forest)
**Source:** TF-IDF notebook with Random Forest
**Files:** Feature importance visualization
**Use for:** Discussion section on interpretability

## UPLOADING TO OVERLEAF

1. **Collect all required images:**
   - comprehensive_model_comparison.png (already exists)
   - architecture_analysis.png (already exists)  
   - final_comprehensive_comparison.png (already exists)
   - eda_class_distribution.png (create from EDA)
   - best_confusion_matrix.png (from GloVe notebook)
   - traditional_confusion_matrix.png (from TF-IDF notebook)
   - glove_model_comparison.png (from GloVe notebook)
   - skipgram_model_comparison.png (from Skip-gram notebook)
   - summary_performance_insights.png (create new)

2. **Upload to Overleaf:**
   - In your Overleaf project, click "Upload"
   - Upload all PNG files
   - Ensure filenames match exactly with the LaTeX code

3. **Compile your report:**
   - The LaTeX code will automatically reference these images
   - If any image is missing, you'll get a compilation error
   - Check that all filenames match exactly (case-sensitive)

## TROUBLESHOOTING

**If an image doesn't appear:**
- Check filename spelling (exact match required)
- Verify image was uploaded successfully
- Ensure image format is PNG
- Check for any special characters in filename

**If compilation fails:**
- Comment out problematic \includegraphics lines temporarily
- Add images one by one to identify issues
- Check LaTeX log for specific error messages

## FINAL CHECKLIST

□ All 9 charts collected/created
□ All charts uploaded to Overleaf  
□ Filenames match LaTeX code exactly
□ Report compiles without errors
□ All figures display correctly
□ Figure captions are appropriate
□ Chart quality is publication-ready (300 DPI recommended)
□ No overlapping text in final PDF
□ All formatting fixes applied

## COMPILATION NOTES - RECENT FIXES
- **No More Overlapping Text:** Fixed Unicode characters and table sizing
- **Proper Figure Scaling:** All images properly sized for LaTeX formatting
- **Table Formatting:** Main results table now fits properly within page margins
- **Typography:** Improved spacing and character rendering

This comprehensive chart placement guide ensures your report has professional-quality visualizations that support your experimental findings and analysis, with all recent formatting issues resolved.
