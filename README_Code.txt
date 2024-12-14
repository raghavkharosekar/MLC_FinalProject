
# README for Stock Price Movement Prediction Code

## Project Description
This project aims to predict stock price movements (“up”, “neutral”, or “down”) based on financial news headlines using machine learning models. The codebase integrates natural language processing (NLP) and structured financial data to train predictive models. Key models used include Random Forest, Logistic Regression, and Support Vector Machines (SVM).

## Code Structure
### 1. **Data Preprocessing**
- **Purpose**: Align financial news data with stock price data, compute sentiment scores, and engineer features for model training.
- **Key Files/Functions**:
  - `preprocess_news.py`: Prepares the news data, computes sentiment scores using VADER.
  - `merge_data.py`: Merges news and stock price data by date and stock ticker.
  - `feature_engineering.py`: Adds derived features like `change`, `movement`, and `sentiment_score`.

### 2. **Model Training**
- **Purpose**: Train machine learning models for binary and multi-class classification of stock movements.
- **Key Files/Functions**:
  - `train_random_forest.py`: Implements Random Forest for both 2-label and 3-label classification.
  - `train_logistic_regression.py`: Trains Logistic Regression for comparison.
  - `train_svm.py`: Applies Support Vector Machines for non-linear classification.

### 3. **Evaluation and Metrics**
- **Purpose**: Evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrices.
- **Key Files/Functions**:
  - `evaluate_model.py`: Calculates metrics and generates confusion matrices.
  - `plot_results.py`: Visualizes the results for comparison across models.

### 4. **Prediction Pipeline**
- **Purpose**: Make predictions on new data using the trained models.
- **Key Files/Functions**:
  - `predict_stock_movement.py`: Loads trained models and predicts stock movements for new news headlines.

---

## Setup Instructions
### Requirements
- **Python Version**: Python 3.8+
- **Libraries**:
  - pandas
  - numpy
  - scikit-learn
  - nltk (for VADER sentiment analysis)
  - matplotlib, seaborn (for visualization)
  - yfinance (for stock price retrieval)

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-repo>/stock-price-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd stock-price-prediction
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the Kaggle dataset for financial news and place it in the `data/` directory.
5. Ensure you have API access for Yahoo Finance to retrieve stock price data.

---

## How to Run the Code
### 1. Preprocess Data
Run the preprocessing script to align and engineer features:
```bash
python preprocess_news.py
```

### 2. Train Models
Train specific models by running their respective scripts:
- Random Forest:
  ```bash
  python train_random_forest.py
  ```
- Logistic Regression:
  ```bash
  python train_logistic_regression.py
  ```
- SVM:
  ```bash
  python train_svm.py
  ```

### 3. Evaluate Models
Run the evaluation script to generate metrics and confusion matrices:
```bash
python evaluate_model.py
```

### 4. Make Predictions
Use the prediction script for new data:
```bash
python predict_stock_movement.py --input new_headlines.csv
```

---

## Key Features in Code
### Sentiment Analysis
- Sentiment scores for news headlines are calculated using the VADER sentiment analysis tool.

### Feature Engineering
- **Change**: Difference between opening and closing prices.
- **Movement**: Categorical label based on price change thresholds.

### Model Configurations
- **Random Forest**:
  - Handles binary and multi-class classification.
  - Configurable hyperparameters for tree depth, number of trees, etc.
- **Logistic Regression**:
  - Baseline model with regularization tuning.
- **SVM**:
  - Uses non-linear kernels for complex relationships.

---

## Results Summary
| **Model**                          | **Accuracy (%)** | **Remarks**                                              |
|------------------------------------|------------------|----------------------------------------------------------|
| Random Forest (2 Labels, 1-Day)   | 54.56            | Balanced but struggled with false positives/negatives.   |
| Random Forest (3 Labels, Sentiment-Based) | 67.31   | Best overall performance with sentiment integration.     |
| Logistic Regression (3 Labels, 1-Day) | 56.52         | Performed well for `neutral`, struggled with other labels. |
| Logistic Regression (3 Labels, 3-Day) | 51.34         | Noise introduced in extended time frame.                |
| SVM (3 Labels, Sentiment-Based)   | 58.42            | Strong precision for `neutral`, weaker for minority classes. |

---

## Future Improvements
1. **Advanced NLP Models**:
   - Integrate transformer-based models like BERT for better sentiment analysis.
2. **Dynamic Thresholds**:
   - Implement adaptive thresholds based on volatility or historical trends.
3. **Real-Time Integration**:
   - Build a pipeline for real-time prediction using streaming data.

---

## Acknowledgments
- Kaggle for the financial news dataset.
- Yahoo Finance for stock price data.
- Open-source libraries for tools used in preprocessing, training, and evaluation.

For further inquiries or contributions, please contact **Raghavendra Kharosekar**.
