# IQTrader

## Inspiration

I have always been fascinated by physics and game theory, and I'm captivated by the idea that our seemingly chaotic world can be accurately predicted. Mathematics provides the framework to forecast the behavior of the physical world; the more information we have about a system, the more accurately we can predict its outcomes. If we can predict something as complex as our natural environment, then it should be relatively straightforward to determine the outcome of a virtual system where most of the factors are visible. Motivated by this belief, I developed a trading model to test how effectively artificial intelligence—leveraging vast amounts of data—can predict the outcome of such a virtual system.

## What It Does

The model predicts the price change of any stock regardless of the company. It uses only the hourly OHLCV (Open, High, Low, Close, Volume) data from the past 5 days, with all features derived from these data points, to forecast the expected price change over the next 8 hours. The performance of these predictions is evaluated using several metrics that offer insight into both the statistical properties and practical effectiveness of the model:

- **Trend Accuracy:**  
  *55% Match (Correlation)* – How well the model matches general price movements.
- **Direction Accuracy:**  
  *72% Correct* – How often the model correctly predicts whether prices will go up or down.
- **Stability of Predictions:**  
  *Lower than Real Volatility (0.24 vs. 0.40)* – How consistent the model’s predictions are compared to the natural fluctuations in the data.
- **Prediction Bias:**  
  *Slightly Overpredicts Upward Movements* – Indicates whether the model has a tendency to over- or underpredict.
- **Prediction Error:**  
  *Moderately Low Error (0.30)* – How far off the predictions are on average.

## How We Built It

### Feature Engineering

- **Diverse Indicators:**  
  Created a wide range of indicators, including momentum-based features and metrics describing trend and volatility from raw OHLCV data.  
  *Examples:* ATR, ATR14, long-term rolling standard deviations, MACD, and Stochastic Oscillator.
- **Composite Features:**  
  Developed composite indicators by multiplying different features together, which proved to have strong predictive capabilities.

### Data Exploration

- **Statistical Analysis:**  
  Explored the data to understand its statistical characteristics, such as skewness and long tails, which are crucial for identifying potential biases or overfitting issues.
- **Normalization:**  
  Recognized the need to address skewness and implemented appropriate data standardization techniques.

### Feature Selection

Implemented six techniques to select the most important features:

- **Pearson Correlation:**  
  Measures linear relationships between features and the target outcome (though it misses non-linear dependencies).
- **Mutual Information:**  
  Quantifies how much knowing a feature reduces uncertainty about the output.
- **Permutation Importance:**  
  Assesses feature importance by shuffling feature values and observing the impact on prediction performance using a Random Forest model.
- **Tree-Based Importance:**  
  Ranks features based on their contribution in a trained Random Forest model.
- **SHAP Analysis & Partial Dependence/ICE Plots:**  
  Used for understanding feature influence and interactions.

### Data Scaling

- **Quantile Transformer:**  
  Applied to normalize the data distribution.
- **Standard Scaler:**  
  Ensured the data had a mean of 0 and a standard deviation of 1.

### Model Architecture

- **Sliding-Window Interaction Block:**  
  Captures short-term temporal dependencies using multiple convolutional filters, including a redundant convolutional path to enhance feature robustness.
- **Convolutional Neural Networks (CNNs):**  
  Extracts hierarchical temporal patterns via sequential convolutional layers and residual connections for improved training stability.
- **Bidirectional Long Short-Term Memory (BiLSTM):**  
  Models long-term temporal relationships in both forward and backward directions, using batch normalization and dropout to reduce overfitting.
- **Lightweight Multi-Headed Attention Transformer:**  
  Dynamically prioritizes significant temporal segments using multiple attention heads, integrated with feed-forward neural networks for enhanced feature representation.

## Challenges We Ran Into

- **Loss Function Selection:**  
  Initially used Mean Squared Error (MSE) which underperformed at capturing variance, then tried Mean Absolute Error (MAE) which overshot the variance. Ultimately, Huber Loss was chosen to balance both extremes.
- **Enhancing Sign Accuracy:**  
  After reaching 58% sign accuracy, a sliding window interaction block was implemented to better capture short-term temporal patterns. This addition introduced complexity and required extensive debugging.
- **Minor Alignment Issues:**  
  Numerous minor challenges arose throughout the project, necessitating careful adjustments and alignment efforts.

## Accomplishments We’re Proud Of

Despite training on just one year of data, I managed to develop a relatively complex model that delivered excellent results without overfitting or underfitting. Achieving a sign accuracy of 72%—well above the 60% usability threshold—while maintaining a robust variance ratio (5:3 between true and predicted variance) highlights the model's precision and robustness. This project stands as my most elaborate hackathon effort to date, and I’m incredibly proud of its performance.

## What We Learned

- **Data Standardization Techniques:**  
  Gained deeper insights into normalizing and standardizing data for improved model performance.
- **ML Model Architectures:**  
  Explored various architectures capable of capturing complex patterns, including advanced feature selection and evaluation strategies.
- **Loss Functions & Regularization:**  
  Experimented with different loss functions and regularization methods to effectively balance model complexity with generalization.
- **Hyperparameter Tuning:**  
  Learned the critical importance of harmonizing the learning rate and schedule for efficient model convergence.

## What’s Next for IQTrader

- **Incorporating Fourier Transform:**  
  To capture periodic and frequency-based data.
- **Exploring Long-Term Autocorrelation:**  
  Investigate additional methods such as the Hurst Exponent to enhance predictions.
- **Expanding the Dataset:**  
  Train models on longer historical data to improve prediction accuracy and avoid catastrophic forgetting, using techniques like dynamic freezing and curriculum learning.
- **Sentiment Analysis Integration:**  
  Map data from social media and Yahoo Finance to discrete emotional scores using a GoEmotions classifier, then aggregate these into a public sentiment indicator ranging from +1 to -1.

---

Contributions and feedback are welcome!
