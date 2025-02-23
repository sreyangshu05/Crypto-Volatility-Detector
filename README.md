# 🚀 Crypto Volatility Detector

## 📜 Problem Statement
The objective of this project is to build a **Crypto Volatility Detector** that analyzes historical cryptocurrency data, calculates volatility, and predicts future trends using machine learning.

## 🛠️ Approach
1. **Data Collection:**  
   - Used a historical dataset of the top 10 cryptocurrencies.
   - The dataset includes `Date`, `Open`, `High`, `Low`, `Close`, and `Volume` fields.

2. **Feature Engineering:**  
   - Calculated `Daily_Return` as the percentage change between consecutive closing prices.
   - Computed `Volatility` using the 7-day rolling standard deviation of daily returns.

3. **Model Preparation:**  
   - Selected `Open`, `High`, `Low`, and `Volume` as features and `Volatility` as the target.
   - Split the dataset into training (70%) and testing (30%) sets.

4. **Model Training and Evaluation:**  
   - Applied **Linear Regression** to predict volatility.
   - Evaluated model performance using:
     - Mean Absolute Error (MAE)
     - Mean Squared Error (MSE)
     - R-squared (R²)

5. **Visualization:**  
   - Plotted actual vs. predicted volatility trends to visually assess model accuracy.

## 🔧 Tools Used
- Python
- Pandas
- Scikit-Learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## 📄 Documentation
The project documentation covers:
1. **Data Exploration:** Understanding dataset structure and key statistics.
2. **Feature Engineering:** Calculating daily returns and volatility.
3. **Model Training:** Training a linear regression model to predict volatility.
4. **Evaluation & Visualization:** Analyzing results using metrics and plots.

## ✨ Additional Insights
- The model provides reasonably accurate volatility predictions.
- Performance can further improve with advanced models like Random Forest or LSTM.
- Real-time API integration can make the project production-ready.