import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # type: ignore

# Load dataset
url = "https://raw.githubusercontent.com/sreyangshu05/crypto-dataset/main/top_10_crypto.csv"
data = pd.read_csv(url)

# Display basic dataset info
print("Dataset Head:\n", data.head())
print("\nDataset Info:\n")
data.info()

# Feature engineering: Calculate daily returns and volatility
data['Date'] = pd.to_datetime(data['Date'])
data['Daily_Return'] = data['Close'].pct_change()
data['Volatility'] = data['Daily_Return'].rolling(window=7).std()

# Drop NaN values
crypto_data = data.dropna()

# Define features and target
X = crypto_data[['Open', 'High', 'Low', 'Volume']]
y = crypto_data['Volatility']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Performance:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R-squared (R2):", r2_score(y_test, y_pred))

# Visualization
plt.figure(figsize=(12, 7))
sns.lineplot(x=crypto_data['Date'], y=crypto_data['Volatility'], label='Actual Volatility')
sns.lineplot(x=crypto_data['Date'][-len(y_test):], y=y_pred, label='Predicted Volatility')
plt.title('Crypto Volatility Detection')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.show()