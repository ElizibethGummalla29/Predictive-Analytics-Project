import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("data/sales_data.csv")

print("Original Data:")
print(data.head())


# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())


# Convert Date column
data['Date'] = pd.to_datetime(data['Date'])


# Create Year and Month columns
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month


print("\nAfter Preprocessing:")
print(data.head())


# Plot historical trend
plt.figure(figsize=(10,5))

plt.plot(
    data['Date'],
    data['Sales'],
    marker='o'
)

plt.title("Historical Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.xticks(rotation=45)

plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import pickle


# Select input and output
X = data[['Year','Month']]
y = data['Sales']


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create model
model = LinearRegression()


# Train model
model.fit(
    X_train,
    y_train
)


print("\nModel trained successfully")


# Prediction
predictions = model.predict(X_test)


print("\nActual vs Predicted:")

result = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

print(result)


# Accuracy
mae = mean_absolute_error(
    y_test,
    predictions
)

accuracy = r2_score(
    y_test,
    predictions
)


print("\nMean Absolute Error:", mae)
print("Model Accuracy:", accuracy)



# Save model

pickle.dump(
    model,
    open("model/predictor.pkl","wb")
)

print("\nModel saved as predictor.pkl")
# Future Prediction

future_data = pd.DataFrame({
    "Year":[2026,2026,2026,2026],
    "Month":[1,2,3,4]
})


future_sales = model.predict(future_data)


future_result = pd.DataFrame({
    "Year":future_data["Year"],
    "Month":future_data["Month"],
    "Predicted Sales":future_sales
})


print("\nFuture Sales Forecast:")
print(future_result)

plt.figure(figsize=(10,5))


plt.plot(
    future_result["Month"],
    future_result["Predicted Sales"],
    marker='o'
)


plt.title("Future Sales Forecast")

plt.xlabel("Month")

plt.ylabel("Predicted Sales")

plt.show()