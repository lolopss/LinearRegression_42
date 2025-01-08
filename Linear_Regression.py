import csv
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = 'data.csv'
MODEL_FILE = 'model.txt'

data = []
t_mileage = []
t_price = []


# Normalize the array by dividing each element by the maximum value
def normalize_array(arr, max_val):
    return [x / max_val for x in arr]

# First program of the subject
def estimatePrice(mileage):
    return theta0 + theta1 * mileage

# Evaluate the model using Mean Squared Error (MSE) and Mean Absolute Error (MAE).
def evaluate_model(true_prices, predicted_prices):
    n = len(true_prices)
    
    mse = sum([(true_prices[i] - predicted_prices[i]) ** 2 for i in range(n)]) / n
    mae = sum([abs(true_prices[i] - predicted_prices[i]) for i in range(n)]) / n
    
    return mse, mae

# Load the data from the CSV file
try:
    with open(DATA_FILE, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            try:
                mileage, price = map(float, row)
                t_mileage.append(mileage)
                t_price.append(price)
            except ValueError:
                print(f"Error: Non-numeric data found in row: {row}")
                continue
except FileNotFoundError:
    print("Error: {DATA_FILE} file not found.")
    exit()

theta0 = 0
theta1 = 0
best_mse = float('inf')
learning_rate = 0.1
iterations = 10000
patience = 100
patience_counter = 0
i = 0

max_mileage = max(t_mileage)
max_price = max(t_price)
m = len(t_mileage)
norm_mileage = normalize_array(t_mileage, max_mileage)
norm_price = normalize_array(t_price, max_price)

print(f"Training started with {m} data points.")

stored_params = []

for i in range(iterations):
    tmptheta0 = learning_rate * (1/m) * sum([estimatePrice(norm_mileage[i]) - norm_price[i] for i in range(m)])
    tmptheta1 = learning_rate * (1/m) * sum([(estimatePrice(norm_mileage[i]) - norm_price[i]) * norm_mileage[i] for i in range(m)])
    theta0 -= tmptheta0
    theta1 -= tmptheta1
    
    # Used to calculate MAE and MSE
    predicted_prices = [(theta0 + theta1 * (mileage / max_mileage)) * max_price for mileage in t_mileage]
    mse, mae = evaluate_model(t_price, predicted_prices)
    if mse < best_mse:
        best_mse = mse
        patience_counter = 0
    else:
        patience_counter += 1

    # Store parameters at regular intervals to later see progression during training (only for the bonus, not the actual project)
    if i in [1, 25, 100, 500, 1000]:
        stored_params.append((theta0, theta1))

    # If no improvement within patience, we stop the training
    if patience_counter >= patience:
        print(f"Early stopping at iteration {i} with MSE: {mse} and MAE: {mae}")
        break

# store the final parameters (only for the bonus, not the actual project)
stored_params.append((theta0, theta1))

with open(MODEL_FILE, 'w') as file:
    file.write(f"theta0 = {theta0}\ntheta1 = {theta1}\nmax_mileage = {max_mileage}\nmax_price = {max_price}")

print(f"Training complete. Model saved to {MODEL_FILE}.")
print(f"Theta0: {theta0}, Theta1: {theta1}")


#Exemple of difference between price in data and prediction : 
print("Km : 63060 | Price : 6390")
print(f"Prediction : {estimatePrice(63060/max_mileage) * max_price}")


# Print the results
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")

# Graph to display the regression line and the scatter plot of the data points
df = pd.DataFrame({'Mileage': t_mileage, 'Price': t_price})

plt.figure(figsize=(10, 6))
plt.scatter(df['Mileage'], df['Price'], color='blue', label='Data Points')
plt.xlabel('Mileage (Km)')
plt.ylabel('Price (€)')
plt.title('Mileage vs Price')
plt.grid(True)

# Plotting the regression line
x = list(range(int(min(t_mileage)), int(max(t_mileage)), 1000))
y = [(theta0 + theta1 * (i / max_mileage)) * max_price for i in x]
plt.plot(x, y, color='red', label='Regression Line')

plt.legend()
plt.show()

# Bonus: Plot the evolution of the prediction
# This part is not part of the project, but is a bonus to visualize the progression of the regression line during training
plt.figure(figsize=(10, 6))
plt.scatter(df['Mileage'], df['Price'], color='blue', label='Data Points')
plt.xlabel('Mileage (Km)')
plt.ylabel('Price (€)')
plt.title('Evolution of Predictions')
plt.grid(True)


# Define distinct colors for better recognition
colors = ['yellow', 'red', 'green', 'blue', 'orange', 'purple']

# Plotting the regression lines at intervals with distinct colors
for idx, (theta0, theta1) in enumerate(stored_params):
    y = [(theta0 + theta1 * (i / max_mileage)) * max_price for i in x]
    plt.plot(x, y, color=colors[idx], label=f'Iteration {["1", "25", "100", "500", "1000", "final"][idx]}')


plt.legend()
plt.show()