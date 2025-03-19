import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv("POY.CSV")  # Replace with your CSV file path

# Assuming CSV has 'x' and 'y' columns
x = df["x"].values
y = df["y"].values

# Compute the area using the Trapezoidal rule
area = np.trapz(y, x)
print(f"Area under the curve: {area:.4f}")

# Plot the function
plt.plot(x, y, 'r', label="Curve")
plt.fill_between(x, y, alpha=0.3, color='blue', label="Area under curve")

# Labels and title
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Area Under Curve :{area:.4f}")
plt.legend()
plt.grid(True)

plt.show()
