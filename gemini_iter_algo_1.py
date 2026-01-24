# https://share.google/aimode/ECHy3FBRimDIXwbAt
import numpy as np

# Initial vectors and target
x = np.array([0.018, 0.032, 0.044, 0.056, 0.069,
             0.085, 0.103, 0.128, 0.167, 0.298])
i = np.array([0.000, 0.000, 0.000, 0.001, 0.002,
             0.003, 0.004, 0.007, 0.012, 0.044])
r_target = 0.023

# Convergence parameters
tolerance = 0.00001
learning_rate = 0.5  # Increased step size for faster convergence
max_iterations = 10000

for step in range(max_iterations):
    current_product = np.dot(x, i)
    diff = r_target - current_product

    # Stop when the difference is at most 0.00001
    if abs(diff) <= tolerance:
        print(f"Target reached in {step} iterations.")
        break

    # 1. Update x based on the gradient (vector i)
    # This pushes x in the direction that most effectively increases the dot product
    x = x + learning_rate * diff * i

    # 2. Maintain the constraint Sum(x) = 1
    # Distribute any excess sum equally back to all elements
    x = x - (np.sum(x) - 1.0) / len(x)

    # 3. Ensure values remain non-negative (standard for sum-to-one vectors)
    x = np.maximum(x, 0)
    x = x / np.sum(x)
else:
    print("Maximum iterations reached without meeting tolerance.")

print("\nFinal x vector:")
print(np.round(x, 6))
print(f"\nFinal Sum of x: {np.sum(x):.6f}")
print(
    f"Final Product x * i: {np.dot(x, i):.6f} (Difference: {abs(np.dot(x, i) - r_target):.6f})")
