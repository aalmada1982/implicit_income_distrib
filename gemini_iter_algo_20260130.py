# Proportional Adjustment algorithm

# x_int	0.0177	0.0321	0.0437	0.0556	0.0688	0.0845	0.1025	0.1272	0.1672	0.3007
# i	0.0000	0.0000	0.0000	0.0000	0.0010	0.0030	0.0040	0.0070	0.0120	0.0430

import numpy as np
from scipy.optimize import minimize

# Initial data
x_init = np.array([0.0177,	0.0321,	0.0437,	0.0556,	0.0688,
                  0.0845,	0.1025,	0.1272,	0.1672,	0.3007])
i = np.array([0.0000,	0.0000,	0.0000,	0.0000,	0.0010,
             0.0030,	0.0040,	0.0070,	0.0120,	0.0430])
r_target = 0.020

# Constraint: Elements cannot decrease more than 75% (lower floor = 25% of original)
lower_bounds = x_init * 0.25
# Format bounds for SciPy: (min, max) for each element
bounds = [(low, 1.0) for low in lower_bounds]

# Objective function: Minimize the squared difference to hit r_target exactly


def objective(x):
    return (np.dot(x, i) - r_target)**2


# Equality constraint: Sum(x) must be exactly 1.0
constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]

# Run the solver
result = minimize(
    objective,
    x_init,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    tol=1e-10  # High precision tolerance
)

if result.success:
    x_final = result.x
    final_product = np.dot(x_final, i)
    diff = abs(final_product - r_target)

    print("Optimization Successful!")
    print("\nFinal x vector:")
    print(np.round(x_final, 6))
    print(f"\nFinal Product x * i: {final_product:.8f}")
    print(f"Target r: {r_target}")
    print(f"Difference: {diff:.8f} (Constraint: <= 0.0001)")
    print(f"Final Sum of x: {np.sum(x_final):.8f}")

    # Verify bounds
    below_bounds = x_final < (lower_bounds - 1e-12)
    print(f"Any values below floor? {np.any(below_bounds)}")
else:
    print("Solver failed:", result.message)
