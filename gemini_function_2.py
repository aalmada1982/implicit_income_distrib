# To achieve a precision of 0.0001 while strictly maintaining your lower bound and sum constraints, we can use a more advanced Sequential Least Squares Programming (SLSQP) solver from the SciPy Optimize library. This method is designed to handle multiple linear constraints (like your sum constraint) and bounds simultaneously with high accuracy.

# To make the function versatile for Pandas, we can use .to_numpy() to handle both Series and DataFrame inputs seamlessly.
# The function below is "format-agnostic"—it will accept Pandas columns (Series), single-row/column DataFrames, or standard NumPy arrays.

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def solve_vector_incidence_pandas(x_init, i_vector, r_target, precision=0.0001):
    """
    Solves the vector incidence problem accepting Pandas or NumPy inputs.

    Args:
        x_init: Pandas Series, DataFrame (1 row/col), or NumPy array.
        i_vector: Pandas Series, DataFrame (1 row/col), or NumPy array.
        r_target (float): Target scalar product (e.g., 0.023).
        precision (float): Target accuracy for the product.
    """
    # Convert Pandas inputs to flat NumPy arrays for the solver
    x_0 = x_init.to_numpy().flatten() if hasattr(
        x_init, 'to_numpy') else np.array(x_init).flatten()
    inc = i_vector.to_numpy().flatten() if hasattr(
        i_vector, 'to_numpy') else np.array(i_vector).flatten()

    # 1. Constraints: -75% lower bound
    lower_bounds = x_0 * 0.25
    bounds = [(low, 1.0) for low in lower_bounds]

    # 2. Objective: Minimize squared error
    def objective(x):
        return (np.dot(x, inc) - r_target)**2

    # 3. Constraint: Sum(x) == 1
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]

    # 4. Solve
    result = minimize(objective, x_0, method='SLSQP', bounds=bounds,
                      constraints=constraints, tol=1e-12)

    if result.success:
        # Return as a Pandas Series if the input was Pandas, matching the original index
        if isinstance(x_init, (pd.Series, pd.DataFrame)):
            return pd.Series(result.x, index=x_init.index if isinstance(x_init, pd.Series) else None)
        return result.x
    else:
        raise ValueError(f"Optimization failed: {result.message}")


# --- Example with Pandas --- Uruguay
df = pd.DataFrame({
    'initial_x': [0.018, 0.032, 0.044, 0.056, 0.069, 0.085, 0.103, 0.128, 0.167, 0.298],
    'incidence': [0.000, 0.000, 0.000, 0.001, 0.002, 0.003, 0.004, 0.007, 0.012, 0.044]
})

# Passing DataFrame columns directly
final_series = solve_vector_incidence_pandas(
    df['initial_x'], df['incidence'], 0.0390)

print("Resulting x Series:")
print(final_series)

df_2018 = pd.read_csv('data_pit_lac_2018.csv')

print(df_2018)

# filtered_df = df[df['column_name'] == 'some_value']

# print(
#     df_2018['variable'] == 'inc_dist_cedlas'
# )

# print("\n")

# print(
#     df_2018[df_2018['variable'] == 'inc_dist_cedlas']
# )

# print(df_2018[df_2018['variable'] == 'inc_dist_cedlas']['decile'])

# Reindexing of CEDLAS Income Distribution

df_2018_inc_dist_cedlas = df_2018[df_2018['variable'] == 'inc_dist_cedlas']

df_2018_inc_dist_cedlas.set_index('decile', inplace=True)

print(df_2018_inc_dist_cedlas, "\n")

# Reindexing of LACIT PIT Incidence

df_2018_pit_incid_lacir = df_2018[df_2018['variable'] == 'pit_incid_lacir']

df_2018_pit_incid_lacir.set_index('decile', inplace=True)

print(df_2018_pit_incid_lacir, "\n")

# Extracting pit_rev_gdp_pct

df_2018_pit_rev_gdp_pct = df_2018[df_2018['variable'] == 'pit_rev_gdp_pct']

df_2018_pit_rev_gdp_pct.set_index('decile', inplace=True)

print(df_2018_pit_rev_gdp_pct, "\n")
