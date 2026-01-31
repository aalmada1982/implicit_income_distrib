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

# Addressing it using a Data Frame

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

# Extracting inc_dist_cedlas for Argentina

print(df_2018_inc_dist_cedlas, "\n")
print(df_2018_inc_dist_cedlas['Argentina'], "\n")

# Extracting pit_incid_lacir for Argentina

print(df_2018_pit_incid_lacir, "\n")
print(df_2018_pit_incid_lacir['Argentina'], "\n")

# Extracting pit_rev_gdp_pct for Argentina

print(df_2018_pit_rev_gdp_pct, "\n")
print(df_2018_pit_rev_gdp_pct['Argentina'], "\n")

# Running the algorithm on Argentina

x_arg = solve_vector_incidence_pandas(
    df_2018_inc_dist_cedlas['Argentina'],
    df_2018_pit_incid_lacir['Argentina'],
    df_2018_pit_rev_gdp_pct['Argentina']
)

print(x_arg, "\n")

print(x_arg.dot(df_2018_pit_incid_lacir['Argentina']), "\n")

# Running the algorithm on Bolivia

x_bol = solve_vector_incidence_pandas(
    df_2018_inc_dist_cedlas['Bolivia'],
    df_2018_pit_incid_lacir['Bolivia'],
    df_2018_pit_rev_gdp_pct['Bolivia']
)

print(x_bol, "\n")

print(x_bol.dot(df_2018_pit_incid_lacir['Bolivia']), "\n")

# Running the algorithm on Brazil

x_bra = solve_vector_incidence_pandas(
    df_2018_inc_dist_cedlas['Brazil'],
    df_2018_pit_incid_lacir['Brazil'],
    df_2018_pit_rev_gdp_pct['Brazil']
)

print(x_bra, "\n")

print(x_bra.dot(df_2018_pit_incid_lacir['Brazil']), "\n")

# Running the algorithm on Chile

x_chi = solve_vector_incidence_pandas(
    df_2018_inc_dist_cedlas['Chile'],
    df_2018_pit_incid_lacir['Chile'],
    df_2018_pit_rev_gdp_pct['Chile']
)

print(x_chi, "\n")

print(x_chi.dot(df_2018_pit_incid_lacir['Chile']), "\n")

# Running the algorithm on Colombia

x_col = solve_vector_incidence_pandas(
    df_2018_inc_dist_cedlas['Colombia'],
    df_2018_pit_incid_lacir['Colombia'],
    df_2018_pit_rev_gdp_pct['Colombia']
)

print(x_col, "\n")

print(x_col.dot(df_2018_pit_incid_lacir['Colombia']), "\n")

# Running the algorithm on Dominican R.

x_rd = solve_vector_incidence_pandas(
    df_2018_inc_dist_cedlas['Dominican R.'],
    df_2018_pit_incid_lacir['Dominican R.'],
    df_2018_pit_rev_gdp_pct['Dominican R.']
)

print(x_rd, "\n")

print(x_rd.dot(df_2018_pit_incid_lacir['Dominican R.']), "\n")

# Running the algorithm on Honduras

x_hond = solve_vector_incidence_pandas(
    df_2018_inc_dist_cedlas['Honduras'],
    df_2018_pit_incid_lacir['Honduras'],
    df_2018_pit_rev_gdp_pct['Honduras']
)

print(x_hond, "\n")

print(x_hond.dot(df_2018_pit_incid_lacir['Honduras']), "\n")

# Running the algorithm on Mexico

x_mex = solve_vector_incidence_pandas(
    df_2018_inc_dist_cedlas['Mexico'],
    df_2018_pit_incid_lacir['Mexico'],
    df_2018_pit_rev_gdp_pct['Mexico']
)

print(x_mex, "\n")

print(x_mex.dot(df_2018_pit_incid_lacir['Mexico']), "\n")

# Running the algorithm on Peru

x_per = solve_vector_incidence_pandas(
    df_2018_inc_dist_cedlas['Peru'],
    df_2018_pit_incid_lacir['Peru'],
    df_2018_pit_rev_gdp_pct['Peru']
)

print(x_per, "\n")

print(x_per.dot(df_2018_pit_incid_lacir['Peru']), "\n")

# Running the algorithm on Uruguay

x_uru = solve_vector_incidence_pandas(
    df_2018_inc_dist_cedlas['Uruguay'],
    df_2018_pit_incid_lacir['Uruguay'],
    df_2018_pit_rev_gdp_pct['Uruguay']
)

print(x_uru, "\n")

print(x_uru.dot(df_2018_pit_incid_lacir['Uruguay']), "\n")

# -------------------------------------------------------------------------
# To bring the table to 2023
# 1. we have to infer the LACIR incidence
# 2. after that we should infer the gross income distribution
# Lets try it for Argentina
# No, the algo will need to be changed, SOLUTION, USE LACIR INC AND ADJUST GROSS INCOME DISTRIBUTION AND REV

# Importing the new data frame

print("\n", "Importing the new data frame: 2023", "\n")

df_2023 = pd.read_csv('data_pit_lac_2023.csv')

print(df_2023)

# Reindexing of CEDLAS Income Distribution 2023

df_2023_inc_dist_cedlas = df_2023[df_2023['variable'] == 'inc_dist_cedlas']

df_2023_inc_dist_cedlas.set_index('decile', inplace=True)

print(df_2023_inc_dist_cedlas, "\n")

# Reindexing of LACIT PIT Incidence 2023 (in fact it is the same as 2018)

df_2023_pit_incid_lacir = df_2023[df_2023['variable'] == 'pit_incid_lacir']

df_2023_pit_incid_lacir.set_index('decile', inplace=True)

print(df_2023_pit_incid_lacir, "\n")

# Extracting pit_rev_gdp_pct 2023

df_2023_pit_rev_gdp_pct = df_2023[df_2023['variable'] == 'pit_rev_gdp_pct']

df_2023_pit_rev_gdp_pct.set_index('decile', inplace=True)

print(df_2023_pit_rev_gdp_pct, "\n")

# Testing the compromise solution for Argentina 2023

# print(df_2023_pit_rev_gdp_pct['Argentina'], "\n")

# ARGENTINA 2023

x_arg_2023 = solve_vector_incidence_pandas(
    df_2023_inc_dist_cedlas['Argentina'],
    df_2018_pit_incid_lacir['Argentina'],
    df_2023_pit_rev_gdp_pct['Argentina']
)

print(x_arg_2023, "\n")

print(x_arg_2023.dot(df_2018_pit_incid_lacir['Argentina']), "\n")
print(x_arg_2023.dot(df_2023_pit_incid_lacir['Argentina']), "\n")


# VSCode Keyboard Shortcut: Replace All Occurrences Matching Current Text Selection
# CTRL + D
# Continue pressing Ctrl+D to select additional occurrences you want to replace.
# To skip a match, press Ctrl+K then Ctrl+D.


# BOLIVIA 2023

x_bol_2023 = solve_vector_incidence_pandas(
    df_2023_inc_dist_cedlas['Bolivia'],
    df_2018_pit_incid_lacir['Bolivia'],
    df_2023_pit_rev_gdp_pct['Bolivia']
)

print(x_bol_2023, "\n")

print(x_bol_2023.dot(df_2018_pit_incid_lacir['Bolivia']), "\n")

# BRAZIL 2023

x_bra_2023 = solve_vector_incidence_pandas(
    df_2023_inc_dist_cedlas['Brazil'],
    df_2018_pit_incid_lacir['Brazil'],
    df_2023_pit_rev_gdp_pct['Brazil']
)

print(x_bra_2023, "\n")

print(x_bra_2023.dot(df_2018_pit_incid_lacir['Brazil']), "\n")

# CHILE 2023

x_chi_2023 = solve_vector_incidence_pandas(
    df_2023_inc_dist_cedlas['Chile'],
    df_2018_pit_incid_lacir['Chile'],
    df_2023_pit_rev_gdp_pct['Chile']
)

print(x_chi_2023, "\n")

print(x_chi_2023.dot(df_2018_pit_incid_lacir['Chile']), "\n")

# COLOMBIA 2023

x_col_2023 = solve_vector_incidence_pandas(
    df_2023_inc_dist_cedlas['Colombia'],
    df_2018_pit_incid_lacir['Colombia'],
    df_2023_pit_rev_gdp_pct['Colombia']
)

print(x_col_2023, "\n")

print(x_col_2023.dot(df_2018_pit_incid_lacir['Colombia']), "\n")

# DOMINICAN REPUBLIC 2023

x_dr_2023 = solve_vector_incidence_pandas(
    df_2023_inc_dist_cedlas['Dominican R.'],
    df_2018_pit_incid_lacir['Dominican R.'],
    df_2023_pit_rev_gdp_pct['Dominican R.']
)

print(x_dr_2023, "\n")

print(x_dr_2023.dot(df_2018_pit_incid_lacir['Dominican R.']), "\n")

# HONDURAS 2023

x_hond_2023 = solve_vector_incidence_pandas(
    df_2023_inc_dist_cedlas['Honduras'],
    df_2018_pit_incid_lacir['Honduras'],
    df_2023_pit_rev_gdp_pct['Honduras']
)

print(x_hond_2023, "\n")

print(x_hond_2023.dot(df_2018_pit_incid_lacir['Honduras']), "\n")

# MEXICO 2023

x_mex_2023 = solve_vector_incidence_pandas(
    df_2023_inc_dist_cedlas['Mexico'],
    df_2018_pit_incid_lacir['Mexico'],
    df_2023_pit_rev_gdp_pct['Mexico']
)

print(x_mex_2023, "\n")

print(x_mex_2023.dot(df_2018_pit_incid_lacir['Mexico']), "\n")

# PERU 2023

x_per_2023 = solve_vector_incidence_pandas(
    df_2023_inc_dist_cedlas['Peru'],
    df_2018_pit_incid_lacir['Peru'],
    df_2023_pit_rev_gdp_pct['Peru']
)

print(x_per_2023, "\n")

print(x_per_2023.dot(df_2018_pit_incid_lacir['Peru']), "\n")

# URUGUAY 2023

x_uru_2023 = solve_vector_incidence_pandas(
    df_2023_inc_dist_cedlas['Uruguay'],
    df_2018_pit_incid_lacir['Uruguay'],
    df_2023_pit_rev_gdp_pct['Uruguay']
)

print(x_uru_2023, "\n")

print(x_uru_2023.dot(df_2018_pit_incid_lacir['Uruguay']), "\n")
