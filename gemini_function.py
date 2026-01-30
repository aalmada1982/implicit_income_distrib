# https://www.google.com/search?sca_esv=038229efecd48614&sxsrf=ANbL-n57FmZxlqiqtY6_Uc_HvvqtkXeFjg%3A1769782809075&source=hp&ei=Gb58abqiAq3N1sQP-YKs6A8&iflsig=AFdpzrgAAAAAaXzMKdqNj-yoj19fQkoQxnGUtzBxuHuA&aep=26&udm=50&ved=0ahUKEwj6kqbdurOSAxWtppUCHXkBC_0QteYPCBc&oq=&gs_lp=Egdnd3Mtd2l6IgBIAFAAWABwAHgAkAEAmAEAoAEAqgEAuAEByAEAmAIAoAIAmAMAkgcAoAcAsgcAuAcAwgcAyAcAgAgA&sclient=gws-wiz&mstk=AUtExfAT4JZNUdIHfm-N9v3lwWd-WDEbn3-ftp9nVs5TcgGqFtX-8-AsIy6psIczZWDsGdO_Kl6sogsmTRD27iaGiL6QkGck5oZOkPWpwiat8U_hhoZhAn2OG6EOZpOtdIE_aZt7C3XJ0YGCO8ukolrx7nFXbTrK9HBHjaTX4GiRRdM3xQXEt1kj1e2sEzCHLeP-qDdjNB0iktCbQmds5-JGiYlWahwhe0Fc5n-ShZ6RPTQrqxMZrGMs5x4M8J5cgfJ8YyBN0KL9zeEYBM3c52ZGxnAJOmD1ErLt_Yq5Kg0Vz_dS8n3m3nqr2IaK7jemQUD_X3JmYuIHFsWGHFqKX6GFLQBgOQZgioXk014pm-5PLWgCoPFAFp7-wP8&mtid=jiN0adf8Apzb5OUPkYDdsQI&csuir=1&q=using+python%2C+solve+the+following+problem+iteratively%3A%0Ax+vector+is+1x10+vector+of+unknown+values+which+sum+is+1%0Ax_init+is+1x10+vector+of+initial+estimates+values+of+x+which+sum+is+1%2C+the+aproximations+in+x_init+are+the+starting+point+for+the+algorithm%0Ax_init+%3D+%5B+0.018%2C+0.032%2C+0.044%2C+0.056%2C+0.069%2C+0.085%2C+0.103%2C+0.128%2C+0.167%2C+0.298%5D%0Ai+vector+is+a+10x1+vector+of+incidence%2C+with+the+following+values%0Ai+%3D+%5B0.000%2C+0.000%2C+0.000%2C+0.001%2C+0.002%2C+0.003%2C+0.004%2C+0.007%2C+0.012%2C+0.044%5D%0Athe+vector+product+x+*+i+is+equal+to+an+scalar+r+%3D+0.023%0Astarting+from++x_init+iterative+adjust+vector+x+values+so+the+vector+product+x+*+i+converges+to+the+scalar+r+%3D+0.023&atvm=2


import numpy as np
from scipy.optimize import minimize


def solve_vector_incidence(x_init, i_vector, r_target, precision=0.0001):
    """
    Adjusts x_init iteratively so that dot product with i_vector hits r_target.

    Args:
        x_init (np.array): 1x10 vector of initial estimates (must sum to 1).
        i_vector (np.array): 10x1 incidence vector.
        r_target (float): The target scalar product result (e.g., 0.023).
        precision (float): The maximum allowed difference between result and target.

    Returns:
        np.array: The adjusted x vector if successful, otherwise None.
    """
    # 1. Define the floor constraint (-75% of initial values)
    lower_bounds = x_init * 0.25
    bounds = [(low, 1.0) for low in lower_bounds]

    # 2. Define the objective: Minimize squared error to reach r_target
    def objective(x):
        return (np.dot(x, i_vector) - r_target)**2

    # 3. Define constraint: Sum of x elements must equal 1.0
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]

    # 4. Execute the SLSQP solver
    # We use a tighter tolerance (tol) to guarantee the user's precision
    result = minimize(
        objective,
        x_init,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        tol=1e-12
    )

    if result.success:
        diff = abs(np.dot(result.x, i_vector) - r_target)
        if diff <= precision:
            return result.x
        else:
            print(f"Solver finished but precision was only {diff:.6f}")
            return None
    else:
        print(f"Optimization failed: {result.message}")
        return None


# --- Example Usage: Uruguay (2018)

# x_int	0.0227	0.0359	0.0468	0.0577	0.0697	0.0835	0.1005	0.1241	0.1622	0.2968
# i	0.0000	0.0000	0.0000	0.0010	0.0030	0.0060	0.0110	0.0200	0.0340	0.0920
# target	3.9000

x_start = np.array([0.0227,	0.0359,	0.0468,	0.0577,	0.0697,
                   0.0835,	0.1005,	0.1241,	0.1622,	0.2968])
incidence = np.array([0.0000, 0.0000, 0.0000, 0.0010,
                     0.0030, 0.0060, 0.0110, 0.0200, 0.0340, 0.0920])
target = 0.0390

final_x = solve_vector_incidence(x_start, incidence, target)

if final_x is not None:
    print("Final Adjusted x Vector:")
    print(np.round(final_x, 6))
    print(f"\nFinal Product: {np.dot(final_x, incidence):.8f}")
    print(f"Final Sum: {np.sum(final_x):.8f}")
