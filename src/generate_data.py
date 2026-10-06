from pathlib import Path
import numpy as np
import pandas as pd


def main():
    rng = np.random.default_rng(42)
    n = 5000

    tenure = rng.integers(1, 73, n)
    monthly = np.round(rng.normal(75, 25, n).clip(20, 160), 2)
    tickets = rng.poisson(1.4, n).clip(0, 8)
    contract = rng.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.27, 0.18])
    autopay = rng.choice(['Yes', 'No'], n, p=[0.58, 0.42])
    internet = rng.choice(['Fiber optic', 'DSL', 'None'], n, p=[0.52, 0.38, 0.10])

    score = (
        -0.035 * tenure
        + 0.012 * (monthly - 75)
        + 0.33 * tickets
        + np.where(contract == 'Month-to-month', 1.0, np.where(contract == 'One year', 0.15, -0.55))
        + np.where(autopay == 'No', 0.45, -0.10)
        + np.where(internet == 'Fiber optic', 0.35, np.where(internet == 'None', -0.30, 0.0))
        - 0.45
    )
    prob = 1 / (1 + np.exp(-score))
    churn = rng.binomial(1, prob)

    df = pd.DataFrame({
        'tenure_months': tenure,
        'monthly_charges': monthly,
        'support_tickets': tickets,
        'contract_type': contract,
        'autopay': autopay,
        'internet_service': internet,
        'churn': churn,
    })
    Path('data').mkdir(exist_ok=True)
    df.to_csv('data/customer_churn.csv', index=False)
    print(f'Generated {len(df):,} rows; churn rate={df.churn.mean():.1%}')


if __name__ == '__main__':
    main()
