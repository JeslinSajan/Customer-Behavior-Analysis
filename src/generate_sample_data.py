import numpy as np
import pandas as pd


def generate_customer_data(num_customers: int = 2000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    customer_ids = np.arange(1, num_customers + 1)
    ages = rng.integers(18, 70, size=num_customers)
    gender = rng.choice(["Male", "Female", "Other"], p=[0.48, 0.49, 0.03], size=num_customers)
    tenure_months = rng.integers(1, 72, size=num_customers)
    region = rng.choice(["North", "South", "East", "West"], size=num_customers)
    channel = rng.choice(["Online", "Retail", "Partner"], p=[0.55, 0.35, 0.10], size=num_customers)

    annual_income = rng.normal(60000, 18000, size=num_customers).clip(15000, 160000)
    monthly_spend = (
        (annual_income / 12) * rng.uniform(0.04, 0.16, size=num_customers)
        + tenure_months * rng.uniform(0.8, 2.2, size=num_customers)
    ).clip(20, 2500)
    purchase_frequency = rng.poisson(lam=3.2, size=num_customers).clip(0, 20)

    complaints_last_6m = rng.poisson(lam=0.6, size=num_customers).clip(0, 12)
    support_tickets = rng.poisson(lam=1.4, size=num_customers).clip(0, 20)
    discount_usage = rng.uniform(0, 1, size=num_customers)

    satisfaction_score = (
        4.5
        - (complaints_last_6m * 0.35)
        - (support_tickets * 0.08)
        + (purchase_frequency * 0.05)
        + rng.normal(0, 0.5, size=num_customers)
    ).clip(1, 5)

    churn_score = (
        0.4 * (complaints_last_6m > 2).astype(int)
        + 0.25 * (satisfaction_score < 2.8).astype(int)
        + 0.2 * (purchase_frequency < 2).astype(int)
        + 0.15 * (tenure_months < 12).astype(int)
        + rng.uniform(0, 0.3, size=num_customers)
    )
    churn_risk = np.where(churn_score > 0.75, 1, 0)

    purchase_intent = (
        0.45 * (satisfaction_score > 3.8).astype(int)
        + 0.35 * (discount_usage > 0.5).astype(int)
        + 0.2 * (purchase_frequency > 3).astype(int)
    )
    high_purchase_intent = np.where(purchase_intent > 0.6, 1, 0)

    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "age": ages,
            "gender": gender,
            "region": region,
            "acquisition_channel": channel,
            "tenure_months": tenure_months,
            "annual_income": annual_income.round(2),
            "monthly_spend": monthly_spend.round(2),
            "purchase_frequency": purchase_frequency,
            "complaints_last_6m": complaints_last_6m,
            "support_tickets": support_tickets,
            "discount_usage_ratio": discount_usage.round(3),
            "satisfaction_score": satisfaction_score.round(2),
            "churn": churn_risk,
            "high_purchase_intent": high_purchase_intent,
        }
    )

    # Inject a small amount of missingness to reflect real-world data quality.
    for col in ["annual_income", "monthly_spend", "satisfaction_score"]:
        missing_indices = rng.choice(df.index, size=int(0.03 * num_customers), replace=False)
        df.loc[missing_indices, col] = np.nan

    return df


if __name__ == "__main__":
    data = generate_customer_data(num_customers=3000)
    data.to_csv("data/customer_data.csv", index=False)
    print("Generated sample dataset at data/customer_data.csv")
