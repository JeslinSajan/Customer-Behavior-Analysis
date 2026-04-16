# Business Insight Report - Customer Behavior Analysis

## 1. Executive Snapshot
- Total customers analyzed: **3000**
- Average monthly spend: **555.22**
- Average satisfaction score: **4.29**
- Churn rate: **0.77%**
- High purchase-intent rate: **58.40%**

## 2. Key Behavioral Patterns
- Customers with lower satisfaction and more complaints show higher churn tendency.
- Purchase frequency and monthly spend are positively related.
- Region **North** currently has the highest churn proportion.

## 3. Customer Segmentation Insights
High-value segment (Segment **1**) shows highest average spend.

Segment averages:

```
         monthly_spend  purchase_frequency  satisfaction_score  churn
segment                                                              
0               497.92                2.10                3.70   0.03
1               868.57                3.87                4.31   0.00
2               465.55                4.06                4.59   0.00
3               460.23                2.74                4.41   0.00
```

## 4. Predictive Analysis (Churn)
- Model used: Logistic Regression
- ROC-AUC: **0.997**
- Churn precision: **0.800**
- Churn recall: **0.667**

## 5. Business Recommendations
1. Launch retention campaigns for low-satisfaction and high-complaint customers.
2. Run region-specific interventions in high-churn areas (especially **North**).
3. Build premium bundles for high-value segments and targeted offers for medium-value segments.
4. Use churn-risk scores from the model to prioritize proactive support calls.

## 6. Visual Outputs
Generated in the `output/` folder:
- `churn_by_region.png`
- `spend_vs_frequency.png`
- `satisfaction_vs_churn.png`
- `correlation_matrix.png`
