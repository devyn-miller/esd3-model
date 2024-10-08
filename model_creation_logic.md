I decided to use real data to build the model. I used the churn data from the telco dataset. I cleaned the data and then built a logistic regression model to predict churn. I then used the results of the logistic regression model to build the `calculate_market_share()` function.

- The **coefficients from R** represent how factors like contract type, monthly charges, and tenure affect the odds of a customer leaving (churning).
- The Python function uses these coefficients to calculate **churn probability** for each customer based on contract length, monthly charges, and tenure.

### R Code/Dataset Analysis

1. **Data Cleaning**:

   - The dataset was loaded and missing values in `TotalCharges` were handled by converting it to numeric and filtering out rows with missing values.
   - The `Churn` column was converted into a binary format (`Yes = 1`, `No = 0`), and `Contract` was transformed into a factor variable (Month-to-month, One-year, Two-year) with month-to-month as the reference level.

2. **Exploratory Data Analysis**:

   - Several visualizations were created to understand the relationship between `Churn`, `Contract`, `MonthlyCharges`, and `Tenure`.
      - Customers with month-to-month contracts have a much higher churn rate.
      - Higher `MonthlyCharges` are associated with higher churn.
      - Customers with longer `Tenure` are less likely to churn.

3. **Propensity Score Matching (PSM)**:

   - PSM was used to balance key covariates (MonthlyCharges, Tenure) between groups (month-to-month vs. longer-term contracts). The goal of PSM was to reduce selection bias so that differences in churn can more reliably be attributed to contract length rather than other variables like tenure or monthly charges.
   - After matching, balance diagnostics showed that the treated and control groups were well-balanced on MonthlyCharges and Tenure, meaning we can confidently use this matched data to estimate the effect of contract length on churn.

4. **Logistic Regression Model**:

   - A logistic regression model was fitted to predict the likelihood of churn based on `binary_contract` (contract length), `MonthlyCharges`, and `Tenure`.
   - Key results:
      - __Contract Length (`binary_contract`)__: Customers on one-year or two-year contracts have significantly lower odds of churning compared to month-to-month customers.
      - **MonthlyCharges**: Higher monthly charges are associated with an increased likelihood of churn.
      - **Tenure**: Longer tenure reduces the likelihood of churn.

5. **Interaction Model**:

   - An interaction term was added to check if the effect of contract length depends on MonthlyCharges or Tenure.
      - The interaction between `binary_contract` and `MonthlyCharges` was significant, indicating that the relationship between monthly charges and churn is moderated by contract length.
      - The interaction between `binary_contract` and `Tenure` was not significant.

6. **ROC Curve and AUC**:

   - The model's AUC (Area Under the Curve) was 0.7902, indicating that the logistic regression model had good discriminatory power (i.e., it can distinguish between customers who churn and those who do not with reasonable accuracy).


The coefficients in the **logistic regression model** from the R analysis represent how different factors (like contract type, monthly charges, and tenure) affect the likelihood of **churn**. Each coefficient tells us the strength and direction of the relationship between the variable and the **odds of churn**.

#### Logistic Regression Formula Recap:

In logistic regression, the formula for the probability of churn is:

\[
P(\text{churn}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + \beta_2X_2 + ...)}}
\]

Where:
- \(P(\text{churn})\) is the **churn probability**.
- \(\beta_0\) is the **intercept** (base churn probability when all predictors are 0).
- \(\beta_1, \beta_2, ...\) are the **coefficients** for the respective predictor variables \(X_1, X_2, ...\) (like contract type, monthly charges, etc.).
- \(e\) is the mathematical constant for the exponential function.

Each coefficient tells us **how much the log-odds** of churn change with a one-unit increase in the corresponding predictor variable.

### Coefficients from the R Analysis

1. **Intercept (beta_0 = -1.570033)**:
   - The intercept represents the **baseline log-odds of churn** when all other variables (contract type, monthly charges, and tenure) are zero. Since the log-odds of churn are negative, the base probability of churn is relatively low when everything else is held constant.
   - **Interpretation**: This is the "starting point" for the churn probability when no other factors are considered.

2. **Contract Type (beta_1 = -1.357234)**:
   - This coefficient represents the effect of **contract length** (one-year or two-year) compared to the reference group (month-to-month contract).
   - A **negative coefficient** means that customers on longer contracts (one-year or two-year) have **lower odds of churning** compared to those on month-to-month contracts.
   - **Interpretation**: A negative value (-1.357234) indicates that customers with longer contracts are significantly less likely to churn. In other words, longer contracts help retain customers.

3. **Monthly Charges (beta_2 = 0.025150)**:
   - This coefficient represents the effect of **monthly charges** on churn.
   - A **positive coefficient** means that **higher monthly charges increase the odds of churn**. Each additional dollar in monthly charges increases the odds of a customer churning.
   - **Interpretation**: Higher monthly charges make customers more likely to leave. The more they pay, the more likely they are to churn.

4. **Tenure (beta_3 = -0.030779)**:
   - This coefficient represents the effect of **customer tenure** (the number of months a customer has been with the company) on churn.
   - A **negative coefficient** means that **longer tenure decreases the odds of churn**. Each additional month a customer stays reduces their likelihood of leaving.
   - **Interpretation**: Customers who have been with the company longer are less likely to churn.

5. **Interaction Term: Contract Type and Monthly Charges (beta_4 = 0.010766)**:
   - This interaction term represents how **contract type** modifies the relationship between **monthly charges** and churn. Specifically, it shows how the effect of monthly charges on churn changes for customers with longer contracts.
   - A **positive coefficient** here means that while higher monthly charges increase the odds of churn, the effect is **less severe for customers on longer contracts**.
   - **Interpretation**: For customers with longer contracts, the impact of high monthly charges on churn is somewhat reduced. This suggests that longer contracts provide some "protection" against the negative effect of higher charges.

### Using the R Analysis to Build `calculate_market_share()`

Using the R analysis, the key insights for building the `calculate_market_share()` function are:

1. **Contract Length**:

   - Customers on longer contracts (one-year or two-year) are significantly less likely to churn. The market share can be adjusted based on this finding, rewarding longer contract types.

2. **Monthly Charges**:

   - Higher monthly charges increase the likelihood of churn. You can penalize market share based on this, meaning that customers with higher monthly charges may be more likely to switch providers.

3. **Tenure**:

   - Customers with longer tenures are less likely to churn, meaning that retaining customers over time reduces churn risk. This should be reflected in a positive adjustment to market share for customers with higher tenure.

4. **Interaction Between Contract Length and Monthly Charges**:

   - The interaction between contract length and monthly charges was significant, suggesting that the effect of monthly charges on churn is moderated by the contract type. For customers on longer contracts, the negative impact of higher monthly charges is reduced.

### Python Code for `calculate_market_share()` Function

```python {"id":"01J9PTCWMKVCQJM29JRR1JWVJZ"}


def calculate_market_share(customer_data, initial_market_share):
    """
    Calculate the market share based on customer data using logistic regression findings.

    Parameters:
    - customer_data (DataFrame): A DataFrame containing customer information with columns:
      'binary_contract' (0 = Month-to-month, 1 = One/Two Year), 'MonthlyCharges', 'tenure'.
    - initial_market_share (float): The initial market share of the company before considering churn probabilities.

    Returns:
    - adjusted_market_share (float): The adjusted market share after accounting for churn probabilities.
    """
    # Coefficients from the logistic regression model (obtained from R analysis)
    beta_intercept = -1.570033  # Intercept from the logistic regression model
    beta_contract = -1.357234   # Coefficient for binary_contract (contract length)
    beta_monthly_charges = 0.025150  # Coefficient for MonthlyCharges
    beta_tenure = -0.030779    # Coefficient for tenure
    interaction_contract_monthly = 0.010766  # Interaction term for binary_contract * MonthlyCharges
    
    # Calculate churn probability using the logistic regression formula
    # churn_probability = 1 / (1 + exp(-(X * beta)))
    customer_data['churn_probability'] = 1 / (1 + np.exp(-(
        beta_intercept +
        beta_contract * customer_data['binary_contract'] +
        beta_monthly_charges * customer_data['MonthlyCharges'] +
        beta_tenure * customer_data['tenure'] +
        interaction_contract_monthly * customer_data['binary_contract'] * customer_data['MonthlyCharges']
    )))

    # Adjust market share based on the mean churn probability
    # If churn_probability is high, reduce the market share accordingly
    adjusted_market_share = initial_market_share * (1 - customer_data['churn_probability'].mean())

    return adjusted_market_share

# Example usage:
# Assume customer_data is a DataFrame with columns 'binary_contract', 'MonthlyCharges', 'tenure'
# initial_market_share = 0.5  # Starting with 50% market share

# Simulated customer data for demonstration
customer_data = pd.DataFrame({
    'binary_contract': [0, 1, 1, 0, 0],  # 0 = month-to-month, 1 = one-year or two-year
    'MonthlyCharges': [30, 100, 60, 45, 70],  # Monthly charges for the customers
    'tenure': [12, 48, 60, 5, 24]  # Tenure in months
})

initial_market_share = 0.5
adjusted_market_share = calculate_market_share(customer_data, initial_market_share)
print(f"Adjusted Market Share: {adjusted_market_share}")
```

### Explanation of the Code:

1. **Inputs**:

   - `customer_data`: This DataFrame contains columns representing each customer's contract length (`binary_contract`), monthly charges (`MonthlyCharges`), and tenure (`tenure`).
   - `initial_market_share`: The initial market share of the company before accounting for churn probabilities.

2. **Logistic Regression Model**:

   - The coefficients from the R model (`beta_intercept`, `beta_contract`, `beta_monthly_charges`, `beta_tenure`, and `interaction_contract_monthly`) are hardcoded in the function based on the results of the R analysis.
   - Using the logistic regression formula, we calculate the **churn probability** for each customer.

3. **Market Share Adjustment**:

   - The market share is adjusted by the mean churn probability. If the churn probability is high (i.e., many customers are likely to leave), the market share will be reduced.
   - The formula used is: `adjusted_market_share = initial_market_share * (1 - average_churn_probability)`.

4. **Interaction Term**:

   - The interaction term between `binary_contract` and `MonthlyCharges` from the R model is included to capture the moderating effect of contract length on the impact of monthly charges on churn.