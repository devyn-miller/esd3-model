import numpy as np

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
