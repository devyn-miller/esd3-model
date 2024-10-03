def calculate_market_share(prices, inertia=10**10, customers=3000):
    num_companies = len(prices)

    # Calculate inverse prices to distribute customers (lower price gets more customers)
    inverse_prices = [(1/inertia) + (1 / (price + 0.01)) for price in prices]
    total_inverse = sum(inverse_prices)

    # Calculate market share for each company based on inverse price proportion
    market_shares = [(inverse_price / total_inverse) * customers for inverse_price in inverse_prices]

    # Convert to integers and ensure the total is equal to the number of customers (handle any rounding issues)
    market_shares = [int(share) for share in market_shares]
    difference = customers - sum(market_shares)

    # Adjust the rounding difference if necessary
    for i in range(abs(difference)):
        market_shares[i % num_companies] += 1 if difference > 0 else -1

    return market_shares