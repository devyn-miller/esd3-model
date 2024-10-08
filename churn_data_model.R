# Load libraries
library(dplyr)
library(ggplot2)

# Load dataset
df <- read.csv('/Users/devynmiller/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv')


summary(df)
str(df)

# Handle missing values in TotalCharges by converting to numeric
df$TotalCharges <- as.numeric(df$TotalCharges)
df <- df %>% filter(!is.na(TotalCharges))

# Convert 'Churn' to binary (Yes = 1, No = 0)
df$Churn <- ifelse(df$Churn == "Yes", 1, 0)

# Convert 'Contract' to a factor (with Month-to-month as the reference level)
df$Contract <- factor(df$Contract, levels = c("Month-to-month", "One year", "Two year"))

# Quick check of cleaned data
summary(df)

# Churn distribution by contract type
ggplot(df, aes(x = Contract, fill = as.factor(Churn))) +
  geom_bar(position = "fill") +
  labs(title = "Churn by Contract Type", y = "Proportion", fill = "Churn")

# Distribution of MonthlyCharges
ggplot(df, aes(x = MonthlyCharges)) +
  geom_histogram(binwidth = 10, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Monthly Charges", x = "Monthly Charges", y = "Frequency")

# Tenure distribution
ggplot(df, aes(x = tenure)) +
  geom_histogram(binwidth = 5, fill = "lightgreen", color = "black") +
  labs(title = "Distribution of Customer Tenure", x = "Tenure (Months)", y = "Frequency")

# Churn by MonthlyCharges
ggplot(df, aes(x = as.factor(Churn), y = MonthlyCharges)) +
  geom_boxplot() +
  labs(title = "Monthly Charges by Churn", x = "Churn", y = "Monthly Charges")

# Churn by Tenure
ggplot(df, aes(x = as.factor(Churn), y = tenure)) +
  geom_boxplot() +
  labs(title = "Tenure by Churn", x = "Churn", y = "Tenure")

