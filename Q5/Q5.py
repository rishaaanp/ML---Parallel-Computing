import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# B. Dataset (Grocery Store Transactions)
# Example Grocery Transactions Dataset
transactions = [

    ['milk','bread','butter'],
    ['beer','bread'],
    ['milk','bread','butter','eggs'],
    ['beer','diapers'],
    ['milk','diapers','beer','bread'],
    ['bread','butter'],
    ['milk','diapers','bread','butter'],
    ['eggs','bread'],
    ['milk','bread','eggs'],
    ['diapers','beer']

]

print("\nTransactions Dataset:")

for i,t in enumerate(transactions):
    print(f"Transaction {i+1} :", t)

# C. Generate Individual Itemsets
# Convert transactions → One Hot Encoding
encoder = TransactionEncoder()

encoded = encoder.fit(transactions).transform(transactions)

df = pd.DataFrame(encoded, columns=encoder.columns_)

print("\nOne Hot Encoded Dataset:")

print(df.head())

# Calculate Support of Individual Items
support = df.mean()

print("\nSupport of Individual Items:")

print(support.sort_values(ascending=False))

# D. Apriori Algorithm (Frequent Itemsets)
min_support = 0.3

frequent_itemsets = apriori(

    df,
    min_support=min_support,
    use_colnames=True

)

print("\nFrequent Itemsets:")

print(frequent_itemsets)

# E. Association Rules
min_confidence = 0.6

rules = association_rules(

    frequent_itemsets,
    metric="confidence",
    min_threshold=min_confidence

)

print("\nAssociation Rules:")

print(rules[['antecedents',
             'consequents',
             'support',
             'confidence',
             'lift']])

# F. Evaluate Rules
# Strong Rules (High Lift)
strong_rules = rules[rules['lift'] > 1]

print("\nHigh Lift Association Rules:")

print(strong_rules[['antecedents',
                    'consequents',
                    'support',
                    'confidence',
                    'lift']])


print("\nAssociation Rule Mining Completed Successfully ✅")