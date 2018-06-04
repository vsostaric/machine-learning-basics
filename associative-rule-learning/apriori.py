import pandas as pd

dataset = pd.read_csv('../data/Market_Basket_Optimisation.csv')

transactions = []
for i in range(0, 1000):
    if(i % 100 == 0):
        print('Done: ' + str(i))
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

from apyori import apriori
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length = 2)

results = list(rules)
