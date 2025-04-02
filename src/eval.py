from experiment import Query, ExperimentRegistry
from utils import minutes
import ipdb

exp = ExperimentRegistry()

metric = 'occurs_auc'
print(metric)

exp.add_run('random 2 / 1y', "/storage2/payal/EveryQuery/results/2025-02-19_09-40-04_823984")
exp.add_run('random 5 / 1y', "/storage2/payal/EveryQuery/results/2025-02-19_09-41-36_883066")

r2_train_q = exp.get_one_run('random 2 / 1y').training_queries
r5_train_q = exp.get_one_run('random 5 / 1y').training_queries

# look into the first query where r2 is nan
for query in r2_train_q: 
    print(query)
    print('model r2')
    print(exp.evaluate('random 2 / 1y', query)[metric])
    print('model r5')
    print(exp.evaluate('random 5 / 1y', query)[metric])
print()

for query in r5_train_q: 
    print(query)
    print('model r2')
    print(exp.evaluate('random 2 / 1y', query)[metric])
    print('model r5')
    print(exp.evaluate('random 5 / 1y', query)[metric])
print()

print('r2 vs r5 compare on r2 training')
metrics = exp.compare(
    "/storage2/payal/EveryQuery/results/2025-02-19_09-40-04_823984",
    "/storage2/payal/EveryQuery/results/2025-02-19_09-41-36_883066",
    r2_train_q
)
print(metrics[metric])
print()

print('r2 vs r5 compare on r5 training')
metrics = exp.compare(
    "/storage2/payal/EveryQuery/results/2025-02-19_09-40-04_823984",
    "/storage2/payal/EveryQuery/results/2025-02-19_09-41-36_883066",
    r5_train_q
)
print(metrics[metric])
print()