from experiment import Query, ExperimentRegistry
from utils import minutes
import ipdb

exp = ExperimentRegistry()

exp.add_run('random 2 / 1y', "/storage2/payal/EveryQuery/results/2025-02-19_09-40-04_823984")
exp.add_run('random 5 / 1y', "/storage2/payal/EveryQuery/results/2025-02-19_09-41-36_883066")

r2_train_q = exp.get_one_run('random 2 / 1y').training_queries
r5_train_q = exp.get_one_run('random 5 / 1y').training_queries

for query in r2_train_q: 
    print(exp.evaluate('random 2 / 1y', query))
    print(exp.evaluate('random 5 / 1y', query))

for query in r5_train_q: 
    print(exp.evaluate('random 2 / 1y', query))
    print(exp.evaluate('random 5 / 1y', query))

metrics = exp.compare(
    "/storage2/payal/EveryQuery/results/2025-02-19_09-40-04_823984",
    "/storage2/payal/EveryQuery/results/2025-02-19_09-41-36_883066",
    r2_train_q
)
print(metrics)

metrics = exp.compare(
    "/storage2/payal/EveryQuery/results/2025-02-19_09-40-04_823984",
    "/storage2/payal/EveryQuery/results/2025-02-19_09-41-36_883066",
    r5_train_q
)
print(metrics)