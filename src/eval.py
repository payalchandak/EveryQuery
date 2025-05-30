from experiment import Query, ExperimentRegistry
from utils import minutes
import ipdb

exp = ExperimentRegistry()

exp.add_run('stage_0', "/home/pac4279/EveryQuery/results/2025-05-26_17-07-50_945117")
exp.add_run('stage_1', "/home/pac4279/EveryQuery/results/2025-05-26_22-03-31_379915")

eval_queries = exp.get_one_run('stage_0').training_queries

comp = exp.compare_printer('stage_0', 'stage_1', eval_queries)

for query in eval_queries: 
    print(query)
    for s in [0,1]:
        print(f'\t stage_{s}', exp.evaluate(f'stage_{s}', query)['occurs_auc'])
    print()