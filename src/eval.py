from experiment import Query, ExperimentRegistry
from utils import minutes
import ipdb

exp = ExperimentRegistry()

exp.add_run('stage_0', "")
exp.add_run('stage_1', "")
exp.add_run('stage_2', "")
exp.add_run('stage_3', "")

eval_queries = exp.get_one_run('stage_0').training_queries
# also get hold out queries

for query in eval_queries: 
    print(query)
    for s in [3,2,1,0]:
        print(f'\t stage_{s}', exp.evaluate(f'stage_{s}', query)['occurs_auc'])
    print()

# comp = exp.compare_printer('stage_0', 'stage_1', eval_queries)
# comp = exp.compare_printer('stage_0', 'stage_2', eval_queries)
# comp = exp.compare_printer('stage_1', 'stage_2', eval_queries)
# comp = exp.compare_printer('stage_0', 'stage_3', eval_queries)
# comp = exp.compare_printer('stage_1', 'stage_3', eval_queries)
# comp = exp.compare_printer('stage_2', 'stage_3', eval_queries)

exp.plot_auroc_comparison('stage_0','stage_1',eval_queries).figure.savefig('01.png')
exp.plot_auroc_comparison('stage_0','stage_2',eval_queries).figure.savefig('02.png')
exp.plot_auroc_comparison('stage_1','stage_2',eval_queries).figure.savefig('12.png')
exp.plot_auroc_comparison('stage_0','stage_3',eval_queries).figure.savefig('03.png')
exp.plot_auroc_comparison('stage_1','stage_3',eval_queries).figure.savefig('13.png')
exp.plot_auroc_comparison('stage_2','stage_3',eval_queries).figure.savefig('23.png')
exp.plot_auroc_heatmap(['stage_0','stage_1','stage_2', 'stage_3'], eval_queries).figure.savefig('heatmap.png', dpi=600, bbox_inches="tight")
exp.plot_auroc_clustermap(['stage_0','stage_1','stage_2', 'stage_3'], eval_queries).figure.savefig('clustermap.png', dpi=600, bbox_inches="tight")