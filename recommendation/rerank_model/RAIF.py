import numpy as np
from .Abstract_Reranker import Abstract_Reranker
from gurobipy import Model, GRB, quicksum

r"""
RAIF, a model-agnostic repeat-bias-aware item fairness optimization (RAIF) algorithm based on MILP.
Note that we remove repeat bias term, change the item fairness objective to make the exposure of each group closer, and extend RAIF into multi-group cases.

@article{liu2025repeat,
  title={Repeat-bias-aware Optimization of Beyond-accuracy Metrics for Next Basket Recommendation},
  author={Liu, Yuanna and Li, Ming and Aliannejadi, Mohammad and de Rijke, Maarten},
  journal={arXiv e-prints},
  pages={arXiv--2501},
  year={2025}
}

"""


def get_results(num_users, size, topk, solution, topk_items):
    """
    Converts the solution matrix into selected item lists for multiple users.

    Parameters:
    ----------
    num_users: int
        The number of users.
    size: int
        The expected number of items per user in the final rerank list.
    topk: int
        The number of candidate items per user.
    solution: numpy.ndarray, shape (num_users, topk)
        A matrix indicating the final selected items.
    topk_items: list of list of int, shape (num_users, topk)
        A list where each entry contains candidate item IDs corresponding to a user.

    Returns:
    -------
    rerank: list of list of int, shape (num_users, size)
        A list where each entry contains exactly `size` selected items for a user.
    """
  
    rerank = []
    for i in range(num_users):

        rerank_user = []
        for j in range(topk):
            if solution[i, j] > 0.5:
                rerank_user.append(topk_items[i][j])

        assert len(rerank_user) == size
        rerank.append([int(x) for x in rerank_user])

    return rerank

def load_ranking_matrices(relevance, topk): 
    num_users, num_items = relevance.shape
    
    topk_items = np.zeros((num_users, topk), dtype=int)
    topk_scores = np.zeros((num_users, topk))

    for user_idx in range(num_users):
        # Get the indices of the items sorted by their relevance score in descending order
        sorted_indices = np.argsort(relevance[user_idx])[::-1]
        
        # Select the top k indices and corresponding scores
        topk_items[user_idx] = sorted_indices[:topk]
        topk_scores[user_idx] = relevance[user_idx, sorted_indices[:topk]]
    
    return topk_items, topk_scores, num_users


#save item group info
def read_item_index(total_users, topk, no_item_groups, item_group_map, topk_items):
    Ihelp = np.zeros((total_users, topk, no_item_groups))
    for uid in range(total_users):
        for lid in range(topk):
            for k in range(no_item_groups):
                if item_group_map[topk_items[uid][lid]] == k:
                    Ihelp[uid][lid][k] = 1

    return Ihelp


def fairness_optimisation(total_users, alpha, size, topk, group_num, Ihelp, topk_scores, mean):

    print(f"Running RAIF, {format(alpha, 'f')}")
    # V1: No. of users
    # V2: No. of top items (topk)
    # V4: no. of item groups
    V1, V2, V4 = range(total_users), range(topk), range(group_num)

    # initiate model
    model = Model()

    W = model.addVars(V1, V2, vtype=GRB.BINARY)
    item_group = model.addVars(V4, vtype=GRB.CONTINUOUS)
    item_fair = model.addVar(vtype=GRB.CONTINUOUS)
    abs_diff = model.addVars(V4, lb=0, name="abs_diff")
               
    model.setObjective(quicksum(topk_scores[i][j] * W[i, j] for i in V1 for j in V2) - alpha * item_fair, GRB.MAXIMIZE)

    for i in V1:
        model.addConstr(quicksum(W[i, j] for j in V2) == size)
    
    for k in V4:
        model.addConstr(item_group[k] == quicksum(W[i, j] * Ihelp[i][j][k] for i in V1 for j in V2))
    
    for k in V4:
        model.addConstr(abs_diff[k] >= item_group[k] - mean)
        model.addConstr(abs_diff[k] >= -(item_group[k] - mean))

    model.addConstr(item_fair == quicksum(abs_diff[k] for k in V4))


    # optimizing
    model.optimize()
    if model.status == GRB.OPTIMAL:
        solution = model.getAttr('x', W)
        fairness = model.getAttr('x', item_group)


    return solution, fairness



class RAIF(Abstract_Reranker):
    def __init__(self, config, weights = None):
        super().__init__(config, weights)


    def rerank(self, ranking_score, k):
        ## its parameters
        topk = self.config['candidate']
        alpha = self.config['alpha']

        topk_items, topk_scores, num_users = load_ranking_matrices(ranking_score, topk)
        mean = (num_users * k) / self.group_num
        Ihelp = read_item_index(total_users=num_users, topk=topk, no_item_groups=self.group_num, item_group_map=self.iid2pid, topk_items=topk_items) 
        solution, fairness = fairness_optimisation(num_users, alpha, k, topk, self.group_num, Ihelp, topk_scores, mean)
        rerank_list = get_results(num_users, k, topk, solution, topk_items)
        
        return rerank_list

