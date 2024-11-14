import numpy as np
import math
from .Abstract_Reranker import Abstract_Reranker
from tqdm import tqdm,trange

'''
FairRec Model for item-side fair re-ranking

@inproceedings{patro2020fairrec,
  title={Fairrec: Two-sided fairness for personalized recommendations in two-sided platforms},
  author={Patro, Gourab K and Biswas, Arpita and Ganguly, Niloy and Gummadi, Krishna P and Chakraborty, Abhijnan},
  booktitle={Proceedings of the web conference 2020},
  pages={1194--1204},
  year={2020}
}
'''


def greedy_round_robin(m,n,R,l,T,V,U,F):
    # greedy round robin allocation based on a specific ordering of customers (assuming the ordering is done in the relevance scoring matrix before passing it here)

    # creating empty allocations
    B={}
    for u in U:
        B[u]=[]

    # available number of copies of each producer
    Z={} # total availability
    P=range(n) # set of producers
    for p in P:
        Z[p]=l

    # allocating the producers to customers
    for t in range(1,R+1):
        #print("GRR round number==============================",t)
        for i in range(m):
            if T==0:
                return B,F
            u=U[i]
            # choosing the p_ which is available and also in feasible set for the user
            possible=[(Z[p]>0)*(p in F[u])*V[u,p] for p in range(n)]
            p_=np.argmax(possible)

            if (Z[p_]>0) and (p_ in F[u]) and len(F[u])>0:
                B[u].append(p_)
                F[u].remove(p_)
                Z[p_]=Z[p_]-1
                T=T-1
            else:
                return B,F
    # returning the allocation
    return B,F


def FairRec_train(U,P,k,V,alpha, m , n):
    # Allocation set for each customer, initially it is set to empty set
    A={}
    for u in U:
        A[u]=[]

    # feasible set for each customer, initially it is set to P
    F={}
    for u in U:
        F[u]=P[:]
    #print(sum([len(F[u]) for u in U]))

    # number of copies of each producer
    l=int(alpha*m*k/(n+0.0))

    # R= number of rounds of allocation to be done in first GRR
    R=int(math.ceil((l*n)/(m+0.0)))


    # total number of copies to be allocated
    T= l*n

    # first greedy round-robin allocation
    [B,F1]=greedy_round_robin(m,n,R,l,T,V,U[:],F.copy())
    F={}
    F=F1.copy()
    #print("GRR done")
    # adding the allocation
    for u in U:
        A[u]=A[u][:]+B[u][:]

    # second phase
    u_less=[] # customers allocated with <k products till now
    for u in A:
        if len(A[u])<k:
            u_less.append(u)

    # allocating every customer till k products
    for u in u_less:
        scores=V[u,:]
        new=scores.argsort()[-(k+k):][::-1]
        for p in new:
            if p not in A[u]:
                A[u].append(p)
            if len(A[u])==k:
                break

    return A

class FairRec(Abstract_Reranker):
    def __init__(self, config, weights=None):
        super().__init__(config, weights)

    def rerank(self, ranking_score, k):

        user_size = len(ranking_score)
        U=list(range(user_size)) # list of customers
        P=list(range(self.config['item_num'])) # list of producers
        rerank_list =FairRec_train(U,P,k, ranking_score ,alpha=self.config['para'], m=user_size, n=self.config['item_num'])
        return rerank_list

        #rho_reverse = 1/(self.rho*batch_size*self.TopK)


