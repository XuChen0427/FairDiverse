import numpy as np
import math
from .Abstract_Reranker import Abstract_Reranker
from tqdm import tqdm,trange

import gzip,pickle
import random,math
from itertools import permutations
import sys,datetime
import networkx as nx

'''
FairRec+ Model for item-side fair re-ranking improved by the FairRec Model

@article{biswas2021toward,
  title={Toward Fair Recommendation in Two-sided Platforms},
  author={Biswas, Arpita and Patro, Gourab K and Ganguly, Niloy and Gummadi, Krishna P and Chakraborty, Abhijnan},
  journal={ACM Transactions on the Web (TWEB)},
  volume={16},
  number={2},
  pages={1--34},
  year={2021},
  publisher={ACM New York, NY}
}
'''


def remove_envy_cycle(B,U,V):
    r=0
    while True:
        r+=1
        #print("In envy cycle removal:",r)
        # create empty graph
        G=nx.DiGraph()
        # add nodes
        G.add_nodes_from(U)
        # edges
        E=[]
        # find edges
        #print("In envy cycle removal: finding edges")
        for u in U:
            for v in U:
                if u!=v:
                    V_u=0
                    V_v=0
                    for p in B[u]:
                        V_u+=V[u,p]
                    for p in B[v]:
                        V_v+=V[u,p]
                    if V_v>V_u:
                        E.append((u,v))
        # add edges to the graph
        G.add_edges_from(E)
        # find cycle and remove
        #print("In envy cycle removal: graph done, finding and removing cycles")
        try:
            cycle=nx.find_cycle(G,orientation="original")
            temp=B[cycle[0][0]][:]
            for pair in cycle:
                B[pair[0]]=B[pair[1]][:]
            B[cycle[-1][0]]=temp[:]
        except:
            break
    # topological sort
    U=list(nx.topological_sort(G))
    return B.copy(),U[:]

# greedy round robin allocation based on a specific ordering of customers
# This is the modified greedy round robin where we remove envy cycles
def greedy_round_robin(m,n,R,l,T,V,U,F):
    # creating empty allocations
    B={}
    for u in U:
        B[u]=[]

    # available number of copies of each producer
    Z={} # total availability
    P=range(n) # set of producers
    for p in P:
        Z[p]=l

    # number of rounds
    r=0
    while True:
        # number of rounds
        r=r+1
        if r > 200:
            break
        # allocating the producers to customers
        #print("GRR round number==============================",r)

        for i in range(m):
            #user
            u=U[i]

            # choosing the p_ which is available and also in feasible set for the user
            possible=[(Z[p]>0)*(p in F[u])*V[u,p] for p in range(n)]
            p_=np.argmax(possible)

            if (Z[p_]>0) and (p_ in F[u]) and len(F[u])>0:
                B[u].append(p_)
                F[u].remove(p_)
                Z[p_]=Z[p_]-1
                T=T-1


            else: #stopping criteria
                #print("now doing envy cycle removal")
                B,U=remove_envy_cycle(B.copy(),U[:],V)
                return B.copy(),F.copy()

            if T==0: #stopping criteria
                #print("now doing envy cycle removal")
                B,U=remove_envy_cycle(B.copy(),U[:],V)
                return B.copy(),F.copy()
        # envy-based manipulations, m, U, V, B.copy()
        #print("GRR done")

        # remove envy cycle
        #print("now doing envy cycle removal")
        B,U=remove_envy_cycle(B.copy(),U[:],V)
        #print(sum([len(B[u]) for u in B]),T,n*l)
    # returning the allocation
    return B.copy(),F.copy()


def FairRecPlus_train(U,P,k,V,alpha, m, n):
    # Allocation set for each customer, initially it is set to empty set
    A={}
    for u in U:
        A[u]=[]

    # feasible set for each customer, initially it is set to P
    F={}
    for u in U:
        F[u]=P[:]
    #print(sum([len(F[u]) for u in U]))

    # l= number of copies of each producer, equal to the exposure guarantee for producers
    l=int(alpha*m*k/(n+0.0))

    # R= number of rounds of allocation to be done in first GRR
    R=int(math.ceil((l*n)/(m+0.0)))


    # T= total number of products to be allocated
    T= l*n

    # first greedy round-robin allocation
    B={}
    [B,F1]=greedy_round_robin(m,n,R,l,T,V,U[:],F.copy())
    F={}
    F=F1.copy()
    #print("GRR done")
    # adding the allocation
    for u in U:
        A[u]=A[u][:]+B[u][:]


    # filling the recommendation set upto size k
    u_less=[]
    for u in A:
        if len(A[u])<k:
            u_less.append(u)
    for u in u_less:
        scores=V[u,:]
        new=scores.argsort()[-(k+k):][::-1]
        for p in new:
            if p not in A[u]:
                A[u].append(p)
            if len(A[u])==k:
                break
    end_time=datetime.datetime.now()
    return A

class FairRecPlus(Abstract_Reranker):
    def __init__(self, config, weights=None):
        super().__init__(config, weights)

    def rerank(self, ranking_score, k):

        user_size = len(ranking_score)
        U=list(range(user_size)) # list of customers
        P=list(range(self.config['item_num'])) # list of producers
        rerank_list =FairRecPlus_train(U,P,k, ranking_score ,alpha=self.config['para'], m=user_size, n=self.config['item_num'])
        return rerank_list

        #rho_reverse = 1/(self.rho*batch_size*self.TopK)


