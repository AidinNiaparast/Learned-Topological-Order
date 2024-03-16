# Learned Topological Order
The official implementation of the Learned DFS Ordering (LDFS) algorithm and the baselines in the following paper:

McCauley, S., Moseley, B., Niaparast, A. and Singh, S., 2024. Incremental Topological Ordering and Cycle Detection with Predictions.
arXiv preprint arXiv:2402.11028.

## Datasets

We use the following real temporal networks from SNAP Large Network Dataset Collection (https://snap.stanford.edu/data/):
1. email-Eu-core [1] (https://snap.stanford.edu/data/email-Eu-core-temporal.html)
2. CollegeMsg [2] (https://snap.stanford.edu/data/CollegeMsg.html)
3. Math Overflow [1] (https://snap.stanford.edu/data/sx-mathoverflow.html): we use the sx-mathoverflow-a2q file, which contains the Answers-to-Questions network.

## Codes

1. LDFS.py implements the Learned DFS Ordering (LDFS) algorithm in the paper.
2. DFSII.py implements the algorithm for maintaining a topological order described in [3].
3. Test.py prints and plots the results of the experiments.

## Reproducing the results

To get the results of the experiments "Scaling Training Data Size" and "Extreme Stress Test" for real datasets, run getResults function with the appropriate dataset name.
To generate the results of the experiment "Synthetic DAG: Scaling Edge Density", which is explained in the appendix of the paper, run changeDensity(1000).

## References
[1] Ashwin Paranjape, Austin R. Benson, and Jure Leskovec. "Motifs in Temporal Networks." In Proceedings of the Tenth ACM International Conference on Web Search and Data Mining, 2017.

[2] Pietro Panzarasa, Tore Opsahl, and Kathleen M. Carley. "Patterns and dynamics of users' behavior and interaction: Network analysis of an online community." Journal of the American Society for Information Science and Technology 60.5 (2009): 911-932. 

[3] Marchetti-Spaccamela, Alberto, Umberto Nanni, and Hans Rohnert. "On-line graph algorithms for incremental compilation." In International Workshop on Graph-Theoretic Concepts in Computer Science, pp. 70-86. Berlin, Heidelberg: Springer Berlin Heidelberg, 1993.
