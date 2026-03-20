from sklearn.neighbors import NearestNeighbors
import numpy as np

def construct_adjacency_graph(X, n_neighbors, weight_method="simple"):
    """
    Input: 
        X: matrix or array of points
        n_neighbors: number of neighbors

    Output:
        matrix of adjacency with distance if there is an edge 0 if not

        The A.maximum(A.T) is used because the kneighbors_graph is directed, not symmetric. 
        Using the definition of the graph laplacian from:

            Belkin, M. & Niyogi, P. (2003). 
            Laplacian Eigenmaps for Dimensionality Reduction and Data Representation. 
            Neural Computation, 15(6), 1373–1396.
        
        They define an edge existing if i $\\in$ neigbors of j OR j $\\in$ neighbors of i.
        The form A.maximum(A.T) yields this definition. If point i $\\in$ neigbors of j then,
        the graph a_{ij} = d for the distance between them. then there are two cases:

            a) j $\\in$ neigbors of i  then, a_{ji} = d. So, max(a_{ij},a_{ji}) = d and the 
            resulting graph has both. No change is made to the resultant graph

            b) j $\\nin$ neigbors of i  then, a_{ji} = 0. So, max(a_{ij},a_{ji}) = d so the
            a_ji is reassigned to d resulting in a symmetric matrix.

    """
    if weight_method == "weighted":
        mode = "distance"
    if weight_method == "simple":
        mode = "connectivity"
    ## Calling class object. This step does not make the graph, just pre defines index for efficient computing
    neigbors = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    ## Creating graph adjacency matrix representation such that at joining points we have distance between neighbors
    neighborGraph = neigbors.kneighbors_graph(X, mode = mode)

    return  neighborGraph.maximum(neighborGraph.T) 



def add_weights(graph,t):
    """
    Input:
        graph: sparse CSR matrix of adjacency with Euclidean distances as edge values
        t: heat kernel parameter, controls the width of the Gaussian

    Output:
        sparse CSR matrix with heat kernel weights applied to edges

        Applies the heat kernel weighting scheme from:

            Belkin, M. & Niyogi, P. (2003). 
            Laplacian Eigenmaps for Dimensionality Reduction and Data Representation. 
            Neural Computation, 15(6), 1373–1396.

        For connected nodes i and j:

            W_{ij} = e^{-||x_i - x_j||^2 / t}

        The operation acts only on the .data array of the sparse matrix, which contains
        the stored nonzero entries. Zero entries are never materialized, preserving sparsity.
    """
    graph.data = np.exp((graph.data ** 2) * (-1/t))
    return graph


def construct_laplacian(weightedGraph :np.ndarray):
    """
    Input:
        weightedGraph: symmetric matrix of edge weights (dense array)

    Output:
        graph Laplacian matrix L = D - W

        D is the diagonal degree matrix where D_{ii} = sum of row i of W.
        W is the weighted adjacency matrix.

        The Laplacian L = D - W is symmetric positive semi-definite by construction
        when W is symmetric with non-negative entries. Its smallest eigenvalue is 0
        with eigenvector equal to the constant vector. The multiplicity of the zero
        eigenvalue equals the number of connected components in the graph.
    """
    D = np.diag(np.sum(weightedGraph,axis=0))
    return D - weightedGraph 

def eigen_decomposition(X,t, n_neighbors, method="weighted"):
    """
    Input:
        X: matrix or array of data points in R^l
        t: heat kernel parameter (used only when method="weighted")
        n_neighbors: number of nearest neighbors for graph construction
        method: "weighted" applies heat kernel weights, "simple" uses binary connectivity

    Output:
        values: eigenvalues of the graph Laplacian in ascending order
        vectors: corresponding eigenvectors as columns

        Full pipeline for Laplacian Eigenmaps:
            1. Construct symmetric k-NN adjacency graph
            2. Apply heat kernel weights if method="weighted"
            3. Compute graph Laplacian L = D - W
            4. Solve the eigenvalue problem Lv = lambda v

        The embedding is given by the eigenvectors corresponding to the smallest
        non-zero eigenvalues.
    """

    graph = construct_adjacency_graph(X,n_neighbors,weight_method=method)

    if method == "weighted":
        values, vectors =  np.linalg.eigh(construct_laplacian(add_weights(graph,t).toarray()))
        
    else:
        values, vectors =  np.linalg.eigh(construct_laplacian(graph.toarray()))
    
    return values, vectors

