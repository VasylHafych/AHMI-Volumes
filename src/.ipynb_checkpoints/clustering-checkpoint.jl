"""
    Abstract type for clustering algorithms.
"""
abstract type ClusteringAlgorithm end


"""
    KMeans clustering. 

    - nclusters : Specifies how many space partitiong are used in the integration.  
"""
@with_kw struct KMeans <: ClusteringAlgorithm
    nclusters = [1, 2, 4, 8, 16, 32]
end


"""
    Compute KMeans clusters. Returns matrix of cluster assignments and vector of cluster means. 

    - samples : MCMC samples
    - samples : MCMC weights
    - n_clusters : Number of clusters
    - algorithm : KMeans
"""
function find_clusters(
        samples::AbstractArray{F,2}, 
        w::AbstractArray{R}, 
        n_clusters::Integer, 
        algorithm::KMeans
    ) where {F<:AbstractFloat, R<:Real}
    
    clusters = kmeans(samples, n_clusters, weights=w)
    centers = [i for i in eachcol(clusters.centers)]
    return clusters.assignments, centers 
end