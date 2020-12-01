abstract type ClusteringAlgorithm end

@with_kw struct KMeans <: ClusteringAlgorithm
    nclusters = [1, 2, 4, 8, 16, 32]
end

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