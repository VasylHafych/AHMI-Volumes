"""
    Abstract type for whitening algorithms.
"""
abstract type WhiteningAlgorithm end

"""
    Whitening by eigenvalue Decomposition using only dimensions specified by steep_edges_ind .== false. 
    Other dimensions are not whitened to preserve space morphology. 
"""
@with_kw struct PartialEigenDecompos <: WhiteningAlgorithm
    steep_edges_ind = Array{Bool}[]
end

"""
    Whitening by eigenvalue decomposition for all dimensions. 
"""
struct EigenDecompos <: WhiteningAlgorithm end

"""
    Standardize data to unit variance, with no decorrelation. 
"""
struct Standardizing <: WhiteningAlgorithm end

"""
    Given samples, determine which dimension/s contain steep edges.  
"""
function find_steep_edges(
        samples::AbstractArray{F,1}, 
        w::AbstractArray{R}; α=0.01, 
        nbins=100
    ) where {F<:AbstractFloat, R<:Real}
    
    hist_weights = fit(Histogram, samples, weights(w), nbins=nbins).weights
    max_weight = maximum(hist_weights)

    if hist_weights[1] < α*max_weight && hist_weights[end] < α*max_weight
        return false
    else
        return true
    end
    
end

function _find_whiten_trans(
        samples::AbstractArray{F,2}, 
        w::AbstractArray{R}, 
        steep_edges_ind::AbstractArray{B}
    ) where {F<:AbstractFloat, R<:Real, B<:Bool}
      
    μ = mean(samples, FrequencyWeights(w), 2)
    
    cov_m = cov(samples .- μ, FrequencyWeights(w), 2)
    
    for (ind, edge) in enumerate(steep_edges_ind)
        if edge 
            tmp_vals = cov_m[ind,ind]
            cov_m[:,ind] .= 0.0
            cov_m[ind,:] .= 0.0
            cov_m[ind,ind] = tmp_vals
        end
    end
    
    L = eigvals(cov_m)
    E = eigvecs(cov_m)
    W = E'
    W = W ./ sqrt.(L .+ 1e-4)
    
    return (W = W, μ=μ)
    
end

"""
    Find PartialEigenDecompos transformation. Returns named tuple with transformation matrix W and mean vector μ. 
"""
function find_whiten_trans(
        samples::AbstractArray{F,2}, 
        w::AbstractArray{R}, 
        algorithm::PartialEigenDecompos
    ) where {F<:AbstractFloat, R<:Real}
       
    if isempty(algorithm.steep_edges_ind)
        steep_edges_ind = [find_steep_edges(row, w) for row in eachrow(samples)] 
    else 
        steep_edges_ind = algorithm.steep_edges_ind
        if size(samples)[1] .!= length(steep_edges_ind)
            @error "Dimension mismatch"
            throw("Error")
        end
    end
    
    return _find_whiten_trans(samples, w, steep_edges_ind)
end

function find_whiten_trans(
        samples::AbstractArray{F,2}, 
        w::AbstractArray{R}, 
        algorithm::EigenDecompos
    ) where {F<:AbstractFloat, R<:Real}
    
    steep_edges_ind = zeros(Bool, size(samples)[1])
    return _find_whiten_trans(samples, w, steep_edges_ind)
end

function find_whiten_trans(
        samples::AbstractArray{F, 2},
        w::AbstractArray{R}, 
        algorithm::Standardizing
    ) where {F<:AbstractFloat, R<:Real}
    
    μ = mean(samples, FrequencyWeights(w), 2)
    W = std(samples .- μ, FrequencyWeights(w), 2)
    return (W = W, μ=μ)
end

"""
    Apply EigenDecompos/PartialEigenDecompos transformation. 
"""
function apply_whiten_trans(
        samples::AbstractArray{F}, 
        tranform::NamedTuple, 
        algorithm::Union{EigenDecompos, PartialEigenDecompos}
    ) where {F<:AbstractFloat}
    
    return tranform.W*(samples .- tranform.μ)
end

function apply_whiten_trans(
        samples::AbstractArray{F}, 
        tranform::NamedTuple, 
        algorithm::Standardizing
    ) where {F<:AbstractFloat}
    
    return (samples .- tranform.μ)./tranform.W
end