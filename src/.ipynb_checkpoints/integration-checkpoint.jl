"""
    IntegrationAlgorithm

    Abstract type for integration algorithms.
"""
abstract type IntegrationAlgorithm end

@with_kw struct AHMIntegration{
        WA<:WhiteningAlgorithm,
        CA<:ClusteringAlgorithm,
        VT<:VolumeType
    } <: IntegrationAlgorithm
    
    whitening::WA = PartialEigenDecompos()
    clustering::CA = KMeans()
    volumes::VT = HyperRectangle()
end

function bat_integrate(
        target::BAT.DensitySampleVector, 
        algorithm::AHMIntegration
    )
    
    samples, weights, log_d = flatview(unshaped.(target.v)), target.weight, target.logd
    
    log_d_min = minimum(log_d )
    log_d = log_d .- log_d_min # subtract constant pedestal;
    
    tranform = find_whiten_trans(samples, weights, algorithm.whitening)
    samples_trans = apply_whiten_trans(samples, tranform, algorithm.whitening)
    
    debug = Vector{NamedTuple}() # to be deleted after testigns
    int_estimates = Vector{AbstractFloat}() 
    log_vol = Vector{AbstractFloat}() 
    
    for (ncind, nclust) in enumerate(algorithm.clustering.nclusters)
        
        @info "Iteration $(ncind)/$(length(algorithm.clustering.nclusters))"
       
        clst_assign, clst_centers = find_clusters(samples_trans, weights, nclust, algorithm.clustering)
        
        for clust_ind in Base.OneTo(nclust)
            cluster_mask = clst_assign .== clust_ind
            local_tranform = find_whiten_trans(samples_trans[:,cluster_mask], weights[cluster_mask], algorithm.whitening)
            samples_locally_trans = apply_whiten_trans(samples_trans, local_tranform, algorithm.whitening)
            clst_center = apply_whiten_trans(clst_centers[clust_ind], local_tranform, algorithm.whitening)
            mask_volume, volume = find_volumes(samples_locally_trans, weights, log_d, clst_center, cluster_mask, algorithm.volumes)
            
            hm_estimate = compute_hm(samples_locally_trans, weights, log_d, mask_volume, volume, abs(det(tranform.W)), abs(det(local_tranform.W)))

            hm_estimate = hm_estimate*exp(log_d_min)
            
            push!(debug, (smpls = samples_locally_trans, cl_mask = cluster_mask, cl_center = clst_center, volume = mask_volume) )
            push!(int_estimates, hm_estimate)
            push!(log_vol, log(volume))
        end
    end
    
    # Remove tails 
    low_l, up_l = quantile(int_estimates, 0.2), quantile(int_estimates, 0.8)
    mean_estimate = mean(int_estimates[low_l .< int_estimates .< up_l])
    std_estimate = std(int_estimates[low_l .< int_estimates .< up_l])
        
    return (
        result = Measurements.measurement(mean_estimate,std_estimate), 
        int_estimates = int_estimates,
        log_vol = log_vol,
        debug = debug, 
    )
end

function compute_hm(
        samples::AbstractArray{F}, 
        w::AbstractArray{R}, 
        log_d::AbstractArray{F}, 
        mask::AbstractArray{B}, 
        vol::AbstractFloat, 
        det1::AbstractFloat, 
        det2::AbstractFloat
    ) where {F<:AbstractFloat, R<:Real, B<:Bool} 
    
    # To do: Store evidence in a log. scale 
    r = sum(w[mask]) / sum(w) 
    ll_tmp = 1 ./ exp.(log_d[mask])
    x = mean(ll_tmp, weights(w[mask]))
    i_r = vol / (x*det1*det2)
    bias_corr = 1 - (r-1)/r/sum(w) - var(ll_tmp, FrequencyWeights(w[mask]))/sum(w[mask])/x^2 
    int_est = bias_corr*i_r / r
    
    return  int_est
end