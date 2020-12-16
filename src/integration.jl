"""
    Abstract type for integration algorithms.
"""
abstract type IntegrationAlgorithm end

"""
    AHMI integration algorithm. 

    - whitening : Whitening transformation 
    - clustering : Clustering algorithm
    - volumes : Integration volumes 
"""
@with_kw struct AHMIntegration{
        WA<:WhiteningAlgorithm,
        CA<:ClusteringAlgorithm,
        VT<:VolumeType,
    } <: IntegrationAlgorithm
    
    whitening::WA = PartialEigenDecompos()
    clustering::CA = KMeans()
    volumes::VT = HyperRectangle()
end

"""
    Integrate 'target' using using AHMIntegration. 

    Returns named tuple  

    - result : Evidence estimation and std in a form of Measurements.measurement
    - log_estimates : Log. estimates of integral form different clusters 
    - log_vol : Log. volumes 
    - debug : Named tuples with stored cluster masks (to be deleted after debugging).
"""
function bat_integrate(
        target::BAT.DensitySampleVector, 
        algorithm::AHMIntegration
    )
    
    samples, weights, log_d = flatview(unshaped.(target.v)), target.weight, target.logd
    t_rat, t_quant  = maximum(log_d) - minimum(log_d), quantile(log_d, 0.002) # use 0.2 quantile on the log_d to avoid large volumes
    
    tranform = find_whiten_trans(samples, weights, algorithm.whitening)
    samples_trans = apply_whiten_trans(samples, tranform, algorithm.whitening)
    
    debug = Vector{NamedTuple}() # to be deleted after debugging
    int_estimates = Vector{AbstractFloat}() 
    log_vol = Vector{AbstractFloat}() 
    
    for (ncind, nclust) in enumerate(algorithm.clustering.nclusters)
        
        @info "Iteration $(ncind)/$(length(algorithm.clustering.nclusters))"
       
        clst_assign, clst_centers = find_clusters(samples_trans, weights, nclust, algorithm.clustering)
        
        for clust_ind in Base.OneTo(nclust)
            cluster_mask = clst_assign .== clust_ind
            
            if sum(cluster_mask) > 0.005 * length(weights)
                local_tranform = find_whiten_trans(samples_trans[:,cluster_mask], weights[cluster_mask], algorithm.whitening)
                samples_locally_trans = apply_whiten_trans(samples_trans, local_tranform, algorithm.whitening)
                clst_center = apply_whiten_trans(clst_centers[clust_ind], local_tranform, algorithm.whitening)
                
                mask_volume, volume = find_volumes(
                    samples_locally_trans, 
                    weights, log_d, 
                    clst_center, 
                    cluster_mask,
                    (t_rat, t_quant), 
                    algorithm.volumes
                )
                hm_estimate = compute_hm(weights, log_d, mask_volume, volume, log(abs(det(tranform.W))), log(abs(det(local_tranform.W))))
                push!(debug, (smpls = samples_locally_trans, cl_mask = cluster_mask, cl_center = clst_center, volume = mask_volume) )
                push!(int_estimates, hm_estimate)
                push!(log_vol, volume)
            end
            
        end
    end
    
    # Remove tails (requires taking exp, think of better way)
    exp_estimates = exp.(int_estimates)
    low_l, up_l = quantile(exp_estimates, 0.2), quantile(exp_estimates, 0.8)
    mean_estimate = mean(exp_estimates[low_l .< exp_estimates .< up_l])
    std_estimate = std(exp_estimates[low_l .< exp_estimates .< up_l]) # is it legal? 
        
    return (
        result = Measurements.measurement(mean_estimate,std_estimate), 
        log_estimates = int_estimates,
        log_vol = log_vol,
        debug = debug, 
    )
end

"""
    Compute HM Estimate in the given volume. 

    - w : MCMC weights
    - log_d : Log. density
    - mask : mask which specifies which samples are used in HM estimate
    - log_vol : Log. volume
    - log_det1 : Log. determinant of the 1st whitening
    - log_det2 : Log. determinant of the 2nd whitening
"""
function compute_hm(
        w::AbstractArray{R}, 
        log_d::AbstractArray{F}, 
        mask::AbstractArray{B}, 
        log_vol::AbstractFloat, 
        log_det1::AbstractFloat, 
        log_det2::AbstractFloat
    ) where {F<:AbstractFloat, R<:Real, B<:Bool} 
    
    log_d_tmp = log_d[mask]
    ped = maximum(log_d_tmp) # remove constant pedestal 
    log_d_tmp = log_d_tmp .- ped
    
    n_Ω = sum(w) 
    n_Δ = sum(w[mask])
    
    r = n_Δ/n_Ω
    ll_tmp = 1 ./ exp.(log_d_tmp)
    x = mean(ll_tmp, weights(w[mask]))
    
    log_i_r = log_vol - log(x) - log_det1 - log_det2
    bias_corr = 1 - (r-1)/r/n_Ω - var(ll_tmp, FrequencyWeights(w[mask]))/n_Δ/x^2  # bias correction
    
    log_int_est = log(bias_corr) + log_i_r - log(r) + ped
    
    return  log_int_est 
end
