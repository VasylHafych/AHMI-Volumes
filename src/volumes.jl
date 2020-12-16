"""
    Abstract type for a volume.
"""
abstract type VolumeType end

@with_kw struct HyperRectangle{
        F <: AbstractFloat,
        I <: Integer
    } <: VolumeType 
    niter::I = 10
    α::F  = 0.6
    step::F = 0.9
end

@with_kw struct HyperSpheres{
        F <: AbstractFloat,
        I <: Integer
    } <: VolumeType 
    niter::I = 10
    α::F  = 0.6
    step::F = 0.9
end


"""
    Create and adjust HyperRectangle (to be improved)
"""
function find_volumes(
        samples::AbstractArray{F,2}, 
        w::AbstractArray{R},
        log_d::AbstractArray{F},
        clst_center::AbstractArray{F}, 
        clst_mask::AbstractArray{B},
        logd_tr::Tuple,
        algorithm::HyperRectangle;
    ) where {F<:AbstractFloat, R<:Real, B<:Bool}
    
    
    t_treshold, t_quantile = logd_tr
    
    size_run = [quantile(vec(i), FrequencyWeights(w[clst_mask]), 0.85) for i in eachrow(abs.(samples[:,clst_mask] .- clst_center))]
    volume_mask = prod((clst_center .- size_run) .< samples .< (clst_center .+ size_run),  dims=1)[1,:]
    ll_masked = log_d[volume_mask]
    
    bool_check = (maximum(ll_masked) - minimum(ll_masked) > algorithm.α*t_treshold) && (minimum(ll_masked) < t_quantile)
    iter_ind = 1
    
    while  bool_check && (iter_ind < algorithm.niter)
        size_run = size_run .* algorithm.step 
        volume_mask = prod((clst_center .- size_run) .< samples .< (clst_center .+ size_run),  dims=1)[1,:]
        ll_masked = log_d[volume_mask]
        bool_check = (maximum(ll_masked) - minimum(ll_masked) > algorithm.α*t_treshold) && (minimum(ll_masked) < t_quantile)
        iter_ind += 1
        @info "Decreasing Volume "
    end
    
    if bool_check
        volume_mask = zeros(Bool, length(volume_mask))
        log_volume = NaN
    else 
        log_volume = sum(log.(2 .* size_run))
    end
    
    return volume_mask, log_volume
end

"""
    Create and adjust HyperSpheres (to be improved)
"""
function find_volumes(
        samples::AbstractArray{F,2}, 
        w::AbstractArray{R},
        log_d::AbstractArray{F},
        clst_center::AbstractArray{F}, 
        clst_mask::AbstractArray{B},
        logd_tr::Tuple,
        algorithm::HyperSpheres;
    ) where {F<:AbstractFloat, R<:Real, B<:Bool}
    
    
    t_treshold, t_quantile = logd_tr
    n_dims = size(samples)[1]
    
    r_vect = vec(sqrt.(sum((samples[:,clst_mask] .- clst_center).^2, dims=1)))
    size_run = quantile(r_vect, FrequencyWeights(w[clst_mask]), 0.5)
    volume_mask = vec(sqrt.(sum((samples .- clst_center).^2, dims=1))) .< size_run
    ll_masked = log_d[volume_mask]
    bool_check = (maximum(ll_masked) - minimum(ll_masked) < algorithm.α*t_treshold) && (minimum(ll_masked) > t_quantile)
    iter_ind = 1
    
    while  !bool_check && (iter_ind < algorithm.niter)
        @info "Decreasing Volume "
        size_run = size_run .* algorithm.step
        volume_mask = vec(sqrt.(sum((samples .- clst_center).^2, dims=1))) .< size_run
        ll_masked = log_d[volume_mask]
        bool_check = (maximum(ll_masked) - minimum(ll_masked) > 0.8*t_treshold) && (minimum(ll_masked) < t_quantile)
        iter_ind += 1
    end
    
    if !bool_check
        volume_mask = zeros(Bool, length(volume_mask))
        log_volume = NaN
    else 
        log_volume = n_dims/2*log(pi) + n_dims*log(size_run) - log(gamma(n_dims/2+1))
    end
    
    return volume_mask, log_volume
end

