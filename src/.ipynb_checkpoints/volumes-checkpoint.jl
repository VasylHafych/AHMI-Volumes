abstract type VolumeType end

struct HyperRectangle <: VolumeType end

function find_volumes(
        samples::AbstractArray{F,2}, 
        w::AbstractArray{R},
        log_d::AbstractArray{F},
        clst_center::AbstractArray{F}, 
        clst_mask::AbstractArray{B}, 
        algorithm::HyperRectangle
    ) where {F<:AbstractFloat, R<:Real, B<:Bool}
    
    std_arry = std(samples[:,clst_mask], FrequencyWeights(w[clst_mask]), 2)    
    volume_mask = prod((clst_center .- std_arry) .< samples .< (clst_center .+ std_arry),  dims=1)[1,:]
    return volume_mask, prod(2 .* std_arry)
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