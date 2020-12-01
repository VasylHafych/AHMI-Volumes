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