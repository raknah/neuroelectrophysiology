# Session.jl - Streamlined EEG/MEG session loader for Julia
# Purpose: Load Python HDF5 files with consistent dimension ordering
# Conventions: (channels, samples) for 2D, (channels, samples, epochs) for 3D

using HDF5, JSON3

"""
    Session

Minimal session struct for loading EEG/MEG data from HDF5.
Julia conventions: (channels, samples) for 2D, (channels, samples, epochs) for 3D
"""
mutable struct Session
    # Core identifiers
    session::String
    experiment::String
    sampling_rate::Int
    
    # Data arrays (Julia neuroscience standard: channels-first)
    raw::Matrix{Float32}                           # (channels, samples)
    preprocessed::Union{Matrix{Float32}, Nothing}  # (channels, samples) 
    data::Union{Array{Float32, 3}, Nothing}        # (channels, samples, epochs)
    
    # Dimension labels for tracking data transformations
    raw_dimensions::Union{Vector{String}, Nothing}
    preprocessed_dimensions::Union{Vector{String}, Nothing}
    data_dimensions::Union{Vector{String}, Nothing}

    # Essential metadata
    good_channels::Union{Vector{Int}, Nothing}     # Julia 1-based indices
    notes::Dict{String, Any}
    history::Vector{Dict{String, Any}}
    group::Union{String, Nothing}
end

"""
    _fix_dimensions_2d(data, name, dimensions) -> (Matrix{Float32}, Vector{String})

Ensure 2D data is (channels, samples). Transpose if needed based on size heuristics.
Returns the reordered data and corresponding dimension labels.
"""
function _fix_dimensions_2d(data::AbstractMatrix, name::String, dimensions::Union{Vector{String}, Nothing})::Tuple{Matrix{Float32}, Union{Vector{String}, Nothing}}
    data_f32 = Matrix{Float32}(data)
    nrows, ncols = size(data_f32)
    
    # If rows >> cols and rows > 1000, it's likely (samples, channels)
    if nrows > ncols && nrows > 1000
        result = transpose(data_f32)
        reordered_dims = dimensions !== nothing ? reverse(dimensions) : nothing
        @info "$(name): Transposed $(size(data_f32)) → $(size(result)) [channels, samples]"
        return result, reordered_dims
    else
        @info "$(name): Kept $(size(data_f32)) [channels, samples]"
        return data_f32, dimensions
    end
end

"""
    _fix_dimensions_3d(data, name, dimensions) -> (Array{Float32, 3}, Vector{String})

Ensure 3D data is (channels, samples, epochs). Permute dimensions based on size heuristics.
Returns the reordered data and corresponding dimension labels.
"""
function _fix_dimensions_3d(data::AbstractArray{<:Real, 3}, name::String, dimensions::Union{Vector{String}, Nothing})::Tuple{Array{Float32, 3}, Union{Vector{String}, Nothing}}
    data_f32 = Array{Float32, 3}(data)
    dims = collect(size(data_f32))
    
    # Channels are always smallest dimension (2-6 typical)
    channel_idx = argmin(dims)
    other_dims = setdiff(1:3, channel_idx)
    
    # Put larger non-channel dimension as epochs (3rd), smaller as samples (2nd)
    if dims[other_dims[1]] > dims[other_dims[2]]
        epochs_idx, samples_idx = other_dims[1], other_dims[2]
    else
        epochs_idx, samples_idx = other_dims[2], other_dims[1]
    end
    
    # Create permutation to (channels=1, samples=2, epochs=3)
    perm = (channel_idx, samples_idx, epochs_idx)
    
    result = permutedims(data_f32, perm)
    
    # Reorder dimension labels if provided
    reordered_dims = nothing
    if dimensions !== nothing && length(dimensions) == 3
        reordered_dims = [dimensions[channel_idx], dimensions[samples_idx], dimensions[epochs_idx]]
    end
    
    @info "$(name): Permuted $(size(data_f32)) → $(size(result)) [channels, samples, epochs]"
    return result, reordered_dims
end

# Basic accessors for your workflow
Base.size(s::Session) = s.data !== nothing ? size(s.data) : s.preprocessed !== nothing ? size(s.preprocessed) : size(s.raw)
duration(s::Session) = size(s.raw, 2) / s.sampling_rate

"""
    set_dimensions!(session, raw_dims, preprocessed_dims, data_dims)

Update dimension labels for a session. Useful for tracking transformations like FFT.
"""
function set_dimensions!(s::Session; 
                        raw_dimensions::Union{Vector{String}, Nothing} = nothing,
                        preprocessed_dimensions::Union{Vector{String}, Nothing} = nothing,
                        data_dimensions::Union{Vector{String}, Nothing} = nothing)
    if raw_dimensions !== nothing
        s.raw_dimensions = raw_dimensions
    end
    if preprocessed_dimensions !== nothing
        s.preprocessed_dimensions = preprocessed_dimensions
    end
    if data_dimensions !== nothing
        s.data_dimensions = data_dimensions
    end
end

function Base.show(io::IO, s::Session)
    shape = size(s)
    data_type = s.data !== nothing ? "epoched" : s.preprocessed !== nothing ? "preprocessed" : "raw"
    
    # Get appropriate dimension labels
    dims = if s.data !== nothing
        s.data_dimensions
    elseif s.preprocessed !== nothing
        s.preprocessed_dimensions
    else
        s.raw_dimensions
    end
    
    if length(shape) == 3 && dims !== nothing && length(dims) >= 3
        print(io, "Session('$(s.session)', $(shape[1]) × $(shape[2]) × $(shape[3]) [$(dims[1]) × $(dims[2]) × $(dims[3])], $(data_type))")
    elseif length(shape) == 2 && dims !== nothing && length(dims) >= 2
        print(io, "Session('$(s.session)', $(shape[1]) × $(shape[2]) [$(dims[1]) × $(dims[2])], $(data_type))")
    else
        print(io, "Session('$(s.session)', $(join(string.(shape), " × ")), $(data_type))")
    end
end

"""
    from_hdf5(path::String) -> Session

Load session from HDF5, handling Python→Julia dimension conversion and indexing.
"""
function from_hdf5(path::String)::Session
    h5open(path, "r") do f
        # Load dimension labels from HDF5 if available (will be nothing for existing files)
        raw_dims = nothing
        preprocessed_dims = nothing
        data_dims = nothing
        
        try
            if haskey(attrs(f), "raw_dimensions")
                raw_dims_str = HDF5.read_attribute(f, "raw_dimensions")
                if raw_dims_str isa String
                    raw_dims = JSON3.read(raw_dims_str)
                end
            end
        catch
            raw_dims = nothing
        end
        
        try
            if haskey(attrs(f), "preprocessed_dimensions")
                preprocessed_dims_str = HDF5.read_attribute(f, "preprocessed_dimensions")
                if preprocessed_dims_str isa String
                    preprocessed_dims = JSON3.read(preprocessed_dims_str)
                end
            end
        catch
            preprocessed_dims = nothing
        end
        
        try
            if haskey(attrs(f), "data_dimensions")
                data_dims_str = HDF5.read_attribute(f, "data_dimensions")
                if data_dims_str isa String
                    data_dims = JSON3.read(data_dims_str)
                end
            end
        catch
            data_dims = nothing
        end
        
        # Load and fix data dimensions
        # Note: raw_dims, preprocessed_dims, data_dims will be nothing for existing HDF5 files
        raw, raw_dimensions = _fix_dimensions_2d(read(f["raw"]), "raw", raw_dims)
        
        preprocessed = nothing
        preprocessed_dimensions = nothing
        if haskey(f, "preprocessed")
            preprocessed, preprocessed_dimensions = _fix_dimensions_2d(read(f["preprocessed"]), "preprocessed", preprocessed_dims)
        end
        
        data = nothing
        data_dimensions = nothing
        if haskey(f, "data")
            data, data_dimensions = _fix_dimensions_3d(read(f["data"]), "data", data_dims)
        end
        
        # Load essential metadata
        attrs_dict = Dict{String, Any}()
        for name in keys(attrs(f))
            try
                val = HDF5.read_attribute(f, name)
                if val isa String && !(name in ["session", "experiment", "group"])
                    try
                        attrs_dict[name] = JSON3.read(val)
                    catch
                        attrs_dict[name] = val
                    end
                else
                    attrs_dict[name] = val
                end
            catch
                attrs_dict[name] = nothing
            end
        end
        
        # Convert Python 0-based indices to Julia 1-based
        good_channels = nothing
        if haskey(attrs_dict, "good_channels") && attrs_dict["good_channels"] !== nothing
            try
                gc_raw = attrs_dict["good_channels"]
                if gc_raw isa String
                    parsed = JSON3.read(gc_raw)
                    good_channels = [Int(x) + 1 for x in parsed]
                else
                    good_channels = [Int(x) + 1 for x in gc_raw]
                end
            catch
                good_channels = nothing
            end
        end
        
        # Handle notes as Dict{String,Any}
        notes_any = get(attrs_dict, "notes", Dict{String, Any}())
        notes = Dict{String,Any}()
        if notes_any isa String
            try
                parsed = JSON3.read(notes_any)
                notes = parsed isa Dict ? Dict{String,Any}(parsed) : parsed isa JSON3.Object ? Dict{String,Any}(parsed) : Dict{String,Any}("raw_notes" => notes_any)
            catch
                notes = Dict{String, Any}("raw_notes" => notes_any)
            end
        elseif notes_any isa JSON3.Object
            notes = Dict{String,Any}(notes_any)
        elseif notes_any isa Dict
            notes = Dict{String,Any}(notes_any)
        else
            notes = Dict{String,Any}()
        end
        
        # Handle history as Vector{Dict{String,Any}}
        history_any = get(attrs_dict, "history", Dict{String, Any}[])
        history = Vector{Dict{String,Any}}()
        if history_any isa String
            try
                parsed = JSON3.read(history_any)
                if parsed isa Vector
                    history = [d isa Dict ? Dict{String,Any}(d) : d isa JSON3.Object ? Dict{String,Any}(d) : Dict{String,Any}() for d in parsed]
                else
                    history = Dict{String,Any}[]
                end
            catch
                history = Dict{String,Any}[]
            end
        elseif history_any isa Vector
            history = [d isa Dict ? Dict{String,Any}(d) : d isa JSON3.Object ? Dict{String,Any}(d) : Dict{String,Any}() for d in history_any]
        else
            history = Dict{String,Any}[]
        end
        
        return Session(
            get(attrs_dict, "session", "unknown"),
            get(attrs_dict, "experiment", "unknown"),
            get(attrs_dict, "sampling_rate", 30000),
            raw,
            preprocessed,
            data,
            raw_dimensions,
            preprocessed_dimensions,
            data_dimensions,
            good_channels,
            notes,
            history,
            get(attrs_dict, "group", nothing)
        )
    end
end

# Export main functions
export Session, from_hdf5, duration, set_dimensions!
