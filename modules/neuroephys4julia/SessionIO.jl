# Session.jl - Julia implementation for loading and working with EEG/MEG session data
# Optimized for Julia with compatibility for Python-generated HDF5 files
# 
# DIMENSION CONVENTIONS (Julia neuroscience standard):
# - Raw/Preprocessed: (n_channels, n_samples) - channels-first for neuroscience readability
# - Epoched data: (n_channels, n_samples_per_trial, n_trials) - consistent across EEG epochs, MEP events, etc.
# - Python files are automatically transposed on load to match Julia conventions
#
# MEMORY LAYOUT RATIONALE:
# Julia uses column-major memory layout (first dimension varies fastest).
# For optimal performance, we want frequently accessed data to vary along the first axis.
# 
# Performance comparison for different conventions:
# 1. (samples, channels, epochs)     - Best Julia performance, but less readable
# 2. (channels, samples, epochs)     - Good compromise: readable + reasonably fast
# 3. (epochs, channels, samples)     - Matches Python/MNE but slower in Julia
#
# We chose #2 following common Julia neuroscience packages (MNE.jl, etc.)

using HDF5, JSON3, Dates

"""
    Session

Represents an EEG/MEG recording session with raw data, preprocessing,
metadata, and convenient serialization to/from HDF5 for cross-language analysis.

DIMENSION CONVENTIONS (Julia neuroscience standard):
- Raw/Preprocessed data: (n_channels, n_samples) - channels-first for readability
- Epoched data: (n_channels, n_samples_per_trial, n_trials) - consistent across all data types  
- Python files are automatically transposed to match these conventions
"""
mutable struct Session
    # Core metadata
    session::String
    experiment::String
    sampling_rate::Int
    schema_version::Int
    
    # Data arrays (Julia neuroscience standard: channels-first)
    raw::Matrix{Float32}                           # (n_channels, n_samples)
    preprocessed::Union{Matrix{Float32}, Nothing}  # (n_channels, n_samples) 
    data::Union{Array{Float32, 3}, Nothing}        # (n_channels, n_samples_per_trial, n_trials)
    
    # Channel and spatial info
    ch_names::Vector{String}
    montage::Union{Matrix{Float32}, Nothing}       # (n_channels, 3) for x,y,z positions
    location::Union{String, Nothing}
    
    # Channel mapping and provenance
    original_channels::Union{Vector{Int}, Nothing} # Hardware channel indices (e.g., [3,4,5,6,7,8])
    good_channels::Union{Vector{Int}, Nothing}     # Array indices of good channels (e.g., [1,2,4,5])
    
    # Analysis metadata
    notes::Dict{String, Any}
    group::Union{String, Nothing}
    events::Union{Vector{Tuple{Int, String}}, Nothing}
    stats::Dict{String, Any}
    history::Vector{Dict{String, Any}}
    
    # Optional analysis results
    states::Any
    ica_model::Any
    ica_sources::Union{Matrix{Float32}, Nothing}
    bad_ics::Union{Vector{Int}, Nothing}
end

"""
    Session(session, experiment, raw; kwargs...)

Create a new Session with required fields and optional parameters.
Automatically handles dimension ordering: ensures (n_channels, n_samples) for raw data.
"""
function Session(
    session::String,
    experiment::String,
    raw::AbstractMatrix;
    preprocessed::Union{AbstractMatrix, Nothing} = nothing,
    data::Union{AbstractArray{<:Real, 3}, Nothing} = nothing,
    sampling_rate::Int = 30000,
    ch_names::Vector{String} = String[],
    montage::Union{AbstractMatrix, Nothing} = nothing,
    location::Union{String, Nothing} = nothing,
    original_channels::Union{Vector{Int}, Nothing} = nothing
)
    # Ensure proper Julia neuroscience dimension ordering and memory layout
    raw_julia = _ensure_julia_dims(raw, "raw")
    preprocessed_julia = preprocessed !== nothing ? _ensure_julia_dims(preprocessed, "preprocessed") : nothing
    data_julia = data !== nothing ? _ensure_julia_dims_3d(data, "data") : nothing
    montage_julia = montage !== nothing ? Matrix{Float32}(montage) : nothing
    
    return Session(
        session,
        experiment,
        sampling_rate,
        1,  # schema_version
        raw_julia,
        preprocessed_julia,
        data_julia,
        ch_names,
        montage_julia,
        location,
        original_channels,
        nothing,  # good_channels
        Dict{String, Any}("schema_version" => 1),
        nothing,  # group
        nothing,  # events
        Dict{String, Any}(),
        Dict{String, Any}[],
        nothing,  # states
        nothing,  # ica_model
        nothing,  # ica_sources
        nothing   # bad_ics
    )
end

"""
    _ensure_julia_dims(data, name) -> Matrix{Float32}

Ensure 2D data has Julia neuroscience-standard dimensions: (n_channels, n_samples).
Based on our data: n_channels ≈ 5-6, n_samples ≈ 30k+. Use size heuristics for robust detection.
"""
function _ensure_julia_dims(data::AbstractMatrix, name::String)::Matrix{Float32}
    data_f32 = Matrix{Float32}(data)
    nrows, ncols = size(data_f32)
    
    # For EEG/neural data: channels (5-6) << samples (30k+)
    # If first dimension >> second dimension, it's (samples, channels) and needs transposing
    if nrows > ncols && nrows > 1000
        transposed = transpose(data_f32)
        @info "$(name): Transposing $(size(data_f32)) to $(size(transposed)) [channels, samples]"
        return transposed
    else
        @info "$(name): Keeping $(size(data_f32)) [channels, samples]"
        return data_f32
    end
end

"""
    _ensure_julia_dims_3d(data, name) -> Array{Float32, 3}

Ensure 3D data has Julia neuroscience-standard dimensions: (n_channels, n_samples_per_trial, n_trials).
Handles multiple data types:
- EEG epochs: ~3000 epochs, ~5 channels, ~100 samples  
- MEP events: ~60 events, ~2 channels, ~4000 samples
Uses consistent heuristics to identify channels (always smallest), then orders remaining dimensions.
"""
function _ensure_julia_dims_3d(data::AbstractArray{<:Real, 3}, name::String)::Array{Float32, 3}
    data_f32 = Array{Float32, 3}(data)
    d1, d2, d3 = size(data_f32)
    dims = [d1, d2, d3]
    
    # HEURISTIC: Channels are always the smallest dimension (2-6 typical)
    channel_idx = argmin(dims)
    channel_count = dims[channel_idx]
    
    # Get the two non-channel dimensions
    other_indices = setdiff(1:3, channel_idx)
    dim1_idx, dim2_idx = other_indices[1], other_indices[2]
    dim1_val, dim2_val = dims[dim1_idx], dims[dim2_idx]
    
    # CONSISTENT ORDERING: Always put larger dimension as "trials" (3rd), smaller as "samples" (2nd)
    if dim1_val > dim2_val
        # dim1 is trials, dim2 is samples
        trials_idx, samples_idx = dim1_idx, dim2_idx
    else
        # dim2 is trials, dim1 is samples  
        trials_idx, samples_idx = dim2_idx, dim1_idx
    end
    
    trials_count = dims[trials_idx]
    samples_count = dims[samples_idx]
    
    # Create permutation: (channels=1, samples=2, trials=3)
    perm = zeros(Int, 3)
    perm[channel_idx] = 1   # channels → position 1
    perm[samples_idx] = 2   # samples → position 2  
    perm[trials_idx] = 3    # trials → position 3
    
    result = permutedims(data_f32, perm)
    
    @info "$(name): Permuting $(size(data_f32)) to $(size(result)) [$(channel_count) channels, $(samples_count) samples, $(trials_count) trials]"
    return result
end

# Property accessors for Julia convenience
"""Get the shape of the active data array (data → preprocessed → raw)"""
function Base.size(session::Session)
    if session.data !== nothing
        return size(session.data)
    elseif session.preprocessed !== nothing
        return size(session.preprocessed)
    else
        return size(session.raw)
    end
end

"""Get number of channels"""
function n_channels(session::Session)
    sz = size(session)
    return length(sz) == 3 ? sz[1] : sz[1]  # First dimension is always channels
end

"""Get number of samples per epoch (or total samples if not epoched)"""
function n_samples(session::Session)
    sz = size(session)
    return length(sz) == 3 ? sz[2] : sz[2]  # Second dimension is samples
end

"""Get number of epochs (if epoched data exists)"""
function n_epochs(session::Session)
    sz = size(session)
    return length(sz) == 3 ? sz[3] : 1  # Third dimension is epochs (1 if not epoched)
end

"""Get number of samples per epoch (or total samples if not epoched)"""
Base.length(session::Session) = n_samples(session)

"""Index into the active data array"""
function Base.getindex(session::Session, key...)
    if session.data !== nothing
        return session.data[key...]
    elseif session.preprocessed !== nothing
        return session.preprocessed[key...]
    else
        return session.raw[key...]
    end
end

"""Get recording duration in seconds"""
duration(session::Session) = length(session) / session.sampling_rate

"""Get time axis vector in seconds"""
times(session::Session) = (0:length(session)-1) ./ session.sampling_rate

"""Display session information with clear dimension info"""
function Base.show(io::IO, session::Session)
    active_shape = size(session)
    data_type = session.data !== nothing ? "epoched" : 
                session.preprocessed !== nothing ? "preprocessed" : "raw"
    
    if length(active_shape) == 3
        print(io, "Session('$(session.session)', $(active_shape[1]) channels × $(active_shape[2]) samples × $(active_shape[3]) epochs, group=$(session.group), type=$data_type)")
    else
        print(io, "Session('$(session.session)', $(active_shape[1]) channels × $(active_shape[2]) samples, group=$(session.group), type=$data_type)")
    end
end

"""Add notes to session metadata"""
function add_notes!(session::Session, notes::Dict{String, Any})
    merge!(session.notes, notes)
    if haskey(notes, "group")
        session.group = notes["group"]
    end
end

"""Add event annotations"""
function annotate!(session::Session, onsets::Vector{Int}, labels::Vector{String})
    length(onsets) == length(labels) || throw(ArgumentError("onsets and labels must have equal length"))
    session.events = [(onset, label) for (onset, label) in zip(onsets, labels)]
end

"""Log a processing step with timestamp"""
function log_step!(session::Session, step_name::String, params::Dict{String, Any})
    entry = Dict{String, Any}(
        "step" => step_name,
        "params" => params,
        "time" => string(now())
    )
    push!(session.history, entry)
end

"""
    _convert_python_indices(raw_data) -> Union{Vector{Int}, Nothing}

Convert Python 0-based indices to Julia 1-based indices.
Handles String (JSON), Vector, JSON3.Array, or nothing.
"""
function _convert_python_indices(raw_data)::Union{Vector{Int}, Nothing}
    if raw_data isa String
        try
            parsed_data = JSON3.read(raw_data)
            return parsed_data !== nothing ? [idx + 1 for idx in parsed_data] : nothing
        catch
            return nothing
        end
    elseif (raw_data isa Vector || raw_data isa JSON3.Array) && !isempty(raw_data)
        return [idx + 1 for idx in raw_data]
    else
        return raw_data
    end
end

"""
    from_hdf5(path::String) -> Session

Load a Session from an HDF5 file created by Python or Julia.
Automatically handles dimension ordering and index conversion:
- Ensures consistent (channels, samples, trials) layout in Julia
- Converts Python 0-based indices to Julia 1-based indices
"""
function from_hdf5(path::String)::Session
    local session_obj
    
    h5open(path, "r") do f
        # Load datasets - handle dimension ordering automatically
        raw_python = read(f["raw"])  # Python: (n_channels, n_samples)
        raw = _ensure_julia_dims(raw_python, "raw")
        
        preprocessed = nothing
        if haskey(f, "preprocessed")
            preprocessed_python = read(f["preprocessed"])
            preprocessed = _ensure_julia_dims(preprocessed_python, "preprocessed")
        end
        
        # Handle epoched data - Python: (n_epochs, n_channels, n_samples) 
        data = nothing
        if haskey(f, "data")
            data_python = read(f["data"])
            data = _ensure_julia_dims_3d(data_python, "data")
        end
        
        # Load metadata attributes
        metadata = Dict{String, Any}()
        attrs_obj = attrs(f)
        
        for name in keys(attrs_obj)
            try
                attr_val = HDF5.read_attribute(f, name)
                if attr_val isa AbstractString && length(attr_val) > 0
                    # For session and experiment, keep as strings, don't parse as JSON
                    if name in ["session", "experiment", "group", "location"]
                        metadata[name] = attr_val
                    else
                        try
                            metadata[name] = JSON3.read(attr_val)
                        catch
                            metadata[name] = attr_val
                        end
                    end
                else
                    metadata[name] = attr_val
                end
            catch e
                @warn "Could not read attribute $name: $e"
                metadata[name] = missing
            end
        end
        
        # Extract core fields with defaults
        session_id = get(metadata, "session", "unknown")
        experiment = get(metadata, "experiment", "unknown")
        
        sampling_rate = get(metadata, "sampling_rate", 30000)
        location = get(metadata, "location", nothing)
        
        # Handle channel names
        ch_names_raw = get(metadata, "ch_names", String[])
        ch_names = if ch_names_raw isa Vector && !isempty(ch_names_raw)
            String[string(name) for name in ch_names_raw]
        else
            String[]
        end
        
        # Handle montage
        montage = get(metadata, "montage", nothing)
        if montage isa Vector
            try
                montage = reduce(hcat, montage)'  # Convert to proper matrix
            catch
                montage = nothing
            end
        end
        
        # Create session object with proper Julia dimensions
        session_obj = Session(
            session_id,
            experiment,
            raw;
            preprocessed = preprocessed,
            data = data,
            sampling_rate = sampling_rate,
            ch_names = ch_names,
            montage = montage,
            location = location,
            original_channels = get(metadata, "original_channels", nothing)
        )
        
        # Process and add remaining metadata safely
        # Handle notes 
        notes_raw = get(metadata, "notes", Dict{String, Any}())
        if notes_raw isa String
            try
                processed_notes = JSON3.read(notes_raw)
                session_obj.notes = isa(processed_notes, Dict) ? processed_notes : Dict{String, Any}("raw_notes" => notes_raw)
            catch
                session_obj.notes = Dict{String, Any}("raw_notes" => notes_raw)
            end
        elseif notes_raw isa Dict
            session_obj.notes = notes_raw
        else
            session_obj.notes = Dict{String, Any}()
        end
        
        session_obj.group = get(metadata, "group", nothing)
        
        # Handle stats
        stats_raw = get(metadata, "stats", Dict{String, Any}())
        if stats_raw isa String
            try
                parsed_stats = JSON3.read(stats_raw)
                session_obj.stats = Dict{String, Any}(parsed_stats)  # Convert to native Dict
            catch
                session_obj.stats = Dict{String, Any}("raw_stats" => stats_raw)
            end
        elseif stats_raw isa Dict
            session_obj.stats = stats_raw
        else
            session_obj.stats = Dict{String, Any}()
        end
        
        # Handle history 
        history_raw = get(metadata, "history", Dict{String, Any}[])
        if history_raw isa String
            try
                parsed_history = JSON3.read(history_raw)
                session_obj.history = Vector{Dict{String, Any}}(parsed_history)  # Convert to native Vector
            catch
                session_obj.history = Dict{String, Any}[Dict("raw_history" => history_raw)]
            end
        elseif history_raw isa Vector
            session_obj.history = history_raw
        else
            session_obj.history = Dict{String, Any}[]
        end
        # Handle good_channels and bad_ics - convert Python 0-based to Julia 1-based indexing
        session_obj.good_channels = _convert_python_indices(get(metadata, "good_channels", nothing))
        session_obj.bad_ics = _convert_python_indices(get(metadata, "bad_ics", nothing))
        
        # Handle events (convert onset indices from Python 0-based to Julia 1-based)
        events_raw = get(metadata, "events", nothing)
        if events_raw isa Vector && !isempty(events_raw)
            try
                # Convert event onsets from Python 0-based to Julia 1-based indexing
                session_obj.events = [(Int(evt[1]) + 1, String(evt[2])) for evt in events_raw]
            catch
                session_obj.events = nothing
            end
        end
    end
    
    return session_obj
end

"""
    to_hdf5(session::Session, path::String)

Save a Session to HDF5 format compatible with Python loading.
Saves Julia dimensions as-is: Python will need to transpose if needed.
"""
function to_hdf5(session::Session, path::String)
    println("Saving Session to $path...")
    println("  - Raw data shape: $(size(session.raw)) [channels, samples]")
    session.preprocessed !== nothing && println("  - Preprocessed data shape: $(size(session.preprocessed)) [channels, samples]")
    session.data !== nothing && println("  - Final data shape: $(size(session.data)) [channels, samples, epochs]")
    
    h5open(path, "w") do f
        # Save datasets with compression
        write(f, "raw", session.raw)
        
        if session.preprocessed !== nothing
            write(f, "preprocessed", session.preprocessed)
        end
        
        if session.data !== nothing
            write(f, "data", session.data)
        end
        
        # Save metadata as attributes
        metadata = to_dict(session)
        for (key, value) in metadata
            if value !== nothing
                if value isa Union{Int, Float64, String, Bool}
                    attrs(f)[key] = value
                else
                    attrs(f)[key] = JSON3.write(value)
                end
            end
        end
    end
    
    println("✅ Session saved successfully!")
end

"""Convert session metadata to dictionary for serialization"""
function to_dict(session::Session)
    # Convert Julia 1-based indices back to Python 0-based for compatibility
    good_channels_python = session.good_channels !== nothing ? [idx - 1 for idx in session.good_channels] : nothing
    bad_ics_python = session.bad_ics !== nothing ? [idx - 1 for idx in session.bad_ics] : nothing
    events_python = session.events !== nothing ? [(evt[1] - 1, evt[2]) for evt in session.events] : nothing
    
    return Dict{String, Any}(
        "session" => session.session,
        "experiment" => session.experiment,
        "sampling_rate" => session.sampling_rate,
        "location" => session.location,
        "ch_names" => session.ch_names,
        "montage" => session.montage,
        "notes" => session.notes,
        "group" => session.group,
        "events" => events_python,
        "stats" => session.stats,
        "history" => session.history,
        "good_channels" => good_channels_python,
        "original_channels" => session.original_channels,  # Hardware indices - keep as-is
        "bad_ics" => bad_ics_python
    )
end

"""Convenience function: save to directory with session name"""
function save(session::Session, directory::String)
    mkpath(directory)
    to_hdf5(session, joinpath(directory, "$(session.session).h5"))
end

"""Convenience function: load from directory by session ID"""
function load(directory::String, session_id::String)::Session
    return from_hdf5(joinpath(directory, "$(session_id).h5"))
end

# Utility functions for channel mapping
"""
    hardware_to_array_indices(session::Session, hardware_indices::Vector{Int}) -> Vector{Int}

Convert hardware channel indices to array indices.
E.g., if original_channels=[3,4,5,6,7,8] and you want hardware channels [4,7,8],
returns [2,5,6] (the array positions of those channels).
"""
function hardware_to_array_indices(session::Session, hardware_indices::Vector{Int})::Vector{Int}
    if session.original_channels === nothing
        @warn "No original_channels mapping available, assuming hardware_indices are array indices"
        return hardware_indices
    end
    
    array_indices = Int[]
    for hw_idx in hardware_indices
        array_pos = findfirst(==(hw_idx), session.original_channels)
        if array_pos !== nothing
            push!(array_indices, array_pos)
        else
            @warn "Hardware channel $hw_idx not found in original_channels $(session.original_channels)"
        end
    end
    
    return array_indices
end

"""
    array_to_hardware_indices(session::Session, array_indices::Vector{Int}) -> Vector{Int}

Convert array indices to hardware channel indices.
E.g., if original_channels=[3,4,5,6,7,8] and array_indices=[2,5,6],
returns [4,7,8] (the hardware channels at those array positions).
"""
function array_to_hardware_indices(session::Session, array_indices::Vector{Int})::Vector{Int}
    if session.original_channels === nothing
        @warn "No original_channels mapping available, assuming array_indices are hardware indices"
        return array_indices
    end
    
    hardware_indices = Int[]
    for arr_idx in array_indices
        if 1 <= arr_idx <= length(session.original_channels)
            push!(hardware_indices, session.original_channels[arr_idx])
        else
            @warn "Array index $arr_idx out of bounds for original_channels (length $(length(session.original_channels)))"
        end
    end
    
    return hardware_indices
end

"""
    good_hardware_channels(session::Session) -> Union{Vector{Int}, Nothing}

Get the hardware channel indices corresponding to good_channels.
"""
function good_hardware_channels(session::Session)::Union{Vector{Int}, Nothing}
    if session.good_channels === nothing
        return nothing
    end
    return array_to_hardware_indices(session, session.good_channels)
end

# Export main functions
export Session, from_hdf5, to_hdf5, save, load, add_notes!, annotate!, log_step!
export duration, times, n_samples, n_channels, n_epochs
export hardware_to_array_indices, array_to_hardware_indices, good_hardware_channels
