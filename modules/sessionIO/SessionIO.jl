# Session.jl - Julia implementation for loading and working with EEG/MEG session data
# Optimized for Julia with compatibility for Python-generated HDF5 files

using HDF5, JSON3, Dates

"""
    Session

Represents an EEG/MEG recording session with raw data, preprocessing,
metadata, and convenient serialization to/from HDF5 for cross-language analysis.

Optimized for Julia performance with proper type annotations and memory layout.
"""
mutable struct Session
    # Core metadata
    session::String
    experiment::String
    sampling_rate::Int
    schema_version::Int
    
    # Data arrays (Julia column-major optimized)
    raw::Matrix{Float32}                    # (samples, channels)
    preprocessed::Union{Matrix{Float32}, Nothing}  # (samples, channels)
    data::Union{Array{Float32, 3}, Nothing}        # (samples_per_epoch, channels, epochs)
    
    # Channel and spatial info
    ch_names::Vector{String}
    montage::Union{Matrix{Float32}, Nothing}  # (channels, 3) for x,y,z positions
    location::Union{String, Nothing}
    
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
    good_channels::Union{Vector{Int}, Nothing}
end

"""
    Session(session, experiment, raw; kwargs...)

Create a new Session with required fields and optional parameters.
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
    location::Union{String, Nothing} = nothing
)
    # Ensure proper Julia memory layout (column-major)
    raw_julia = Matrix{Float32}(raw)
    preprocessed_julia = preprocessed !== nothing ? Matrix{Float32}(preprocessed) : nothing
    data_julia = data !== nothing ? Array{Float32, 3}(data) : nothing
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
        Dict{String, Any}("schema_version" => 1),
        nothing,  # group
        nothing,  # events
        Dict{String, Any}(),
        Dict{String, Any}[],
        nothing,  # states
        nothing,  # ica_model
        nothing,  # ica_sources
        nothing,  # bad_ics
        nothing   # good_channels
    )
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

"""Get number of samples in the active data array"""
Base.length(session::Session) = size(session)[1]

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

"""Display session information"""
function Base.show(io::IO, session::Session)
    print(io, "Session('$(session.session)', size=$(size(session)), group=$(session.group))")
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
    from_hdf5(path::String) -> Session

Load a Session from an HDF5 file created by Python or Julia.
Handles the data layout differences between Python (row-major) and Julia (column-major).
"""
function from_hdf5(path::String)::Session
    local session_obj
    
    h5open(path, "r") do f
        # Load datasets - handle dimension ordering
        raw = read(f["raw"])  # Will be (samples, channels) from Python
        
        preprocessed = haskey(f, "preprocessed") ? read(f["preprocessed"]) : nothing
        
        # Handle epoched data - Python saves as (epochs, channels, samples_per_epoch)
        # Julia loads as (samples_per_epoch, channels, epochs) due to memory layout
        data = haskey(f, "data") ? read(f["data"]) : nothing
        
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
        
        # Create session object
        session_obj = Session(
            session_id,
            experiment,
            raw;
            preprocessed = preprocessed,
            data = data,
            sampling_rate = sampling_rate,
            ch_names = ch_names,
            montage = montage,
            location = location
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
                session_obj.stats = JSON3.read(stats_raw)
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
                session_obj.history = JSON3.read(history_raw)
            catch
                session_obj.history = Dict{String, Any}[Dict("raw_history" => history_raw)]
            end
        elseif history_raw isa Vector
            session_obj.history = history_raw
        else
            session_obj.history = Dict{String, Any}[]
        end
        session_obj.good_channels = get(metadata, "good_channels", nothing)
        session_obj.bad_ics = get(metadata, "bad_ics", nothing)
        
        # Handle events
        events_raw = get(metadata, "events", nothing)
        if events_raw isa Vector && !isempty(events_raw)
            try
                session_obj.events = [(Int(evt[1]), String(evt[2])) for evt in events_raw]
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
"""
function to_hdf5(session::Session, path::String)
    println("Saving Session to $path...")
    println("  - Raw data shape: $(size(session.raw))")
    session.preprocessed !== nothing && println("  - Preprocessed data shape: $(size(session.preprocessed))")
    session.data !== nothing && println("  - Final data shape: $(size(session.data))")
    
    h5open(path, "w") do f
        # Save datasets with compression
        h5_write(f, "raw", session.raw, compress=3)
        
        if session.preprocessed !== nothing
            h5_write(f, "preprocessed", session.preprocessed, compress=3)
        end
        
        if session.data !== nothing
            h5_write(f, "data", session.data, compress=3)
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
    return Dict{String, Any}(
        "session" => session.session,
        "experiment" => session.experiment,
        "sampling_rate" => session.sampling_rate,
        "location" => session.location,
        "ch_names" => session.ch_names,
        "montage" => session.montage,
        "notes" => session.notes,
        "group" => session.group,
        "events" => session.events,
        "stats" => session.stats,
        "history" => session.history,
        "good_channels" => session.good_channels,
        "bad_ics" => session.bad_ics
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

# Export main functions
export Session, from_hdf5, to_hdf5, save, load, add_notes!, annotate!, log_step!
export duration, times
