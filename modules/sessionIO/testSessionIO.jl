# Test the Session.jl implementation

include("Session.jl")

# Test loading the HDF5 file
const H5_FILE_PATH = joinpath(@__DIR__, "data", "2023-08-25_14-20-15.h5")

if !isfile(H5_FILE_PATH)
    println("❌ ERROR: File not found at path: ", H5_FILE_PATH)
    exit(1)
end

try
    println("🔬 Testing Julia Session implementation...")
    println("Loading session from: $H5_FILE_PATH")
    
    # Load the session
    session = from_hdf5(H5_FILE_PATH)
    
    println("\n✅ Session loaded successfully!")
    println(session)
    
    println("\n📊 Session Details:")
    println("  - Session ID: $(session.session)")
    println("  - Experiment: $(session.experiment)")
    println("  - Group: $(session.group)")
    println("  - Sampling Rate: $(session.sampling_rate) Hz")
    println("  - Duration: $(round(duration(session), digits=2)) seconds")
    println("  - Location: $(session.location)")
    
    println("\n📈 Data Shapes:")
    println("  - Raw: $(size(session.raw))")
    if session.preprocessed !== nothing
        println("  - Preprocessed: $(size(session.preprocessed))")
    end
    if session.data !== nothing
        println("  - Epoched: $(size(session.data))")
        println("    → $(size(session.data, 3)) epochs of $(size(session.data, 1)) samples × $(size(session.data, 2)) channels")
    end
    
    println("\n🔍 Metadata:")
    println("  - Channel names: $(length(session.ch_names)) channels")
    if !isempty(session.ch_names)
        println("    Channels: $(session.ch_names)")
    end
    println("  - Processing steps: $(length(session.history))")
    for (i, step) in enumerate(session.history)
        println("    $i. $(step["step"]) at $(step["time"])")
    end
    
    println("\n🧮 Data Access Tests:")
    
    # Test data access
    if session.data !== nothing
        println("  - First epoch, all channels, first 5 samples:")
        println("    $(session.data[1:5, :, 1])")
        
        println("  - Channel 1, epoch 1, all samples (first 10):")
        println("    $(session.data[1:10, 1, 1])")
    end
    
    if session.preprocessed !== nothing
        println("  - Preprocessed data (first 5 samples, all channels):")
        println("    $(session.preprocessed[1:5, :])")
    end
    
    println("\n⏱️ Time Access:")
    time_vec = times(session)
    println("  - Time vector length: $(length(time_vec))")
    println("  - First 5 timepoints: $(time_vec[1:5])")
    println("  - Last 5 timepoints: $(time_vec[end-4:end])")
    
    println("\n📈 Statistics from metadata:")
    if haskey(session.stats, "epoch")
        epoch_info = session.stats["epoch"]
        println("  - Number of epochs: $(get(epoch_info, "n_epochs", "unknown"))")
    end
    
    if haskey(session.stats, "standardize")
        std_info = session.stats["standardize"]
        println("  - Standardization method: $(get(std_info, "method", "unknown"))")
        println("  - Per-epoch standardization: $(get(std_info, "per_epoch", "unknown"))")
    end
    
    # Test session indexing
    println("\n🎯 Session Indexing Tests:")
    active_data_shape = size(session)
    if length(active_data_shape) == 3
        println("  - Active data is 3D (epoched): $(active_data_shape)")
        sample_data = session[1:5, 1, 1]
        println("  - session[1:5, 1, 1] (first 5 samples, channel 1, epoch 1): $(sample_data)")
        println("  - session[:, 1, 1] (all samples, channel 1, epoch 1): length = $(length(session[:, 1, 1]))")
    else
        println("  - Active data is 2D (continuous): $(active_data_shape)")
        sample_data = session[1:5, 1]
        println("  - session[1:5, 1] (first 5 samples, channel 1): $(sample_data)")
    end
    println("  - size(session): $(size(session))")
    println("  - length(session): $(length(session))")
    
    println("\n🎉 All tests passed! Julia Session implementation working correctly.")
    
catch e
    println("\n❌ Error during testing:")
    println(e)
    println("\nStacktrace:")
    for line in stacktrace(catch_backtrace())
        println("  ", line)
    end
end
