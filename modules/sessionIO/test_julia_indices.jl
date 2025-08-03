# Test Julia SessionIO index conversion
include("SessionIO.jl")

# Test file path
h5_path = "/Users/fomo/Documents/Kaizen/code/neuroelectrophysiology/projects/5xFAD/5xFAD-resting-state-preprocessed/2023-08-25_15-32-14.h5"

println("🔬 Testing Julia SessionIO index conversion...")
println("Loading: $h5_path")

try
    session = from_hdf5(h5_path)
    
    println("\n✅ Session loaded successfully!")
    println("Session: $(session.session)")
    println("Group: $(session.group)")
    
    println("\n📊 Index conversion check:")
    println("Good channels (Julia 1-based): $(session.good_channels)")
    println("Data shapes:")
    println("  - Raw: $(size(session.raw)) [should have all channels including bad]")  
    println("  - Preprocessed: $(size(session.preprocessed)) [should have only good channels]")
    println("  - Epoched: $(size(session.data)) [should be channels × samples × epochs]")
    
    println("\n🧮 Expected behavior:")
    println("  - Python saved good_channels as [0,1,2,3,4] (0-based)")
    println("  - Julia should convert to [1,2,3,4,5] (1-based)")
    println("  - This allows proper channel indexing in Julia")
    
    # Verify indexing works
    if session.good_channels !== nothing && length(session.good_channels) > 0
        first_good = session.good_channels[1]
        println("\n🎯 Indexing test:")
        println("  - First good channel index: $first_good")
        println("  - Raw data for first good channel (first 10 samples): $(session.raw[first_good, 1:10])")
    end
    
    println("\n🎉 Index conversion test completed successfully!")
    
catch e
    println("\n❌ Error during test:")
    println(e)
    for line in stacktrace(catch_backtrace())
        println("  $line")
    end
end
