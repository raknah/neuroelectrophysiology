using NeuroAnalyzer
using Statistics

path = "/Users/fomo/Documents/Research/UNIC Research/Neuroelectrophysiology/data/TEPAK EEG workshop/Brainvision_Day2"
raw = import_bv(joinpath(path, "Subject_1_2024-05-09_18-42-01.vhdr"))

# EDA 
begin
        
    @info "Raw Type:" typeof(raw)
    @info "Raw Object Properties:" propertynames(raw)
    @show "Raw Data Metadata" size(raw.data), eltype(raw.data)

    @info "Header properties" propertynames(raw.header)
    @info "Header contents" keys(raw.header.recording)

    @info "Sampling rate" fs = raw.header.recording[:sampling_rate]
    @info "Channel labels" labels = raw.header.recording[:label] 

    @info "Markers preview" first=first(raw.markers, 10)
    return nothing
end

# Markers
begin
    M = raw.markers
    stims = M[(M.id .== "Stimulus") .& (M.value .== "s1000"), :]

    starts = stims.start[1:2:end]
    stops = stims.start[2:2:end]
    durations = stops .- starts

    start_samples = floor.(Int, starts .* fs)
    stop_samples = floor.(Int, stops .* fs)

    conditions = [raw.data[:, start:stop, :] for (start, stop) in zip(start_samples, stop_samples)];
    @info "Built conditions" n_conditions=length(conditions) sizes=map(size, conditions)
end

# Preprocessing

begin
    flattened = [dropdims(condition, dims=3) for condition in conditions]

    eeg_idx = findall(l -> !(l in ["M1", "M2"]), labels)   # 30 channels
    mast_idx = findall(l ->  (l in ["M1", "M2"]), labels)  # 2 channels

    X = flattened[1]

    quick_rms(ch) = sqrt(mean(ch .^ 2))

    eeg_rms = [quick_rms(view(X, i, :)) for i in eeg_idx]
    mst_rms = [quick_rms(view(X, i, :)) for i in mast_idx]

    # Scalp-only average reference (exclude M1/M2 from the reference computation)
    # For each time point t, subtract mean across scalp channels from every channel.
    scalp_mean = vec(mean(@view(X[eeg_idx, :]), dims=1))  # length T
    X_ref = X .- reshape(scalp_mean, 1, :)                # 32 × T

    eeg_rms_ref = [quick_rms(view(X_ref, i, :)) for i in eeg_idx]
    mst_rms_ref = [quick_rms(view(X_ref, i, :)) for i in mast_idx]

    @info "RMS (raw)" eeg_mean=mean(eeg_rms) eeg_std=std(eeg_rms) mastoids=mst_rms
    @info "RMS (scalp-avg-ref)" eeg_mean=mean(eeg_rms_ref) eeg_std=std(eeg_rms_ref) mastoids=mst_rms_ref
    return nothing
end    

# referenced is contains scalp electrodes after subtracting mean of scalp electrodes

begin
    referenced = Vector{Matrix{Float64}}(undef, length(flattened))

    for c in eachindex(flattened)
        μ = vec(mean(@view(flattened[c][eeg_idx, :]), dims=1))
        referenced[c] = flattened[c] .- reshape(μ, 1, :)
    end

end

# checking if 50Hz notch necessary

begin
    cz = findfirst(==("Cz"), labels)
    x = @view referenced[1][cz, :]
    P = NeuroAnalyzer.psd(x; fs=fs)
    
    mask_50   = (P.f .>= 49) .& (P.f .<= 51)
    mask_near = ((P.f .>= 45) .& (P.f .< 49)) .| ((P.f .> 51) .& (P.f .<= 55))
    ratio50 = mean(P.p[mask_50]) / mean(P.p[mask_near])

    @show ratio50
    @show length(P.f) length(P.p) P.f[1] P.f[end]
end


using Plots
begin
    cz = findfirst(==("Cz"), labels)
    x = @view referenced[1][cz, :]
    P = NeuroAnalyzer.psd(x; fs=fs)
    Plots.plot(P.f, P.p,
         xlabel="Frequency (Hz)",
         ylabel="Power",
         title="PSD – Cz, condition 1",
         legend=false,
         xlims=[0, 100])
end

# Quantifying alpha/control power and their ratio
# bands: Dict(:alpha => (8,12), :control => (13,20)) by default
function compare_alpha_power(condition::Int,
                             electrodes::Vector{String};
                             bands::Dict{Symbol,<:Tuple{<:Real,<:Real}} = Dict(
                                :alpha => (8, 12),
                                :control => (13, 20)
                            ))
    results = Dict{String, NamedTuple}()

    for electrode in electrodes
        e = findfirst(==(electrode), labels)
        x = @view referenced[condition][e, :]
        P = NeuroAnalyzer.psd(x; fs = fs)

        a1, a2 = bands[:alpha]
        c1, c2 = bands[:control]

        mask_alpha   = (P.f .>= a1) .& (P.f .<= a2)
        mask_control = (P.f .>= c1) .& (P.f .<= c2)

        alpha_abs   = mean(P.p[mask_alpha])
        control_abs = mean(P.p[mask_control])
        ratio       = alpha_abs / control_abs

        results[electrode] = (
            alpha_abs = alpha_abs,
            control_abs = control_abs,
            ratio = ratio
        )
    end

    return results
end

for c in 1:5
    res = compare_alpha_power(c, ["Fz", "Cz", "Oz"])
    @info "alpha summary" condition=c bands=Dict(:alpha=>(8,12), :control=>(13,20)) results=res
end
