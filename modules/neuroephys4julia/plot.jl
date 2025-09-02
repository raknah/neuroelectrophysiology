using CairoMakie, Statistics

"""
    gaussian_smooth(vec::AbstractVector, σ::Float64) -> Vector

Apply Gaussian smoothing to a 1D signal.
- Uses a symmetric kernel of radius ceil(3σ), normalized to sum to 1
- Edges handled by replicating boundary values (reflect-like padding)

Arguments:
- vec: input vector
- σ: Gaussian standard deviation in samples

Returns:
- Smoothed vector of the same length as input

Example:
- y = gaussian_smooth(x, 2.0)
"""
function gaussian_smooth(vec::AbstractVector, σ::Float64)
    radius = ceil(Int, 3σ)
    kernel = exp.(-((-radius:radius).^2) ./ (2σ^2))
    kernel ./= sum(kernel)

    # pad manually with replicated edge values
    padded = vcat(fill(first(vec), radius), vec, fill(last(vec), radius))
    
    # convolution
    fullconv = conv(padded, kernel)

    # extract the valid part with same length as input
    start_idx = radius + 1
    end_idx   = start_idx + length(vec) - 1
    return fullconv[start_idx:end_idx]
end


"""
    plot_raw_eeg(session; kwargs...) -> Figure

Plot raw or preprocessed EEG traces with options for standardization, log scaling,
and smoothing. Colors channels by good/bad if `session.good_channels` is available.

Required shape:
- session.raw or session.preprocessed: (channels, samples)

Keywords:
- sample_start::Int = 1
- sample_end::Int = size(session.raw, 2)
- name::AbstractString = "EEG"
- dataset::Symbol = :raw                 # :raw | :preprocessed
- standardize::Bool = false              # z-score per channel, clipped to [-3,3]
- log_scale::Bool = false                # log10(abs(x)+eps)
- window_size::Int = 1                   # >1 enables smoothing
- smooth_method::Symbol = :gaussian      # :moving_average | :gaussian
- time_units::Symbol = :ms               # :ms | :s
- good_color = :darkcyan; bad_color = :red; alpha=0.7; linewidth=2.1
- show::Bool = true                      # display the figure

Returns:
- CairoMakie Figure

Notes:
- Time axis is computed from `session.sampling_rate`.
- Safely handles missing `good_channels`.
"""
function plot_raw_eeg(
    session;
    sample_start=1,
    sample_end=size(session.raw, 2),
    name="EEG",
    log_scale=false,
    standardize=false,
    window_size=1,
    smooth_method=:gaussian,
    dataset=:raw,
    # consistency options
    time_units=:ms,           # :ms or :s
    good_color=:darkcyan,
    bad_color=:red,
    alpha=0.7,
    linewidth=2.1,
    show::Bool=true
)
    # choose source
    if dataset == :raw
        data_full = session.raw
    elseif dataset == :preprocessed
        session.preprocessed === nothing && error("session.preprocessed is nothing; dataset=:preprocessed is invalid")
        data_full = session.preprocessed
    else
        error("dataset must be :raw or :preprocessed")
    end

    # clamp sample range
    sample_end = min(sample_end, size(data_full, 2))
    sample_start = max(sample_start, 1)
    sample_start <= sample_end || error("sample_start must be <= sample_end")
    sample_range = sample_start:sample_end

    data = data_full[:, sample_range]

    # standardize per channel
    if standardize
        μ = mean(data, dims=2)
        σ = std(data, dims=2)
        σ[σ .== 0] .= 1
        data = (data .- μ) ./ σ
        data = clamp.(data, -3, 3)
    end

    # log scale transform
    if log_scale
        data = log10.(abs.(data) .+ eps(Float32))
    end

    # smoothing
    if window_size > 1
        for ch in axes(data, 1)
            if smooth_method == :moving_average
                v = data[ch, :]
                data[ch, :] = [mean(v[max(1, i - window_size + 1):i]) for i in 1:length(v)]
            elseif smooth_method == :gaussian
                σ = window_size / 2
                data[ch, :] = gaussian_smooth(data[ch, :], σ)
            else
                error("smooth_method must be :moving_average or :gaussian")
            end
        end
    end

    # time axis from sampling rate
    if time_units == :ms
        dt = 1000 / session.sampling_rate
        time_axis = (sample_range .- 1) .* dt
        xlabel = "Time (ms)"
    elseif time_units == :s
        dt = 1 / session.sampling_rate
        time_axis = (sample_range .- 1) .* dt
        xlabel = "Time (s)"
    else
        error("time_units must be :ms or :s")
    end

    # plot
    fig = Figure(size = (1500, 700))
    ax = Axis(fig[1, 1],
        title = "Raw Resting State EEG ($name)",
        titlesize = 30,
        titlegap = 21,
        xlabel = xlabel,
        xlabelsize = 21,
        ylabel = "Amplitude",
        ylabelsize = 21
    )

    # good/bad coloring (handle nothing)
    goodset = session.good_channels === nothing ? nothing : Set(session.good_channels)

    for channel in axes(data, 1)
        isgood = goodset === nothing ? true : (channel in goodset)
        color = isgood ? (good_color, alpha) : (bad_color, alpha)
        label = isgood ? "Channel $channel (good)" : "Channel $channel (bad)"
        lines!(ax, time_axis, data[channel, :], color=color, label=label, linewidth=linewidth)
    end

    axislegend(ax)
    if show
        display(fig)
    end
    return fig
end

"""
    plot_spectrogram(times, freqs, power; kwargs...) -> Figure

Plot a time–frequency spectrogram per-channel or aggregated across channels.
Supports dB/log scaling, optional per-frequency normalization, and robust color ranges.

Required shape:
- power: (channels, freqs, times)

Keywords:
- mode::Symbol = :per_channel            # :per_channel | :aggregate
- scale::Symbol = :db                    # :db | :log10 | :none
- normalize_per_freq::Bool = true        # z-score each frequency bin over time
- prange::Tuple = (0.05, 0.95)           # robust color scaling percentiles
- scale_mode::Symbol = :per_channel      # :per_channel | :global
- colormap = :viridis; interpolate=false; show=true

Returns:
- Figure (per-channel grid or a single aggregate figure)

Notes:
- Power should be nonnegative before log/dB transforms.
- In aggregate mode, the mean over channels is taken after scaling/normalization.
"""
function plot_spectrogram(times, freqs, power;
    mode=:per_channel,           # :per_channel or :aggregate
    normalize_per_freq=true,
    prange=(0.05, 0.95),
    scale_mode=:per_channel,
    colormap=:viridis,
    interpolate=false,
    scale=:db,              # :db, :log10, :none
    show::Bool=true
)
    n_channels = size(power, 1)
    fig = Figure(size=(1100, 220 * n_channels))

    # precompute scaled power + optional per-frequency normalization
    z_all = Array{Float64}(undef, size(power))
    for ch in 1:n_channels
        Z = @view z_all[ch, :, :]
        if scale == :db
            Z .= 10 .* log10.(power[ch, :, :] .+ 1e-12)
        elseif scale == :log10
            Z .= log10.(power[ch, :, :] .+ 1e-12)
        elseif scale == :none
            Z .= power[ch, :, :]
        else
            error("scale must be :db, :log10 or :none")
        end

        if normalize_per_freq
            for f in axes(Z, 1)
                μ = mean(@view Z[f, :])
                σ = std(@view Z[f, :]) + 1e-6
                Z[f, :] .= (Z[f, :] .- μ) ./ σ
            end
        end
    end

    # Aggregate mode: mean across channels
    if mode == :aggregate
        fig = Figure(size=(1100, 400))
        ax = Axis(fig[1, 1], title="Mean across channels", xlabel="Time (s)", ylabel="Freq (Hz)",
                  titlesize=30, xlabelsize=21, ylabelsize=21)
        z_mean = dropdims(mean(z_all, dims=1); dims=1)  # (freqs, times)
        colorrange = scale_mode == :per_channel ? quantile(vec(z_mean), prange) : quantile(vec(z_all), prange)
        hm = heatmap!(ax, times, freqs, z_mean; colormap=colormap, colorrange=colorrange, interpolate=interpolate)
        Colorbar(fig[1, 2], hm, label="Power $(scale == :db ? "(dB)" : scale == :log10 ? "(log10)" : "") $(normalize_per_freq ? "norm" : "")")
        if show
            display(fig)
        end
        return fig
    end

    # Global color range if needed
    global_crange = scale_mode == :global ? quantile(vec(z_all), prange) : nothing

    # Plot per channel
    for ch in 1:n_channels
        ax = Axis(fig[ch, 1], title="Channel $ch", xlabel="Time (s)", ylabel="Freq (Hz)",
                  titlesize=30, xlabelsize=21, ylabelsize=21)
        z = @view z_all[ch, :, :]
        colorrange = scale_mode == :per_channel ? quantile(vec(z), prange) : global_crange
        hm = heatmap!(ax, times, freqs, z; colormap=colormap, colorrange=colorrange, interpolate=interpolate)
        Colorbar(fig[ch, 2], hm, label="Power $(scale == :db ? "(dB)" : scale == :log10 ? "(log10)" : "") $(normalize_per_freq ? "norm" : "")")
    end

    if show
        display(fig)
    end
    return fig
end

"""
    plot_bands(session; kwargs...) -> Union{Figure, Vector{Figure}}

Heatmap of band features over time, per channel or aggregated across channels.
Supports optional log scaling and per-band normalization across epochs.

Required shape:
- session.data: (channels, bands, epochs)

Keywords:
- mode::Symbol = :per_channel            # :per_channel | :aggregate
- band_labels::Union{Nothing,Vector} = nothing
- overlap::Real = 0.9                    # epoch overlap used for time axis spacing (1-overlap)
- normalize::Symbol = :none              # :none | :per_band
- method::Symbol = :zscore               # :zscore | :minmax (used if normalize=:per_band)
- log_scale::Bool = false                # log10 on values before normalization
- colormap=:viridis; interpolate=false
- prange=(0.05, 0.95); scale_mode=:per_channel
- show::Bool = true

Returns:
- Vector{Figure} for :per_channel mode, or a single Figure for :aggregate

Notes:
- If `band_labels` is not provided, generic labels are generated.
- Color range can be per plot or global via `scale_mode`.
"""
function plot_bands(session;
    mode=:per_channel,
    interpolate=false,
    band_labels=nothing,
    overlap=0.9,
    normalize=:none,        # :none or :per_band
    method=:zscore,         # :zscore or :minmax (used if normalize=:per_band)
    colormap=:viridis,
    prange=(0.05, 0.95),
    scale_mode=:per_channel,  # color scaling
    log_scale=false,
    show::Bool=true
)
    epoch_dt = 1 - overlap
    data = session.data
    n_chans, n_bands, n_epochs = size(data)  # Updated: now (channels, bands, epochs)

    # optional log scaling
    plot_data = log_scale ? log10.(abs.(data) .+ 1e-12) : data

    # Optional per-band normalization (per channel across epochs)
    if normalize == :per_band
        plot_data = copy(plot_data)
        @inbounds for ch in 1:n_chans, b in 1:n_bands  # Updated: swapped loop order
            v = @view plot_data[ch, b, :]  # Updated: now [ch, b, :]
            if method == :zscore
                μ = mean(v); σ = std(v); σ = σ == 0 ? one(eltype(v)) : σ
                v .= (v .- μ) ./ σ
            elseif method == :minmax
                mn = minimum(v); mx = maximum(v); denom = mx - mn; denom = denom == 0 ? one(eltype(v)) : denom
                v .= (v .- mn) ./ denom
            else
                error("Unsupported method: $(method). Use :zscore or :minmax.")
            end
        end
    elseif normalize != :none
        error("Unsupported normalize option: $(normalize). Use :none or :per_band.")
    end

    if isnothing(band_labels)
        band_labels = ["Band $i" for i in 1:n_bands]
    elseif length(band_labels) != n_bands
        error("band_labels length must match second dimension of session.data")  # Updated: second dimension
    end

    time_axis = (0:n_epochs - 1) .* epoch_dt  # seconds

    if mode == :per_channel
        figs = Figure[]
        for ch in 1:n_chans
            fig = Figure(size = (1500, 500))
            ax = Axis(fig[1, 1], title = "Channel $ch", xlabel = "Time (s)", ylabel = "Band",
                      yticks = (1:n_bands, band_labels), titlesize=30, xlabelsize=21, ylabelsize=21)
            z = permutedims(@view plot_data[ch, :, :])  # Updated: now [ch, :, :] -> (epochs, bands)
            # color scaling
            colorrange = scale_mode == :per_channel ? quantile(vec(z), prange) : quantile(vec(plot_data), prange)
            hm = heatmap!(ax, time_axis, 1:n_bands, z; colormap=colormap, interpolate=interpolate, colorrange=colorrange)
            Colorbar(fig[1, 2], hm, label = (normalize == :per_band ? "Normalized " : "") * (log_scale ? "log-" : "") * "Power")
            push!(figs, fig)
            if show
                display(fig)
            end
        end
        return figs

    elseif mode == :aggregate
        mean_over_ch = dropdims(mean(plot_data, dims=1); dims=1)  # Updated: mean over channels (dim 1) -> (bands, epochs)
        fig = Figure(size = (1000, 400))
        ax = Axis(fig[1, 1], title = "Mean across channels", xlabel = "Time (s)", ylabel = "Band",
                  yticks = (1:n_bands, band_labels), titlesize=30, xlabelsize=21, ylabelsize=21)
        z = permutedims(mean_over_ch)  # (epochs, bands)
        colorrange = scale_mode == :per_channel ? quantile(vec(z), prange) : quantile(vec(plot_data), prange)
        hm = heatmap!(ax, time_axis, 1:n_bands, z; colormap=colormap, interpolate=interpolate, colorrange=colorrange)
        Colorbar(fig[1, 2], hm, label = (normalize == :per_band ? "Normalized " : "") * (log_scale ? "log-" : "") * "Power")
        if show
            display(fig)
        end
        return fig

    else
        error("mode must be :per_channel or :aggregate")
    end
end
