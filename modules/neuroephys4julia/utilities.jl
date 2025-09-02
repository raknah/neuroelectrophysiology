using Statistics, Printf

# -- tiny helpers --------------------------------------------------------------

@inline _hasprop(x, sym) = hasproperty(x, sym) && getfield(x, sym) !== nothing

# summarize a 2D matrix as per-channel quick stats
function _summarize_matrix(name::AbstractString, M::AbstractMatrix)
    n_ch, n_samp = size(M)
    println("[$name] shape = ($n_ch, $n_samp)")
    n_nans = count(isnan, M); n_infs = count(isinf, M)
    println("  NaNs: $n_nans | Infs: $n_infs")
    for ch in 1:n_ch
        x = @view M[ch, :]
        @printf("  Ch %-2d  mean=% .4f  std=% .4f  min=% .4f  max=% .4f  med=% .4f\n",
                ch, mean(x), std(x), minimum(x), maximum(x), median(x))
    end
end

# summarize a 3D epochs array (channels × samples × epochs)
function _summarize_epochs(name::AbstractString, X::AbstractArray{<:Real,3}; fs::Real=NaN, overlap::Union{Nothing,Real}=nothing)
    n_ch, n_samp, n_ep = size(X)
    println("[$name] shape = ($n_ch, $n_samp, $n_ep)  (ch × samples/epoch × epochs)")
    if !isnan(fs)
        if overlap === nothing
            println("  fs = $(fs) Hz  | (pass `overlap=` to compute hop/time span)")
        else
            hop_sec = (n_samp * (1 - overlap)) / fs
            total_sec = (n_ep - 1) * hop_sec + (n_samp / fs)
            @printf("  fs = %.3f Hz  | overlap = %.1f%%  | hop = %.3f s  | span ≈ %.2f s\n",
                    fs, overlap*100, hop_sec, total_sec)
        end
    end
    n_nans = count(isnan, X); n_infs = count(isinf, X)
    println("  NaNs: $n_nans | Infs: $n_infs")

    # per-epoch std (variability over time) per channel
    per_epoch_std = [std(@view X[ch, :, ep]) for ch in 1:n_ch, ep in 1:n_ep]
    for ch in 1:n_ch
        x = vec(@view X[ch, :, :])  # flatten across epochs for overall stats
        mu, sd = mean(x), std(x)
        mn, mx, med = minimum(x), maximum(x), median(x)
        μσ_epochs = mean(per_epoch_std[ch, :]), std(per_epoch_std[ch, :])
        @printf("  Ch %-2d  mean=% .4f  std=% .4f  min=% .4f  max=% .4f  med=% .4f  || per-epoch std: %.4f ± %.4f\n",
                ch, mu, sd, mn, mx, med, μσ_epochs...)
    end
end

# -- main user-facing function -------------------------------------------------

"""
    describe_session_overview(session; overlap=nothing, show_pre=true, show_raw=true)

Concise overview of:
- `session.raw` (if present): (all_channels, n_samples)
- `session.preprocessed` (if present): (good_channels, n_samples)
- `session.data`: epochs array (channels × samples/epoch × n_epochs)

Keyword args:
- `overlap`: fraction (e.g., 0.9). If provided, reports hop and total span for epochs.
- `show_pre`, `show_raw`: toggle printing those sections if present.
"""
function overview(session; overlap::Union{Nothing,Real}=nothing, show_pre::Bool=true, show_raw::Bool=true)
    println("=== SESSION OVERVIEW ===\n")
    println("Session ID: $(session.session)")
    println("Sampling Rate: $(session.sampling_rate) Hz")
    println("Raw Shape: $(size(session.raw))")
    println("Preprocessed Shape: $(size(session.preprocessed))")
    println("Epoch Shapes: $(size(session.data))")
    println("Good Channels: $(session.good_channels)")

    fs = getproperty(session, :sampling_rate)

    # RAW
    if show_raw && _hasprop(session, :raw)
        raw = getfield(session, :raw)
        _summarize_matrix("raw", raw)
        println()
    end

    # PREPROCESSED
    if show_pre && _hasprop(session, :preprocessed)
        pre = getfield(session, :preprocessed)
        _summarize_matrix("preprocessed", pre)
        println()
        if _hasprop(session, :good_channels)
            gch = getfield(session, :good_channels)
            println("good_channels = $(collect(gch))  (n=$(length(gch)))\n")
        end
    end

    # EPOCHS (session.data)
    if !_hasprop(session, :data)
        error("session.data not found; expected epochs in (channels × samples × epochs).")
    end
    data = getfield(session, :data)
    _summarize_epochs("epochs", data; fs=fs, overlap=overlap)
    println("=== END OVERVIEW ===")
    return nothing
end
