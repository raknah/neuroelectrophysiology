using LinearAlgebra
using DSP, FFTW, Statistics

function mydpss(N::Int, NW::Float64, Kmax::Int)
    Ω = π * (NW/N) # time bandwidth converted to radians per sample

    # setting up matrix kernel
    A = zeros(Float64, N, N)
    for m in 1:N
        for n in 1:N
            τ = m - n
            if τ == 0
                A[m, n] = 2*Ω
            else
                A[m, n] = (2*sin(Ω*τ))/τ
            end
        end
    end
    
    # solving for eigenvectors
    eigvals, eigvecs = eigen(Symmetric(A))
    
    # sort eigenvalues and eigenvectors in descending order
    idx = sortperm(eigvals, rev = true)
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    # keep only Kmax orthogonal vectors or tapers
    tapers = eigvecs[:, 1:Kmax]  # Fixed: was eigvecs[:, Kmax]
    lambdas = eigvals[1:Kmax]
    
    # normalize tapers
    for k in 1:Kmax
        tapers[:, k] ./= norm(tapers[:, k])
    end

    return tapers, lambdas
end


function myspectrogram(session; overlap = 0.9)
    channels, samples, epochs = size(session.data)
    fs = session.sampling_rate
    nyquist = fs/2

    # RFFT 
    freqs = rfftfreq(samples, fs)  
    fmask = freqs .<= nyquist

    # power spectrogram
    power = zeros(channels, length(freqs), epochs)  
    for channel in 1:channels
        for epoch in 1:epochs
            signal = session.data[channel, :, epoch]
            spectrum = abs2.(rfft(signal))
            power[channel, :, epoch] .= spectrum[fmask] 
        end
    end

    epoch_duration = (samples * (1-overlap))/fs
    times = (0:epochs-1) .* epoch_duration

    return collect(times), collect(freqs[fmask]), power  

end

function bandpower!(session::Session; NW=3.0, BANDS=nothing)
    if BANDS === nothing
        BANDS = Dict(
            "delta" => (1, 4),
            "theta" => (4, 8),
            "alpha" => (8, 12),
            "beta"  => (12, 25),
            "gamma" => (25, 50)
        )
    end

    channels, samples, epochs = size(session.data)
    fs = session.sampling_rate

    # precompute tapers and frequency bins
    K = Int(2*NW - 1)
    tapers = dpss(samples, NW, K)
    frequencies = rfftfreq(samples, fs)

    # prepare band masks
    band_names = collect(keys(BANDS))
    masks = [(frequencies .>= low) .& (frequencies .<= high) for (low, high) in values(BANDS)]

    # allocate output: (channels, bands, epochs)
    features = zeros(Float64, channels, length(BANDS), epochs)
    spectrum_sum = zeros(Float64, length(frequencies))  # reused buffer

    for epoch in 1:epochs
        for channel in 1:channels
            signal = session.data[channel, :, epoch]
            fill!(spectrum_sum, 0.0)

            # multitaper PSD
            for k in 1:K
                tapered = signal .* tapers[:, k]
                spectrum_sum .+= abs2.(rfft(tapered))
            end
            spectrum_sum ./= K

            # bandpower extraction
            for (idx, mask) in enumerate(masks)
                features[channel, idx, epoch] = sum(@view spectrum_sum[mask])
            end
        end
    end
    
    # Efficient copy - only copy metadata, replace data array
    result_session = Session(
        session.session, session.experiment, session.sampling_rate,
        session.raw, session.preprocessed, features,
        session.raw_dimensions, session.preprocessed_dimensions, 
        ["channels", "bands", "epochs"],  # Set appropriate dimensions
        session.good_channels, session.notes, session.history, session.group
    )
    return result_session, band_names
end

function logistic_scaler(session)  # Removed ! to make non-mutating
    features = session.data  # channels × bands × epochs
    channels, bands, epochs = size(features)
    scaled = similar(features)

    for ch in 1:channels
        for b in 1:bands
            x = @view features[ch, b, :]  # all epochs for this band–channel
            q1 = quantile(x, 0.25)
            q3 = quantile(x, 0.75)
            med = median(x)
            iqr = q3 - q1

            if iqr == 0
                minx, maxx = minimum(x), maximum(x)
                scaled[ch, b, :] .= maxx == minx ? 0.5 : (x .- minx) ./ (maxx - minx)
            else
                λ = (2 * log(3)) / iqr
                scaled[ch, b, :] .= 1 ./ (1 .+ exp.(-λ .* (x .- med)))
            end
        end
    end
    
    # Efficient copy - only copy metadata, replace data array
    result_session = Session(
        session.session, session.experiment, session.sampling_rate,
        session.raw, session.preprocessed, scaled,
        session.raw_dimensions, session.preprocessed_dimensions, session.data_dimensions,
        session.good_channels, session.notes, session.history, session.group
    )
    return result_session
end