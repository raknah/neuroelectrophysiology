using LinearAlgebra

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

using DSP, FFTW, Statistics

function bandpower(session::Session; NW=3.0, BANDS=nothing)
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

    # allocate output: (bands, channels, epochs)
    features = zeros(Float64, length(BANDS), channels, epochs)
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
                features[idx, channel, epoch] = sum(@view spectrum_sum[mask])
            end
        end
    end

    return features, band_names
end
