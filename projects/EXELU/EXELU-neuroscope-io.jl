# Goal: load one Neuroscope-style session

using EzXML

using Statistics, CairoMakie

using GLMakie

# 1. DATA LOCATION

DIRECTORY = "/Users/fomo/Local Data/EXELU"
SESSION = "2025-09-18_09-59-40"

xml_path = joinpath(DIRECTORY, SESSION, SESSION * ".xml")
lfp_path = joinpath(DIRECTORY, SESSION, SESSION * ".lfp")

isfile(xml_path) || error("Missing XML: $xml_path")
isfile(lfp_path) || error("Missing LFP: $lfp_path") 

@info "Using files:" xml_path, lfp_path


# 2. PARSE XML FILE

xml_doc = EzXML.readxml(xml_path)
xml_root = EzXML.root(xml_doc)

# extract metadata

doc  = EzXML.readxml(xml_path)
root = EzXML.root(doc)

function find_text(root::EzXML.Node, tag::AbstractString)
    if root.name == tag
        return strip(root.content)
    end
    for child in EzXML.eachelement(root)
        value = find_text(child, tag)
        value === nothing || return value
    end
    return nothing
end

fs = parse(Float64, find_text(root, "lfpSamplingRate"))
n_channels = parse(Int, find_text(root, "nChannels"))
nBits = parse(Int, find_text(root, "nBits"))

@info "XML" fs n_channels nBits


# 3. LOAD LFP

using Mmap, Statistics, Plots

T = Int16 # raw data type inferred from nBits

bytes = filesize(lfp_path)
total_samples = bytes ÷ sizeof(Int16)
n_samples = total_samples ÷ n_channels

# load LFP data memory-mapped

X_raw = Mmap.mmap(lfp_path, Matrix{T},(n_channels, n_samples)) 

@info "LFP" size=size(X_raw) duration_s=(n_samples / fs)

# 4. 

using MAT

chanmap_path = joinpath(DIRECTORY, SESSION, SESSION * ".chanmap.mat")
isfile(chanmap_path) || error("Missing chanmap: $chanmap_path") 

cm = matread(chanmap_path)

channel_idx = vec(Int.(cm["chanMap"]))
connected = vec(Bool.(cm["connected"]))
ycoords = vec(Float64.(cm["ycoords"]))

@info "Channel Map" n_connected=sum(connected) n_channels=length(connected)

# 5. PLOT EXAMPLE

duration = 5.0 # seconds

start_sample = 10000
n_plot_samples = Int(round(duration * fs))
end_sample = start_sample + n_plot_samples - 1

X = @view X_raw[connected, start_sample:end_sample] 

t = (0:n_plot_samples-1) ./ fs



GLMakie.activate!()

fig = Figure(size = (3000, 1000))

ax = Axis(fig[1, 1];
    title = "LFP ($SESSION) Example",
    xlabel = "Time (s)",
    ylabel = "Channels (sorted by depth)",
)

offset = 0.0
for i in 1:size(X,1)
    y = Float64.(X[i, 1:n_plot_samples])
    y = (y .- mean(y)) ./ std(y)
    lines!(ax, t, y .+ offset)
    offset += 10.0
end

fig

save(expanduser("~/Desktop/EXELU_LFP_example.png"), fig)


# heatmap
Y = Float64.(X)
Yz = (Y .- mean(Y, dims=2)) ./ (std(Y, dims=2) .+ eps())

fig2 = Figure(size=(n_channels*10,1000))
ax2 = Axis(fig2[1,1], xlabel="time (s)", ylabel="connected channel index", title="Heatmap (z per channel)")
GLMakie.heatmap!(ax2, t, 1:size(Yz,1), Yz, colormap=:curl)
fig2

save(expanduser("~/Desktop/EXELU_LFP_heatmap.png"), fig2)


# spectrogram
using DSP

X_win = @view X_raw[connected, :]

ch = 1
x = Float64.(X_win[ch, :])
x .-= mean(x)

# windowing 
win_s  = 2.0      # theta: 1–2 s 
step_s = 0.2
nwin   = Int(round(win_s * fs))
nover  = nwin - Int(round(step_s * fs))

# DPSS multitaper params
nw = 4         
ntapers = 5

S = DSP.mt_spectrogram(x, nwin, nover; fs=fs, nw=nw, ntapers=ntapers)  

P  = log10.(S.power .+ eps())   # freq × time
tt = S.time                     # seconds (relative to x start)
ff = S.freq

fig3 = Figure(size=(2100, 700))
ax3 = Axis(fig3[1,1], xlabel="time (s)", ylabel="Hz", title="DPSS multitaper spectrogram (theta optimised)")
GLMakie.heatmap!(ax3, tt, ff, P)
GLMakie.ylims!(ax3, 0, 100)
fig3

save(expanduser("~/Desktop/EXELU_LFP_spectrogram_ch$(ch).png"), fig3)