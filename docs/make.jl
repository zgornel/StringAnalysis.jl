using Pkg
Pkg.add("Documenter")
using Documenter, StringAnalysis, Languages

# Make src directory available
push!(LOAD_PATH,"../src/")

# Make documentation
makedocs(
    modules = [StringAnalysis],
    format = :html,
    sitename = "  ",
    authors = "Corneliu Cofaru, 0x0α Research",
    clean = true,
    debug = true,
    pages = [
        "Introduction" => "index.md",
        "Usage examples" => "examples.md",
        "API Reference" => "api.md",
    ]
)

# Deploy documentation
deploydocs(
    repo = "github.com/zgornel/StringAnalysis.jl.git",
    target = "build",
    deps = nothing,
    make = nothing
)
