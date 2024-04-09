module NearDuplicates

using TextSearch, SimilaritySearch, BagOfWords
using JSON, CodecZstd, CodecBzip2, CodecZlib, DataFrames, Statistics, Comonicon

function neardup_bow(textconfig::TextConfig, corpus, qs; samplesize::Int)
    corpus_tokens = tokenize_corpus(textconfig, corpus)
    voc = Vocabulary(textconfig, corpus_tokens)
    X = VectorDatabase([bagofwords(voc, tok.tokens) |> keys |> collect |> sort!
                            for tok in corpus_tokens])
    dist = JaccardDistance()
    epsilonlist = quantile(distsample(dist, rand(X, samplesize); samplesize=samplesize), qs)
    epsilon = epsilonlist[findfirst(x -> x > 0f0, epsilonlist)]
    neardup(dist, X, epsilon)
end

function read_corpus(filename)
    codec = if endswith(filename, ".gz")
        GzipDecompressorStream
    elseif endswith(filename, ".bz2")
        Bz2DecompressorStream
    elseif endswith(".zst")
        ZstdDecompressorStream
    else
        nothing
    end

    if codec === nothing
        open(filename) do f
            read_json_lines(f, 10^8)
        end
    else
        open(codec, filename) do f
            read_json_lines(f, 10^8)
        end
    end
end

function neardup_bow(filename::String; outname::String, field::String, textconfig::TextConfig, qs::Vector, samplesize::Int, minsize=samplesize ÷ 10, print_stdout=false)
    @info "$filename -> $outname"
    Corpus = read_corpus(filename)
    text = get.(Corpus, field, nothing)
    if length(text) > minsize
        D = neardup_bow(textconfig, text, qs; samplesize)
        C = Corpus[sort!(unique(D.nn))]
    else
        println(stderr, "$filename is too short to filter neardups; copying without filtering")
        C = Corpus
    end

    if print_stdout
        for r in C
            println(stdout, json(r))
        end
    else
        open(GzipCompressorStream, outname, "w") do f
            for r in C
                println(f, json(r))
            end
        end
    end
end

"""
removes near duplicates of each input file (one json per line files) and stores the filtered files in the given output directory.
It will preserve only documents being at most an epsilon distant, where epsilon is a small distance value estimated from input parameters and the datasets.
Documents are preprocessed and distance is measured with `1 - jaccardcoefficient` of their bag of word representations (using multiple tokenizers)

# Args

- `files`: input files

# Options

- `-q, --quantiles=<quantil-values>`: a list of quantiles to define what is near based on distances-values, using a small sample
- `-f, --field=<field-key>`: field name to use as input text in each json 
- `-o, --output=<directory name>`: output directory to store filtered data
- `-s, --samplesize=<int>`: size of the samplesize to estimate epsilon

# Flags

- `--store`: save files into an output directory print instead of printing to stdout

"""
@main function main(files...; quantiles::String="0.001,0.003,0.01,0.03,0.1", field::String="text", outdir::String="output", store::Bool=false, samplesize::Int=1000)
    textconfig = TextConfig(del_punc=true, group_usr=true)
    qs = [parse(Float64, n) for n in split(quantiles, ',')]
    
    store && mkpath(outdir)
    for filename in files
        if store
            outname = let outname = basename(filename)
                outname = splitext(outname) |> first
                joinpath(outdir, outname * ".gz")
            end
            !isfile(outname) & neardup_bow(filename; outname, textconfig, field, qs, samplesize, print_stdout=false)
        else
            neardup_bow(filename; outname="", textconfig, field, qs, samplesize, print_stdout=true)
        end
    end
end

end
