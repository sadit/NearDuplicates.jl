using NearDuplicates
using Test, Downloads

@testset "NearDuplicates.jl" begin
    # Write your tests here.
    fname = "emo50k.json.gz"
    !isfile(fname) && Downloads.download("https://github.com/sadit/TextClassificationTutorial/raw/main/data/$fname", fname)
    NearDuplicates.command_main(["--print", "--quantiles=0.01,0.03", "--field=text", fname])
end
