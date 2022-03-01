# NAME
#************************************************************************
# Intro definitions
using JuMP
using GLPK
using Gurobi
using CSV
using DataFrames
#************************************************************************
clearconsole()

#************************************************************************
# PARAMETERS
pathNS = "C:\\Users\\Nicklas\\OneDrive - Danmarks Tekniske Universitet\\Undevisning\\FetalMaternal_project\\NS.csv"
pathD = "C:\\Users\\Nicklas\\OneDrive - Danmarks Tekniske Universitet\\Undevisning\\FetalMaternal_project\\D.csv"
#df = CSV.read(path)
dfNS = DataFrame(CSV.File(pathNS))
dfD = DataFrame(CSV.File(pathD))

NS = length(dfNS.paths)
D = length(dfD.paths)
#************************************************************************
data_sets = 3
T = sum(dfD.Num) + sum(dfNS.Num)
#************************************************************************
# Model
NAME = Model(with_optimizer(GLPK.Optimizer, tm_lim = 60000, msg_lev = GLPK.OFF))
#NAME = Model(with_optimizer(Gurobi.Optimizer, TimeLimit = 300, MIPGap = 0.0))


@variable(NAME, x[1:D, 1:data_sets] >= 0, Bin)
@variable(NAME, y[1:NS, 1:data_sets] >= 0, Bin)
@variable(NAME, s[1:data_sets])
@variable(NAME, a[1:data_sets])
@variable(NAME, b[1:data_sets])
#************************************************************************
#@objective(NAME, Min, sum(s[d] for d=1:data_sets)) # Min cost
@objective(NAME, Min, sum(s[d] for d = 1:data_sets)) # Min cost
#************************************************************************

@constraint(
    NAME,
    test_con1[i = 1],
    sum(x[d, i]*dfD.Num[d] for d = 1:D) + sum(y[n, i]*dfNS.Num[n] for n = 1:NS) >= 0.50 * T
) # Cap limit

@constraint(
    NAME,
    test_con2[i = 2],
    sum(x[d, i]*dfD.Num[d] for d = 1:D) + sum(y[n, i]*dfNS.Num[n] for n = 1:NS) >= 0.20 * T
) # Cap limit

@constraint(
    NAME,
    test_con3[i = 3],
    sum(x[d, i]*dfD.Num[d] for d = 1:D) + sum(y[n, i]*dfNS.Num[n] for n = 1:NS) >= 0.20 * T
) # Cap limit
@constraint(NAME, oneset[d = 1:D], sum(x[d, i] for i = 1:data_sets) == 1)
@constraint(NAME, onesetn[n = 1:NS], sum(y[n, i] for i = 1:data_sets) == 1)

# @constraint(
#     NAME,
#     calc[i = 1:data_sets],
#     s[i] ==
#     sum(x[d, i] * dfD.Num[d] for d = 1:D) -
#     sum(y[n, i] * dfD.Num[n] for n = 1:NS)
# )
@constraint(
    NAME,
    abs0[i = 1:data_sets],
    sum(x[d, i] * dfD.Num[d] for d = 1:D) -
    sum(y[n, i] * dfNS.Num[n] for n = 1:NS) <= s[i]
)
@constraint(
    NAME,
    abs1[i = 1:data_sets],
    sum(y[n, i] * dfNS.Num[n] for n = 1:NS) -
    sum(x[d, i] * dfD.Num[d] for d = 1:D) <= s[i]
)
#************************************************************************
#************************************************************************
# solve
solution = optimize!(NAME)
#************************************************************************
#************************************************************************
# Print model
print(NAME)
#************************************************************************
#************************************************************************
# Report results
if termination_status(NAME) == MOI.OPTIMAL
    println("RESULTS:")
    println("Maximum profit: $(JuMP.objective_value(NAME))")
    for i = 1:data_sets
        for d = 1:D
            println("  x$d$i = $(JuMP.value(x[d,i]))")
        end
        for n = 1:NS
            println("  y$n$i = $(JuMP.value(y[n,i]))")
        end
    end
    println("$(JuMP.value.(s))")
    println("$(JuMP.value.(x))")
else
    println("  No solution")
end
#************************************************************************
B = [1;2;3];
dfDonor=DataFrame(X=collect(eachslice(JuMP.value.(x), dims=2)))
dfNString=DataFrame(X=collect(eachslice(JuMP.value.(y), dims=2)))


dfD_out = hcat(dfD, JuMP.value.(x)*B)
CSV.write("C:\\Users\\Nicklas\\OneDrive - Danmarks Tekniske Universitet\\Undevisning\\FetalMaternal_project\\Solution\\D_out.csv", dfD_out)

dfNS_out = hcat(dfNS, JuMP.value.(y)*B)
CSV.write("C:\\Users\\Nicklas\\OneDrive - Danmarks Tekniske Universitet\\Undevisning\\FetalMaternal_project\\Solution\\NS_out.csv", dfNS_out)
#x = DataFrame(train = JuMP.value.(x)[1,:])
