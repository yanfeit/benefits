import traffic as tr
import json
import argparse

parser = argparse.ArgumentParser(description="Knapsack Solver")
parser.add_argument('-N', type=int, help="number of users")
parser.add_argument('-M', type=int, help="number of coupons")
parser.add_argument('-eps', type=float, help="relaxation number")
parser.add_argument('-seed', type = int, help="random number seed")
parser.add_argument('-time', type=int, help="max seconds for MIP solver")
# parser.add_argument('-f', type=str, help = "output file name")

args=parser.parse_args()

N = args.N
M = args.M
eps = args.eps
seed = args.seed
tlimit = args.time
# filename = args.f

fp = open(f"{N}_{M}_{seed}.txt", 'w')

fp.write(f"Parameters: {N} {M} {eps} {seed} {tlimit}\n")

para = tr.TrafficPara(N, M, eps, seed)
model_dual = tr.TrafficDual(para)
model_dual.optimize()

fp.write(f"Dual: {N} {M} {model_dual.objective_value} {model_dual.checkAbsConstraint()}\n")

model_mip  = tr.TrafficMIP(para, maxseconds=tlimit)
fp.write(f"MIP: {N} {M} {model_mip.model.objective_value} {model_mip.checkAbsConstraint()}\n")

q  = model_dual.objective_value
qs = model_mip.model.objective_value
fp.write(f"OPT: {N} {M} {tr.optimality(q, qs)}\n")

fp.close()