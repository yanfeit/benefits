import traffic as tr
import json

Ns = [1000, 2000, 4000, 8000,  16000, 32000, 64000]
# Ns = [1000, 2000]
Ms = [10, 20, 50]
eps = 0.01

dual_cons = []
mip_cons = []
opt = []

for N in Ns:
    for M in Ms:
        
        print(f"Parameters: {N} {M}\n")

        para = tr.TrafficPara(N, M, 0.01, 17)
        model_dual = tr.TrafficDual(para)
        model_dual.optimize()
        dual_cons.append([N, M, model_dual.checkAbsConstraint()])

        model_mip  = tr.TrafficMIP(para, maxseconds=1000)
        mip_cons.append([N, M, model_mip.checkAbsConstraint()])

        q  = model_dual.objective_value
        qs = model_mip.model.objective_value

        opt.append([N, M, tr.optimality(q, qs)])


data = {
    "mip_cons" : mip_cons,
    "dual_cons" : dual_cons,
    "opt" : opt
    }

with open('lists.json', 'w') as file:
    # Write the data into the file using json.dump()
    json.dump(data, file)

with open('lists.json', 'r') as file:
    # Load the data from the file using json.load()
    data = json.load(file)

# Access the lists from the loaded data
list1 = data['mip_cons']
list2 = data['dual_cons']
list3 = data['opt']

# Print the lists
print(list1)
print(list2)
print(list3)