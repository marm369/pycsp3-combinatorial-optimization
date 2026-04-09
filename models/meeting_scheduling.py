from pycsp3 import *

x = VarArray(size=5, dom=range(5))

satisfy(
    AllDifferent(x)
)

if solve(solver="ortools") is SAT:
    print("Z3 a trouvé une solution :", values(x))
else:
    print("Z3 n'a pas trouvé de solution")