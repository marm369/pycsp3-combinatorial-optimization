from pycsp3 import *
import json
import os
from datetime import datetime

# 1. Chargement des données
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'frequency', 'instance.json')

with open(data_path) as f:
    data = json.load(f)

nbCells  = data['nbCells']    
nbFreqs  = data['nbFreqs']    
nbTrans  = data['nbTrans']    
distance = data['distance']

# 2. Construction de la liste des émetteurs
transmitters = []

for c in range(nbCells):
    for t in range(nbTrans[c]):
        transmitters.append((c, t))

nbTransmitters = len(transmitters)

# Construction de l'index
idx = {}
i = 0
for c, t in transmitters:
    idx[(c, t)] = i
    i = i + 1

# 3. Variables de décision
freq = VarArray(size=nbTransmitters, dom=range(1, nbFreqs + 1))
nFreqUsed = Var(dom=range(1, nbFreqs + 1))

# 4. Contraintes
satisfy(
    abs(freq[idx[c, t1]] - freq[idx[c, t2]]) >= 16
    for c in range(nbCells)
    for t1 in range(nbTrans[c])
    for t2 in range(t1 + 1, nbTrans[c])
)

satisfy(
    abs(freq[idx[c1, t1]] - freq[idx[c2, t2]]) >= distance[c1][c2]
    for c1 in range(nbCells)
    for c2 in range(c1 + 1, nbCells)
    if distance[c1][c2] > 0
    for t1 in range(nbTrans[c1])
    for t2 in range(nbTrans[c2])
)

satisfy(
    nFreqUsed == NValues(freq)
)

minimize(nFreqUsed)


# 5. Résolution avec timeout 10 min

print("Lancement ACE pour 10 minutes...")
result = solve(solver="ace", options={"timeout": 600})

# 6. Sauvegarde


results_dir = os.path.join(os.path.dirname(__file__), '..', 'results','frequency')
os.makedirs(results_dir, exist_ok=True)

with open(os.path.join(results_dir, 'ace_result.txt'), 'w') as f:
    f.write(f"Statut: {result}\n")
    if result in [OPTIMUM, SAT]:
        f.write(f"Fréquences utilisées: {value(nFreqUsed)}\n")
        f.write(f"Solution: {value(freq)}\n")
    f.write(f"Temps: {600} secondes max\n")

print(f"Résultat: {value(nFreqUsed)} fréquences" if result in [OPTIMUM, SAT] else "Pas de solution")