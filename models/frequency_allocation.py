from pycsp3 import *
import json
import os
import subprocess
import sys
import time

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

idx = {}
i = 0
for c, t in transmitters:
    idx[(c, t)] = i
    i += 1

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

satisfy(nFreqUsed == NValues(freq))
minimize(nFreqUsed)

# 5. Génération du fichier XCSP3 (sans lancer le solveur)
compile(filename="frequency_model")
print("Modèle compilé : frequency_model.xcsp3")

# 6. Lancement ACE via subprocess avec timeout strict
TIMEOUT = 600
ACE_JAR = os.path.expanduser("~/.pycsp3/solvers/ace/ACE.jar")  # chemin ACE

print(f"Lancement ACE pour {TIMEOUT//60} minutes...")
start = time.time()

try:
    proc = subprocess.run(
        ["java", "-jar", ACE_JAR, "frequency_model.xcsp3", f"-t={TIMEOUT}s"],
        capture_output=True,
        text=True,
        timeout=TIMEOUT + 10  # +10s de marge pour arrêt propre
    )
    elapsed = time.time() - start
    output = proc.stdout
    print(output)

except subprocess.TimeoutExpired:
    proc.kill()
    elapsed = time.time() - start
    output = ""
    print(f"⏰ Timeout forcé après {elapsed:.1f}s")

# 7. Parsing du résultat ACE
status = "TIMEOUT"
nb_freq = None
solution = None

for line in output.splitlines():
    if "o " in line:                        # ligne objectif : "o 12"
        try:
            nb_freq = int(line.strip().split()[1])
        except:
            pass
    if line.startswith("s OPTIMUM"):
        status = "OPTIMUM"
    elif line.startswith("s SATISFIABLE"):
        status = "SAT"
    elif line.startswith("s UNSATISFIABLE"):
        status = "UNSAT"
    elif line.startswith("v "):             # ligne valeurs
        solution = line.strip()

# 8. Sauvegarde
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'frequency')
os.makedirs(results_dir, exist_ok=True)

with open(os.path.join(results_dir, 'ace_result.txt'), 'w') as f:
    f.write(f"Statut: {status}\n")
    f.write(f"Temps écoulé: {elapsed:.1f}s (max {TIMEOUT}s)\n")
    if nb_freq is not None:
        f.write(f"Fréquences utilisées: {nb_freq}\n")
    if solution:
        f.write(f"Solution: {solution}\n")

if nb_freq is not None:
    print(f"Résultat: {nb_freq} fréquences ({status})")
else:
    print(f"Pas de solution optimale — statut: {status}")