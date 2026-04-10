from pycsp3 import *
from pycsp3 import clear

import argparse
import glob
import json
import os
import re
# Pour gérer le timeout via signaux
import signal   
import time


DEFAULT_SOLVER = "ace"
DEFAULT_TIMEOUT = 3600
INTRA_CELL_DISTANCE = 16


class SolverTimeout(Exception):
    pass


def load_instance():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "frequency", "instance.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# Préparation des données : construction de la liste des émetteurs et indexation  
def prepare_instance(data):
    nb_cells = data["nbCells"]
    nb_freqs = data["nbFreqs"]
    nb_trans = data["nbTrans"]
    distance = data["distance"]
    # Liste de tous les couples (cellule, numéro d'émetteur dans cette cellule)
    transmitters = [(c, t) for c in range(nb_cells) for t in range(nb_trans[c])]
    # Dictionnaire pour obtenir l'indice de la variable à partir d'un couple (cellule, émetteur)
    idx = {(c, t): i for i, (c, t) in enumerate(transmitters)}
    return nb_cells, nb_freqs, nb_trans, distance, transmitters, idx

# Vérification qu'une solution (liste de fréquences) respecte toutes les contraintes
def verify_solution(solution, data):
    if solution is None:
        return False, "Aucune solution"

    nb_cells, _, nb_trans, distance, transmitters, idx = prepare_instance(data)

    if len(solution) != len(transmitters):
        return False, f"Taille incorrecte: {len(solution)} au lieu de {len(transmitters)}"

    # Deux émetteurs de la même cellule doivent être distants d'au moins 16
    for c in range(nb_cells):
        for t1 in range(nb_trans[c]):
            for t2 in range(t1 + 1, nb_trans[c]):
                gap = abs(solution[idx[c, t1]] - solution[idx[c, t2]])
                if gap < INTRA_CELL_DISTANCE:
                    return False, f"Conflit intra-cellule ({c},{t1})-({c},{t2})"

    # La distance entre les fréquences doit respecter la matrice
    for c1 in range(nb_cells):
        for c2 in range(c1 + 1, nb_cells):
            if distance[c1][c2] <= 0:
                continue
            for t1 in range(nb_trans[c1]):
                for t2 in range(nb_trans[c2]):
                    gap = abs(solution[idx[c1, t1]] - solution[idx[c2, t2]])
                    if gap < distance[c1][c2]:
                        return False, f"Conflit inter-cellules ({c1},{t1})-({c2},{t2})"

    return True, "Solution valide"


def build_model(data, upper_bound=None):
    clear()

    nb_cells, nb_freqs, nb_trans, distance, transmitters, idx = prepare_instance(data)
    # Domaine des fréquences : si une borne supérieure est donnée, on restreint
    max_freq = upper_bound if upper_bound is not None else nb_freqs
    freq = VarArray(size=len(transmitters), dom=range(1, max_freq + 1))
    objective = Var(dom=range(1, max_freq + 1))
    # Contraintes intra-cellule
    satisfy(
        abs(freq[idx[c, t1]] - freq[idx[c, t2]]) >= INTRA_CELL_DISTANCE
        for c in range(nb_cells)
        for t1 in range(nb_trans[c])
        for t2 in range(t1 + 1, nb_trans[c])
    )

    # Contraintes inter-cellules
    satisfy(
        abs(freq[idx[c1, t1]] - freq[idx[c2, t2]]) >= distance[c1][c2]
        for c1 in range(nb_cells)
        for c2 in range(c1 + 1, nb_cells)
        if distance[c1][c2] > 0
        for t1 in range(nb_trans[c1])
        for t2 in range(nb_trans[c2])
    )

    # Symétrie : fixer la première fréquence à 1
    satisfy(freq[0] == 1)

    if upper_bound is not None:
        satisfy(objective <= upper_bound)

    # L'objectif est la fréquence maximale utilisée (on veut la minimiser)
    satisfy(objective == Maximum(freq))
    minimize(objective)

    return freq, objective

# Gestionnaire de signal pour le timeout
def timeout_handler(signum, frame):
    raise SolverTimeout()

# Recherche du fichier de log le plus récent créé après le début de la résolution
def latest_solver_log(start_time):
    root = os.path.join(os.path.dirname(__file__), "..")
    candidates = []
    for path in glob.glob(os.path.join(root, "solver_*.log")):
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if mtime >= start_time - 1:
            candidates.append((mtime, path))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


# Extraction de la solution à partir du fichier de log du solveur
def parse_solver_log(log_path):
    if not log_path or not os.path.exists(log_path):
        return None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read()

    text = re.sub(r"\x1b\[[0-9;]*m", "", text)

    status = None
    if "s OPTIMUM" in text:
        status = "OPTIMUM"
    elif "s SATISFIABLE" in text:
        status = "SAT"
    elif "s UNSATISFIABLE" in text:
        status = "UNSAT"

    matches = re.findall(r"<values>\s*([^<]+?)\s*</values>", text, flags=re.DOTALL)
    if not matches:
        return {"status": status, "solution": None}

    values_text = matches[-1]
    solution = [int(x) for x in re.findall(r"-?\d+", values_text)]
    return {"status": status, "solution": solution}

# Lancement du solveur avec gestion du timeout
def solve_with_timeout(solver_name, timeout_seconds):
    start = time.time()
    try:
        status = solve(solver=solver_name, options=f"-t={max(1, int(timeout_seconds))}s")
        return status, time.time() - start, latest_solver_log(start)
    except SolverTimeout:
        return None, time.time() - start, latest_solver_log(start)
    except TypeError:
        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(max(1, int(timeout_seconds)))
        start = time.time()
        try:
            status = solve(solver=solver_name)
            return status, time.time() - start, latest_solver_log(start)
        except SolverTimeout:
            return None, time.time() - start, latest_solver_log(start)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)


# Fonction principale : résout le problème d'allocation de fréquences
def solve_frequency_allocation(solver_name, timeout_seconds):
    data = load_instance()
    start = time.time()
    freq, objective = build_model(data)
    solve_status, _, log_path = solve_with_timeout(solver_name, timeout_seconds)
    elapsed = round(time.time() - start, 3)

    parsed = parse_solver_log(log_path)

    # Si le solveur a trouvé une solution complète (status SAT ou OPTIMUM)
    if solve_status in {SAT, OPTIMUM}:
        solution = values(freq)
        valid, message = verify_solution(solution, data)
        return {
            "solver": solver_name,
            "status": "OPTIMUM" if solve_status is OPTIMUM else "SAT",
            "time_seconds": elapsed,
            "timeout_seconds": timeout_seconds,
            "objective": value(objective),
            "solution": solution,
            "valid": valid,
            "verification": message,
        }

    # Si le log contient une solution (cas de timeout avec solution partielle)
    if parsed and parsed["solution"] is not None:
        solution = parsed["solution"]
        valid, message = verify_solution(solution, data)
        return {
            "solver": solver_name,
            "status": parsed["status"] or "TIMEOUT",
            "time_seconds": elapsed,
            "timeout_seconds": timeout_seconds,
            "objective": max(solution) if solution else None,
            "solution": solution,
            "valid": valid,
            "verification": message,
        }

    # Timeout sans solution
    if solve_status is None:
        return {
            "solver": solver_name,
            "status": "TIMEOUT",
            "time_seconds": elapsed,
            "timeout_seconds": timeout_seconds,
            "objective": None,
            "solution": None,
            "valid": False,
            "verification": "Aucune solution recuperee avant le timeout",
        }
    # Sinon, le problème est insatisfiable
    return {
        "solver": solver_name,
        "status": "UNSAT",
        "time_seconds": elapsed,
        "timeout_seconds": timeout_seconds,
        "objective": None,
        "solution": None,
        "valid": False,
        "verification": "Aucune solution trouvee",
    }


def save_results(result, solver_name):
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "frequency")
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, f"results_{solver_name}.json")

    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    return out_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default=DEFAULT_SOLVER, choices=["ace", "choco"])
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    print(f"Lancement de {args.solver} (timeout {args.timeout}s)...")
    result = solve_frequency_allocation(args.solver, args.timeout)
    out_file = save_results(result, args.solver)

    print(f"Statut: {result['status']}")
    print(f"Objectif courant: {result['objective']}")
    print(f"Verification: {result['verification']}")
    print(f"Resultat sauvegarde dans : {out_file}")


if __name__ == "__main__":
    main()
