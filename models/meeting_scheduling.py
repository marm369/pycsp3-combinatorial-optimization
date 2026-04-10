from pycsp3 import *
from pycsp3 import clear
import os
import json
import time

def load_instance(json_file, instance_id):
    with open(json_file, 'r') as f:
        data = json.load(f)
    for inst in data:
        if inst["id"] == instance_id:
            return inst
    raise ValueError("Instance not found")

def verify_solution(instance, solution):
    for idx, val in enumerate(solution):
        if not isinstance(val, int):
            return False, f"Solution invalide : la valeur à l'indice {idx} n'est pas un entier ({val})"
    
    n_meetings = instance["NumberOfMeetings"]
    if len(solution) != n_meetings:
        return False, f"Taille de la solution incorrecte : {len(solution)} au lieu de {n_meetings}"
    
    agents_meetings = [instance["AgentsMeetings"][str(i)] for i in range(instance["NumberOfAgents"])]
    dist = instance["BetweenMeetingsDistance"]
    
    for agent_id, meetings in enumerate(agents_meetings):
        for i_idx, i in enumerate(meetings):
            for j in meetings[i_idx+1:]:
                d = dist[i][j]
                ecart = abs(solution[i] - solution[j])
                required = d + 1
                if ecart < required:
                    return False, (f"Agent {agent_id}: meetings {i} et {j} "
                                   f"(horaires {solution[i]} et {solution[j]}) "
                                   f"écart={ecart} < {required} (distance={d})")
    return True, "Solution valide"

def solve_instance(instance_id, solver="ace", json_file="instances.json"):
    clear()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "..", "data", "meeting", json_file)
    instance = load_instance(json_path, instance_id)

    print("\n--- Instance chargée ---")
    print("id:", instance.get("id"))
    print("NumberOfMeetings:", instance["NumberOfMeetings"])
    print("NumberOfAgents:", instance["NumberOfAgents"])
    print("DomainSize:", instance["DomainSize"])
    print("------------------------\n")

    nMeetings = instance["NumberOfMeetings"]
    nAgents = instance["NumberOfAgents"]
    domainSize = instance["DomainSize"]
    agents_meetings = [instance["AgentsMeetings"][str(i)] for i in range(nAgents)]
    dist = instance["BetweenMeetingsDistance"]

    x = VarArray(size=nMeetings, dom=range(domainSize))

    satisfy(
        [(x[i] - x[j] >= dist[i][j] + 1) | (x[j] - x[i] >= dist[i][j] + 1)
         for meetings in agents_meetings
         for i in meetings
         for j in meetings
         if i < j]
    )
    
    makespan = Var(dom=range(domainSize))
    satisfy(makespan == Maximum(x))
    minimize(makespan)

    if solve(solver=solver):
        try:
            sol = [value(x[i]) for i in range(nMeetings)]
            if any(v == '*' or v is None for v in sol):
                raise ValueError("Solution incomplète (présence de '*')")
            print(f"Solution trouvée pour l'instance {instance_id}")
            print("Horaires:", sol)
            return instance, sol
        except Exception as e:
            print(f"Instance {instance_id} : pas de solution ({solver} a indiqué une solution invalide : {e})")
            return instance, None
    else:
        print(f"Instance {instance_id} : aucune solution")
        return instance, None

def save_results(results_list, solver_name, total_time):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir) 
    results_dir = os.path.join(root_dir, "results", "meeting")
    os.makedirs(results_dir, exist_ok=True)
    filename = os.path.join(results_dir, f"results_{solver_name}.json")
    output = {
        "solver": solver_name,
        "total_time_seconds": round(total_time, 3),
        "instances": results_list
    }
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Résultats sauvegardés dans {filename}")

if __name__ == "__main__":
    SOLVER = "ace"   
    
    print("\n" + "="*60)
    print(f"Test de toutes les instances 1 à 27 avec le solveur {SOLVER}")
    print("="*60)
    
    start_total = time.time()
    results_list = []
    
    for instance_id in range(1, 28):
        print(f"\n--- Traitement de l'instance {instance_id} ---")
        start_instance = time.time()
        try:
            instance, solution = solve_instance(instance_id, solver=SOLVER)
            elapsed_instance = time.time() - start_instance
            if solution is not None:
                valide, msg = verify_solution(instance, solution)
                print(f"Vérification : {msg}")
                status = "solved"
                verification_msg = msg
            else:
                print("Aucune solution à vérifier.")
                status = "unsolved"
                verification_msg = "No solution found"
            results_list.append({
                "instance_id": instance_id,
                "status": status,
                "solution": solution,
                "verification": verification_msg,
                "time_seconds": round(elapsed_instance, 3)
            })
            print(f"Temps écoulé pour l'instance {instance_id} : {elapsed_instance:.2f} secondes")
        except Exception as e:
            elapsed_instance = time.time() - start_instance
            print(f"Erreur lors du traitement de l'instance {instance_id}: {e}")
            print(f"Temps écoulé avant erreur : {elapsed_instance:.2f} secondes")
            results_list.append({
                "instance_id": instance_id,
                "status": "error",
                "solution": None,
                "verification": str(e),
                "time_seconds": round(elapsed_instance, 3)
            })
    
    total_time = time.time() - start_total
    print("\n" + "="*60)
    print(f"Temps total pour les 27 instances : {total_time:.2f} secondes")
    print("="*60)
    
    save_results(results_list, SOLVER, total_time)