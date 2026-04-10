from ortools.sat.python import cp_model
import os
import json
import time

SOLVER = "ortools"


def load_instance(json_file, instance_id):
    with open(json_file, 'r') as f:
        data = json.load(f)
    for inst in data:
        if inst["id"] == instance_id:
            return inst
    raise ValueError(f"Instance {instance_id} non trouvée dans {json_file}")


def save_results(results_list, solver_name, total_time):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir) 
    results_dir = os.path.join(root_dir, "results", "meeting")
    os.makedirs(results_dir, exist_ok=True)
    filename = os.path.join(results_dir, f"results_{solver_name}_opt.json")
    output = {
        "solver": solver_name,
        "total_time_seconds": round(total_time, 3),
        "instances": results_list
    }
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Résultats sauvegardés dans {filename}")



def solve_instance(instance_id, json_file="instances.json"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "..", "data", "meeting", json_file)

    instance = load_instance(json_path, instance_id)

    print(f"\n--- Instance {instance_id} chargée ---")
    print(f"  Meetings : {instance['NumberOfMeetings']}")
    print(f"  Agents   : {instance['NumberOfAgents']}")
    print(f"  Domain   : {instance['DomainSize']}")
    print("-------------------------------\n")

    nMeetings  = instance["NumberOfMeetings"]
    nAgents    = instance["NumberOfAgents"]
    domainSize = instance["DomainSize"]

    agents_meetings = [instance["AgentsMeetings"][str(i)] for i in range(nAgents)]
    dist = instance["BetweenMeetingsDistance"]

    # ---------------- MODEL ----------------
    model = cp_model.CpModel()

    x = [
        model.NewIntVar(0, domainSize - 1, f"x{i}")
        for i in range(nMeetings)
    ]

    # ---------------- CONSTRAINTS ----------------
    for meetings in agents_meetings:
        for a in range(len(meetings)):
            for b_idx in range(a + 1, len(meetings)):
                i = meetings[a]
                j = meetings[b_idx]
                d = dist[i][j]

                b = model.NewBoolVar(f"b_{i}_{j}")
                model.Add(x[i] - x[j] >= d + 1).OnlyEnforceIf(b)
                model.Add(x[j] - x[i] >= d + 1).OnlyEnforceIf(b.Not())

    # ---------------- OBJECTIF (MAYSPAN) ----------------
    #makespan = model.NewIntVar(0, domainSize - 1, "makespan")
    #model.AddMaxEquality(makespan, x)
    #model.Minimize(makespan)

    # ---------------- OBJECTIF (SOMME DES CRÉNEAUX) ----------------
    model.Minimize(sum(x))

    # ---------------- SOLVE ----------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0

    start = time.time()
    status = solver.Solve(model)
    elapsed = time.time() - start

    # ---------------- RESULT ----------------
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        sol = [solver.Value(x[i]) for i in range(nMeetings)]

        print(f"✔ Solution trouvée pour l'instance {instance_id}")
        print(f"  Horaires : {sol}")
        print(f"  Makespan : {solver.Value(makespan)}")
        print(f"  Temps    : {elapsed:.3f}s")

        return instance, sol, elapsed
    else:
        print(f"✘ Instance {instance_id} : aucune solution trouvée (status={solver.StatusName(status)})")
        return instance, None, elapsed

# run all instances and save results
if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  OR-Tools - Meeting Scheduling (instances 1 à 27)")
    print("=" * 60)

    start_total = time.time()
    results_list = []

    for instance_id in range(1, 28):

        print(f"\n{'='*60}")
        print(f"  Instance {instance_id}")
        print(f"{'='*60}")
        start_inst = time.time()

        try:
            instance, solution, elapsed = solve_instance(instance_id)

            if solution is not None:
                status = "solved"
                verification = "OK"
            else:
                status = "unsolved"
                verification = "No solution found"

            results_list.append({
                "instance_id":  instance_id,
                "status":       status,
                "solution":     solution,
                "time_seconds": round(elapsed, 3),
                "verification": verification
            })

        except Exception as e:
            elapsed = time.time() - start_inst
            print(f"  ERREUR : {e}")
            results_list.append({
                "instance_id":  instance_id,
                "status":       "error",
                "solution":     None,
                "time_seconds": round(elapsed, 3),
                "verification": str(e)
            })

    total_time = time.time() - start_total

    print(f"\n  Temps total : {total_time:.2f}s")

    save_results(results_list, SOLVER, total_time)