from pycsp3 import *
from pycsp3 import clear

import argparse
import json
import os
import time
from collections import defaultdict


SHIFT_TO_INDEX = {"early": 0, "late": 1, "night": 2}


def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dataset_dir(dataset):
    base = os.path.join(project_root(), "data", "healthcare")
    if dataset == "competition":
        return os.path.join(base, "ihtc2024_competition_instances")
    if dataset == "test":
        return os.path.join(base, "ihtc2024_test_dataset")
    raise ValueError(f"Dataset inconnu: {dataset}")


def solution_dir(dataset):
    return os.path.join(project_root(), "results", "healthcare", "pycsp3_solutions", dataset)


def summary_path(dataset):
    return os.path.join(project_root(), "results", "healthcare", f"pycsp3_summary_{dataset}.json")


def instance_filename(dataset, instance_id):
    if dataset == "competition":
        return f"i{instance_id:02d}.json"
    return f"test{instance_id:02d}.json"


def solution_filename(dataset, instance_id):
    if dataset == "competition":
        return f"sol_i{instance_id:02d}.json"
    return f"sol_test{instance_id:02d}.json"


def load_instance(dataset, instance_id):
    path = os.path.join(dataset_dir(dataset), instance_filename(dataset, instance_id))
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def list_instance_ids(dataset):
    ids = []
    for name in sorted(os.listdir(dataset_dir(dataset))):
        if dataset == "competition" and name.startswith("i") and name.endswith(".json"):
            ids.append(int(name[1:3]))
        if dataset == "test" and name.startswith("test") and name.endswith(".json"):
            ids.append(int(name[4:6]))
    return ids


def shift_offset(relative_day, shift_name):
    return relative_day * 3 + SHIFT_TO_INDEX[shift_name]


def value_from_profile(profile, relative_day, shift_name):
    idx = shift_offset(relative_day, shift_name)
    return profile[idx] if 0 <= idx < len(profile) else 0


def build_fixed_occupancy(instance):
    n_days = instance["days"]
    room_ids = [room["id"] for room in instance["rooms"]]
    fixed_room_occupancy = {room_id: [0] * n_days for room_id in room_ids}
    fixed_room_age_groups = {room_id: [set() for _ in range(n_days)] for room_id in room_ids}

    for occupant in instance.get("occupants", []):
        room_id = occupant["room_id"]
        for day in range(min(n_days, occupant["length_of_stay"])):
            fixed_room_occupancy[room_id][day] += 1
            fixed_room_age_groups[room_id][day].add(occupant["age_group"])

    return fixed_room_occupancy, fixed_room_age_groups


def build_patient_model(instance):
    clear()

    patients = instance["patients"]
    rooms = instance["rooms"]
    theaters = instance["operating_theaters"]
    surgeons = instance["surgeons"]
    weights = instance["weights"]
    n_days = instance["days"]

    n_patients = len(patients)
    n_rooms = len(rooms)
    n_theaters = len(theaters)
    n_surgeons = len(surgeons)

    room_id_to_idx = {room["id"]: idx for idx, room in enumerate(rooms)}
    theater_id_to_idx = {theater["id"]: idx for idx, theater in enumerate(theaters)}
    surgeon_id_to_idx = {surgeon["id"]: idx for idx, surgeon in enumerate(surgeons)}

    fixed_room_occupancy, fixed_room_age_groups = build_fixed_occupancy(instance)

    unscheduled_day = n_days

    start = VarArray(
        size=n_patients,
        dom=lambda p: list(
            range(
                patients[p]["surgery_release_day"],
                min(
                    patients[p].get("surgery_due_day", n_days - patients[p]["length_of_stay"]),
                    n_days - patients[p]["length_of_stay"],
                )
                + 1,
            )
        )
        + ([unscheduled_day] if not patients[p]["mandatory"] else []),
    )

    room = VarArray(size=n_patients, dom=range(n_rooms))
    theater = VarArray(size=n_patients, dom=range(n_theaters))
    scheduled = VarArray(size=n_patients, dom={0, 1})
    present = VarArray(size=[n_patients, n_days], dom={0, 1})
    operated_day = VarArray(size=[n_patients, n_days], dom={0, 1})
    in_room = VarArray(size=[n_patients, n_rooms, n_days], dom={0, 1})
    in_theater = VarArray(size=[n_patients, n_theaters, n_days], dom={0, 1})
    open_theater = VarArray(size=[n_theaters, n_days], dom={0, 1})

    optional_patients = [p for p in range(n_patients) if not patients[p]["mandatory"]]
    mandatory_patients = [p for p in range(n_patients) if patients[p]["mandatory"]]

    satisfy(
        [scheduled[p] == (start[p] != unscheduled_day) for p in range(n_patients)],

        [scheduled[p] == 1 for p in mandatory_patients],

        [
            present[p][d] == ((scheduled[p] == 1) & (start[p] <= d) & (d < start[p] + patients[p]["length_of_stay"]))
            for p in range(n_patients)
            for d in range(n_days)
        ],

        [operated_day[p][d] == ((scheduled[p] == 1) & (start[p] == d)) for p in range(n_patients) for d in range(n_days)],

        [
            in_room[p][r][d] == ((present[p][d] == 1) & (room[p] == r))
            for p in range(n_patients)
            for r in range(n_rooms)
            for d in range(n_days)
        ],

        [
            in_theater[p][t][d] == ((operated_day[p][d] == 1) & (theater[p] == t))
            for p in range(n_patients)
            for t in range(n_theaters)
            for d in range(n_days)
        ],

        [
            room[p] != room_id_to_idx[room_id]
            for p in range(n_patients)
            for room_id in patients[p].get("incompatible_room_ids", [])
            if room_id in room_id_to_idx
        ],

        [
            Sum(in_room[p][r][d] for p in range(n_patients)) + fixed_room_occupancy[rooms[r]["id"]][d] <= rooms[r]["capacity"]
            for r in range(n_rooms)
            for d in range(n_days)
        ],

        [
            Sum(
                patients[p]["surgery_duration"] * operated_day[p][d]
                for p in range(n_patients)
                if surgeon_id_to_idx[patients[p]["surgeon_id"]] == s
            )
            <= surgeons[s]["max_surgery_time"][d]
            for s in range(n_surgeons)
            for d in range(n_days)
        ],

        [
            Sum(patients[p]["surgery_duration"] * in_theater[p][t][d] for p in range(n_patients))
            <= theaters[t]["availability"][d]
            for t in range(n_theaters)
            for d in range(n_days)
        ],

        [
            open_theater[t][d] == (Sum(in_theater[p][t][d] for p in range(n_patients)) > 0)
            for t in range(n_theaters)
            for d in range(n_days)
        ],
    )

    delay_terms = Sum(
        (start[p] - patients[p]["surgery_release_day"]) * scheduled[p]
        for p in range(n_patients)
    )
    unscheduled_optional_terms = Sum(1 - scheduled[p] for p in optional_patients) if optional_patients else 0
    open_ot_terms = Sum(open_theater[t][d] for t in range(n_theaters) for d in range(n_days))

    objective = (
        weights["patient_delay"] * delay_terms
        + weights["unscheduled_optional"] * unscheduled_optional_terms
        + weights["open_operating_theater"] * open_ot_terms
    )

    minimize(objective)

    return {
        "start": start,
        "room": room,
        "theater": theater,
        "scheduled": scheduled,
        "present": present,
        "room_id_to_idx": room_id_to_idx,
        "theater_id_to_idx": theater_id_to_idx,
        "surgeon_id_to_idx": surgeon_id_to_idx,
        "unscheduled_day": unscheduled_day,
        "fixed_room_age_groups": fixed_room_age_groups,
    }


def extract_patient_solution(instance, model_vars):
    patients = instance["patients"]
    rooms = instance["rooms"]
    theaters = instance["operating_theaters"]
    n_patients = len(patients)
    unscheduled_day = model_vars["unscheduled_day"]

    patient_assignments = {}
    patients_json = []

    for p in range(n_patients):
        start_day = value(model_vars["start"][p])
        if start_day == unscheduled_day:
            patient_assignments[patients[p]["id"]] = None
            patients_json.append({"id": patients[p]["id"], "admission_day": "none"})
            continue

        room_id = rooms[value(model_vars["room"][p])]["id"]
        theater_id = theaters[value(model_vars["theater"][p])]["id"]
        patient_assignments[patients[p]["id"]] = {
            "admission_day": start_day,
            "room": room_id,
            "operating_theater": theater_id,
        }
        patients_json.append(
            {
                "id": patients[p]["id"],
                "admission_day": start_day,
                "room": room_id,
                "operating_theater": theater_id,
            }
        )

    return patient_assignments, patients_json


def build_room_shift_demands(instance, patient_assignments):
    room_shift_demands = defaultdict(lambda: {"load": 0, "skill": 0})

    for occupant in instance.get("occupants", []):
        for day in range(min(instance["days"], occupant["length_of_stay"])):
            for shift_name in instance["shift_types"]:
                key = (day, shift_name, occupant["room_id"])
                room_shift_demands[key]["load"] += value_from_profile(occupant["workload_produced"], day, shift_name)
                room_shift_demands[key]["skill"] = max(
                    room_shift_demands[key]["skill"],
                    value_from_profile(occupant["skill_level_required"], day, shift_name),
                )

    patient_by_id = {patient["id"]: patient for patient in instance["patients"]}
    for patient_id, assignment in patient_assignments.items():
        if assignment is None:
            continue
        patient = patient_by_id[patient_id]
        start_day = assignment["admission_day"]
        for day in range(start_day, start_day + patient["length_of_stay"]):
            relative_day = day - start_day
            for shift_name in instance["shift_types"]:
                key = (day, shift_name, assignment["room"])
                room_shift_demands[key]["load"] += value_from_profile(
                    patient["workload_produced"], relative_day, shift_name
                )
                room_shift_demands[key]["skill"] = max(
                    room_shift_demands[key]["skill"],
                    value_from_profile(patient["skill_level_required"], relative_day, shift_name),
                )

    return room_shift_demands


def assign_nurses_greedily(instance, patient_assignments):
    room_shift_demands = build_room_shift_demands(instance, patient_assignments)
    weights = instance["weights"]
    nurse_solutions = [{"id": nurse["id"], "assignments": []} for nurse in instance["nurses"]]
    nurse_output_by_id = {entry["id"]: entry for entry in nurse_solutions}

    shifts_by_slot = defaultdict(list)
    for nurse in instance["nurses"]:
        for shift in nurse["working_shifts"]:
            shifts_by_slot[(shift["day"], shift["shift"])].append(
                {
                    "id": nurse["id"],
                    "skill_level": nurse["skill_level"],
                    "max_load": shift["max_load"],
                }
            )

    previous_room_nurse = {}
    skill_penalty_units = 0
    excess_penalty_units = 0
    continuity_penalty_units = 0

    for day in range(instance["days"]):
        for shift_name in instance["shift_types"]:
            slot = (day, shift_name)
            available_nurses = shifts_by_slot.get(slot, [])
            nurse_state = {
                nurse["id"]: {"rooms": [], "load": 0, "max_load": nurse["max_load"], "skill": nurse["skill_level"]}
                for nurse in available_nurses
            }

            room_ids = sorted(
                {
                    room_id
                    for demand_day, demand_shift, room_id in room_shift_demands
                    if demand_day == day and demand_shift == shift_name and room_shift_demands[(demand_day, demand_shift, room_id)]["load"] > 0
                },
                key=lambda room_id: room_shift_demands[(day, shift_name, room_id)]["load"],
                reverse=True,
            )

            for room_id in room_ids:
                demand = room_shift_demands[(day, shift_name, room_id)]
                if not available_nurses:
                    skill_penalty_units += demand["skill"]
                    excess_penalty_units += demand["load"]
                    continue

                best_nurse_id = None
                best_score = None

                for nurse in available_nurses:
                    state = nurse_state[nurse["id"]]
                    overload = max(0, state["load"] + demand["load"] - state["max_load"])
                    skill_gap = max(0, demand["skill"] - state["skill"])
                    continuity_gap = 0
                    prev_nurse = previous_room_nurse.get((room_id, shift_name))
                    if prev_nurse is not None and prev_nurse != nurse["id"]:
                        continuity_gap = 1

                    score = (
                        weights["nurse_eccessive_workload"] * overload
                        + weights["room_nurse_skill"] * skill_gap
                        + weights["continuity_of_care"] * continuity_gap
                        + len(state["rooms"])
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best_nurse_id = nurse["id"]

                selected = nurse_state[best_nurse_id]
                selected["rooms"].append(room_id)
                selected["load"] += demand["load"]
                skill_penalty_units += max(0, demand["skill"] - selected["skill"])
                excess_penalty_units += max(0, selected["load"] - selected["max_load"])

                prev_nurse = previous_room_nurse.get((room_id, shift_name))
                if prev_nurse is not None and prev_nurse != best_nurse_id:
                    continuity_penalty_units += 1
                previous_room_nurse[(room_id, shift_name)] = best_nurse_id

            for nurse in available_nurses:
                nurse_output_by_id[nurse["id"]]["assignments"].append(
                    {"day": day, "shift": shift_name, "rooms": sorted(nurse_state[nurse["id"]]["rooms"])}
                )

    return nurse_solutions, skill_penalty_units, excess_penalty_units, continuity_penalty_units


def compute_age_mix_units(instance, patient_assignments):
    room_day_age_groups = defaultdict(set)

    for occupant in instance.get("occupants", []):
        for day in range(min(instance["days"], occupant["length_of_stay"])):
            room_day_age_groups[(occupant["room_id"], day)].add(occupant["age_group"])

    patient_by_id = {patient["id"]: patient for patient in instance["patients"]}
    for patient_id, assignment in patient_assignments.items():
        if assignment is None:
            continue
        patient = patient_by_id[patient_id]
        for day in range(assignment["admission_day"], assignment["admission_day"] + patient["length_of_stay"]):
            room_day_age_groups[(assignment["room"], day)].add(patient["age_group"])

    return sum(1 for groups in room_day_age_groups.values() if len(groups) > 1)


def compute_delay_units(instance, patient_assignments):
    patient_by_id = {patient["id"]: patient for patient in instance["patients"]}
    return sum(
        assignment["admission_day"] - patient_by_id[patient_id]["surgery_release_day"]
        for patient_id, assignment in patient_assignments.items()
        if assignment is not None
    )


def compute_unscheduled_optional_units(instance, patient_assignments):
    patient_by_id = {patient["id"]: patient for patient in instance["patients"]}
    return sum(
        1
        for patient_id, assignment in patient_assignments.items()
        if assignment is None and not patient_by_id[patient_id]["mandatory"]
    )


def compute_open_ot_units(patient_assignments):
    return len(
        {
            (assignment["operating_theater"], assignment["admission_day"])
            for assignment in patient_assignments.values()
            if assignment is not None
        }
    )


def compute_surgeon_transfer_units(instance, patient_assignments):
    patient_by_id = {patient["id"]: patient for patient in instance["patients"]}
    surgeon_day_theaters = defaultdict(set)
    for patient_id, assignment in patient_assignments.items():
        if assignment is None:
            continue
        surgeon_id = patient_by_id[patient_id]["surgeon_id"]
        surgeon_day_theaters[(surgeon_id, assignment["admission_day"])].add(assignment["operating_theater"])
    return sum(max(0, len(theaters) - 1) for theaters in surgeon_day_theaters.values())


def build_cost_line(instance, patient_assignments, nurse_skill_units, nurse_excess_units, continuity_units):
    weights = instance["weights"]

    unscheduled_units = compute_unscheduled_optional_units(instance, patient_assignments)
    delay_units = compute_delay_units(instance, patient_assignments)
    open_ot_units = compute_open_ot_units(patient_assignments)
    age_mix_units = compute_age_mix_units(instance, patient_assignments)
    surgeon_transfer_units = compute_surgeon_transfer_units(instance, patient_assignments)

    unscheduled_cost = unscheduled_units * weights["unscheduled_optional"]
    delay_cost = delay_units * weights["patient_delay"]
    open_ot_cost = open_ot_units * weights["open_operating_theater"]
    age_mix_cost = age_mix_units * weights["room_mixed_age"]
    skill_cost = nurse_skill_units * weights["room_nurse_skill"]
    excess_cost = nurse_excess_units * weights["nurse_eccessive_workload"]
    continuity_cost = continuity_units * weights["continuity_of_care"]
    surgeon_transfer_cost = surgeon_transfer_units * weights["surgeon_transfer"]

    total_cost = (
        unscheduled_cost
        + delay_cost
        + open_ot_cost
        + age_mix_cost
        + skill_cost
        + excess_cost
        + continuity_cost
        + surgeon_transfer_cost
    )

    return (
        f"Cost: {total_cost}, Unscheduled: {unscheduled_cost},  Delay: {delay_cost},  "
        f"OpenOT: {open_ot_cost},  AgeMix: {age_mix_cost},  Skill: {skill_cost},  Excess: {excess_cost},  "
        f"Continuity: {continuity_cost},  SurgeonTransfer: {surgeon_transfer_cost}"
    )


def build_solution(instance, patient_assignments, patients_json):
    nurses_json, nurse_skill_units, nurse_excess_units, continuity_units = assign_nurses_greedily(
        instance, patient_assignments
    )
    cost_line = build_cost_line(
        instance, patient_assignments, nurse_skill_units, nurse_excess_units, continuity_units
    )
    return {"patients": patients_json, "nurses": nurses_json, "costs": [cost_line]}


def validate_solution_shape(instance, solution):
    patient_ids = {patient["id"] for patient in instance["patients"]}
    nurse_ids = {nurse["id"] for nurse in instance["nurses"]}
    room_ids = {room["id"] for room in instance["rooms"]}
    theater_ids = {theater["id"] for theater in instance["operating_theaters"]}

    if set(solution.keys()) != {"patients", "nurses", "costs"}:
        raise ValueError("Le JSON solution doit contenir exactement patients, nurses et costs")

    if len(solution["patients"]) != len(patient_ids):
        raise ValueError("Nombre de patients incorrect dans la solution")

    if len(solution["nurses"]) != len(nurse_ids):
        raise ValueError("Nombre de nurses incorrect dans la solution")

    for patient in solution["patients"]:
        if patient["id"] not in patient_ids:
            raise ValueError(f"Patient inconnu dans la solution: {patient['id']}")
        if patient["admission_day"] != "none":
            if patient["room"] not in room_ids:
                raise ValueError(f"Room inconnue: {patient['room']}")
            if patient["operating_theater"] not in theater_ids:
                raise ValueError(f"Bloc inconnu: {patient['operating_theater']}")

    for nurse in solution["nurses"]:
        if nurse["id"] not in nurse_ids:
            raise ValueError(f"Nurse inconnue dans la solution: {nurse['id']}")


def save_solution(dataset, instance_id, solution):
    os.makedirs(solution_dir(dataset), exist_ok=True)
    path = os.path.join(solution_dir(dataset), solution_filename(dataset, instance_id))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(solution, fh, indent=2)
    return path


def solve_instance(dataset, instance_id, solver_name):
    instance = load_instance(dataset, instance_id)
    model_vars = build_patient_model(instance)

    started = time.time()
    status = solve(solver=solver_name)
    elapsed = round(time.time() - started, 3)

    if not status:
        return {
            "instance_id": instance_id,
            "status": "unsolved",
            "time_seconds": elapsed,
            "solution_path": None,
        }

    patient_assignments, patients_json = extract_patient_solution(instance, model_vars)
    solution = build_solution(instance, patient_assignments, patients_json)
    validate_solution_shape(instance, solution)
    path = save_solution(dataset, instance_id, solution)

    return {
        "instance_id": instance_id,
        "status": "solved",
        "time_seconds": elapsed,
        "solution_path": path,
        "cost": solution["costs"][0],
    }


def solve_all_instances(dataset, instance_ids, solver_name):
    results = []
    started = time.time()

    for instance_id in instance_ids:
        result = solve_instance(dataset, instance_id, solver_name)
        results.append(result)
        print(
            f"{instance_filename(dataset, instance_id)} -> {result['status']} "
            f"({result['time_seconds']}s)"
        )

    summary = {
        "solver": solver_name,
        "dataset": dataset,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_seconds": round(time.time() - started, 3),
        "instances": results,
    }

    os.makedirs(os.path.dirname(summary_path(dataset)), exist_ok=True)
    with open(summary_path(dataset), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Solveur healthcare IHTC en PyCSP3")
    parser.add_argument("--dataset", choices=["competition", "test"], default="competition")
    parser.add_argument("--instance", type=int, help="Numero d'instance a resoudre")
    parser.add_argument("--all", action="store_true", help="Resoudre toutes les instances du dataset")
    parser.add_argument("--solver", default="ace", help="Solveur PyCSP3 a utiliser")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.instance is not None:
        ids = [args.instance]
    else:
        ids = list_instance_ids(args.dataset)

    summary = solve_all_instances(args.dataset, ids, args.solver)
    print(f"Resume ecrit dans {summary_path(args.dataset)}")
