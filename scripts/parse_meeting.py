# scripts/parse_meeting.py

import json
import os
import re


def parse_msp_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    instances = []

    # Découpage sur **Instance #N** ou __Instance #N__
    blocks = re.split(r'(?:\*\*|__)Instance #(\d+)(?:\*\*|__)', content)
    pairs = [(blocks[i], blocks[i + 1]) for i in range(1, len(blocks) - 1, 2)]

    for num_str, body in pairs:
        instance_id = int(num_str)
        inst = {"id": instance_id}

        def extract_int(key, text):
            m = re.search(rf'{key}\s*=\s*(\d+)', text)
            return int(m.group(1)) if m else None

        inst["NumberOfMeetings"]          = extract_int("NumberOfMeetings",          body)
        inst["NumberOfAgents"]            = extract_int("NumberOfAgents",            body)
        inst["NumberOfMeetingPerAgent"]   = extract_int("NumberOfMeetingPerAgent",   body)
        inst["MinDisTimeBetweenMeetings"] = extract_int("MinDisTimeBetweenMeetings", body)
        inst["MaxDisTimeBetweenMeetings"] = extract_int("MaxDisTimeBetweenMeetings", body)
        inst["DomainSize"]                = extract_int("DomainSize",                body)

        p = re.search(r'Estimated P1=([\d.]+)\s+P2=([\d.]+)', body)
        inst["EstimatedP1"] = float(p.group(1)) if p else None
        inst["EstimatedP2"] = float(p.group(2)) if p else None

        agents_meetings = {}
        agents_section = re.search(
            r'Agents Meetings:\s*((?:[ \t]*Agents\s*\(\d+\):.*\n?)+)',
            body
        )
        if agents_section:
            for line in agents_section.group(1).splitlines():
                m = re.match(r'\s*Agents\s*\((\d+)\):\s*([\d\s]+)', line)
                if m:
                    agents_meetings[int(m.group(1))] = list(map(int, m.group(2).split()))
        inst["AgentsMeetings"] = agents_meetings

        dist_section = re.search(
            r'Between Meetings Distance:\s*\n(.*?)(?=\n\s*Estimated|\Z)',
            body, re.DOTALL
        )
        distance_matrix = []
        if dist_section:
            for line in dist_section.group(1).splitlines():
                line = line.strip()
                if not line or not re.match(r'^\d+\s*:', line):
                    continue
                data_part = re.sub(r'^\d+\s*:\s*', '', line)
                nums = list(map(int, re.findall(r'\d+', data_part)))
                if nums:
                    distance_matrix.append(nums)

        inst["BetweenMeetingsDistance"] = distance_matrix
        instances.append(inst)

    return instances


if __name__ == '__main__':
    base       = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base, '..', 'data', 'meeting', 'instances.md')
    if not os.path.exists(input_file):
        input_file = os.path.join(base, 'instances.md')

    print(f"Lecture de : {input_file}\n")
    if not os.path.exists(input_file):
        print("Fichier introuvable."); exit(1)

    instances = parse_msp_file(input_file)

    output_file = os.path.join(base, '..', 'data', 'meeting', 'instances.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(instances, f, indent=2)

    print(f"{len(instances)} instances parsées → {output_file}\n")

    all_ok = True
    for inst in instances:
        n, na = inst["NumberOfMeetings"], inst["NumberOfAgents"]
        mat, agents = inst["BetweenMeetingsDistance"], inst["AgentsMeetings"]
        ok_mat    = len(mat) == n and all(len(row) == n for row in mat)
        ok_agents = len(agents) == na
        status = "Ok" if (ok_mat and ok_agents) else "KO"
        if not (ok_mat and ok_agents): all_ok = False
        print(f"Instance #{inst['id']:>2}  meetings={n}  agents={na}  matrix={len(mat)}x{len(mat[0]) if mat else 0}  agents_parsed={len(agents)}  {status}")

    print("\nToutes les instances sont valides !" if all_ok else "\nCertaines instances ont des anomalies.")
