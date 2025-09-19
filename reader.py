import pandas as pd
from collections import defaultdict

import pandas as pd
from collections import defaultdict

def read_tntp_network(filepath):
    # Find where the actual table starts (after metadata & headers)
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the header line (starts with "~")
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("~"):
            header_idx = i
            break
    
    # Read table with pandas
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        skiprows=header_idx+1,  # skip metadata & header row
        usecols=range(11),      # only first 11 cols before the trailing ";"
        names=["init_node","term_node","capacity","length",
               "free_flow_time","B","power","speed","toll","link_type","end"],
        comment="~"
    )

    # Drop last "end" column (it is just ';')
    df = df.drop(columns=["end"])
    
    # Convert node IDs to strings (so '1', '2', '24', … instead of floats)
    df["init_node"] = df["init_node"].astype(int).astype(str)
    df["term_node"] = df["term_node"].astype(int).astype(str)

    if '6665' in df['term_node'].values:
        print(" term node 6665 is in the data frame ")
    return df


def build_graph(df, weight_col="free_flow_time"):
    graph = defaultdict(list)
    for _, row in df.iterrows():
        start = row["init_node"]  # string now
        end   = row["term_node"]  # string now
        w     = row[weight_col]
        graph[start].append((end, w))
    return dict(graph)


def read_trips_tntp(filepath):
    demand = defaultdict(dict)
    with open(filepath, "r") as f:
        lines = f.readlines()

    origin = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("<"):  # skip metadata and blanks
            continue

        if line.startswith("Origin"):
            # e.g., "Origin   24"
            origin = str(line.split()[1])
            continue

        # Otherwise, we are inside an origin block: destinations and flows
        parts = line.split(";")
        for part in parts:
            if ":" in part:
                dest, val = part.split(":")
                dest = str(dest.strip())
                val = float(val.strip())
                demand[origin][dest] = val

    return demand



def get_all_od_pairs_from_trips(demand, nonzero_only=True):
    """
    Returns a list of all origin–destination pairs in the trips data as strings "XY".

    Parameters:
        demand        : dict of dict
                        e.g. demand[origin][dest] = flow
        nonzero_only  : bool (default True)
                        if True, only include OD pairs with positive flow

    Returns:
        List[str]     : e.g. ['12', '13', '24', ...]
    """
    od_pairs = []
    for o, dests in demand.items():
        for d, flow in dests.items():
            if o == d:  # skip same node
                continue
            if nonzero_only and flow <= 0:
                continue
            od_pairs.append(f"{o}{d}")
    return od_pairs


def build_od_mappings(demand):
    """
    Creates mappings from OD pair string to origin and destination.

    Parameters:
        demand (dict): demand[origin][destination] = value

    Returns:
        od_to_origin (dict): { "12": "1", "13": "1", ... }
        od_to_dest   (dict): { "12": "2", "13": "3", ... }
    """
    od_to_origin = {}
    od_to_dest = {}

    for orig, dests in demand.items():
        if orig == "0":
            continue  # skip zone 0
        for dest in dests:
            if dest == "0":
                continue
            od_pair = f"{orig}{dest}"
            od_to_origin[od_pair] = orig
            od_to_dest[od_pair] = dest

    return od_to_origin, od_to_dest

import random

def initialize_y_with_random_removals(links, fraction, seed=2):
    """
    Initialize y dict for all links, randomly deactivating some.

    Parameters
    ----------
    links : list of str
        List of all links (e.g., ["12", "23", ...]).
    fraction : float, default=0.01
        Fraction of links to deactivate (set y=0).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    y : dict
        Dictionary where y[link] = 1 (active) or 0 (inactive).
    """
    if seed is not None:
        random.seed(seed)

    # Start with all links active
    y = {link: 1 for link in links}

    # Randomly choose subset to deactivate
    num_remove = max(1, int(len(links) * fraction))
    removed_links = random.sample(links, num_remove)

    for link in removed_links:
        y[link] = 0

    return y, removed_links




# ---- Example usage ----
# # df = read_tntp_network("SiouxFalls_trips.tntp")
# demand = read_trips_tntp("SiouxFalls_trips.tntp")


# # graph = build_graph(demand, weight_col="free_flow_time")

# print(sum(demand[orig][dest] for orig in demand for dest in demand[orig]))  # show first few entries


# pairs = get_all_od_pairs_from_trips(demand, nonzero_only=True)
# print(pairs[:])  # first 10 OD pairs like ['241', '243', '109', ...]
