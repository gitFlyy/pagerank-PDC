# Converts web-Google.txt edge list to METIS graph format
# Usage:
#   python convert.py
#   python convert.py web-Google.txt web-Google.metis

import sys
from collections import defaultdict

def convert(input_file, output_file):
    edges = []
    nodes = set()
    
    print("Reading edge list...")
    with open(input_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            edges.append((u, v))
            nodes.add(u)
            nodes.add(v)
    
    # Remap node IDs to 0..N-1
    node_list = sorted(nodes)
    node_map = {old: new for new, old in enumerate(node_list)}
    N = len(node_list)
    
    print(f"Nodes: {N}, Edges: {len(edges)}")
    
    # Build adjacency list (undirected for METIS partitioning)
    adj = defaultdict(set)
    for u, v in edges:
        mu, mv = node_map[u], node_map[v]
        if mu != mv:
            adj[mu].add(mv)
            adj[mv].add(mu)
    
    # Count actual edges (undirected, no self-loops)
    total_edges = sum(len(neighbors) for neighbors in adj.values()) // 2
    
    print(f"Writing METIS format to {output_file}...")
    with open(output_file, 'w') as f:
        f.write(f"{N} {total_edges}\n")
        for i in range(N):
            neighbors = sorted(adj[i])
            # METIS uses 1-based indexing
            line = ' '.join(str(n + 1) for n in neighbors)
            f.write(line + '\n')
    
    # Also save the node mapping and original directed edges
    with open(output_file + '.mapping', 'w') as f:
        for old, new in node_map.items():
            f.write(f"{old} {new}\n")
    
    # Save directed edge list with remapped IDs (for PageRank)
    with open(output_file + '.edges', 'w') as f:
        f.write(f"{N} {len(edges)}\n")
        for u, v in edges:
            f.write(f"{node_map[u]} {node_map[v]}\n")
    
    print("Done!")
    print(f"Output: {output_file} (METIS format)")
    print(f"Output: {output_file}.edges (directed edges for PageRank)")

if __name__ == '__main__':
    default_input = 'web-Google.txt'
    default_output = 'web-Google.metis'

    if len(sys.argv) == 1:
        input_file = default_input
        output_file = default_output
        print(f'No arguments provided. Using defaults: {input_file} -> {output_file}')
    elif len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        script_name = sys.argv[0] if sys.argv else 'convert.py'
        print('Usage: python convert.py [input_edge_list.txt output_metis_file]')
        print(f'Example: {script_name} web-Google.txt web-Google.metis')
        sys.exit(1)

    convert(input_file, output_file)

