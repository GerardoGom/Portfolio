import sys
import re
from typing import Dict, List, Set, Tuple


# Parses DOT file and extract vertices and edges
def parse_dot_file(filename: str) -> Tuple[Set[str], List[Tuple[str, str]]]:
    vertices = set()
    edges = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Matches edge pattern: V1 -> V2
            match = re.match(r'(\w+)\s*->\s*(\w+)', line)
            if match:
                from_vertex = match.group(1)
                to_vertex = match.group(2)
                vertices.add(from_vertex)
                vertices.add(to_vertex)
                edges.append((from_vertex, to_vertex))
    
    return vertices, edges


# Builds adjacency list representation of the graph
def build_graph(vertices: Set[str], edges: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    graph = {v: [] for v in vertices}
    
    for from_v, to_v in edges:
        graph[from_v].append(to_v)
    
    return graph


# Computes PageRank using iterative power method with damping factor
# Handles dead ends and spider traps
def pagerank(graph: Dict[str, List[str]], damping: float = 0.85, 
             max_iterations: int = 100, tolerance: float = 1e-8) -> Dict[str, float]:
    n = len(graph)
    if n == 0:
        return {}
    
    # Initializes PageRank values uniformly
    pagerank_values = {v: 1.0 / n for v in graph}
    
    # Identifys dead ends (vertices with no outgoing edges)
    dead_ends = [v for v in graph if len(graph[v]) == 0]
    
    for iteration in range(max_iterations):
        new_pagerank = {}
        teleport_contrib = 0.0
        
        # Calculates contribution from dead ends
        # Dead ends distribute their PageRank to all vertices equally
        for v in dead_ends:
            teleport_contrib += pagerank_values[v] / n
        
        # Calculates new PageRank for each vertex
        for v in graph:
            # Teleportation contribution (random jumps)
            rank = (1 - damping) / n
            
            # Adds the dead end contribution
            rank += damping * teleport_contrib
            
            # Adds contributions from incoming edges
            for u in graph:
                if v in graph[u]:
                    # u links to v
                    out_degree = len(graph[u])
                    if out_degree > 0:
                        rank += damping * pagerank_values[u] / out_degree
            
            new_pagerank[v] = rank
        
        # Checks for convergence
        diff = sum(abs(new_pagerank[v] - pagerank_values[v]) for v in graph)
        pagerank_values = new_pagerank
        
        if diff < tolerance:
            break
    
    # Normalizes to ensure sum equals 1
    total = sum(pagerank_values.values())
    if total > 0:
        pagerank_values = {v: pr / total for v, pr in pagerank_values.items()}
    
    return pagerank_values


# Writes PageRank results to CSV file, sorted by PageRank (desc) then vertex name (asc)
def write_output(pagerank_values: Dict[str, float], output_filename: str):
    sorted_vertices = sorted(pagerank_values.items(), 
                            key=lambda x: (-x[1], x[0]))
    
    with open(output_filename, 'w') as f:
        f.write("vertex,pagerank\n")
        for vertex, pr in sorted_vertices:
            f.write(f"{vertex},{pr:.10f}\n")


# Main entry point
def main():
    # Parses command line arguments
    if len(sys.argv) < 3:
        print("Usage: python pagerank.py <input_graph.dot> <output_pagerank.csv>")
        print("\nOptional parameters:")
        print("  python pagerank.py <input_graph.dot> <output_pagerank.csv> <damping_factor>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    damping = 0.85
    
    if len(sys.argv) >= 4:
        damping = float(sys.argv[3])
    
    print(f"Reading graph from: {input_file}")
    vertices, edges = parse_dot_file(input_file)
    
    print(f"Graph has {len(vertices)} vertices and {len(edges)} edges")
    
    graph = build_graph(vertices, edges)
    
    print(f"Computing PageRank (damping factor = {damping})...")
    pr_values = pagerank(graph, damping=damping)
    
    print(f"Writing results to: {output_file}")
    write_output(pr_values, output_file)
    
    print("Done!")
    print(f"\nTop 5 vertices by PageRank:")
    sorted_vertices = sorted(pr_values.items(), key=lambda x: (-x[1], x[0]))
    for i, (vertex, pr) in enumerate(sorted_vertices[:5]):
        print(f"  {i+1}. {vertex}: {pr:.6f}")


if __name__ == "__main__":
    main()

