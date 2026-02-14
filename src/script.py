import matplotlib.pyplot as plt
import networkx as nx
from graph_ats_ranker import GraphBasedATSRanker

# Simple example data
job_requirements = {
    'skills': {
        'Python': 0.4,
        'SQL': 0.3,
        'Git': 0.3
    },
    'min_years_experience': 3,
    'preferred_years_experience': 5
}

candidates = [
    {
        'id': 'Alice',
        'years_of_experience': 5,
        'skills': {
            'Python': 0.9,
            'SQL': 0.8,
            'Git': 0.6
        }
    },
    {
        'id': 'Bob',
        'years_of_experience': 2,
        'skills': {
            'Python': 0.7,
            'SQL': 0.5
        }
    }
]

# Create ranker and build graph
ranker = GraphBasedATSRanker(
    experience_weight=0.3,
    experience_mode='both',
    normalize_edges=True
)
G = ranker.build_graph(job_requirements, candidates)

# Print graph structure
print("=" * 60)
print("GRAPH STRUCTURE")
print("=" * 60)
print(f"\nTotal Nodes: {G.number_of_nodes()}")
print(f"Total Edges: {G.number_of_edges()}")
print("\n" + "-" * 60)
print("NODES:")
print("-" * 60)
for node in G.nodes(data=True):
    print(f"  {node[0]:30} type={node[1].get('node_type', 'N/A')}")

print("\n" + "-" * 60)
print("EDGES (with weights):")
print("-" * 60)
for source, target, data in G.edges(data=True):
    weight = data['weight']
    edge_type = data.get('edge_type', 'skill_flow')
    print(f"  {source:30} -> {target:30} weight={weight:.4f} [{edge_type}]")

# Calculate outgoing weight totals per node (should be ~1.0 if normalized)
print("\n" + "-" * 60)
print("OUTGOING WEIGHT TOTALS (should be ~1.0 after normalization):")
print("-" * 60)
for node in G.nodes():
    out_edges = G.out_edges(node, data=True)
    if out_edges:
        total = sum(data['weight'] for _, _, data in out_edges)
        print(f"  {node:30} total={total:.6f}")

# Visualize the graph
plt.figure(figsize=(14, 10))

# Define node positions using a hierarchical layout
pos = {}
# Job at top
pos['JOB'] = (0, 3)
# Perfect candidate at top right
pos['PERFECT_CANDIDATE'] = (1.5, 0.5)
# Skills in middle
skills = [n for n in G.nodes() if n.startswith('SKILL_')]
for i, skill in enumerate(skills):
    pos[skill] = (-1 + i * 1, 2)
# Candidates at bottom
candidates_nodes = [n for n in G.nodes() if n.startswith('CAND_')]
for i, cand in enumerate(candidates_nodes):
    pos[cand] = (-0.5 + i * 1, 0.5)

# Color nodes by type
node_colors = []
for node in G.nodes():
    if node == 'JOB':
        node_colors.append('#ff6b6b')  # Red
    elif node == 'PERFECT_CANDIDATE':
        node_colors.append('#ffd93d')  # Gold
    elif node.startswith('SKILL_'):
        node_colors.append('#6bcf7f')  # Green
    else:  # Candidates
        node_colors.append('#4ecdc4')  # Teal

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, alpha=0.9)

# Draw labels
labels = {}
for node in G.nodes():
    if node.startswith('SKILL_'):
        labels[node] = node.replace('SKILL_', '')
    elif node.startswith('CAND_'):
        labels[node] = node.replace('CAND_', '')
    else:
        labels[node] = node
nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')

# Draw edges with different styles for different types
skill_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') != 'experience_match']
exp_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'experience_match']

# Draw skill flow edges
nx.draw_networkx_edges(G, pos, edgelist=skill_edges, edge_color='gray',
                       arrows=True, arrowsize=20, width=2, alpha=0.6,
                       connectionstyle='arc3,rad=0.1')

# Draw experience edges
nx.draw_networkx_edges(G, pos, edgelist=exp_edges, edge_color='red',
                       arrows=True, arrowsize=20, width=2, alpha=0.6,
                        connectionstyle='arc3,rad=0.2')

# Draw edge labels (weights)
edge_labels = {(u, v): f"{d['weight']:.3f}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7,
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.title("ATS Ranking Graph Structure\n" +
          "(Gris = skill flow, rouge = experience match)",
          fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('graph_structure.png', dpi=300, bbox_inches='tight')
print("\n" + "=" * 60)
print("Graph visualization saved to graph_structure.png")
print("=" * 60)

# Show what happens with PageRank
print("\n" + "-" * 60)
print("RUNNING PAGERANK...")
print("-" * 60)
rankings = ranker.compute_rankings()
print("\nCandidate Rankings:")
print(rankings.to_string(index=False))

print("\n" + "-" * 60)
print("PAGERANK SCORES (all nodes):")
print("-" * 60)
for node, score in sorted(ranker.pagerank_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  {node:30} score={score:.6f}")