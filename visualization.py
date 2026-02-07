"""
Visualization and Analysis Tools for Graph-Based ATS Ranker
============================================================

Provides tools to:
1. Visualize the bipartite graph
2. Create ranking comparison charts
3. Analyze skill importance and coverage
4. Generate candidate reports
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List
import seaborn as sns

sns.set_style("whitegrid")


def visualize_graph(ranker, figsize=(14, 10), save_path=None):
    """
    Visualize the bipartite graph structure.
    
    Parameters:
    -----------
    ranker : GraphBasedATSRanker
        A ranker with a built graph
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    if ranker.graph is None:
        raise ValueError("Ranker must have a graph built")
    
    G = ranker.graph
    
    # Create layout - three layers
    pos = {}
    
    # Job node at top
    pos[ranker.job_node] = (0, 2)
    
    # Skill nodes in middle
    skill_nodes = [n for n in G.nodes() if n.startswith(ranker.skill_prefix)]
    num_skills = len(skill_nodes)
    for i, node in enumerate(skill_nodes):
        x = (i - num_skills/2) * 1.5
        pos[node] = (x, 1)
    
    # Candidate nodes at bottom
    cand_nodes = [n for n in G.nodes() if n.startswith(ranker.candidate_prefix)]
    num_cands = len(cand_nodes)
    for i, node in enumerate(cand_nodes):
        x = (i - num_cands/2) * 1.5
        pos[node] = (x, 0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw nodes
    job_color = '#FF6B6B'
    skill_color = '#4ECDC4'
    cand_color = '#95E1D3'
    
    nx.draw_networkx_nodes(G, pos, nodelist=[ranker.job_node], 
                          node_color=job_color, node_size=1000, 
                          node_shape='s', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=skill_nodes,
                          node_color=skill_color, node_size=800, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=cand_nodes,
                          node_color=cand_color, node_size=600, ax=ax)
    
    # Draw edges with varying thickness based on weight
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        nx.draw_networkx_edges(G, pos, [(u, v)], width=weight*3,
                              alpha=0.6, edge_color='gray', ax=ax)
    
    # Labels
    labels = {}
    labels[ranker.job_node] = "JOB"
    for node in skill_nodes:
        labels[node] = node.replace(ranker.skill_prefix, "")
    for node in cand_nodes:
        labels[node] = node.replace(ranker.candidate_prefix, "")
    
    nx.draw_networkx_labels(G, pos, labels, font_size=9, ax=ax)
    
    # Legend
    job_patch = mpatches.Patch(color=job_color, label='Job')
    skill_patch = mpatches.Patch(color=skill_color, label='Skills')
    cand_patch = mpatches.Patch(color=cand_color, label='Candidates')
    ax.legend(handles=[job_patch, skill_patch, cand_patch], loc='upper right')
    
    ax.set_title("ATS Graph Structure\n(Edge thickness = weight)", fontsize=14, weight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_rankings(rankings, top_n=10, figsize=(10, 6), save_path=None):
    """
    Plot candidate rankings as a horizontal bar chart.
    
    Parameters:
    -----------
    rankings : pd.DataFrame
        Rankings from ranker.compute_rankings()
    top_n : int
        Number of top candidates to show
    """
    data = rankings.head(top_n).copy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(data)))
    
    ax.barh(range(len(data)), data['score'], color=colors)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data['candidate_id'])
    ax.invert_yaxis()
    
    ax.set_xlabel('PageRank Score', fontsize=12)
    ax.set_title(f'Top {top_n} Candidates by Graph-Based Ranking', 
                fontsize=14, weight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add score labels
    for i, (idx, row) in enumerate(data.iterrows()):
        ax.text(row['score'] + 0.001, i, f"{row['score']:.4f}", 
               va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_skill_coverage_heatmap(
    job_skills: Dict[str, float],
    candidates: Dict[str, Dict[str, float]],
    rankings: pd.DataFrame = None,
    figsize=(12, 8),
    save_path=None
):
    """
    Create a heatmap showing which candidates have which skills.
    
    Parameters:
    -----------
    job_skills : Dict[str, float]
        Skill importances
    candidates : Dict[str, Dict[str, float]]
        Candidate proficiencies
    rankings : pd.DataFrame, optional
        If provided, sort candidates by rank
    """
    # Build matrix
    skills = sorted(job_skills.keys(), key=lambda s: job_skills[s], reverse=True)
    
    if rankings is not None:
        cand_ids = rankings['candidate_id'].tolist()
    else:
        cand_ids = sorted(candidates.keys())
    
    matrix = []
    for cand_id in cand_ids:
        row = [candidates[cand_id].get(skill, 0.0) for skill in skills]
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(matrix, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(range(len(skills)))
    ax.set_yticks(range(len(cand_ids)))
    ax.set_xticklabels(skills, rotation=45, ha='right')
    ax.set_yticklabels(cand_ids)
    
    # Add importance indicators to skill labels
    skill_labels_with_importance = [
        f"{skill}\n(imp: {job_skills[skill]:.2f})" for skill in skills
    ]
    ax.set_xticklabels(skill_labels_with_importance, rotation=45, ha='right', fontsize=9)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Proficiency', rotation=270, labelpad=20)
    
    # Add values in cells
    for i in range(len(cand_ids)):
        for j in range(len(skills)):
            value = matrix[i, j]
            if value > 0:
                text = ax.text(j, i, f'{value:.2f}',
                             ha='center', va='center',
                             color='white' if value > 0.5 else 'black',
                             fontsize=8)
    
    ax.set_title('Candidate-Skill Coverage Matrix\n(Ordered by importance)', 
                fontsize=14, weight='bold')
    ax.set_xlabel('Skills (with importance)', fontsize=11)
    ax.set_ylabel('Candidates (by rank)' if rankings is not None else 'Candidates', 
                 fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compare_ranking_methods(
    job_skills: Dict[str, float],
    candidates: Dict[str, Dict[str, float]],
    figsize=(14, 6),
    save_path=None
):
    """
    Compare graph-based ranking with simple vector distance.
    
    Shows how graph ranking differs from naive approaches.
    """
    from graph_ats_ranker import rank_candidates
    
    # Graph-based ranking
    graph_rankings = rank_candidates(job_skills, candidates)
    
    # Naive ranking: weighted sum
    naive_scores = []
    for cand_id, skills in candidates.items():
        score = sum(skills.get(s, 0) * job_skills[s] for s in job_skills)
        naive_scores.append({'candidate_id': cand_id, 'score': score})
    
    naive_df = pd.DataFrame(naive_scores)
    naive_df = naive_df.sort_values('score', ascending=False).reset_index(drop=True)
    naive_df['rank'] = naive_df.index + 1
    
    # Merge
    comparison = graph_rankings[['candidate_id', 'rank']].rename(columns={'rank': 'graph_rank'})
    comparison = comparison.merge(
        naive_df[['candidate_id', 'rank']].rename(columns={'rank': 'naive_rank'}),
        on='candidate_id'
    )
    comparison['rank_diff'] = comparison['naive_rank'] - comparison['graph_rank']
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Side-by-side comparison
    x = np.arange(len(comparison))
    width = 0.35
    
    ax1.barh(x - width/2, comparison['graph_rank'], width, 
            label='Graph-based', color='#4ECDC4')
    ax1.barh(x + width/2, comparison['naive_rank'], width,
            label='Weighted Sum', color='#FF6B6B')
    
    ax1.set_yticks(x)
    ax1.set_yticklabels(comparison['candidate_id'])
    ax1.invert_yaxis()
    ax1.invert_xaxis()
    ax1.set_xlabel('Rank (lower is better)')
    ax1.set_title('Ranking Comparison')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    
    # Rank change analysis
    colors = ['green' if d < 0 else 'red' if d > 0 else 'gray' 
             for d in comparison['rank_diff']]
    
    ax2.barh(range(len(comparison)), comparison['rank_diff'], color=colors, alpha=0.7)
    ax2.set_yticks(range(len(comparison)))
    ax2.set_yticklabels(comparison['candidate_id'])
    ax2.invert_yaxis()
    ax2.axvline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Rank Change\n(Negative = Graph ranks higher)')
    ax2.set_title('Graph vs Naive Ranking Difference')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, comparison


def generate_candidate_report(
    ranker,
    candidate_id: str,
    job_skills: Dict[str, float],
    candidates: Dict[str, Dict[str, float]]
):
    """
    Generate a detailed text report for a candidate.
    """
    explanation = ranker.explain_ranking(candidate_id)
    
    report = f"""
{'='*70}
CANDIDATE REPORT: {candidate_id}
{'='*70}

OVERALL RANKING
--------------
Rank:           #{explanation['rank']}
Score:          {explanation['score']:.6f}
Skill Coverage: {explanation['skill_coverage']:.1%}

TOP CONTRIBUTING SKILLS
-----------------------
"""
    
    for skill in explanation['top_skills']:
        report += f"""
{skill['skill']}:
  Proficiency:  {skill['proficiency']:.2f}
  Importance:   {skill['importance']:.2f}
  Contribution: {skill['contribution']:.6f}
"""
    
    if explanation['missing_skills']:
        report += f"""
MISSING SKILLS (GAPS)
---------------------
"""
        for skill in explanation['missing_skills']:
            report += f"  • {skill['skill']} (importance: {skill['importance']:.2f})\n"
    else:
        report += "\nMISSING SKILLS: None ✓\n"
    
    report += f"""
{'='*70}
"""
    
    return report


if __name__ == "__main__":
    from graph_ats_ranker import GraphBasedATSRanker
    
    # Example usage
    job_skills = {
        "Python": 0.9,
        "SQL": 0.7,
        "Leadership": 0.5,
        "Communication": 0.6
    }
    
    candidates = {
        "Alice": {"Python": 0.8, "SQL": 0.6, "Leadership": 0.4, "Communication": 0.7},
        "Bob": {"Python": 0.5, "SQL": 0.9, "Leadership": 0.7},
        "Charlie": {"Python": 0.9, "SQL": 0.8},
        "Diana": {"Python": 0.7, "SQL": 0.7, "Leadership": 0.6, "Communication": 0.8}
    }
    
    # Build and rank
    ranker = GraphBasedATSRanker()
    ranker.build_graph(job_skills, candidates)
    rankings = ranker.compute_rankings()
    
    # Create visualizations
    print("Creating visualizations...")
    
    visualize_graph(ranker, save_path='image/graph_structure.png')
    print("✓ Graph structure saved")
    
    plot_rankings(rankings, save_path='image/rankings.png')
    print("✓ Rankings plot saved")
    
    plot_skill_coverage_heatmap(job_skills, candidates, rankings, 
                               save_path='image/skill_coverage.png')
    print("✓ Skill coverage heatmap saved")
    
    fig, comp = compare_ranking_methods(job_skills, candidates,
                                       save_path='image/comparison.png')
    print("✓ Method comparison saved")
    
    # Generate report
    top_candidate = rankings.iloc[0]['candidate_id']
    report = generate_candidate_report(ranker, top_candidate, job_skills, candidates)
    print(report)
