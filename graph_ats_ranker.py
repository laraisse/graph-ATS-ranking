"""
Core Principle:
- Importance flows from Job → Skills → Candidates
- Ranking emerges from graph structure, not arithmetic comparison
- Naturally penalizes gaps and rewards balanced profiles
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd


class GraphBasedATSRanker:
    """
    Architecture:
    -------------
    Job Node (J)
        ↓ [importance weights I(s)]
    Skill Nodes (S₁, S₂, ..., Sₙ)
        ↓ [proficiency weights P(c,s)]
    Candidate Nodes (C₁, C₂, ..., Cₘ)
    
    Algorithm:
    ----------
    1. Build weighted bipartite graph
    2. Run Personalized PageRank starting from Job node
    3. Extract candidate scores from stationary distribution
    4. Rank candidates by score (descending)
    """
    
    def __init__(
        self,
        damping: float = 0.85,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
        normalize_edges: bool = True
    ):
        """
        Initialize the ranker.
        
        Parameters:
        -----------
        damping : float
            PageRank damping factor (probability of following edges vs teleporting)
            Default 0.85 is standard
        tolerance : float
            Convergence tolerance for PageRank
        max_iterations : int
            Maximum iterations for PageRank
        normalize_edges : bool
            If True, normalize outgoing edge weights to sum to 1 per node
        """
        self.damping = damping
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.normalize_edges = normalize_edges
        
        self.graph = None
        self.job_node = "JOB"
        self.skill_prefix = "SKILL_"
        self.candidate_prefix = "CAND_"
        
        self.pagerank_scores = None
        self.candidate_rankings = None
        
    def build_graph(
        self,
        job_skills: Dict[str, float],
        candidates: Dict[str, Dict[str, float]]
    ) -> nx.DiGraph:
        """
        Build the bipartite graph from job requirements and candidate profiles.
        
        Parameters:
        -----------
        job_skills : Dict[str, float]
            Maps skill_name → importance (0-1)
            Example: {"Python": 0.9, "SQL": 0.7, "Leadership": 0.5}
            
        candidates : Dict[str, Dict[str, float]]
            Maps candidate_id → {skill_name → proficiency (0-1)}
            Example: {
                "Alice": {"Python": 0.8, "SQL": 0.6, "Leadership": 0.4},
                "Bob": {"Python": 0.5, "SQL": 0.9, "Leadership": 0.7}
            }
            
        Returns:
        --------
        nx.DiGraph
            The constructed graph
        """
        G = nx.DiGraph()
        
        # Add job node
        G.add_node(self.job_node, node_type='job')
        
        # Add skill nodes and edges from job
        for skill, importance in job_skills.items():
            skill_node = f"{self.skill_prefix}{skill}"
            G.add_node(skill_node, node_type='skill', skill_name=skill)
            G.add_edge(self.job_node, skill_node, weight=importance)
        
        # Add candidate nodes and edges to skills
        for cand_id, skills in candidates.items():
            cand_node = f"{self.candidate_prefix}{cand_id}"
            G.add_node(cand_node, node_type='candidate', candidate_id=cand_id)
            
            for skill, proficiency in skills.items():
                skill_node = f"{self.skill_prefix}{skill}"
                
                # Only add edge if skill exists in job requirements AND proficiency > 0
                if skill_node in G and proficiency > 0:
                    G.add_edge(skill_node, cand_node, weight=proficiency)
        
        # Normalize edge weights if requested
        if self.normalize_edges:
            self._normalize_graph_weights(G)
        
        self.graph = G
        return G
    
    def _normalize_graph_weights(self, G: nx.DiGraph):
        """
        Normalize outgoing edge weights so they sum to 1 per node.
        This makes the graph a proper transition probability matrix.
        """
        for node in G.nodes():
            out_edges = G.out_edges(node, data=True)
            if out_edges:
                total_weight = sum(data['weight'] for _, _, data in out_edges)
                if total_weight > 0:
                    for _, target, data in out_edges:
                        data['weight'] /= total_weight
    
    def compute_rankings(self) -> pd.DataFrame:
        """
        Run Personalized PageRank and extract candidate rankings.
        
        Returns:
        --------
        pd.DataFrame
            Columns: ['candidate_id', 'score', 'rank']
            Sorted by rank (1 = best)
        """
        if self.graph is None:
            raise ValueError("Must call build_graph() first")
        
        # Personalized PageRank: start random walk from job node
        personalization = {node: 1.0 if node == self.job_node else 0.0 
                          for node in self.graph.nodes()}
        
        # Run PageRank
        self.pagerank_scores = nx.pagerank(
            self.graph,
            alpha=self.damping,
            personalization=personalization,
            max_iter=self.max_iterations,
            tol=self.tolerance,
            weight='weight'
        )
        
        # Extract candidate scores
        candidate_scores = []
        for node, score in self.pagerank_scores.items():
            if node.startswith(self.candidate_prefix):
                candidate_id = node.replace(self.candidate_prefix, "")
                candidate_scores.append({
                    'candidate_id': candidate_id,
                    'score': score
                })
        
        # Sort by score descending
        df = pd.DataFrame(candidate_scores)
        df = df.sort_values('score', ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        
        self.candidate_rankings = df
        return df
    
    def explain_ranking(
        self,
        candidate_id: str,
        top_k_skills: int = 5
    ) -> Dict:
        """
        Explain why a candidate ranked where they did.
        
        Parameters:
        -----------
        candidate_id : str
            The candidate to explain
        top_k_skills : int
            Number of top contributing skills to show
            
        Returns:
        --------
        Dict with explanation details
        """
        if self.pagerank_scores is None:
            raise ValueError("Must call compute_rankings() first")
        
        cand_node = f"{self.candidate_prefix}{candidate_id}"
        
        if cand_node not in self.graph:
            raise ValueError(f"Candidate {candidate_id} not found in graph")
        
        # Get candidate's rank and score
        rank_row = self.candidate_rankings[
            self.candidate_rankings['candidate_id'] == candidate_id
        ]
        rank = int(rank_row['rank'].values[0])
        score = float(rank_row['score'].values[0])
        
        # Get incoming edges (from skills)
        incoming_skills = []
        for skill_node, _, data in self.graph.in_edges(cand_node, data=True):
            skill_name = skill_node.replace(self.skill_prefix, "")
            proficiency = data['weight']
            
            # Get importance from job → skill edge
            job_edge = self.graph.get_edge_data(self.job_node, skill_node)
            importance = job_edge['weight'] if job_edge else 0.0
            
            # Get skill's PageRank score (how much flow it received)
            skill_score = self.pagerank_scores[skill_node]
            
            incoming_skills.append({
                'skill': skill_name,
                'proficiency': proficiency,
                'importance': importance,
                'skill_flow': skill_score,
                'contribution': proficiency * skill_score  # Approximate contribution
            })
        
        # Sort by contribution
        incoming_skills.sort(key=lambda x: x['contribution'], reverse=True)
        
        # Get missing skills (gaps)
        job_skills = [node.replace(self.skill_prefix, "") 
                     for node in self.graph.nodes() 
                     if node.startswith(self.skill_prefix)]
        candidate_skills = [s['skill'] for s in incoming_skills]
        missing_skills = [s for s in job_skills if s not in candidate_skills]
        
        # Get importance of missing skills
        missing_details = []
        for skill in missing_skills:
            skill_node = f"{self.skill_prefix}{skill}"
            job_edge = self.graph.get_edge_data(self.job_node, skill_node)
            importance = job_edge['weight'] if job_edge else 0.0
            missing_details.append({
                'skill': skill,
                'importance': importance
            })
        missing_details.sort(key=lambda x: x['importance'], reverse=True)
        
        return {
            'candidate_id': candidate_id,
            'rank': rank,
            'score': score,
            'top_skills': incoming_skills[:top_k_skills],
            'missing_skills': missing_details,
            'skill_coverage': len(candidate_skills) / len(job_skills) if job_skills else 0.0
        }
    
    def get_graph_stats(self) -> Dict:
        """Get statistics about the constructed graph."""
        if self.graph is None:
            raise ValueError("Must call build_graph() first")
        
        num_candidates = len([n for n in self.graph.nodes() 
                             if n.startswith(self.candidate_prefix)])
        num_skills = len([n for n in self.graph.nodes() 
                         if n.startswith(self.skill_prefix)])
        
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_candidates': num_candidates,
            'num_skills': num_skills,
            'avg_skills_per_candidate': self.graph.number_of_edges() / num_candidates if num_candidates > 0 else 0
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def rank_candidates(
    job_skills: Dict[str, float],
    candidates: Dict[str, Dict[str, float]],
    damping: float = 0.85,
    normalize: bool = True
) -> pd.DataFrame:
    """
    One-shot function to rank candidates.
    
    Parameters:
    -----------
    job_skills : Dict[str, float]
        skill_name → importance
    candidates : Dict[str, Dict[str, float]]
        candidate_id → {skill_name → proficiency}
    damping : float
        PageRank damping factor
    normalize : bool
        Normalize edge weights
        
    Returns:
    --------
    pd.DataFrame with rankings
    """
    ranker = GraphBasedATSRanker(damping=damping, normalize_edges=normalize)
    ranker.build_graph(job_skills, candidates)
    return ranker.compute_rankings()


def explain_candidate(
    job_skills: Dict[str, float],
    candidates: Dict[str, Dict[str, float]],
    candidate_id: str,
    top_k_skills: int = 5
) -> Dict:
    """
    One-shot function to explain a candidate's ranking.
    """
    ranker = GraphBasedATSRanker()
    ranker.build_graph(job_skills, candidates)
    ranker.compute_rankings()
    return ranker.explain_ranking(candidate_id, top_k_skills)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example data
    job_skills = {
        "Python": 0.9,
        "SQL": 0.7,
        "Leadership": 0.5,
        "Communication": 0.6
    }
    
    candidates = {
        "Alice": {
            "Python": 0.8,
            "SQL": 0.6,
            "Leadership": 0.4,
            "Communication": 0.7
        },
        "Bob": {
            "Python": 0.5,
            "SQL": 0.9,
            "Leadership": 0.7
            # Missing Communication
        },
        "Charlie": {
            "Python": 0.9,
            "SQL": 0.8,
            # Missing Leadership and Communication
        },
        "Diana": {
            "Python": 0.7,
            "SQL": 0.7,
            "Leadership": 0.6,
            "Communication": 0.8
        }
    }
    
    # Create ranker
    ranker = GraphBasedATSRanker()
    
    # Build graph
    ranker.build_graph(job_skills, candidates)
    print("Graph Statistics:")
    print(ranker.get_graph_stats())
    print()
    
    # Compute rankings
    rankings = ranker.compute_rankings()
    print("Rankings:")
    print(rankings)
    print()
    
    # Explain top candidate
    top_candidate = rankings.iloc[0]['candidate_id']
    explanation = ranker.explain_ranking(top_candidate)
    
    print(f"\nExplanation for {top_candidate} (Rank {explanation['rank']}):")
    print(f"Score: {explanation['score']:.6f}")
    print(f"Skill Coverage: {explanation['skill_coverage']:.1%}")
    print("\nTop Contributing Skills:")
    for skill in explanation['top_skills']:
        print(f"  - {skill['skill']}: proficiency={skill['proficiency']:.2f}, "
              f"importance={skill['importance']:.2f}, "
              f"contribution={skill['contribution']:.6f}")
    
    if explanation['missing_skills']:
        print("\nMissing Skills:")
        for skill in explanation['missing_skills']:
            print(f"  - {skill['skill']}: importance={skill['importance']:.2f}")
