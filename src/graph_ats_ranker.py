import networkx as nx
from typing import Dict, List, Optional
import pandas as pd


class GraphBasedATSRanker:
    def __init__(
            self,
            damping: float = 0.85,
            tolerance: float = 1e-6,
            max_iterations: int = 100,
            normalize_edges: bool = True,
            experience_weight: float = 0.3,
            experience_mode: str = 'both'  # 'boost', 'direct', or 'both'
    ):
        self.damping = damping
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.normalize_edges = normalize_edges
        self.experience_weight = experience_weight
        self.experience_mode = experience_mode

        self.graph = None
        self.job_node = "JOB"
        self.skill_prefix = "SKILL_"
        self.candidate_prefix = "CAND_"
        self.perfect_candidate_node = "PERFECT_CANDIDATE"

        self.pagerank_scores = None
        self.candidate_rankings = None
        self.perfect_candidate_score = None

    def _calculate_experience_boost(
            self,
            candidate_years: float,
            required_years: float,
            preferred_years: Optional[float] = None
    ) -> float:
        if preferred_years is None:
            preferred_years = required_years * 1.5 if required_years > 0 else 5.0

        if required_years == 0:
            # No experience requirement - give small bonus for any experience
            if candidate_years == 0:
                return 1.0
            elif candidate_years <= 3:
                return 1.0 + 0.1 * (candidate_years / 3)
            else:
                return 1.1

        if candidate_years < required_years:
            # Penalty for insufficient experience (50% to 100%)
            ratio = candidate_years / required_years if required_years > 0 else 0
            return 0.5 + 0.5 * ratio
        elif candidate_years < preferred_years:
            # Bonus for meeting requirements (100% to 130%)
            excess = candidate_years - required_years
            range_size = preferred_years - required_years
            ratio = excess / range_size if range_size > 0 else 1
            return 1.0 + 0.3 * ratio
        else:
            # Diminishing returns for exceeding preferred (130% to 150%)
            excess = min(candidate_years - preferred_years, preferred_years)
            ratio = excess / preferred_years if preferred_years > 0 else 1
            return 1.3 + 0.2 * ratio

    def _calculate_experience_match_score(
            self,
            candidate_years: float,
            required_years: float,
            preferred_years: Optional[float] = None
    ) -> float:
        if preferred_years is None:
            preferred_years = required_years * 1.5 if required_years > 0 else 5.0

        if required_years == 0:
            # No experience requirement - small bonus for any experience
            if candidate_years == 0:
                return 0.5
            elif candidate_years <= 3:
                return 0.5 + 0.2 * (candidate_years / 3)
            else:
                return 0.7

        if candidate_years < required_years * 0.5:
            # Severely underqualified
            return 0.2
        elif candidate_years < required_years:
            # Somewhat underqualified
            ratio = candidate_years / required_years
            return 0.2 + 0.5 * ratio
        elif candidate_years <= preferred_years:
            # Well qualified
            return 0.7 + 0.3 * ((candidate_years - required_years) /
                                (preferred_years - required_years))
        else:
            # Overqualified (slight diminishing return)
            excess = min(candidate_years - preferred_years, preferred_years * 2)
            max_excess = preferred_years * 2
            return 1.0 - 0.1 * (excess / max_excess)

    def build_graph(
            self,
            job_requirements: Dict,
            candidates: List[Dict]
    ) -> nx.DiGraph:
        G = nx.DiGraph()

        # Extract job requirements
        job_skills = job_requirements.get('skills', {})
        required_years = job_requirements.get('min_years_experience', 0)
        preferred_years = job_requirements.get('preferred_years_experience', None)

        # Add job node
        G.add_node(
            self.job_node,
            node_type='job',
            required_years=required_years,
            preferred_years=preferred_years
        )

        # Add perfect candidate node
        G.add_node(
            self.perfect_candidate_node,
            node_type='perfect_candidate',
            years_experience=preferred_years if preferred_years else required_years
        )

        # Add skill nodes and edges from job
        for skill, importance in job_skills.items():
            skill_node = f"{self.skill_prefix}{skill}"
            G.add_node(skill_node, node_type='skill', skill_name=skill)
            G.add_edge(self.job_node, skill_node, weight=importance)

            # Connect perfect candidate to all required skills with maximum proficiency
            # Perfect candidate gets maximum experience boost
            perfect_years = preferred_years if preferred_years else required_years
            perfect_boost = 1.0
            if self.experience_mode in ['boost', 'both']:
                perfect_boost = self._calculate_experience_boost(
                    perfect_years, required_years, preferred_years
                )
                perfect_boost = 1.0 + (perfect_boost - 1.0) * self.experience_weight

            # Add edge with maximum proficiency (1.0) and experience boost
            G.add_edge(skill_node, self.perfect_candidate_node,
                       weight=1.0 * perfect_boost)

        # Add candidate nodes and edges to skills
        for candidate in candidates:
            cand_id = candidate['id']
            cand_node = f"{self.candidate_prefix}{cand_id}"
            years = candidate.get('years_of_experience', 0)
            skills = candidate.get('skills', {})

            G.add_node(
                cand_node,
                node_type='candidate',
                candidate_id=cand_id,
                years_experience=years
            )

            # Calculate experience boost if applicable
            experience_boost = 1.0
            if self.experience_mode in ['boost', 'both']:
                experience_boost = self._calculate_experience_boost(
                    years, required_years, preferred_years
                )
                # Apply experience weight to modulate the boost effect
                experience_boost = 1.0 + (experience_boost - 1.0) * self.experience_weight

            # Add edges from skills to candidate
            for skill, proficiency in skills.items():
                skill_node = f"{self.skill_prefix}{skill}"

                # Only add edge if skill exists in job requirements AND proficiency > 0
                if skill_node in G and proficiency > 0:
                    # Apply experience boost to proficiency
                    adjusted_proficiency = proficiency * experience_boost
                    G.add_edge(skill_node, cand_node, weight=adjusted_proficiency)

            # Add direct job→candidate edge based on experience match
            if self.experience_mode in ['direct', 'both']:
                experience_match = self._calculate_experience_match_score(
                    years, required_years, preferred_years
                )
                # Scale by experience weight
                edge_weight = experience_match * self.experience_weight
                if edge_weight > 0:
                    G.add_edge(self.job_node, cand_node,
                               weight=edge_weight,
                               edge_type='experience_match')

        # Add direct experience edge to perfect candidate
        if self.experience_mode in ['direct', 'both']:
            perfect_years = preferred_years if preferred_years else required_years
            perfect_exp_match = self._calculate_experience_match_score(
                perfect_years, required_years, preferred_years
            )
            edge_weight = perfect_exp_match * self.experience_weight
            if edge_weight > 0:
                G.add_edge(self.job_node, self.perfect_candidate_node,
                           weight=edge_weight,
                           edge_type='experience_match')

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

        # Store perfect candidate score for normalization
        self.perfect_candidate_score = self.pagerank_scores.get(
            self.perfect_candidate_node, 1.0
        )

        # Extract candidate scores
        candidate_scores = []
        for node, score in self.pagerank_scores.items():
            if node.startswith(self.candidate_prefix):
                candidate_id = node.replace(self.candidate_prefix, "")
                years_exp = self.graph.nodes[node].get('years_experience', 0)
                # Calculate normalized score (as percentage of perfect candidate)
                normalized_score = (
                                               score / self.perfect_candidate_score) * 100 if self.perfect_candidate_score > 0 else 0
                candidate_scores.append({
                    'candidate_id': candidate_id,
                    'score': score,
                    'normalized_score': normalized_score,
                    'years_experience': years_exp
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
            top_k_skills: int = 6
    ) -> Dict:
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
        normalized_score = float(rank_row['normalized_score'].values[0])
        years_exp = float(rank_row['years_experience'].values[0])

        # Get job requirements
        required_years = self.graph.nodes[self.job_node].get('required_years', 0)
        preferred_years = self.graph.nodes[self.job_node].get('preferred_years', None)

        # Get perfect candidate's skill data for normalization
        perfect_skills = {}
        for skill_node, _, data in self.graph.in_edges(self.perfect_candidate_node, data=True):
            if skill_node == self.job_node:
                continue
            skill_name = skill_node.replace(self.skill_prefix, "")
            proficiency = data['weight']
            skill_score = self.pagerank_scores[skill_node]
            perfect_skills[skill_name] = {
                'proficiency': proficiency,
                'skill_flow': skill_score,
                'contribution': proficiency * skill_score
            }

        # Get incoming edges (from skills)
        incoming_skills = []
        for skill_node, _, data in self.graph.in_edges(cand_node, data=True):
            if skill_node == self.job_node:
                # Skip direct job edge for now
                continue

            skill_name = skill_node.replace(self.skill_prefix, "")
            proficiency = data['weight']

            # Get importance from job → skill edge
            job_edge = self.graph.get_edge_data(self.job_node, skill_node)
            importance = job_edge['weight'] if job_edge else 0.0

            # Get skill's PageRank score (how much flow it received)
            skill_score = self.pagerank_scores[skill_node]

            # Calculate normalized values (as percentage of perfect candidate)
            normalized_proficiency = 0.0
            if skill_name in perfect_skills:
                if perfect_skills[skill_name]['proficiency'] > 0:
                    normalized_proficiency = (proficiency / perfect_skills[skill_name]['proficiency'])

            contribution = proficiency * skill_score
            incoming_skills.append({
                'skill': skill_name,
                'proficiency': normalized_proficiency,
                'importance': importance,
                'contribution': contribution
            })

        # Sort by contribution
        incoming_skills.sort(key=lambda x: x['contribution'], reverse=True)

        # Check for direct experience edge
        experience_contribution = 0.0
        job_edge = self.graph.get_edge_data(self.job_node, cand_node)
        if job_edge:
            experience_contribution = job_edge['weight'] * self.pagerank_scores[self.job_node]

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

        # Experience status
        experience_status = "meets requirement"
        if required_years > 0:
            if years_exp < required_years:
                experience_status = "below requirement"
            elif preferred_years and years_exp >= preferred_years:
                experience_status = "exceeds preferred"
            else:
                experience_status = "meets requirement"
        else:
            experience_status = "no requirement"

        return {
            'candidate_id': candidate_id,
            'rank': rank,
            'score': score,
            'normalized_score': normalized_score,
            'years_experience': years_exp,
            'required_years': required_years,
            'preferred_years': preferred_years,
            'experience_status': experience_status,
            'experience_contribution': experience_contribution,
            'top_skills': incoming_skills[:top_k_skills],
            'missing_skills': missing_details,
            'skill_coverage': len(candidate_skills) / len(job_skills) if job_skills else 0.0
        }

    def get_graph_stats(self) -> Dict:
        if self.graph is None:
            raise ValueError("Must call build_graph() first")

        num_candidates = len([n for n in self.graph.nodes()
                              if n.startswith(self.candidate_prefix)])
        num_skills = len([n for n in self.graph.nodes()
                          if n.startswith(self.skill_prefix)])

        # Count direct experience edges
        direct_exp_edges = sum(1 for _, _, data in self.graph.edges(data=True)
                               if data.get('edge_type') == 'experience_match')

        required_years = self.graph.nodes[self.job_node].get('required_years', 0)
        preferred_years = self.graph.nodes[self.job_node].get('preferred_years', None)

        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_candidates': num_candidates,
            'num_skills': num_skills,
            'required_years': required_years,
            'preferred_years': preferred_years,
            'direct_experience_edges': direct_exp_edges,
            'avg_skills_per_candidate': ( self.graph.number_of_edges() - direct_exp_edges - num_skills) / num_candidates if num_candidates > 0 else 0,
            'experience_mode': self.experience_mode,
            'experience_weight': self.experience_weight,
            'perfect_candidate_score': self.perfect_candidate_score
        }


def rank_candidates(
        job_requirements: Dict,
        candidates: List[Dict],
        damping: float = 0.85,
        normalize: bool = True,
        experience_weight: float = 0.3,
        experience_mode: str = 'both'
) -> pd.DataFrame:
    ranker = GraphBasedATSRanker(
        damping=damping,
        normalize_edges=normalize,
        experience_weight=experience_weight,
        experience_mode=experience_mode
    )
    ranker.build_graph(job_requirements, candidates)
    return ranker.compute_rankings()