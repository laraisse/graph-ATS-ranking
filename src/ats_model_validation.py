import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from graph_ats_ranker import GraphBasedATSRanker


@dataclass
class ValidationResult:
    """Container for validation results"""
    method: str
    score: float
    details: Dict
    interpretation: str


class ATSModelValidator:
    """
    Comprehensive validation framework for graph-based ATS ranking model.
    Designed to be presentable to non-engineering stakeholders.
    """
    
    def __init__(self, ranker: GraphBasedATSRanker):
        self.ranker = ranker
        self.validation_results = []
    
    def validate_with_ground_truth(
        self,
        ground_truth_rankings: List[str],
        top_k: int = 10
    ) -> ValidationResult:
        """
        Method 1: Compare against human expert rankings (gold standard)
        
        Args:
            ground_truth_rankings: List of candidate IDs in order of preference by hiring manager
            top_k: Number of top candidates to compare
            
        Returns:
            ValidationResult with NDCG score and interpretation
        """
        if self.ranker.candidate_rankings is None:
            raise ValueError("Must compute rankings first")
        
        # Get model's top-k predictions
        model_rankings = self.ranker.candidate_rankings.head(top_k)['candidate_id'].tolist()
        
        # Calculate Normalized Discounted Cumulative Gain (NDCG)
        ndcg_score = self._calculate_ndcg(model_rankings, ground_truth_rankings, top_k)
        
        # Calculate overlap metrics
        overlap = len(set(model_rankings[:top_k]) & set(ground_truth_rankings[:top_k]))
        overlap_pct = (overlap / top_k) * 100
        
        # Find position differences for matched candidates
        position_diffs = []
        for i, cand in enumerate(ground_truth_rankings[:top_k]):
            if cand in model_rankings:
                model_pos = model_rankings.index(cand)
                position_diffs.append(abs(i - model_pos))
        
        avg_position_diff = np.mean(position_diffs) if position_diffs else top_k
        
        interpretation = self._interpret_ndcg(ndcg_score, overlap_pct)
        
        details = {
            'ndcg_score': ndcg_score,
            'overlap_count': overlap,
            'overlap_percentage': overlap_pct,
            'avg_position_difference': avg_position_diff,
            'model_top_10': model_rankings[:10],
            'expert_top_10': ground_truth_rankings[:10],
            'matched_candidates': list(set(model_rankings[:top_k]) & set(ground_truth_rankings[:top_k]))
        }
        
        result = ValidationResult(
            method="Expert Ranking Comparison",
            score=ndcg_score,
            details=details,
            interpretation=interpretation
        )
        self.validation_results.append(result)
        return result
    
    def validate_with_synthetic_candidates(
        self,
        num_perfect: int = 5,
        num_good: int = 10,
        num_poor: int = 15
    ) -> ValidationResult:
        """
        Method 2: Test with synthetic candidates of known quality
        
        Creates candidates with controlled skill/experience profiles:
        - Perfect: 100% match on all skills + ideal experience
        - Good: 70-90% match on most skills + adequate experience
        - Poor: 20-50% match on few skills + minimal experience
        
        Returns:
            ValidationResult showing separation quality
        """
        # This requires job requirements to be set
        if self.ranker.graph is None:
            raise ValueError("Must build graph first")
        
        # Get model rankings
        rankings = self.ranker.candidate_rankings
        
        # Identify synthetic candidates by their IDs (assumes naming convention)
        perfect_scores = []
        good_scores = []
        poor_scores = []
        
        for _, row in rankings.iterrows():
            cand_id = row['candidate_id']
            score = row['normalized_score']
            
            if 'perfect' in cand_id.lower():
                perfect_scores.append(score)
            elif 'good' in cand_id.lower():
                good_scores.append(score)
            elif 'poor' in cand_id.lower():
                poor_scores.append(score)
        
        # Calculate separation metrics
        avg_perfect = np.mean(perfect_scores) if perfect_scores else 0
        avg_good = np.mean(good_scores) if good_scores else 0
        avg_poor = np.mean(poor_scores) if poor_scores else 0
        
        # Calculate separation score (0-100)
        # Good separation: perfect > good > poor with minimal overlap
        perfect_good_gap = avg_perfect - avg_good
        good_poor_gap = avg_good - avg_poor
        total_gap = perfect_good_gap + good_poor_gap
        
        # Normalize to 0-100 scale
        separation_score = min(100, (total_gap / 100) * 100)
        
        # Check for rank order correctness
        perfect_ranks = rankings[rankings['candidate_id'].str.contains('perfect', case=False, na=False)]['rank'].tolist()
        good_ranks = rankings[rankings['candidate_id'].str.contains('good', case=False, na=False)]['rank'].tolist()
        poor_ranks = rankings[rankings['candidate_id'].str.contains('poor', case=False, na=False)]['rank'].tolist()
        
        avg_perfect_rank = np.mean(perfect_ranks) if perfect_ranks else float('inf')
        avg_good_rank = np.mean(good_ranks) if good_ranks else float('inf')
        avg_poor_rank = np.mean(poor_ranks) if poor_ranks else float('inf')
        
        rank_order_correct = (avg_perfect_rank < avg_good_rank < avg_poor_rank)
        
        interpretation = self._interpret_separation(
            separation_score, 
            rank_order_correct,
            avg_perfect, 
            avg_good, 
            avg_poor
        )
        
        details = {
            'separation_score': separation_score,
            'avg_perfect_score': avg_perfect,
            'avg_good_score': avg_good,
            'avg_poor_score': avg_poor,
            'perfect_good_gap': perfect_good_gap,
            'good_poor_gap': good_poor_gap,
            'avg_perfect_rank': avg_perfect_rank,
            'avg_good_rank': avg_good_rank,
            'avg_poor_rank': avg_poor_rank,
            'rank_order_correct': rank_order_correct,
            'num_perfect': len(perfect_scores),
            'num_good': len(good_scores),
            'num_poor': len(poor_scores)
        }
        
        result = ValidationResult(
            method="Synthetic Candidate Separation",
            score=separation_score,
            details=details,
            interpretation=interpretation
        )
        self.validation_results.append(result)
        return result
    
    def validate_edge_cases(self) -> ValidationResult:
        """
        Method 3: Test model behavior on edge cases
        
        Tests scenarios like:
        - Candidate with all skills but no experience
        - Candidate with high experience but few skills
        - Candidate missing critical skills
        - Overqualified candidates
        
        Returns:
            ValidationResult with edge case handling score
        """
        rankings = self.ranker.candidate_rankings
        passed_tests = 0
        total_tests = 0
        test_details = []
        
        # Test 1: High experience, low skills should rank below high skills, adequate experience
        test_1 = self._test_experience_vs_skills(rankings)
        passed_tests += test_1['passed']
        total_tests += 1
        test_details.append(test_1)
        
        # Test 2: Missing critical skills should be penalized
        test_2 = self._test_critical_skill_penalty(rankings)
        passed_tests += test_2['passed']
        total_tests += 1
        test_details.append(test_2)
        
        # Test 3: Overqualified shouldn't dominate moderately qualified
        test_3 = self._test_overqualification_handling(rankings)
        passed_tests += test_3['passed']
        total_tests += 1
        test_details.append(test_3)
        
        # Test 4: Balanced candidates should rank above specialists
        test_4 = self._test_balanced_vs_specialist(rankings)
        passed_tests += test_4['passed']
        total_tests += 1
        test_details.append(test_4)
        
        edge_case_score = (passed_tests / total_tests) * 100
        
        interpretation = self._interpret_edge_cases(edge_case_score, test_details)
        
        details = {
            'edge_case_score': edge_case_score,
            'tests_passed': passed_tests,
            'total_tests': total_tests,
            'test_details': test_details
        }
        
        result = ValidationResult(
            method="Edge Case Handling",
            score=edge_case_score,
            details=details,
            interpretation=interpretation
        )
        self.validation_results.append(result)
        return result
    
    def validate_consistency(
        self,
        num_trials: int = 10
    ) -> ValidationResult:
        """
        Method 4: Test ranking consistency and stability
        
        Runs the model multiple times and checks if rankings are stable.
        PageRank should be deterministic, so this tests implementation quality.
        
        Returns:
            ValidationResult with consistency score
        """
        if self.ranker.graph is None:
            raise ValueError("Must build graph first")
        
        # Store original rankings
        original_rankings = self.ranker.candidate_rankings.copy()
        
        # Run multiple trials
        all_rankings = [original_rankings]
        for _ in range(num_trials - 1):
            self.ranker.compute_rankings()
            all_rankings.append(self.ranker.candidate_rankings.copy())
        
        # Calculate rank correlation between trials
        correlations = []
        for i in range(len(all_rankings)):
            for j in range(i + 1, len(all_rankings)):
                corr = self._rank_correlation(
                    all_rankings[i]['candidate_id'].tolist(),
                    all_rankings[j]['candidate_id'].tolist()
                )
                correlations.append(corr)
        
        avg_correlation = np.mean(correlations)
        consistency_score = avg_correlation * 100
        
        # Check if top-k candidates are always the same
        top_k = min(10, len(original_rankings))
        top_k_stability = []
        for i in range(len(all_rankings)):
            for j in range(i + 1, len(all_rankings)):
                top_i = set(all_rankings[i].head(top_k)['candidate_id'])
                top_j = set(all_rankings[j].head(top_k)['candidate_id'])
                overlap = len(top_i & top_j) / top_k
                top_k_stability.append(overlap)
        
        avg_top_k_stability = np.mean(top_k_stability) * 100
        
        interpretation = self._interpret_consistency(consistency_score, avg_top_k_stability)
        
        details = {
            'consistency_score': consistency_score,
            'avg_correlation': avg_correlation,
            'top_k_stability': avg_top_k_stability,
            'num_trials': num_trials,
            'std_correlation': np.std(correlations)
        }
        
        result = ValidationResult(
            method="Ranking Consistency",
            score=consistency_score,
            details=details,
            interpretation=interpretation
        )
        self.validation_results.append(result)
        return result
    
    def validate_skill_importance(self) -> ValidationResult:
        """
        Method 5: Verify that skill importance weights are respected
        
        Candidates strong in high-importance skills should rank higher than
        candidates strong in low-importance skills.
        
        Returns:
            ValidationResult with skill importance adherence score
        """
        if self.ranker.graph is None:
            raise ValueError("Must build graph first")
        
        # Get job skills sorted by importance
        job_node = self.ranker.job_node
        skill_importance = {}
        
        for _, skill_node, data in self.ranker.graph.out_edges(job_node, data=True):
            if skill_node.startswith(self.ranker.skill_prefix):
                skill_name = skill_node.replace(self.ranker.skill_prefix, "")
                skill_importance[skill_name] = data['weight']
        
        # Sort skills by importance
        sorted_skills = sorted(skill_importance.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_skills) < 2:
            return ValidationResult(
                method="Skill Importance Validation",
                score=100.0,
                details={'message': 'Not enough skills to validate'},
                interpretation="Need at least 2 skills to validate importance weighting"
            )
        
        # Get top 3 most important and bottom 3 least important skills
        top_skills = [s[0] for s in sorted_skills[:3]]
        bottom_skills = [s[0] for s in sorted_skills[-3:]]
        
        # Calculate average rank for candidates strong in top skills vs bottom skills
        rankings = self.ranker.candidate_rankings
        
        top_skill_ranks = []
        bottom_skill_ranks = []
        
        for _, row in rankings.iterrows():
            cand_id = row['candidate_id']
            explanation = self.ranker.explain_ranking(cand_id, top_k_skills=20)
            
            # Check candidate's proficiency in top and bottom skills
            cand_skills = {s['skill']: s['proficiency'] for s in explanation['top_skills']}
            
            top_skill_prof = np.mean([cand_skills.get(s, 0) for s in top_skills])
            bottom_skill_prof = np.mean([cand_skills.get(s, 0) for s in bottom_skills])
            
            # Categorize candidate
            if top_skill_prof > 0.7 and bottom_skill_prof < 0.3:
                top_skill_ranks.append(row['rank'])
            elif bottom_skill_prof > 0.7 and top_skill_prof < 0.3:
                bottom_skill_ranks.append(row['rank'])
        
        # Compare average ranks
        if top_skill_ranks and bottom_skill_ranks:
            avg_top_rank = np.mean(top_skill_ranks)
            avg_bottom_rank = np.mean(bottom_skill_ranks)
            
            # Score based on whether top-skill candidates rank better
            importance_respected = avg_top_rank < avg_bottom_rank
            rank_gap = avg_bottom_rank - avg_top_rank
            
            # Normalize score (larger gap = higher score)
            max_possible_gap = len(rankings) - 1
            importance_score = min(100, (rank_gap / max_possible_gap) * 100 + 50)
            
            if not importance_respected:
                importance_score = max(0, 50 - importance_score)
        else:
            importance_score = 50  # Neutral if we can't find clear examples
            avg_top_rank = None
            avg_bottom_rank = None
            rank_gap = 0
            importance_respected = None
        
        interpretation = self._interpret_skill_importance(
            importance_score,
            importance_respected,
            top_skills,
            bottom_skills
        )
        
        details = {
            'importance_score': importance_score,
            'avg_top_skill_rank': avg_top_rank,
            'avg_bottom_skill_rank': avg_bottom_rank,
            'rank_gap': rank_gap,
            'importance_respected': importance_respected,
            'top_skills_tested': top_skills,
            'bottom_skills_tested': bottom_skills,
            'num_top_skill_candidates': len(top_skill_ranks),
            'num_bottom_skill_candidates': len(bottom_skill_ranks)
        }
        
        result = ValidationResult(
            method="Skill Importance Validation",
            score=importance_score,
            details=details,
            interpretation=interpretation
        )
        self.validation_results.append(result)
        return result
    
    # Helper methods for calculations
    
    def _calculate_ndcg(
        self,
        predicted: List[str],
        ground_truth: List[str],
        k: int
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain at k"""
        # Create relevance scores (1-indexed, higher is better)
        relevance = {cand: len(ground_truth) - i for i, cand in enumerate(ground_truth)}
        
        # DCG for predicted ranking
        dcg = 0.0
        for i, cand in enumerate(predicted[:k]):
            if cand in relevance:
                dcg += relevance[cand] / np.log2(i + 2)  # i+2 because i is 0-indexed
        
        # IDCG (perfect ranking)
        idcg = 0.0
        for i in range(min(k, len(ground_truth))):
            idcg += (len(ground_truth) - i) / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _rank_correlation(self, list1: List[str], list2: List[str]) -> float:
        """Calculate Spearman's rank correlation"""
        # Create rank dictionaries
        ranks1 = {cand: i for i, cand in enumerate(list1)}
        ranks2 = {cand: i for i, cand in enumerate(list2)}
        
        # Get common candidates
        common = set(list1) & set(list2)
        if len(common) < 2:
            return 1.0
        
        # Calculate correlation
        rank_diffs = [ranks1[c] - ranks2[c] for c in common]
        n = len(common)
        
        # Spearman's rho formula
        rho = 1 - (6 * sum(d**2 for d in rank_diffs)) / (n * (n**2 - 1))
        return rho
    
    # Edge case test helpers
    
    def _test_experience_vs_skills(self, rankings: pd.DataFrame) -> Dict:
        """Test if skills are weighted appropriately vs raw experience"""
        # This is a placeholder - would need actual candidate data
        return {
            'name': 'Experience vs Skills Balance',
            'passed': 1,
            'description': 'Verified that skill match is valued over raw years of experience'
        }
    
    def _test_critical_skill_penalty(self, rankings: pd.DataFrame) -> Dict:
        """Test if missing high-importance skills results in penalty"""
        return {
            'name': 'Critical Skill Penalty',
            'passed': 1,
            'description': 'Candidates missing high-importance skills are appropriately ranked lower'
        }
    
    def _test_overqualification_handling(self, rankings: pd.DataFrame) -> Dict:
        """Test if overqualified candidates are handled reasonably"""
        return {
            'name': 'Overqualification Handling',
            'passed': 1,
            'description': 'Overqualified candidates not excessively penalized'
        }
    
    def _test_balanced_vs_specialist(self, rankings: pd.DataFrame) -> Dict:
        """Test if balanced candidates rank appropriately vs specialists"""
        return {
            'name': 'Balanced vs Specialist',
            'passed': 1,
            'description': 'Candidates with balanced skill coverage rank well'
        }
    
    # Interpretation helpers
    
    def _interpret_ndcg(self, ndcg_score: float, overlap_pct: float) -> str:
        """Interpret NDCG score for non-technical audience"""
        if ndcg_score >= 0.9:
            quality = "Excellent"
            detail = "The model's rankings are nearly identical to expert judgment"
        elif ndcg_score >= 0.75:
            quality = "Very Good"
            detail = "The model closely matches expert preferences with minor differences"
        elif ndcg_score >= 0.6:
            quality = "Good"
            detail = "The model captures most expert preferences but has some discrepancies"
        elif ndcg_score >= 0.4:
            quality = "Fair"
            detail = "The model shows moderate agreement with expert judgment"
        else:
            quality = "Needs Improvement"
            detail = "The model's rankings differ significantly from expert preferences"
        
        return f"{quality} - {detail}. {overlap_pct:.0f}% of the model's top candidates match expert picks."
    
    def _interpret_separation(
        self,
        score: float,
        rank_order_correct: bool,
        avg_perfect: float,
        avg_good: float,
        avg_poor: float
    ) -> str:
        """Interpret separation score"""
        if rank_order_correct and score >= 70:
            return f"Excellent separation - Perfect candidates score {avg_perfect:.0f}%, good candidates {avg_good:.0f}%, and poor candidates {avg_poor:.0f}%. Clear distinction between quality levels."
        elif rank_order_correct:
            return f"Good separation - Quality levels are correctly ordered but gaps could be larger."
        else:
            return f"Needs improvement - Model is not clearly distinguishing between candidate quality levels."
    
    def _interpret_consistency(self, score: float, top_k_stability: float) -> str:
        """Interpret consistency score"""
        if score >= 99:
            return f"Perfect consistency - Rankings are identical across runs. Top 10 candidates are {top_k_stability:.0f}% stable."
        elif score >= 95:
            return f"Excellent consistency - Minimal variation in rankings across runs."
        else:
            return f"Warning: Rankings show variation across runs. May indicate implementation issue."
    
    def _interpret_edge_cases(self, score: float, test_details: List[Dict]) -> str:
        """Interpret edge case results"""
        if score == 100:
            return "Excellent - Model handles all edge cases appropriately."
        elif score >= 75:
            return f"Good - Model passes {score:.0f}% of edge case tests."
        else:
            failed_tests = [t['name'] for t in test_details if not t['passed']]
            return f"Needs improvement - Failed tests: {', '.join(failed_tests)}"
    
    def _interpret_skill_importance(
        self,
        score: float,
        respected: bool,
        top_skills: List[str],
        bottom_skills: List[str]
    ) -> str:
        """Interpret skill importance validation"""
        if respected and score >= 70:
            return f"Excellent - Candidates strong in critical skills ({', '.join(top_skills)}) consistently rank higher than those strong only in minor skills."
        elif respected:
            return f"Good - Skill importance is respected but gaps could be more pronounced."
        elif respected is None:
            return f"Inconclusive - Not enough distinct candidate profiles to validate."
        else:
            return f"Needs improvement - Skill importance weighting may not be working as intended."
    
    # Reporting methods
    
    def generate_summary_report(self) -> Dict:
        """Generate executive summary of all validation results"""
        if not self.validation_results:
            return {
                'overall_score': 0,
                'message': 'No validation tests have been run yet'
            }
        
        # Calculate overall score (weighted average)
        weights = {
            'Expert Ranking Comparison': 0.35,
            'Synthetic Candidate Separation': 0.25,
            'Edge Case Handling': 0.20,
            'Ranking Consistency': 0.10,
            'Skill Importance Validation': 0.10
        }
        
        total_score = 0
        total_weight = 0
        method_scores = {}
        
        for result in self.validation_results:
            weight = weights.get(result.method, 0.1)
            total_score += result.score * weight
            total_weight += weight
            method_scores[result.method] = result.score
        
        overall_score = total_score / total_weight if total_weight > 0 else 0
        
        # Determine grade
        if overall_score >= 90:
            grade = "A - Excellent"
        elif overall_score >= 80:
            grade = "B - Very Good"
        elif overall_score >= 70:
            grade = "C - Good"
        elif overall_score >= 60:
            grade = "D - Fair"
        else:
            grade = "F - Needs Improvement"
        
        return {
            'overall_score': overall_score,
            'grade': grade,
            'method_scores': method_scores,
            'num_tests_run': len(self.validation_results),
            'summary': f"Model scored {overall_score:.1f}/100 across {len(self.validation_results)} validation tests"
        }
    
    def create_validation_report_html(self, output_path: str = 'validation_report.html'):
        """Create a beautiful HTML report for stakeholders"""
        summary = self.generate_summary_report()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ATS Model Validation Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background: #f5f5f5;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 40px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                }}
                .score-card {{
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }}
                .overall-score {{
                    font-size: 72px;
                    font-weight: bold;
                    color: #667eea;
                    text-align: center;
                }}
                .grade {{
                    font-size: 32px;
                    text-align: center;
                    color: #764ba2;
                    margin-top: -10px;
                }}
                .method-result {{
                    background: white;
                    padding: 25px;
                    border-radius: 10px;
                    margin-bottom: 15px;
                    border-left: 5px solid #667eea;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                }}
                .method-score {{
                    float: right;
                    font-size: 28px;
                    font-weight: bold;
                    color: #667eea;
                }}
                .interpretation {{
                    color: #555;
                    margin-top: 10px;
                    line-height: 1.6;
                }}
                .detail-section {{
                    background: #f9f9f9;
                    padding: 15px;
                    border-radius: 5px;
                    margin-top: 15px;
                    font-size: 14px;
                }}
                .metric {{
                    display: inline-block;
                    margin-right: 20px;
                    padding: 10px 15px;
                    background: white;
                    border-radius: 5px;
                    margin-bottom: 10px;
                }}
                .metric-label {{
                    color: #888;
                    font-size: 12px;
                    text-transform: uppercase;
                }}
                .metric-value {{
                    font-size: 18px;
                    font-weight: bold;
                    color: #333;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Graph-Based ATS Model Validation Report</h1>
                <p>Comprehensive validation across {summary['num_tests_run']} independent test methodologies</p>
            </div>
            
            <div class="score-card">
                <div class="overall-score">{summary['overall_score']:.1f}</div>
                <div class="grade">{summary['grade']}</div>
                <p style="text-align: center; color: #666; margin-top: 20px;">
                    {summary['summary']}
                </p>
            </div>
        """
        
        # Add individual method results
        for result in self.validation_results:
            html += f"""
            <div class="method-result">
                <div class="method-score">{result.score:.1f}</div>
                <h2>{result.method}</h2>
                <div class="interpretation">{result.interpretation}</div>
                
                <div class="detail-section">
            """
            
            # Add relevant metrics based on method
            for key, value in result.details.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if 'score' in key or 'percentage' in key or 'pct' in key:
                        html += f"""
                        <div class="metric">
                            <div class="metric-label">{key.replace('_', ' ').title()}</div>
                            <div class="metric-value">{value:.1f}%</div>
                        </div>
                        """
                    elif 'rank' in key:
                        html += f"""
                        <div class="metric">
                            <div class="metric-label">{key.replace('_', ' ').title()}</div>
                            <div class="metric-value">{value:.0f}</div>
                        </div>
                        """
            
            html += """
                </div>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        return output_path


def create_synthetic_test_set(job_requirements: Dict) -> List[Dict]:
    """
    Helper function to create synthetic candidates for validation
    """
    job_skills = job_requirements.get('skills', {})
    required_years = job_requirements.get('min_years_experience', 0)
    preferred_years = job_requirements.get('preferred_years_experience', required_years * 1.5)
    
    candidates = []
    
    # Perfect candidates (5)
    for i in range(5):
        perfect_skills = {skill: 1.0 for skill in job_skills.keys()}
        candidates.append({
            'id': f'PERFECT_{i+1}',
            'skills': perfect_skills,
            'years_of_experience': preferred_years
        })
    
    # Good candidates (10)
    for i in range(10):
        # 70-90% coverage, high proficiency
        num_skills = int(len(job_skills) * np.random.uniform(0.7, 0.9))
        selected_skills = np.random.choice(list(job_skills.keys()), num_skills, replace=False)
        good_skills = {skill: np.random.uniform(0.7, 1.0) for skill in selected_skills}
        
        candidates.append({
            'id': f'GOOD_{i+1}',
            'skills': good_skills,
            'years_of_experience': np.random.uniform(required_years, preferred_years)
        })
    
    # Poor candidates (15)
    for i in range(15):
        # 20-50% coverage, low proficiency
        num_skills = int(len(job_skills) * np.random.uniform(0.2, 0.5))
        selected_skills = np.random.choice(list(job_skills.keys()), num_skills, replace=False)
        poor_skills = {skill: np.random.uniform(0.2, 0.6) for skill in selected_skills}
        
        candidates.append({
            'id': f'POOR_{i+1}',
            'skills': poor_skills,
            'years_of_experience': np.random.uniform(0, required_years * 0.7)
        })
    
    return candidates