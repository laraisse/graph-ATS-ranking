"""
Test Suite for Graph-Based ATS Ranker
======================================

Validates that the ranker correctly handles:
1. Different candidate profiles (complete, partial, specialist)
2. Skill gaps and their penalties
3. Complementarity vs pure vector matching
4. Edge cases (no skills, all skills, duplicate scores)
"""

import sys
sys.path.append('/home/claude')

from src.graph_ats_ranker import GraphBasedATSRanker, rank_candidates
import pandas as pd
import numpy as np


def test_basic_ranking():
    """Test 1: Basic ranking with clear winner"""
    print("=" * 70)
    print("TEST 1: Basic Ranking")
    print("=" * 70)
    
    job_skills = {
        "Python": 0.9,
        "SQL": 0.7,
        "Leadership": 0.5
    }
    
    candidates = {
        "Perfect": {"Python": 0.9, "SQL": 0.9, "Leadership": 0.9},
        "Good": {"Python": 0.7, "SQL": 0.7, "Leadership": 0.7},
        "Weak": {"Python": 0.3, "SQL": 0.3, "Leadership": 0.3}
    }
    
    rankings = rank_candidates(job_skills, candidates)
    print(rankings)
    
    assert rankings.iloc[0]['candidate_id'] == "Perfect", "Perfect should rank first"
    assert rankings.iloc[1]['candidate_id'] == "Good", "Good should rank second"
    assert rankings.iloc[2]['candidate_id'] == "Weak", "Weak should rank third"
    
    print("✅ PASSED\n")


def test_gap_penalty():
    """Test 2: Missing critical skills should be penalized"""
    print("=" * 70)
    print("TEST 2: Gap Penalty")
    print("=" * 70)
    
    job_skills = {
        "Python": 0.9,
        "SQL": 0.8,
        "Leadership": 0.7
    }
    
    candidates = {
        "Complete": {"Python": 0.7, "SQL": 0.7, "Leadership": 0.7},
        "MissingCritical": {"Python": 1.0, "SQL": 1.0},  # Missing Leadership
        "MissingLessCritical": {"Python": 0.7, "Leadership": 0.7}  # Missing SQL
    }
    
    rankings = rank_candidates(job_skills, candidates)
    print(rankings)
    print()
    
    ranker = GraphBasedATSRanker()
    ranker.build_graph(job_skills, candidates)
    ranker.compute_rankings()
    
    for cand_id in ["Complete", "MissingCritical", "MissingLessCritical"]:
        exp = ranker.explain_ranking(cand_id)
        print(f"{cand_id}:")
        print(f"  Rank: {exp['rank']}, Score: {exp['score']:.6f}, Coverage: {exp['skill_coverage']:.1%}")
        if exp['missing_skills']:
            print(f"  Missing: {[s['skill'] for s in exp['missing_skills']]}")
        print()
    
    # Complete profile should win despite lower scores
    assert rankings.iloc[0]['candidate_id'] == "Complete", \
        "Complete profile should rank highest"
    
    print("✅ PASSED\n")


def test_complementarity_vs_compensation():
    """Test 3: Balanced profile should beat unbalanced even with same sum"""
    print("=" * 70)
    print("TEST 3: Complementarity vs Compensation")
    print("=" * 70)
    
    job_skills = {
        "Skill_A": 0.5,
        "Skill_B": 0.5,
        "Skill_C": 0.5
    }
    
    candidates = {
        "Balanced": {"Skill_A": 0.6, "Skill_B": 0.6, "Skill_C": 0.6},  # Sum = 1.8
        "Specialist": {"Skill_A": 1.0, "Skill_B": 0.5, "Skill_C": 0.3},  # Sum = 1.8
        "Superstar": {"Skill_A": 1.0, "Skill_B": 0.9, "Skill_C": 0.1}   # Sum = 2.0, but big gap
    }
    
    rankings = rank_candidates(job_skills, candidates)
    print(rankings)
    print()
    
    ranker = GraphBasedATSRanker()
    ranker.build_graph(job_skills, candidates)
    ranker.compute_rankings()
    
    for cand_id in candidates.keys():
        exp = ranker.explain_ranking(cand_id)
        total_prof = sum(candidates[cand_id].values())
        print(f"{cand_id}: Rank={exp['rank']}, Score={exp['score']:.6f}, Sum={total_prof:.1f}")
    print()
    
    # Balanced should win despite same or lower sum
    balanced_rank = rankings[rankings['candidate_id'] == 'Balanced']['rank'].values[0]
    assert balanced_rank == 1, "Balanced profile should win"
    
    print("✅ PASSED\n")


def test_importance_weighting():
    """Test 4: More important skills should have stronger influence"""
    print("=" * 70)
    print("TEST 4: Importance Weighting")
    print("=" * 70)
    
    job_skills = {
        "Critical": 1.0,
        "Important": 0.5,
        "Nice": 0.1
    }
    
    candidates = {
        "GoodAtCritical": {"Critical": 0.9, "Important": 0.5, "Nice": 0.3},
        "GoodAtNice": {"Critical": 0.5, "Important": 0.5, "Nice": 0.9}
    }
    
    rankings = rank_candidates(job_skills, candidates)
    print(rankings)
    print()
    
    # Candidate strong in critical skill should win
    assert rankings.iloc[0]['candidate_id'] == "GoodAtCritical", \
        "Candidate strong in critical skill should rank higher"
    
    print("✅ PASSED\n")


def test_scale_invariance():
    """Test 5: Ranking should be stable across different scales"""
    print("=" * 70)
    print("TEST 5: Scale Invariance")
    print("=" * 70)
    
    job_skills_v1 = {
        "Skill_A": 0.8,
        "Skill_B": 0.6,
        "Skill_C": 0.4
    }
    
    # Scale all importances by 2
    job_skills_v2 = {k: v * 2 for k, v in job_skills_v1.items()}
    
    candidates = {
        "Alice": {"Skill_A": 0.7, "Skill_B": 0.8, "Skill_C": 0.5},
        "Bob": {"Skill_A": 0.9, "Skill_B": 0.4, "Skill_C": 0.6},
        "Charlie": {"Skill_A": 0.5, "Skill_B": 0.7, "Skill_C": 0.8}
    }
    
    rankings_v1 = rank_candidates(job_skills_v1, candidates)
    rankings_v2 = rank_candidates(job_skills_v2, candidates)
    
    print("Version 1 (original scale):")
    print(rankings_v1)
    print("\nVersion 2 (2x scale):")
    print(rankings_v2)
    print()
    
    # Ranking order should be identical
    assert list(rankings_v1['candidate_id']) == list(rankings_v2['candidate_id']), \
        "Ranking should be invariant to importance scaling"
    
    print("✅ PASSED\n")


def test_tie_breaking():
    """Test 6: Graph should break ties between identical vectors"""
    print("=" * 70)
    print("TEST 6: Tie Breaking")
    print("=" * 70)
    
    job_skills = {
        "Python": 0.8,
        "SQL": 0.6,
        "Java": 0.4
    }
    
    candidates = {
        "Twin_A": {"Python": 0.7, "SQL": 0.7, "Java": 0.7},
        "Twin_B": {"Python": 0.7, "SQL": 0.7, "Java": 0.7},
        "Different": {"Python": 0.8, "SQL": 0.6, "Java": 0.5}
    }
    
    rankings = rank_candidates(job_skills, candidates)
    print(rankings)
    print()
    
    twin_a_score = rankings[rankings['candidate_id'] == 'Twin_A']['score'].values[0]
    twin_b_score = rankings[rankings['candidate_id'] == 'Twin_B']['score'].values[0]
    
    # Twins should have same score
    assert abs(twin_a_score - twin_b_score) < 1e-9, \
        "Identical profiles should have identical scores"
    
    print("✅ PASSED\n")


def test_zero_proficiency():
    """Test 7: Zero proficiency in a skill = no edge = penalty"""
    print("=" * 70)
    print("TEST 7: Zero Proficiency Handling")
    print("=" * 70)
    
    job_skills = {
        "Skill_A": 0.7,
        "Skill_B": 0.7
    }
    
    candidates = {
        "Has_Both": {"Skill_A": 0.5, "Skill_B": 0.5},
        "Zero_B": {"Skill_A": 0.9, "Skill_B": 0.0},  # Explicit zero
        "Missing_B": {"Skill_A": 0.9}  # Implicit missing
    }
    
    rankings = rank_candidates(job_skills, candidates)
    print(rankings)
    print()
    
    # Has_Both should win despite lower proficiencies
    assert rankings.iloc[0]['candidate_id'] == "Has_Both", \
        "Candidate with both skills should rank highest"
    
    # Zero and Missing should be treated similarly
    zero_score = rankings[rankings['candidate_id'] == 'Zero_B']['score'].values[0]
    missing_score = rankings[rankings['candidate_id'] == 'Missing_B']['score'].values[0]
    assert abs(zero_score - missing_score) < 1e-9, \
        "Zero proficiency and missing skill should have same score"
    
    print("✅ PASSED\n")


def test_large_scale():
    """Test 8: Should handle many candidates and skills efficiently"""
    print("=" * 70)
    print("TEST 8: Large Scale Performance")
    print("=" * 70)
    
    import time
    
    # Generate large dataset
    num_skills = 20
    num_candidates = 100
    
    job_skills = {f"Skill_{i}": np.random.uniform(0.3, 1.0) 
                  for i in range(num_skills)}
    
    candidates = {}
    for c in range(num_candidates):
        # Each candidate has 60-90% of skills
        num_cand_skills = np.random.randint(int(0.6 * num_skills), int(0.9 * num_skills))
        selected_skills = np.random.choice(list(job_skills.keys()), num_cand_skills, replace=False)
        candidates[f"Candidate_{c}"] = {
            skill: np.random.uniform(0.3, 1.0) for skill in selected_skills
        }
    
    start = time.time()
    ranker = GraphBasedATSRanker()
    ranker.build_graph(job_skills, candidates)
    rankings = ranker.compute_rankings()
    elapsed = time.time() - start
    
    print(f"Processed {num_candidates} candidates with {num_skills} skills")
    print(f"Time: {elapsed:.3f} seconds")
    print(f"\nGraph Stats: {ranker.get_graph_stats()}")
    print(f"\nTop 5 candidates:")
    print(rankings.head())
    print()
    
    assert len(rankings) == num_candidates, "Should rank all candidates"
    assert elapsed < 5.0, "Should complete in reasonable time"
    
    print("✅ PASSED\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("GRAPH-BASED ATS RANKER TEST SUITE")
    print("=" * 70 + "\n")
    
    tests = [
        test_basic_ranking,
        test_gap_penalty,
        test_complementarity_vs_compensation,
        test_importance_weighting,
        test_scale_invariance,
        test_tie_breaking,
        test_zero_proficiency,
        test_large_scale
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"💥 ERROR: {e}\n")
            failed += 1
    
    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
