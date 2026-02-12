"""
Example usage with the exact data structure requested.
"""

from graph_ats_ranker import GraphBasedATSRanker, rank_candidates

# Define job requirements with your structure
job_requirements = {
    "title": "Senior HR Manager",
    "min_years_experience": 5,  # Required years
    "preferred_years_experience": 8,  # Preferred years (optional)
    "skills": {
        "HRIS": 0.92,
        "Employee Relations": 0.88,
        "Communication": 0.90,
        "Recruitment": 0.85,
        "Performance Management": 0.80
    }
}

# Define candidates with your structure
candidates = [
    {
        "id": "Ethan",
        "years_of_experience": 7,
        "skills": {
            "HRIS": 0.92,
            "Employee Relations": 0.88,
            "Communication": 0.90
        }
    },
    {
        "id": "Sophia",
        "years_of_experience": 4,
        "skills": {
            "HRIS": 0.95,
            "Employee Relations": 0.85,
            "Communication": 0.92,
            "Recruitment": 0.88,
            "Performance Management": 0.82
        }
    },
    {
        "id": "Marcus",
        "years_of_experience": 9,
        "skills": {
            "HRIS": 0.80,
            "Employee Relations": 0.90,
            "Communication": 0.85,
            "Recruitment": 0.75,
            "Performance Management": 0.88
        }
    },
    {
        "id": "Nina",
        "years_of_experience": 3,
        "skills": {
            "HRIS": 0.88,
            "Communication": 0.87,
            "Recruitment": 0.90
        }
    },
    {
        "id": "Oliver",
        "years_of_experience": 10,
        "skills": {
            "HRIS": 0.85,
            "Employee Relations": 0.92,
            "Communication": 0.88,
            "Performance Management": 0.90
        }
    }
]

print("=" * 80)
print("GRAPH-BASED ATS RANKER - EXAMPLE WITH YOUR DATA STRUCTURE")
print("=" * 80)

print(f"\nJob Title: {job_requirements['title']}")
print(f"Required Years: {job_requirements['min_years_experience']}")
print(f"Preferred Years: {job_requirements.get('preferred_years_experience', 'Not specified')}")

print("\nRequired Skills:")
for skill, importance in sorted(job_requirements['skills'].items(),
                                key=lambda x: x[1], reverse=True):
    print(f"  - {skill}: {importance:.2f}")

print("\n" + "=" * 80)
print("CANDIDATE PROFILES")
print("=" * 80)

for candidate in candidates:
    print(f"\n{candidate['id']}:")
    print(f"  Years of Experience: {candidate['years_of_experience']}")
    print(f"  Skills:")
    for skill, proficiency in candidate['skills'].items():
        print(f"    - {skill}: {proficiency:.2f}")

print("\n" + "=" * 80)
print("SCENARIO 1: Without Experience Consideration")
print("=" * 80)

ranker1 = GraphBasedATSRanker(experience_weight=0.0)
ranker1.build_graph(job_requirements, candidates)
rankings1 = ranker1.compute_rankings()

print("\nRankings (Skills Only):")
print(rankings1.to_string(index=False))

print("\n" + "=" * 80)
print("SCENARIO 2: With Experience (mode='both', weight=0.3)")
print("=" * 80)

ranker2 = GraphBasedATSRanker(
    experience_weight=0.3,
    experience_mode='both'
)
ranker2.build_graph(job_requirements, candidates)
rankings2 = ranker2.compute_rankings()

print("\nRankings (With Experience):")
print(rankings2.to_string(index=False))

print("\n" + "=" * 80)
print("SCENARIO 3: Higher Experience Weight (mode='both', weight=0.5)")
print("=" * 80)

ranker3 = GraphBasedATSRanker(
    experience_weight=0.5,
    experience_mode='both'
)
ranker3.build_graph(job_requirements, candidates)
rankings3 = ranker3.compute_rankings()

print("\nRankings (Higher Experience Weight):")
print(rankings3.to_string(index=False))

print("\n" + "=" * 80)
print("DETAILED EXPLANATION - TOP CANDIDATE")
print("=" * 80)

top_candidate = rankings2.iloc[0]['candidate_id']
explanation = ranker2.explain_ranking(top_candidate, top_k_skills=5)

print(f"\nCandidate: {explanation['candidate_id']}")
print(f"Rank: {explanation['rank']}")
print(f"Score: {explanation['score']:.6f}")
print(f"Years of Experience: {explanation['years_experience']:.1f}")
print(f"Required Years: {explanation['required_years']:.1f}")
print(f"Preferred Years: {explanation['preferred_years']}")
print(f"Experience Status: {explanation['experience_status']}")
print(f"Experience Contribution: {explanation['experience_contribution']:.6f}")
print(f"Skill Coverage: {explanation['skill_coverage']:.1%}")

print(f"\nTop Contributing Skills:")
for skill_info in explanation['top_skills']:
    print(f"  - {skill_info['skill']}")
    print(f"      Proficiency: {skill_info['proficiency']:.2f}")
    print(f"      Importance: {skill_info['importance']:.2f}")
    print(f"      Contribution: {skill_info['contribution']:.6f}")

if explanation['missing_skills']:
    print(f"\nMissing Skills:")
    for skill_info in explanation['missing_skills']:
        print(f"  - {skill_info['skill']} (importance: {skill_info['importance']:.2f})")

print("\n" + "=" * 80)
print("COMPARISON OF ALL CANDIDATES")
print("=" * 80)

print("\n{:<10} {:<8} {:<15} {:<12}".format(
    "Candidate", "Years", "Status", "Rank Change"
))
print("-" * 50)

for i, row1 in rankings1.iterrows():
    cand_id = row1['candidate_id']
    years = row1['years_experience']

    # Find rank in scenario 2
    rank2 = rankings2[rankings2['candidate_id'] == cand_id]['rank'].values[0]

    # Determine status
    if years < job_requirements['min_years_experience']:
        status = "Below Req"
    elif years >= job_requirements.get('preferred_years_experience', 8):
        status = "Exceeds Pref"
    else:
        status = "Meets Req"

    # Calculate rank change
    rank_change = row1['rank'] - rank2
    change_str = f"+{rank_change}" if rank_change > 0 else str(rank_change)
    if rank_change == 0:
        change_str = "No change"

    print("{:<10} {:<8} {:<15} {:<12}".format(
        cand_id,
        f"{years} yrs",
        status,
        change_str
    ))

print("\n" + "=" * 80)
print("USING CONVENIENCE FUNCTION")
print("=" * 80)

# Quick ranking with default parameters
quick_rankings = rank_candidates(
    job_requirements=job_requirements,
    candidates=candidates,
    experience_mode='both',
    experience_weight=0.3
)

print("\nQuick Rankings:")
print(quick_rankings.to_string(index=False))

print("\n" + "=" * 80)
print("EXAMPLE: JOB WITH NO MINIMUM EXPERIENCE (defaults to 0)")
print("=" * 80)

job_no_experience = {
    "title": "Junior HR Coordinator",
    # min_years_experience not specified - defaults to 0
    "skills": {
        "Communication": 0.90,
        "Organization": 0.85,
        "MS Office": 0.75
    }
}

candidates_mixed = [
    {
        "id": "Alex",
        "years_of_experience": 0,  # Fresh graduate
        "skills": {
            "Communication": 0.85,
            "Organization": 0.90,
            "MS Office": 0.80
        }
    },
    {
        "id": "Blake",
        "years_of_experience": 2,  # Some experience
        "skills": {
            "Communication": 0.80,
            "Organization": 0.75,
            "MS Office": 0.85
        }
    }
]

print(f"\nJob: {job_no_experience['title']}")
print(f"Min Years Required: 0 (default - entry level)")

ranker_entry = GraphBasedATSRanker(experience_weight=0.3, experience_mode='both')
ranker_entry.build_graph(job_no_experience, candidates_mixed)
rankings_entry = ranker_entry.compute_rankings()

print("\nRankings:")
print(rankings_entry.to_string(index=False))

print("\nNote: With no experience requirement (0), candidates with some experience")
print("get a small bonus, but it's minimal since experience isn't a key factor.")

print("\n" + "=" * 80)
print("GRAPH STATISTICS")
print("=" * 80)

stats = ranker2.get_graph_stats()
print(f"\nGraph Details:")
for key, value in stats.items():
    print(f"  {key}: {value}")



"""
    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    @staticmethod
    def save_to_json(
        job_requirements: 'Dict | List[Dict]',
        candidates: List[Dict],
        job_output_file: str,
        candidates_output_file: str
    ):
        with open(job_output_file, 'w') as f:
            json.dump(job_requirements, f, indent=2)

        with open(candidates_output_file, 'w') as f:
            json.dump(candidates, f, indent=2)

def rank_from_json(
    job_file: str,
    candidates_file: str,
    experience_weight: float = 0.3,
    experience_mode: str = 'both'
) -> Dict[str, list]:
    from graph_ats_ranker import rank_candidates

    jobs, candidates = ATSDataLoader.load_from_json(job_file, candidates_file)

    is_valid, errors = ATSDataLoader.validate_data(jobs, candidates)
    if not is_valid:
        raise ValueError("Data validation failed:\n" + "\n".join(errors))

    results = {}
    for job in jobs:
        title = job.get('title', 'Job Position')
        results[title] = rank_candidates(
            job_requirements=job,
            candidates=candidates,
            experience_weight=experience_weight,
            experience_mode=experience_mode
        )

    return results"""