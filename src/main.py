from json_loader import ATSDataLoader
from graph_ats_ranker import GraphBasedATSRanker


if __name__ == "__main__":
    job_skills, candidates = ATSDataLoader.load_from_json(job_file='../data/job_structured.json',candidates_file='../data/candidates_structured.json')

    ranker = GraphBasedATSRanker()
    ranker.build_graph(job_skills, candidates)

    ranking = ranker.compute_rankings()
    print(ranking)
    top_candidate = ranking.iloc[0]['candidate_id']
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