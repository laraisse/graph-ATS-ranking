import json
from typing import Dict, List, Tuple, Optional


class ATSDataLoader:
    @staticmethod
    def load_from_json(
        job_file: str,
        candidates_file: str
    ) -> Tuple[Dict, List[Dict]]:
        job_requirements = ATSDataLoader.load_job_requirements(job_file)
        candidates = ATSDataLoader.load_candidates(candidates_file)
        return job_requirements, candidates

    @staticmethod
    def load_job_requirements(filepath: str) -> Dict:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Check if it's the new structure with title/min_years_experience/skills
        if isinstance(data, dict) and 'skills' in data and isinstance(data['skills'], dict):
            # New structure - extract everything
            result = {
                'title': data.get('title', 'Job Position'),
                'skills': {str(k): float(v) for k, v in data['skills'].items()}
            }

            # Add years if specified
            if 'min_years_experience' in data:
                result['min_years_experience'] = float(data['min_years_experience'])

            if 'preferred_years_experience' in data:
                result['preferred_years_experience'] = float(data['preferred_years_experience'])

            return result

        # Format 2: Simple dict (skill: importance) - backward compatible
        if isinstance(data, dict) and all(isinstance(v, (int, float)) for v in data.values()):
            return {
                'title': 'Job Position',
                'skills': {str(k): float(v) for k, v in data.items()}
            }

        # Format 3: List of skill objects
        if isinstance(data, dict) and 'skills' in data and isinstance(data['skills'], list):
            return {
                'title': data.get('title', 'Job Position'),
                'skills': {
                    skill['name']: float(skill['importance'])
                    for skill in data['skills']
                }
            }

        # Format 4: Required/Preferred categories
        if isinstance(data, dict) and any(k in data for k in ['required', 'preferred', 'nice_to_have']):
            skills = {}
            if 'required' in data:
                skills.update({str(k): float(v) for k, v in data['required'].items()})
            if 'preferred' in data:
                skills.update({str(k): float(v) for k, v in data['preferred'].items()})
            if 'nice_to_have' in data:
                skills.update({str(k): float(v) for k, v in data['nice_to_have'].items()})

            return {
                'title': data.get('title', 'Job Position'),
                'skills': skills
            }

        raise ValueError(f"Unrecognized job requirements format in {filepath}")

    @staticmethod
    def load_candidates(filepath: str) -> List[Dict]:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Format 1: Direct list of candidate objects
        if isinstance(data, list):
            result = []
            for candidate in data:
                cand_obj = {
                    'id': str(candidate.get('id') or candidate.get('name') or candidate.get('candidate_id')),
                    'years_of_experience': float(candidate.get('years_of_experience', 0))
                }

                # Skills as dict
                if 'skills' in candidate and isinstance(candidate['skills'], dict):
                    cand_obj['skills'] = {
                        str(k): float(v) for k, v in candidate['skills'].items()
                    }
                # Skills as list
                elif 'skills' in candidate and isinstance(candidate['skills'], list):
                    cand_obj['skills'] = {
                        skill['name']: float(skill.get('proficiency') or skill.get('score') or skill.get('level'))
                        for skill in candidate['skills']
                    }
                else:
                    cand_obj['skills'] = {}

                result.append(cand_obj)
            return result

        # Format 2: Wrapped in 'candidates' key
        if isinstance(data, dict) and 'candidates' in data and isinstance(data['candidates'], list):
            result = []
            for candidate in data['candidates']:
                cand_obj = {
                    'id': str(candidate.get('id') or candidate.get('name') or candidate.get('candidate_id')),
                    'years_of_experience': float(candidate.get('years_of_experience', 0))
                }

                # Skills as dict
                if 'skills' in candidate and isinstance(candidate['skills'], dict):
                    cand_obj['skills'] = {
                        str(k): float(v) for k, v in candidate['skills'].items()
                    }
                # Skills as list
                elif 'skills' in candidate and isinstance(candidate['skills'], list):
                    cand_obj['skills'] = {
                        skill['name']: float(skill.get('proficiency') or skill.get('score') or skill.get('level'))
                        for skill in candidate['skills']
                    }
                else:
                    cand_obj['skills'] = {}

                result.append(cand_obj)
            return result

        # Format 3: Simple dict of dicts (backward compatible - no years of experience)
        if isinstance(data, dict) and all(
            isinstance(v, dict) and all(isinstance(vv, (int, float)) for vv in v.values())
            for v in data.values()
        ):
            result = []
            for cand_id, skills in data.items():
                result.append({
                    'id': str(cand_id),
                    'years_of_experience': 0,  # Default to 0
                    'skills': {str(k): float(v) for k, v in skills.items()}
                })
            return result

        raise ValueError(f"Unrecognized candidates format in {filepath}")

    @staticmethod
    def validate_data(
        job_requirements: Dict,
        candidates: List[Dict]
    ) -> Tuple[bool, List[str]]:
        errors = []

        # Check job requirements
        if not job_requirements:
            errors.append("Job requirements dictionary is empty")

        if 'skills' not in job_requirements:
            errors.append("Job requirements must have 'skills' key")
        else:
            job_skills = job_requirements['skills']
            if not job_skills:
                errors.append("Job skills dictionary is empty")

            for skill, importance in job_skills.items():
                if not isinstance(importance, (int, float)):
                    errors.append(f"Job skill '{skill}' importance must be numeric, got {type(importance)}")
                elif not 0 <= importance <= 1:
                    errors.append(f"Job skill '{skill}' importance {importance} not in range [0, 1]")

        # Validate years of experience if present
        if 'min_years_experience' in job_requirements:
            min_years = job_requirements['min_years_experience']
            if not isinstance(min_years, (int, float)) or min_years < 0:
                errors.append(f"min_years_experience must be a non-negative number, got {min_years}")

        if 'preferred_years_experience' in job_requirements:
            pref_years = job_requirements['preferred_years_experience']
            if not isinstance(pref_years, (int, float)) or pref_years < 0:
                errors.append(f"preferred_years_experience must be a non-negative number, got {pref_years}")

        # Check candidates
        if not candidates:
            errors.append("Candidates list is empty")

        for candidate in candidates:
            if 'id' not in candidate:
                errors.append(f"Candidate missing 'id' field: {candidate}")
                continue

            cand_id = candidate['id']

            if 'skills' not in candidate:
                errors.append(f"Candidate '{cand_id}' missing 'skills' field")
            elif not candidate['skills']:
                errors.append(f"Candidate '{cand_id}' has no skills")
            else:
                for skill, proficiency in candidate['skills'].items():
                    if not isinstance(proficiency, (int, float)):
                        errors.append(f"Candidate '{cand_id}' skill '{skill}' proficiency must be numeric")
                    elif not 0 <= proficiency <= 1:
                        errors.append(f"Candidate '{cand_id}' skill '{skill}' proficiency {proficiency} not in [0, 1]")

            # Validate years of experience
            if 'years_of_experience' in candidate:
                years = candidate['years_of_experience']
                if not isinstance(years, (int, float)) or years < 0:
                    errors.append(f"Candidate '{cand_id}' years_of_experience must be non-negative number, got {years}")

        return len(errors) == 0, errors

    @staticmethod
    def save_to_json(
        job_requirements: Dict,
        candidates: List[Dict],
        job_output_file: str,
        candidates_output_file: str
    ):

        # Save job
        with open(job_output_file, 'w') as f:
            json.dump(job_requirements, f, indent=2)

        # Save candidates
        with open(candidates_output_file, 'w') as f:
            json.dump(candidates, f, indent=2)


def rank_from_json(
    job_file: str,
    candidates_file: str,
    experience_weight: float = 0.3,
    experience_mode: str = 'both'
):

    from graph_ats_ranker import rank_candidates

    job_requirements, candidates = ATSDataLoader.load_from_json(job_file, candidates_file)

    # Validate
    is_valid, errors = ATSDataLoader.validate_data(job_requirements, candidates)
    if not is_valid:
        raise ValueError(f"Data validation failed:\n" + "\n".join(errors))

    return rank_candidates(
        job_requirements=job_requirements,
        candidates=candidates,
        experience_weight=experience_weight,
        experience_mode=experience_mode
    )