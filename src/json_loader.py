import json
from typing import Dict, List, Tuple, Optional


class ATSDataLoader:
    @staticmethod
    def load_from_json(
        job_file: str,
        candidates_file: str
    ) -> Tuple[List[Dict], List[Dict]]:
        job_requirements_list = ATSDataLoader.load_job_requirements_list(job_file)
        candidates = ATSDataLoader.load_candidates(candidates_file)
        return job_requirements_list, candidates

    # ------------------------------------------------------------------
    # Internal helper: parse one job dict into the standard format
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_single_job(data: Dict) -> Dict:
        """Parse a single job dict into the standard internal format."""

        # Format 1: New structure with title/min_years_experience/skills as dict
        if 'skills' in data and isinstance(data['skills'], dict):
            result = {
                'title': data.get('title', 'Job Position'),
                'skills': {str(k): float(v) for k, v in data['skills'].items()}
            }
            if 'min_years_experience' in data:
                result['min_years_experience'] = float(data['min_years_experience'])
            if 'preferred_years_experience' in data:
                result['preferred_years_experience'] = float(data['preferred_years_experience'])
            return result

        # Format 2: Simple dict (skill: importance) - backward compatible
        if all(isinstance(v, (int, float)) for v in data.values()):
            return {
                'title': 'Job Position',
                'skills': {str(k): float(v) for k, v in data.items()}
            }

        # Format 3: skills as list of objects
        if 'skills' in data and isinstance(data['skills'], list):
            return {
                'title': data.get('title', 'Job Position'),
                'skills': {
                    skill['name']: float(skill['importance'])
                    for skill in data['skills']
                }
            }

        # Format 4: Required/Preferred categories
        if any(k in data for k in ['required', 'preferred', 'nice_to_have']):
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

        raise ValueError(f"Unrecognized job format: {data}")

    # ------------------------------------------------------------------
    # Primary multi-job loader
    # ------------------------------------------------------------------
    @staticmethod
    def load_job_requirements_list(filepath: str) -> List[Dict]:
        """Load one or more jobs from a file. Always returns a list of job dicts."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Multi-job: top-level JSON array
        if isinstance(data, list):
            return [ATSDataLoader._parse_single_job(item) for item in data]

        # Multi-job: wrapped under a 'jobs' key
        if isinstance(data, dict) and 'jobs' in data and isinstance(data['jobs'], list):
            return [ATSDataLoader._parse_single_job(item) for item in data['jobs']]

        # Single job: any supported dict format
        if isinstance(data, dict):
            return [ATSDataLoader._parse_single_job(data)]

        raise ValueError(f"Unrecognized job requirements format in {filepath}")

    # ------------------------------------------------------------------
    # Legacy single-job loader (backward compatible)
    # ------------------------------------------------------------------
    @staticmethod
    def load_job_requirements(filepath: str) -> Dict:
        """Legacy single-job loader. Returns the first job found in the file."""
        return ATSDataLoader.load_job_requirements_list(filepath)[0]

    # ------------------------------------------------------------------
    # Candidates loader
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_candidates_list(candidates: list) -> List[Dict]:
        result = []
        for candidate in candidates:
            cand_obj = {
                'id': str(candidate.get('id') or candidate.get('name') or candidate.get('candidate_id')),
                'years_of_experience': float(candidate.get('years_of_experience', 0))
            }
            if 'skills' in candidate and isinstance(candidate['skills'], dict):
                cand_obj['skills'] = {str(k): float(v) for k, v in candidate['skills'].items()}
            elif 'skills' in candidate and isinstance(candidate['skills'], list):
                cand_obj['skills'] = {
                    skill['name']: float(skill.get('proficiency') or skill.get('score') or skill.get('level'))
                    for skill in candidate['skills']
                }
            else:
                cand_obj['skills'] = {}
            result.append(cand_obj)
        return result

    @staticmethod
    def load_candidates(filepath: str) -> List[Dict]:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Format 1: Direct list of candidate objects
        if isinstance(data, list):
            return ATSDataLoader._parse_candidates_list(data)

        # Format 2: Wrapped in 'candidates' key
        if isinstance(data, dict) and 'candidates' in data and isinstance(data['candidates'], list):
            return ATSDataLoader._parse_candidates_list(data['candidates'])

        # Format 3: Simple dict of dicts (backward compatible - no years of experience)
        if isinstance(data, dict) and all(
            isinstance(v, dict) and all(isinstance(vv, (int, float)) for vv in v.values())
            for v in data.values()
        ):
            return [
                {
                    'id': str(cand_id),
                    'years_of_experience': 0,
                    'skills': {str(k): float(v) for k, v in skills.items()}
                }
                for cand_id, skills in data.items()
            ]

        raise ValueError(f"Unrecognized candidates format in {filepath}")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_single_job(job: Dict, label: str = "Job") -> List[str]:
        """Validate one job dict and return a list of error strings."""
        errors = []

        if not job:
            errors.append(f"{label}: requirements dictionary is empty")
            return errors

        if 'skills' not in job:
            errors.append(f"{label}: must have 'skills' key")
        else:
            job_skills = job['skills']
            if not job_skills:
                errors.append(f"{label}: skills dictionary is empty")
            for skill, importance in job_skills.items():
                if not isinstance(importance, (int, float)):
                    errors.append(f"{label}: skill '{skill}' importance must be numeric, got {type(importance)}")
                elif not 0 <= importance <= 1:
                    errors.append(f"{label}: skill '{skill}' importance {importance} not in range [0, 1]")

        if 'min_years_experience' in job:
            min_years = job['min_years_experience']
            if not isinstance(min_years, (int, float)) or min_years < 0:
                errors.append(f"{label}: min_years_experience must be a non-negative number, got {min_years}")

        if 'preferred_years_experience' in job:
            pref_years = job['preferred_years_experience']
            if not isinstance(pref_years, (int, float)) or pref_years < 0:
                errors.append(f"{label}: preferred_years_experience must be a non-negative number, got {pref_years}")

        return errors

    @staticmethod
    def validate_data(
        job_requirements: 'Dict | List[Dict]',
        candidates: List[Dict]
    ) -> Tuple[bool, List[str]]:
        errors = []

        # Accept either a single job dict or a list of job dicts
        jobs_to_validate = [job_requirements] if isinstance(job_requirements, dict) else job_requirements

        if not jobs_to_validate:
            errors.append("Job requirements list is empty")
        else:
            for i, job in enumerate(jobs_to_validate):
                label = job.get('title', f'Job {i + 1}')
                errors.extend(ATSDataLoader._validate_single_job(job, label=label))

        # Validate candidates
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
                        errors.append(
                            f"Candidate '{cand_id}' skill '{skill}' proficiency {proficiency} not in [0, 1]"
                        )

            if 'years_of_experience' in candidate:
                years = candidate['years_of_experience']
                if not isinstance(years, (int, float)) or years < 0:
                    errors.append(
                        f"Candidate '{cand_id}' years_of_experience must be non-negative number, got {years}"
                    )

        return len(errors) == 0, errors

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


# ----------------------------------------------------------------------
# Top-level helper
# ----------------------------------------------------------------------
def rank_from_json(
    job_file: str,
    candidates_file: str,
    experience_weight: float = 0.3,
    experience_mode: str = 'both'
) -> Dict[str, list]:
    """
    Load all jobs from job_file and rank candidates against each one.
    Returns a dict keyed by job title, e.g.:
        {
            "Software Engineer": [...ranked candidates...],
            "Data Scientist":    [...ranked candidates...],
        }
    """
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

    return results