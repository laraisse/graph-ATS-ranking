"""
JSON Data Loader for Graph-Based ATS Ranker
============================================

This module provides utilities to load job requirements and candidate profiles
from JSON files and convert them to the format required by the ranker.

Supports multiple JSON formats for flexibility.
"""

import json
from typing import Dict, List, Tuple


class ATSDataLoader:
    """
    Load job and candidate data from JSON files.
    
    Supports multiple JSON formats for maximum flexibility.
    """
    
    @staticmethod
    def load_from_json(
        job_file: str,
        candidates_file: str
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Load data from JSON files.
        
        Parameters:
        -----------
        job_file : str
            Path to JSON file containing job requirements
        candidates_file : str
            Path to JSON file containing candidate profiles
            
        Returns:
        --------
        Tuple of (job_skills, candidates) ready for the ranker
        
        Example:
        --------
        job_skills, candidates = ATSDataLoader.load_from_json(
            'job.json',
            'candidates.json'
        )
        """
        job_skills = ATSDataLoader.load_job_requirements(job_file)
        candidates = ATSDataLoader.load_candidates(candidates_file)
        return job_skills, candidates
    
    @staticmethod
    def load_job_requirements(filepath: str) -> Dict[str, float]:
        """
        Load job requirements from JSON.
        
        Supports multiple formats:
        
        Format 1 - Simple dict:
        {
            "Python": 0.9,
            "SQL": 0.7,
            "Leadership": 0.5
        }
        
        Format 2 - List of skill objects:
        {
            "skills": [
                {"name": "Python", "importance": 0.9},
                {"name": "SQL", "importance": 0.7}
            ]
        }
        
        Format 3 - Job object with skills:
        {
            "job_id": "DS_2024_001",
            "title": "Senior Data Scientist",
            "skills": {
                "Python": 0.9,
                "SQL": 0.7
            }
        }
        
        Format 4 - Skills with categories:
        {
            "required": {
                "Python": 0.9,
                "SQL": 0.8
            },
            "preferred": {
                "Leadership": 0.5,
                "Communication": 0.4
            }
        }
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Format 1: Simple dict (skill: importance)
        if isinstance(data, dict) and all(isinstance(v, (int, float)) for v in data.values()):
            return {str(k): float(v) for k, v in data.items()}
        
        # Format 2: List of skill objects
        if isinstance(data, dict) and 'skills' in data and isinstance(data['skills'], list):
            return {
                skill['name']: float(skill['importance'])
                for skill in data['skills']
            }
        
        # Format 3: Job object with skills dict
        if isinstance(data, dict) and 'skills' in data and isinstance(data['skills'], dict):
            return {str(k): float(v) for k, v in data['skills'].items()}
        
        # Format 4: Required/Preferred categories
        if isinstance(data, dict) and any(k in data for k in ['required', 'preferred', 'nice_to_have']):
            result = {}
            if 'required' in data:
                result.update({str(k): float(v) for k, v in data['required'].items()})
            if 'preferred' in data:
                result.update({str(k): float(v) for k, v in data['preferred'].items()})
            if 'nice_to_have' in data:
                result.update({str(k): float(v) for k, v in data['nice_to_have'].items()})
            return result
        
        raise ValueError(f"Unrecognized job requirements format in {filepath}")
    
    @staticmethod
    def load_candidates(filepath: str) -> Dict[str, Dict[str, float]]:
        """
        Load candidate profiles from JSON.
        
        Supports multiple formats:
        
        Format 1 - Simple dict of dicts:
        {
            "Alice": {
                "Python": 0.8,
                "SQL": 0.6
            },
            "Bob": {
                "Python": 0.5,
                "SQL": 0.9
            }
        }
        
        Format 2 - List of candidate objects:
        {
            "candidates": [
                {
                    "id": "Alice",
                    "skills": {
                        "Python": 0.8,
                        "SQL": 0.6
                    }
                },
                {
                    "id": "Bob",
                    "skills": {
                        "Python": 0.5,
                        "SQL": 0.9
                    }
                }
            ]
        }
        
        Format 3 - List of candidate objects with skill arrays:
        {
            "candidates": [
                {
                    "id": "Alice",
                    "skills": [
                        {"name": "Python", "proficiency": 0.8},
                        {"name": "SQL", "proficiency": 0.6}
                    ]
                }
            ]
        }
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Format 1: Simple dict of dicts
        if isinstance(data, dict) and all(
            isinstance(v, dict) and all(isinstance(vv, (int, float)) for vv in v.values())
            for v in data.values()
        ):
            return {
                str(cand_id): {str(k): float(v) for k, v in skills.items()}
                for cand_id, skills in data.items()
            }
        
        # Format 2: List of candidate objects with skills dict
        if isinstance(data, dict) and 'candidates' in data and isinstance(data['candidates'], list):
            result = {}
            for candidate in data['candidates']:
                cand_id = candidate.get('id') or candidate.get('name') or candidate.get('candidate_id')
                
                # Skills as dict
                if 'skills' in candidate and isinstance(candidate['skills'], dict):
                    result[str(cand_id)] = {
                        str(k): float(v) for k, v in candidate['skills'].items()
                    }
                
                # Skills as list of objects
                elif 'skills' in candidate and isinstance(candidate['skills'], list):
                    result[str(cand_id)] = {
                        skill['name']: float(skill.get('proficiency') or skill.get('score') or skill.get('level'))
                        for skill in candidate['skills']
                    }
            
            return result
        
        raise ValueError(f"Unrecognized candidates format in {filepath}")
    
    @staticmethod
    def validate_data(
        job_skills: Dict[str, float],
        candidates: Dict[str, Dict[str, float]]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that the loaded data is correct.
        
        Returns:
        --------
        Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check job skills
        if not job_skills:
            errors.append("Job skills dictionary is empty")
        
        for skill, importance in job_skills.items():
            if not isinstance(importance, (int, float)):
                errors.append(f"Job skill '{skill}' importance must be numeric, got {type(importance)}")
            elif not 0 <= importance <= 1:
                errors.append(f"Job skill '{skill}' importance {importance} not in range [0, 1]")
        
        # Check candidates
        if not candidates:
            errors.append("Candidates dictionary is empty")
        
        for cand_id, skills in candidates.items():
            if not skills:
                errors.append(f"Candidate '{cand_id}' has no skills")
            
            for skill, proficiency in skills.items():
                if not isinstance(proficiency, (int, float)):
                    errors.append(f"Candidate '{cand_id}' skill '{skill}' proficiency must be numeric")
                elif not 0 <= proficiency <= 1:
                    errors.append(f"Candidate '{cand_id}' skill '{skill}' proficiency {proficiency} not in [0, 1]")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def save_to_json(
        job_skills: Dict[str, float],
        candidates: Dict[str, Dict[str, float]],
        job_output_file: str,
        candidates_output_file: str,
        format: str = 'simple'
    ):
        """
        Save data to JSON files.
        
        Parameters:
        -----------
        format : str
            'simple' - Simple dict format (default)
            'structured' - More detailed format with metadata
        """
        # Save job
        if format == 'simple':
            job_data = job_skills
        else:
            job_data = {
                'job_id': 'JOB_001',
                'skills': job_skills
            }
        
        with open(job_output_file, 'w') as f:
            json.dump(job_data, f, indent=2)
        
        # Save candidates
        if format == 'simple':
            candidates_data = candidates
        else:
            candidates_data = {
                'candidates': [
                    {
                        'id': cand_id,
                        'skills': skills
                    }
                    for cand_id, skills in candidates.items()
                ]
            }
        
        with open(candidates_output_file, 'w') as f:
            json.dump(candidates_data, f, indent=2)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def rank_from_json(job_file: str, candidates_file: str):
    """
    One-liner: Load from JSON and rank.

    Example:
    --------
    rankings = rank_from_json('job.json', 'candidates.json')
    """
    from graph_ats_ranker import rank_candidates

    job_skills, candidates = ATSDataLoader.load_from_json(job_file, candidates_file)

    # Validate
    is_valid, errors = ATSDataLoader.validate_data(job_skills, candidates)
    if not is_valid:
        raise ValueError(f"Data validation failed:\n" + "\n".join(errors))

    return rank_candidates(job_skills, candidates)


