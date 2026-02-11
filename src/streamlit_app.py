import streamlit as st
import pandas as pd
import json
import io
import sys
sys.path.insert(0, '/mnt/user-data/uploads')

from json_loader import ATSDataLoader
from graph_ats_ranker import GraphBasedATSRanker
import plotly.graph_objects as go
import plotly.express as px
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="ATS Candidate Ranker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .candidate-card {
        background-color: #1e1e1e;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .rank-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .rank-1 { background-color: #ffd700; color: #000; }
    .rank-2 { background-color: #c0c0c0; color: #000; }
    .rank-3 { background-color: #cd7f32; color: #fff; }
    .rank-other { background-color: #4CAF50; color: #fff; }
    .exp-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    .exp-below { background-color: #ffcccc; color: #cc0000; }
    .exp-meets { background-color: #ccffcc; color: #006600; }
    .exp-exceeds { background-color: #cce5ff; color: #0066cc; }
</style>
""", unsafe_allow_html=True)


def create_skill_coverage_chart(explanation, job_skills):
    """Create a radar chart showing skill coverage"""
    categories = list(job_skills.keys())

    # Get candidate's proficiencies
    candidate_profs = []
    for skill in categories:
        skill_data = next((s for s in explanation['top_skills'] if s['skill'] == skill), None)
        if skill_data:
            # The proficiency returned is already adjusted, so we need to use it directly
            candidate_profs.append(min(skill_data['proficiency'], 1.0))
        else:
            candidate_profs.append(0)

    # Get importances
    importances = [job_skills[skill] for skill in categories]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=candidate_profs,
        theta=categories,
        fill='toself',
        name='Candidate Proficiency',
        line_color='#1f77b4'
    ))

    fig.add_trace(go.Scatterpolar(
        r=importances,
        theta=categories,
        fill='toself',
        name='Job Importance',
        line_color='#ff7f0e',
        opacity=0.5
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=400
    )

    return fig


def create_contribution_chart(top_skills):
    """Create a bar chart showing skill contributions"""
    df = pd.DataFrame(top_skills)

    fig = px.bar(
        df,
        x='contribution',
        y='skill',
        orientation='h',
        title='Skill Contributions to Final Score',
        labels={'contribution': 'Contribution', 'skill': 'Skill'},
        color='contribution',
        color_continuous_scale='Blues'
    )

    fig.update_layout(height=300, showlegend=False)

    return fig


def validate_json_file(uploaded_file, file_type):
    try:
        content = uploaded_file.read()
        data = json.loads(content)
        uploaded_file.seek(0)

        if file_type == "job":
            if isinstance(data, dict):
                return True, "Valid job requirements file"
            else:
                return False, "Job file must be a JSON object"

        elif file_type == "candidates":
            if isinstance(data, (dict, list)):
                return True, "Valid candidates file"
            else:
                return False, "Candidates file must be a JSON object or array"

        return True, "Valid JSON file"

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON format: {str(e)}"
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


def main():
    # Header
    st.markdown('<div class="main-header">🎯 ATS Candidate Ranking System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Graph-Based Intelligent Candidate Ranking with Experience</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("📤 Upload Data")

        st.markdown("### Job Requirements")
        job_file = st.file_uploader(
            "Upload job requirements JSON",
            type=['json'],
            help="JSON file containing required skills, importance, and years of experience"
        )

        st.markdown("### Candidate Profiles")
        candidates_file = st.file_uploader(
            "Upload candidates JSON",
            type=['json'],
            help="JSON file containing candidate IDs, skills, and years of experience"
        )

        st.markdown("---")

        st.markdown("### ⚙️ Ranking Settings")

        experience_mode = st.selectbox(
            "Experience Integration Mode",
            options=['both', 'boost', 'direct', 'none'],
            index=0,
            help="How to incorporate years of experience:\n"
                 "• both: Use skill boost + direct edges (recommended)\n"
                 "• boost: Multiply skill weights by experience\n"
                 "• direct: Add job→candidate edges for experience\n"
                 "• none: Ignore experience (skills only)"
        )

        if experience_mode != 'none':
            experience_weight = st.slider(
                "Experience Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="How much to weight experience (0.0-1.0)\n"
                     "• 0.1-0.2: Minimal influence\n"
                     "• 0.3-0.4: Moderate (recommended)\n"
                     "• 0.5+: Strong influence"
            )
        else:
            experience_weight = 0.0

        st.markdown("---")

        st.markdown("### 📊 Display Settings")

        top_percentage = st.slider(
            "Top candidates to show (%)",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
            help="Show only the top X% of candidates"
        )

        show_all_skills = st.checkbox(
            "Show all skills in report",
            value=False,
            help="Show all skills or just top 5 per candidate"
        )

        st.markdown("---")
        st.markdown("### 📋 JSON Format Examples")

        with st.expander("Job Requirements Format"):
            st.code('''
{
  "title": "Senior HR Manager",
  "min_years_experience": 5,
  "preferred_years_experience": 8,
  "skills": {
    "HRIS": 0.92,
    "Employee Relations": 0.88,
    "Communication": 0.90
  }
}
            ''', language='json')

        with st.expander("Candidates Format"):
            st.code('''
[
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
      "Employee Relations": 0.85
    }
  }
]
            ''', language='json')

    # Main content
    if job_file is None or candidates_file is None:
        st.info("👈 Please upload both job requirements and candidates JSON files to begin")

        st.markdown("## 📖 How It Works")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 1️⃣ Define Job")
            st.write("Upload job requirements with:")
            st.write("• Required skills & importance")
            st.write("• Minimum years of experience")
            st.write("• Preferred years (optional)")

        with col2:
            st.markdown("### 2️⃣ Upload Candidates")
            st.write("Provide candidate profiles with:")
            st.write("• Skill proficiencies (0-1)")
            st.write("• Years of experience")
            st.write("• Unique identifiers")

        with col3:
            st.markdown("### 3️⃣ Get Rankings")
            st.write("System generates:")
            st.write("• Ranked candidate list")
            st.write("• Skill gap analysis")
            st.write("• Experience matching")
            st.write("• Downloadable reports")

        st.markdown("---")
        st.markdown("### ✨ Key Features")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Graph-Based Ranking**")
            st.write("Uses PageRank algorithm to flow importance from job requirements through skills to candidates")

            st.markdown("**Experience Integration**")
            st.write("Considers years of experience in ranking with configurable weight and modes")

        with col2:
            st.markdown("**Natural Gap Penalty**")
            st.write("Missing skills automatically reduce ranking without manual scoring")

            st.markdown("**Detailed Explanations**")
            st.write("See exactly why each candidate ranked where they did")

        return

    # Process files
    try:
        # Validate files
        is_valid_job, job_msg = validate_json_file(job_file, "job")
        is_valid_cand, cand_msg = validate_json_file(candidates_file, "candidates")

        if not is_valid_job:
            st.error(f"❌ Job file error: {job_msg}")
            return

        if not is_valid_cand:
            st.error(f"❌ Candidates file error: {cand_msg}")
            return

        # Load data
        with st.spinner("📂 Loading data..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path_job = os.path.join(tmpdir, "job.json")
                file_path_candidates = os.path.join(tmpdir, "candidates.json")

                with open(file_path_job, "wb") as f:
                    f.write(job_file.getvalue())
                with open(file_path_candidates, "wb") as f:
                    f.write(candidates_file.getvalue())

                job_requirements, candidates = ATSDataLoader.load_from_json(
                    file_path_job,
                    file_path_candidates
                )

                is_valid, errors = ATSDataLoader.validate_data(job_requirements, candidates)
                if not is_valid:
                    st.error("❌ Data validation failed:")
                    for error in errors:
                        st.error(f"  • {error}")
                    return

        with st.spinner("🧮 Ranking candidates..."):
            ranker = GraphBasedATSRanker(
                experience_weight=experience_weight,
                experience_mode=experience_mode if experience_mode != 'none' else 'boost'
            )
            ranker.build_graph(job_requirements, candidates)
            rankings = ranker.compute_rankings()

            num_total = len(rankings)
            num_top = max(1, int(num_total * top_percentage / 100))
            top_rankings = rankings.head(num_top)

        st.success(f"✅ Successfully ranked {num_total} candidates!")

        # Extract job details
        job_title = job_requirements.get('title', 'Job Position')
        job_skills = job_requirements['skills']
        min_years = job_requirements.get('min_years_experience', 0)
        pref_years = job_requirements.get('preferred_years_experience', None)

        # Summary metrics
        st.markdown("## 📊 Summary")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Job Title", job_title)

        with col2:
            st.metric("Total Candidates", num_total)

        with col3:
            st.metric("Top Candidates", num_top)

        with col4:
            avg_coverage = sum(
                ranker.explain_ranking(row['candidate_id'])['skill_coverage']
                for _, row in top_rankings.iterrows()
            ) / len(top_rankings)
            st.metric("Avg. Skill Coverage", f"{avg_coverage:.1%}")

        with col5:
            top_score = top_rankings.iloc[0]['score']
            st.metric("Top Score", f"{top_score:.4f}")

        # Experience requirements display
        if min_years > 0 or pref_years:
            st.markdown("### 📅 Experience Requirements")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Minimum Years", f"{min_years:.0f} years")
            with col2:
                if pref_years:
                    st.metric("Preferred Years", f"{pref_years:.0f} years")
                else:
                    st.metric("Preferred Years", "Not specified")

        st.markdown("---")

        # Rankings overview
        st.markdown(f"## 🏆 Top {top_percentage}% Candidates")

        display_rankings = top_rankings.copy()
        display_rankings['score'] = display_rankings['score'].apply(lambda x: f"{x:.6f}")
        display_rankings['years_experience'] = display_rankings['years_experience'].apply(lambda x: f"{x:.1f}")

        st.dataframe(
            display_rankings,
            use_container_width=True,
            hide_index=True
        )

        # CSV EXPORTS
        st.markdown("### 📥 Download Ranking Reports")

        st.markdown("""
        Choose your export format:
        - **Basic CSV**: Simple ranking table (rank, candidate ID, score, years)
        - **Detailed CSV**: Comprehensive breakdown with skills, experience, and recommendations
        """)

        col1, col2 = st.columns(2)

        with col1:
            # Basic CSV
            basic_csv = display_rankings.to_csv(index=False)
            st.download_button(
                label="📄 Download Basic Rankings",
                data=basic_csv,
                file_name="ats_rankings_basic.csv",
                mime="text/csv",
                help="Quick overview with rank, ID, score, and years"
            )

        with col2:
            # DETAILED CSV
            detailed_csv_data = []

            for _, row in top_rankings.iterrows():
                candidate_id = row['candidate_id']
                explanation = ranker.explain_ranking(candidate_id)

                # Format skills possessed with detailed breakdown
                skills_detail = " | ".join([
                    f"{s['skill']}(Prof:{s['proficiency']:.2f},Imp:{s['importance']:.2f},Contrib:{s['contribution']:.4f})"
                    for s in explanation['top_skills']
                ])

                # Format missing skills
                missing_detail = " | ".join([
                    f"{m['skill']}(Imp:{m['importance']:.2f})"
                    for m in explanation['missing_skills']
                ]) if explanation['missing_skills'] else "None"

                # Calculate useful metrics
                avg_prof = sum(s['proficiency'] for s in explanation['top_skills']) / len(explanation['top_skills']) if explanation['top_skills'] else 0
                weighted_match = sum(s['proficiency'] * s['importance'] for s in explanation['top_skills']) / len(explanation['top_skills']) if explanation['top_skills'] else 0

                # Generate recommendation
                years_exp = explanation['years_experience']
                exp_status = explanation['experience_status']

                if explanation['skill_coverage'] >= 0.8 and avg_prof >= 0.7 and exp_status in ['meets requirement', 'exceeds preferred']:
                    recommendation = "Excellent Match - Immediate Interview"
                elif explanation['skill_coverage'] >= 0.7 and avg_prof >= 0.6:
                    recommendation = "Strong Match - Highly Recommended"
                elif explanation['skill_coverage'] >= 0.6:
                    recommendation = "Good Match - Consider Interview"
                else:
                    recommendation = "Potential Match - May Need Training"

                detailed_csv_data.append({
                    'Rank': row['rank'],
                    'Candidate_ID': candidate_id,
                    'Years_Experience': years_exp,
                    'Experience_Status': exp_status,
                    'Final_Score': f"{row['score']:.6f}",
                    'Experience_Contribution': f"{explanation['experience_contribution']:.6f}",
                    'Skill_Coverage_%': f"{explanation['skill_coverage'] * 100:.1f}%",
                    'Skills_Matched': len(explanation['top_skills']),
                    'Skills_Missing': len(explanation['missing_skills']),
                    'Avg_Proficiency': f"{avg_prof:.3f}",
                    'Weighted_Match_Score': f"{weighted_match:.3f}",
                    'Recommendation': recommendation,
                    'All_Skills_Detail': skills_detail,
                    'Missing_Skills_Detail': missing_detail
                })

            detailed_df = pd.DataFrame(detailed_csv_data)
            detailed_csv = detailed_df.to_csv(index=False)

            st.download_button(
                label="📊 Download Detailed Rankings",
                data=detailed_csv,
                file_name=f"ats_rankings_detailed_top{top_percentage}pct.csv",
                mime="text/csv",
                help="Full analysis with skill details, experience, gaps, and recommendations"
            )

        st.markdown("---")

        # Detailed candidate reports
        st.markdown("## 📋 Detailed Candidate Reports")

        for idx, row in top_rankings.iterrows():
            candidate_id = row['candidate_id']
            rank = row['rank']
            score = row['score']
            years_exp = row['years_experience']

            explanation = ranker.explain_ranking(candidate_id)
            exp_status = explanation['experience_status']

            # Rank badge
            if rank == 1:
                badge_class = "rank-1"
                badge_text = "🥇 #1"
            elif rank == 2:
                badge_class = "rank-2"
                badge_text = "🥈 #2"
            elif rank == 3:
                badge_class = "rank-3"
                badge_text = "🥉 #3"
            else:
                badge_class = "rank-other"
                badge_text = f"#{rank}"

            # Experience badge
            if exp_status == "below requirement":
                exp_badge_class = "exp-below"
                exp_badge_text = "⚠️ Below Req"
            elif exp_status == "exceeds preferred":
                exp_badge_class = "exp-exceeds"
                exp_badge_text = "⭐ Exceeds Pref"
            elif exp_status == "meets requirement":
                exp_badge_class = "exp-meets"
                exp_badge_text = "✓ Meets Req"
            else:
                exp_badge_class = "exp-meets"
                exp_badge_text = "N/A"

            st.markdown(f"""
            <div class="candidate-card">
                <h3>
                    <span class="rank-badge {badge_class}">{badge_text}</span>
                    {candidate_id}
                    <span class="exp-badge {exp_badge_class}">{exp_badge_text}</span>
                </h3>
            </div>
            """, unsafe_allow_html=True)

            # Metrics
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Score", f"{score:.6f}")

            with col2:
                st.metric("Years Experience", f"{years_exp:.1f}")

            with col3:
                st.metric("Skill Coverage", f"{explanation['skill_coverage']:.1%}")

            with col4:
                num_missing = len(explanation['missing_skills'])
                st.metric("Missing Skills", num_missing)

            with col5:
                st.metric("Exp. Contribution", f"{explanation['experience_contribution']:.6f}")

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(
                    create_skill_coverage_chart(explanation, job_skills),
                    use_container_width=True
                )

            with col2:
                skills_to_show = explanation['top_skills'] if show_all_skills else explanation['top_skills'][:5]
                st.plotly_chart(
                    create_contribution_chart(skills_to_show),
                    use_container_width=True
                )

            # Skills breakdown
            st.markdown("#### 🎯 Skills Breakdown")

            skills_to_display = explanation['top_skills'] if show_all_skills else explanation['top_skills'][:5]

            skills_df = pd.DataFrame(skills_to_display)
            skills_df['proficiency'] = skills_df['proficiency'].apply(lambda x: f"{x:.2f}")
            skills_df['importance'] = skills_df['importance'].apply(lambda x: f"{x:.2f}")
            skills_df['contribution'] = skills_df['contribution'].apply(lambda x: f"{x:.6f}")

            st.dataframe(skills_df, use_container_width=True, hide_index=True)

            # Missing skills
            if explanation['missing_skills']:
                st.warning("⚠️ Missing Skills:")
                missing_df = pd.DataFrame(explanation['missing_skills'])
                missing_df['importance'] = missing_df['importance'].apply(lambda x: f"{x:.2f}")
                st.dataframe(missing_df, use_container_width=True, hide_index=True)
            else:
                st.success("✅ Complete profile - No missing skills!")

            st.markdown("---")

    except Exception as e:
        st.error(f"❌ Error processing files: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()