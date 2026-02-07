"""
Graph-Based ATS Ranking System - Streamlit Interface
=====================================================

Professional web interface for ranking candidates using graph-based algorithms.
Upload job requirements and candidate profiles, get top 10% ranked with detailed reports.
"""

import streamlit as st
import pandas as pd
import json
import io
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
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
</style>
""", unsafe_allow_html=True)


def create_skill_coverage_chart(explanation, job_skills):
    """Create a radar chart showing skill coverage"""
    categories = list(job_skills.keys())

    # Get candidate's proficiencies
    candidate_profs = []
    for skill in categories:
        # Find skill in explanation
        skill_data = next((s for s in explanation['top_skills'] if s['skill'] == skill), None)
        if skill_data:
            candidate_profs.append(skill_data['proficiency'])
        else:
            candidate_profs.append(0)

    # Get importances
    importances = [job_skills[skill] for skill in categories]

    fig = go.Figure()

    # Add candidate proficiency
    fig.add_trace(go.Scatterpolar(
        r=candidate_profs,
        theta=categories,
        fill='toself',
        name='Candidate Proficiency',
        line_color='#1f77b4'
    ))

    # Add job importance
    fig.add_trace(go.Scatterpolar(
        r=importances,
        theta=categories,
        fill='toself',
        name='Job Importance',
        line_color='#ff7f0e',
        opacity=0.5
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
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

    fig.update_layout(
        height=300,
        showlegend=False
    )

    return fig


def validate_json_file(uploaded_file, file_type):
    """Validate uploaded JSON file"""
    try:
        content = uploaded_file.read()
        data = json.loads(content)
        uploaded_file.seek(0)  # Reset file pointer

        if file_type == "job":
            # Check if it's a valid job format
            if isinstance(data, dict):
                return True, "Valid job requirements file"
            else:
                return False, "Job file must be a JSON object"

        elif file_type == "candidates":
            # Check if it's a valid candidates format
            if isinstance(data, dict):
                return True, "Valid candidates file"
            else:
                return False, "Candidates file must be a JSON object"

        return True, "Valid JSON file"

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON format: {str(e)}"
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


def main():
    # Header
    st.markdown('<div class="main-header">🎯 ATS Candidate Ranking System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Graph-Based Intelligent Candidate Ranking</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("📤 Upload Data")

        st.markdown("### Job Requirements")
        job_file = st.file_uploader(
            "Upload job requirements JSON",
            type=['json'],
            help="JSON file containing required skills and their importance (0-1)"
        )

        st.markdown("### Candidate Profiles")
        candidates_file = st.file_uploader(
            "Upload candidates JSON",
            type=['json'],
            help="JSON file containing candidate IDs and their skill proficiencies (0-1)"
        )

        st.markdown("---")

        st.markdown("### ⚙️ Settings")
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
  "Python": 0.9,
  "SQL": 0.7,
  "Leadership": 0.5
}
            ''', language='json')

        with st.expander("Candidates Format"):
            st.code('''
{
  "Alice": {
    "Python": 0.8,
    "SQL": 0.6,
    "Leadership": 0.4
  },
  "Bob": {
    "Python": 0.5,
    "SQL": 0.9
  }
}
            ''', language='json')

    # Main content
    if job_file is None or candidates_file is None:
        st.info("👈 Please upload both job requirements and candidates JSON files to begin")

        # Show example/demo section
        st.markdown("## 📖 How It Works")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 1️⃣ Upload")
            st.markdown("""
            - Job requirements JSON
            - Candidates profiles JSON
            - Set your preferences
            """)

        with col2:
            st.markdown("### 2️⃣ Analyze")
            st.markdown("""
            - Graph-based ranking
            - Skill gap detection
            - Complementarity analysis
            """)

        with col3:
            st.markdown("### 3️⃣ Report")
            st.markdown("""
            - Top 10% candidates
            - Detailed explanations
            - Visual insights
            """)

        st.markdown("---")
        st.markdown("## ✨ Key Features")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ✅ **Automatic Gap Penalties**
            - Missing skills lower rankings automatically
            - No manual threshold tuning needed

            ✅ **Complementarity Rewards**
            - Balanced profiles rank higher
            - Prevents skill compensation
            """)

        with col2:
            st.markdown("""
            ✅ **Importance Weighting**
            - Critical skills matter more
            - Fair, interpretable rankings

            ✅ **Fast & Scalable**
            - 100 candidates in ~5ms
            - Production-ready performance
            """)

        return

    # Validate files
    job_valid, job_msg = validate_json_file(job_file, "job")
    cand_valid, cand_msg = validate_json_file(candidates_file, "candidates")

    if not job_valid:
        st.error(f"❌ Job file error: {job_msg}")
        return

    if not cand_valid:
        st.error(f"❌ Candidates file error: {cand_msg}")
        return

    # Process files
    try:
        with st.spinner("🔄 Loading and validating data..."):
            # Save uploaded files temporarily
            tmp_dir = tempfile.gettempdir()
            file_path_job = os.path.join(tmp_dir, "job.json")
            file_path_candidates = os.path.join(tmp_dir, "candidates.json")
            with open(file_path_job, "wb") as f:
                f.write(job_file.getvalue())
            with open(file_path_candidates, "wb") as f:
                f.write(candidates_file.getvalue())

            # Load data
            job_skills, candidates = ATSDataLoader.load_from_json(
                file_path_job,
                file_path_candidates
            )

            # Validate
            is_valid, errors = ATSDataLoader.validate_data(job_skills, candidates)
            if not is_valid:
                st.error("❌ Data validation failed:")
                for error in errors:
                    st.error(f"  • {error}")
                return

        with st.spinner("🧮 Ranking candidates..."):
            # Build graph and rank
            ranker = GraphBasedATSRanker()
            ranker.build_graph(job_skills, candidates)
            rankings = ranker.compute_rankings()

            # Calculate top percentage
            num_total = len(rankings)
            num_top = max(1, int(num_total * top_percentage / 100))
            top_rankings = rankings.head(num_top)

        # Success message
        st.success(f"✅ Successfully ranked {num_total} candidates!")

        # Summary metrics
        st.markdown("## 📊 Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Candidates", num_total)

        with col2:
            st.metric("Top Candidates", num_top)

        with col3:
            avg_coverage = sum(
                ranker.explain_ranking(row['candidate_id'])['skill_coverage']
                for _, row in top_rankings.iterrows()
            ) / len(top_rankings)
            st.metric("Avg. Skill Coverage", f"{avg_coverage:.1%}")

        with col4:
            top_score = top_rankings.iloc[0]['score']
            st.metric("Top Score", f"{top_score:.4f}")

        st.markdown("---")

        # Rankings overview
        st.markdown(f"## 🏆 Top {top_percentage}% Candidates")

        # Display rankings table
        display_rankings = top_rankings.copy()
        display_rankings['score'] = display_rankings['score'].apply(lambda x: f"{x:.6f}")

        st.dataframe(
            display_rankings,
            use_container_width=True,
            hide_index=True
        )

        # Download button for rankings
        csv = display_rankings.to_csv(index=False)
        st.download_button(
            label="📥 Download Rankings (CSV)",
            data=csv,
            file_name="ats_rankings.csv",
            mime="text/csv"
        )

        st.markdown("---")

        # Detailed candidate reports
        st.markdown("## 📋 Detailed Candidate Reports")

        for idx, row in top_rankings.iterrows():
            candidate_id = row['candidate_id']
            rank = row['rank']
            score = row['score']

            explanation = ranker.explain_ranking(candidate_id)

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

            st.markdown(f"""
            <div class="candidate-card">
                <h3>
                    <span class="rank-badge {badge_class}">{badge_text}</span>
                    {candidate_id}
                </h3>
            </div>
            """, unsafe_allow_html=True)

            # Metrics row
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Score", f"{score:.6f}")

            with col2:
                st.metric("Skill Coverage", f"{explanation['skill_coverage']:.1%}")

            with col3:
                num_missing = len(explanation['missing_skills'])
                st.metric("Missing Skills", num_missing)

            # Create two columns for charts
            col1, col2 = st.columns(2)

            with col1:
                # Skill coverage radar chart
                st.plotly_chart(
                    create_skill_coverage_chart(explanation, job_skills),
                    use_container_width=True
                )

            with col2:
                # Contribution bar chart
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

            st.dataframe(
                skills_df,
                use_container_width=True,
                hide_index=True
            )

            # Missing skills warning
            if explanation['missing_skills']:
                st.warning("⚠️ Missing Skills:")
                missing_df = pd.DataFrame(explanation['missing_skills'])
                missing_df['importance'] = missing_df['importance'].apply(lambda x: f"{x:.2f}")
                st.dataframe(
                    missing_df,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.success("✅ Complete profile - No missing skills!")

            st.markdown("---")

        # Export full report
        st.markdown("## 📄 Export Report")

        # Generate comprehensive report
        report_data = []
        for _, row in top_rankings.iterrows():
            candidate_id = row['candidate_id']
            explanation = ranker.explain_ranking(candidate_id)

            report_data.append({
                'Rank': explanation['rank'],
                'Candidate ID': candidate_id,
                'Score': explanation['score'],
                'Skill Coverage': explanation['skill_coverage'],
                'Missing Skills Count': len(explanation['missing_skills']),
                'Top Skill': explanation['top_skills'][0]['skill'] if explanation['top_skills'] else 'N/A',
                'Top Skill Contribution': explanation['top_skills'][0]['contribution'] if explanation[
                    'top_skills'] else 0
            })

        report_df = pd.DataFrame(report_data)

        col1, col2 = st.columns(2)

        with col1:
            # CSV export
            csv_report = report_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Summary Report (CSV)",
                data=csv_report,
                file_name=f"ats_report_top_{top_percentage}pct.csv",
                mime="text/csv"
            )

        with col2:
            # JSON export
            json_report = report_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download Summary Report (JSON)",
                data=json_report,
                file_name=f"ats_report_top_{top_percentage}pct.json",
                mime="application/json"
            )

    except Exception as e:
        st.error(f"❌ Error processing files: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()