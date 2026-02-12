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

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime

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
    if not job_skills:
        fig = go.Figure()
        fig.update_layout(
            height=400,
            annotations=[dict(text="No job skills defined", x=0.5, y=0.5,
                              xref="paper", yref="paper", showarrow=False, font=dict(size=14))]
        )
        return fig

    categories = list(job_skills.keys())

    # Build a fast lookup: skill name → proficiency (already 0-1 from ranker)
    prof_lookup = {s['skill']: s['proficiency'] for s in explanation['top_skills']}

    # Candidate proficiencies: clamp to [0, 1] in case ranker returns slightly >1
    candidate_profs = [min(max(prof_lookup.get(skill, 0), 0), 1) for skill in categories]

    # Job importances are already 0-1
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
    if not top_skills:
        fig = go.Figure()
        fig.update_layout(
            height=300,
            title="Skill Contributions to Final Score",
            annotations=[dict(text="No matching skills for this job", x=0.5, y=0.5,
                              xref="paper", yref="paper", showarrow=False, font=dict(size=14))]
        )
        return fig

    df = pd.DataFrame(top_skills)
    if 'contribution' not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            height=300,
            title="Skill Contributions to Final Score",
            annotations=[dict(text="No contribution data available", x=0.5, y=0.5,
                              xref="paper", yref="paper", showarrow=False, font=dict(size=14))]
        )
        return fig

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


def generate_hr_pdf_report(top_rankings, ranker, job_skills, job_requirements, top_percentage):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    story = []

    # Extract job details
    job_title = job_requirements.get('title', job_requirements.get('job_title', 'Not Specified'))

    # Create custom styles
    styles = getSampleStyleSheet()

    # Title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    # Subtitle style
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.grey,
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica'
    )

    # Section header style
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )

    # Candidate name style
    candidate_style = ParagraphStyle(
        'CandidateName',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )

    # Body text style
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        fontName='Helvetica'
    )

    # Small text style
    small_style = ParagraphStyle(
        'SmallText',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        fontName='Helvetica'
    )

    # ========== COVER PAGE ==========
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("CANDIDATE RANKING REPORT", title_style))
    story.append(Paragraph(f"Top {top_percentage}% Candidates Analysis", subtitle_style))
    story.append(Spacer(1, 0.3 * inch))

    rankings = ranker.compute_rankings()
    # Report metadata table
    report_data = [
        ['Report Generated:', datetime.now().strftime('%B %d, %Y at %I:%M %p')],
        ['Total Candidates Evaluated:', str(len(rankings))],
        ['Job Position:', job_title],
        ['Analysis Method:', 'Graph-Based Ranking']
    ]

    metadata_table = Table(report_data, colWidths=[2.5 * inch, 4 * inch])
    metadata_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
        ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 10),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1f77b4')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(metadata_table)
    story.append(Spacer(1, 0.5 * inch))

    # Key findings box
    story.append(Paragraph("EXECUTIVE SUMMARY", section_style))

    # Calculate key metrics
    avg_score = rankings['normalized_score'].mean()
    avg_experience = rankings['years_experience'].mean()
    top_candidate = top_rankings.iloc[0]
    num_above_70 = (top_rankings['normalized_score'] > 70).sum()

    summary_text = f"""
    <b>Ranking Overview:</b><br/>
    • <b>Highest Score:</b> {top_candidate['normalized_score']:.4f} - {top_candidate['candidate_id']}<br/>
    • <b>Average Candidate Score:</b> {avg_score:.4f}<br/>
    • <b>Average Experience:</b> {avg_experience:.1f} years<br/>
    • <b>Recommendation:</b> Top {num_above_70:.1f} candidates show strong alignment with job requirements and are recommended for immediate interview scheduling.
    """

    story.append(Paragraph(summary_text, body_style))
    story.append(PageBreak())

    story.append(Paragraph("CANDIDATE RANKINGS AT A GLANCE", section_style))
    story.append(Spacer(1, 0.2 * inch))

    # Prepare summary table data
    summary_table_data = [['Rank', 'Candidate ID', 'normalized_score', 'Experience (Years)', 'Quick Assessment']]

    for idx, row in top_rankings.iterrows():
        explanation = ranker.explain_ranking(row['candidate_id'])

        # Generate quick assessment
        if explanation['normalized_score'] >= 80:
            assessment = "Excellent Match"
        elif explanation['normalized_score'] >= 70:
            assessment = "Strong Match "
        elif explanation['normalized_score'] >= 60:
            assessment = "Good Match"
        else:
            assessment = "Potential Match"

        summary_table_data.append([
            str(row['rank']),
            row['candidate_id'],
            f"{row['normalized_score']:.4f}",
            f"{row['years_experience']:.1f}",
            assessment
        ])

    summary_table = Table(summary_table_data, colWidths=[0.6 * inch, 2 * inch, 1.5 * inch, 1.3 * inch, 1.7 * inch])
    summary_table.setStyle(TableStyle([
        # Header styling
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),

        # Body styling
        ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
        ('ALIGN', (0, 1), (0, -1), 'CENTER'),  # Rank center
        ('ALIGN', (1, 1), (1, -1), 'LEFT'),  # Candidate ID left
        ('ALIGN', (2, 1), (-1, -1), 'CENTER'),  # Rest center

        # Borders and padding
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),

        # Alternating row colors
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
    ]))

    story.append(summary_table)
    story.append(PageBreak())

    story.append(Paragraph("DETAILED CANDIDATE ANALYSIS", section_style))
    story.append(Spacer(1, 0.2 * inch))

    for idx, row in top_rankings.iterrows():
        candidate_id = row['candidate_id']
        rank = row['rank']
        score = row['normalized_score']
        years_exp = row['years_experience']

        explanation = ranker.explain_ranking(candidate_id)
        exp_status = explanation['experience_status']

        # Candidate header with rank badge
        if rank <= 3:
            rank_emoji = ['🥇', '🥈', '🥉'][rank - 1]
            header_text = f"{rank_emoji} RANK #{rank} - {candidate_id}"
        else:
            header_text = f"RANK #{rank} - {candidate_id}"

        story.append(Paragraph(header_text, candidate_style))

        # Key metrics table
        metrics_data = [
            ['Overall Score', f"{score:.4f}"],
            ['Years of Experience', f"{years_exp:.1f} years ({exp_status})"],
            ['Skill Coverage', f"{explanation['skill_coverage'] * 100:.1f}%"],
            ['Skills Matched', f"{len(explanation['top_skills'])} skills"],
            ['Missing Skills', f"{len(explanation['missing_skills'])} skills"]
        ]

        metrics_table = Table(metrics_data, colWidths=[2 * inch, 4.5 * inch])
        metrics_table.setStyle(TableStyle([
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 10),
            ('FONT', (1, 0), (1, -1), 'Helvetica', 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1f77b4')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 0.15 * inch))

        # Calculate useful metrics for recommendation
        avg_prof = sum(s['proficiency'] for s in explanation['top_skills']) / len(explanation['top_skills']) if \
        explanation['top_skills'] else 0

        # HR Recommendation section
        story.append(Paragraph("<b>HR RECOMMENDATION:</b>", body_style))

        if explanation['skill_coverage'] >= 0.8 and avg_prof >= 0.7 and exp_status in ['meets requirement',
                                                                                       'exceeds preferred']:
            recommendation = "  EXCELLENT MATCH - IMMEDIATE INTERVIEW"
            recommendation_detail = "This candidate demonstrates exceptional alignment with job requirements. Strong skills coverage with high proficiency levels and appropriate experience. Recommend fast-tracking to interview stage."
            rec_color = colors.HexColor('#28a745')
        elif explanation['skill_coverage'] >= 0.7 and avg_prof >= 0.6:
            recommendation = "STRONG MATCH - HIGHLY RECOMMENDED"
            recommendation_detail = "Candidate shows very good fit with solid skills and experience. Minor gaps are manageable through onboarding. Recommend scheduling interview."
            rec_color = colors.HexColor('#5bc0de')
        elif explanation['skill_coverage'] >= 0.6:
            recommendation = "GOOD MATCH - CONSIDER FOR INTERVIEW"
            recommendation_detail = "Candidate has foundational skills and meets basic requirements. May need some additional training but shows potential. Consider for interview based on pipeline needs."
            rec_color = colors.HexColor('#ffc107')
        else:
            recommendation = "POTENTIAL MATCH - MAY NEED SIGNIFICANT TRAINING"
            recommendation_detail = "Candidate has some relevant skills but notable gaps exist. Would require substantial training and development. Consider only if candidate pool is limited."
            rec_color = colors.HexColor('#dc3545')

        rec_box = Table([[recommendation]], colWidths=[6.5 * inch])
        rec_box.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), rec_color),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('FONT', (0, 0), (-1, -1), 'Helvetica-Bold', 11),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(rec_box)
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph(recommendation_detail, body_style))
        story.append(Spacer(1, 0.15 * inch))

        # Top Skills section
        story.append(Paragraph("<b>KEY STRENGTHS:</b>", body_style))

        if explanation['top_skills']:
            skills_data = [['Skill', 'Proficiency', 'Importance', 'Match Quality']]

            for skill in explanation['top_skills']:  # Top 6 skills
                proficiency_pct = f"{skill['proficiency']*100 :.0f}%"
                importance_pct = f"{skill['importance'] * 100:.0f}%"

                # Match quality indicator
                match_score = skill['proficiency'] * skill['importance']
                if match_score >= 0.7:
                    match_quality = "Excellent ⭐"
                elif match_score >= 0.5:
                    match_quality = "Good ✓"
                else:
                    match_quality = "Fair"

                skills_data.append([
                    skill['skill'],
                    proficiency_pct,
                    importance_pct,
                    match_quality
                ])

            skills_table = Table(skills_data, colWidths=[2.2 * inch, 1.3 * inch, 1.3 * inch, 1.7 * inch])
            skills_table.setStyle(TableStyle([
                # Header
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e0e0e0')),
                ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 9),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),

                # Body
                ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('ALIGN', (1, 1), (-1, -1), 'CENTER'),

                # Borders
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))

            story.append(skills_table)
        else:
            story.append(Paragraph("No significant skills identified.", small_style))

        story.append(Spacer(1, 0.15 * inch))

        # Missing Skills section
        if explanation['missing_skills']:
            story.append(Paragraph("<b>DEVELOPMENT AREAS:</b>", body_style))

            missing_skills_text = ", ".join([
                f"{m['skill']} (Importance: {m['importance'] * 100:.0f}%)"
                for m in explanation['missing_skills'][:5]  # Top 5 missing
            ])

            story.append(Paragraph(missing_skills_text, small_style))
            story.append(Spacer(1, 0.1 * inch))

            # Training recommendation
            high_importance_missing = [m for m in explanation['missing_skills'] if m['importance'] >= 0.7]
            if high_importance_missing:
                training_text = f"⚠️ <b>Note:</b> Candidate is missing {len(high_importance_missing)} high-importance skill(s). Consider structured training program if hired."
                story.append(Paragraph(training_text, body_style))
        else:
            story.append(Paragraph("<b>DEVELOPMENT AREAS:</b> None - Complete skill set! ✓", body_style))

        # Add separator between candidates
        story.append(Spacer(1, 0.2 * inch))
        separator_table = Table([['']], colWidths=[6.5 * inch])
        separator_table.setStyle(TableStyle([
            ('LINEABOVE', (0, 0), (-1, 0), 1, colors.HexColor('#cccccc'))
        ]))
        story.append(separator_table)
        story.append(Spacer(1, 0.2 * inch))

        # Page break after every 2 candidates for readability
        story.append(PageBreak())

    story.append(Paragraph("APPENDIX: RANKING METHODOLOGY", section_style))
    story.append(Spacer(1, 0.1 * inch))

    methodology_text = """
    <b>How Rankings Are Calculated:</b><br/><br/>

    This report uses an advanced <b>Graph-Based AI Ranking System</b> that evaluates candidates based on multiple factors:<br/><br/>

    <b>1. Skill Matching:</b><br/>
    • Compares candidate skills against job requirements<br/>
    • Weights skills by their importance to the role<br/>
    • Considers proficiency levels, not just presence/absence<br/><br/>

    <b>2. Experience Evaluation:</b><br/>
    • Factors in years of relevant experience<br/>
    • Compares against minimum and preferred experience levels<br/>
    • Balances experience with skill proficiency<br/><br/>

    <b>3. Holistic Scoring:</b><br/>
    • Uses graph algorithms to model relationships between skills<br/>
    • Accounts for skill synergies and combinations<br/>
    • Produces a normalized score (0-100) for fair comparison<br/><br/>

    <b>Score Interpretation:</b><br/>
    • 80+ : Exceptional candidate, very rare<br/>
    • 70-79 : Excellent candidate, strong hire<br/>
    • 60-69 : Good candidate, recommended<br/>
    • 50-59 : Fair candidate, consider based on needs<br/>
    • Below 50 : May need significant development<br/><br/>

    <i>Note: Rankings should be used as one input in the hiring decision. Consider conducting interviews with top candidates to assess cultural fit, communication skills, and other qualitative factors.</i>
    """

    story.append(Paragraph(methodology_text, body_style))

    # Footer on last page
    story.append(Spacer(1, 0.3 * inch))
    footer_text = f"Report generated on {datetime.now().strftime('%B %d, %Y')} | ATS Candidate Ranking System v2.0"
    story.append(Paragraph(footer_text, small_style))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer


def validate_json_file(uploaded_file, file_type):
    try:
        content = uploaded_file.read()
        data = json.loads(content)
        uploaded_file.seek(0)

        if file_type == "job":
            # Accept a single job dict, a list of job dicts, or {"jobs": [...]}
            if isinstance(data, dict):
                return True, "Valid job requirements file"
            elif isinstance(data, list) and all(isinstance(j, dict) for j in data):
                return True, f"Valid multi-job file ({len(data)} jobs)"
            else:
                return False, "Job file must be a JSON object or an array of job objects"

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
    st.markdown('<div class="sub-header">Graph-Based Intelligent Candidate Ranking with Experience</div>',
                unsafe_allow_html=True)

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
            st.markdown("**Single job:**")
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
            st.markdown("**Multiple jobs (array):**")
            st.code('''
[
  {
    "title": "Senior HR Manager",
    "min_years_experience": 5,
    "skills": { "HRIS": 0.92, "Communication": 0.90 }
  },
  {
    "title": "Recruiter",
    "min_years_experience": 2,
    "skills": { "Sourcing": 0.85, "Communication": 0.88 }
  }
]
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

                all_jobs, candidates = ATSDataLoader.load_from_json(
                    file_path_job,
                    file_path_candidates
                )

                is_valid, errors = ATSDataLoader.validate_data(all_jobs, candidates)
                if not is_valid:
                    st.error("❌ Data validation failed:")
                    for error in errors:
                        st.error(f"  • {error}")
                    return

        # ── Job selector (shown only when multiple jobs are present) ────────
        if len(all_jobs) > 1:
            job_titles = [j.get("title", f"Job {i+1}") for i, j in enumerate(all_jobs)]
            selected_job_title = st.selectbox(
                "🗂️ Select job to rank against",
                options=job_titles,
                help="Your job file contains multiple positions. Choose which one to rank candidates against."
            )
            selected_job_index = job_titles.index(selected_job_title)
        else:
            selected_job_index = 0

        job_requirements = all_jobs[selected_job_index]

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

        st.success(f"✅ Successfully ranked {num_total} candidates against **{job_requirements.get('title', 'Job Position')}**!")

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
            top_score = top_rankings.iloc[0]['normalized_score']
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
        display_rankings['normalized_score'] = display_rankings['normalized_score'].apply(lambda x: f"{x:.6f}")
        display_rankings['years_experience'] = display_rankings['years_experience'].apply(lambda x: f"{x:.1f}")

        st.dataframe(
            display_rankings,
            use_container_width=True,
            hide_index=True
        )

        # CSV AND PDF EXPORTS
        st.markdown("### 📥 Download Ranking Reports")

        st.markdown("""
        Choose your export format:
        - **Basic CSV**: Simple ranking table (rank, candidate ID, score, years)
        - **Detailed CSV**: Comprehensive breakdown with skills, experience, and recommendations
        - **HR-Friendly PDF Report**: Professional document with executive summary and detailed candidate profiles
        """)

        col1, col2, col3 = st.columns(3)

        with col1:
            # Basic CSV
            basic_csv = display_rankings.to_csv(index=False)
            st.download_button(
                label="📄 Download Basic CSV",
                data=basic_csv,
                file_name="ats_rankings_basic.csv",
                mime="text/csv",
                help="Quick overview with rank, ID, score, and years"
            )

        with col2:
            # DETAILED CSV - Enhanced for HR readability
            detailed_csv_data = []

            for _, row in top_rankings.iterrows():
                candidate_id = row['candidate_id']
                explanation = ranker.explain_ranking(candidate_id)

                # Format skills possessed with detailed breakdown
                skills_detail = "; ".join([
                    f"{s['skill']} (Proficiency: {s['proficiency'] :.0f}%, Importance: {s['importance'] * 100:.0f}%)"
                    for s in explanation['top_skills']
                ])

                # Format missing skills
                missing_detail = "; ".join([
                    f"{m['skill']} (Importance: {m['importance'] * 100:.0f}%)"
                    for m in explanation['missing_skills']
                ]) if explanation['missing_skills'] else "None - Complete skill set!"

                # Calculate useful metrics
                avg_prof = sum(s['proficiency'] for s in explanation['top_skills']) / len(explanation['top_skills']) if \
                explanation['top_skills'] else 0
                weighted_match = sum(s['proficiency'] * s['importance'] for s in explanation['top_skills']) / len(
                    explanation['top_skills']) if explanation['top_skills'] else 0

                # Generate recommendation
                years_exp = explanation['years_experience']
                exp_status = explanation['experience_status']

                if explanation['skill_coverage'] >= 0.8 and avg_prof >= 0.7 and exp_status in ['meets requirement',
                                                                                               'exceeds preferred']:
                    recommendation = "⭐ EXCELLENT MATCH - Immediate Interview"
                    priority = "High"
                elif explanation['skill_coverage'] >= 0.7 and avg_prof >= 0.6:
                    recommendation = "✓ STRONG MATCH - Highly Recommended"
                    priority = "High"
                elif explanation['skill_coverage'] >= 0.6:
                    recommendation = "◆ GOOD MATCH - Consider Interview"
                    priority = "Medium"
                else:
                    recommendation = "△ POTENTIAL MATCH - May Need Training"
                    priority = "Low"

                # Create training needs assessment
                high_importance_missing = [m for m in explanation['missing_skills'] if m['importance'] >= 0.7]
                if high_importance_missing:
                    training_needs = f"YES - Missing {len(high_importance_missing)} critical skill(s)"
                elif explanation['missing_skills']:
                    training_needs = f"MINIMAL - Missing {len(explanation['missing_skills'])} non-critical skill(s)"
                else:
                    training_needs = "NO - Complete skill match"

                detailed_csv_data.append({
                    'Rank': row['rank'],
                    'Candidate ID': candidate_id,
                    'Overall Score': f"{row['normalized_score']:.4f}",
                    'Hiring Priority': priority,
                    'HR Recommendation': recommendation,

                    # Experience Section
                    'Years of Experience': f"{years_exp:.1f}",
                    'Experience Status': exp_status.title(),
                    'Experience Score Contribution': f"{explanation['experience_contribution']:.4f}",

                    # Skills Overview
                    'Skill Match Rate': f"{explanation['skill_coverage'] * 100:.1f}%",
                    'Number of Skills Matched': len(explanation['top_skills']),
                    'Number of Skills Missing': len(explanation['missing_skills']),
                    'Average Skill Proficiency': f"{avg_prof * 100:.1f}%",
                    'Weighted Skill Match Score': f"{weighted_match:.3f}",

                    # Development Assessment
                    'Training Required': training_needs,

                    # Detailed Skills (for reference)
                    'Possessed Skills (Details)': skills_detail,
                    'Missing Skills (Details)': missing_detail
                })

            detailed_df = pd.DataFrame(detailed_csv_data)
            detailed_csv = detailed_df.to_csv(index=False)

            st.download_button(
                label="📊 Download Detailed CSV",
                data=detailed_csv,
                file_name=f"ats_rankings_detailed_top{top_percentage}pct.csv",
                mime="text/csv",
                help="Enhanced HR-friendly analysis with recommendations and detailed breakdowns"
            )

        with col3:
            # PROFESSIONAL PDF REPORT
            try:
                pdf_buffer = generate_hr_pdf_report(
                    top_rankings,
                    ranker,
                    job_skills,
                    job_requirements,  # Pass the full job_requirements dict
                    top_percentage
                )

                st.download_button(
                    label="📑 Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"ats_candidate_report_top{top_percentage}pct.pdf",
                    mime="application/pdf",
                    help="Professional HR-ready report with executive summary and detailed candidate profiles"
                )
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")

        st.markdown("---")

        # Detailed candidate reports
        st.markdown("## 📋 Detailed Candidate Reports")

        for idx, row in top_rankings.iterrows():
            candidate_id = row['candidate_id']
            rank = row['rank']
            score = row['score']
            display_score = row['normalized_score']
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
                st.metric("Score", f"{display_score:.6f}")

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
                    use_container_width=True,
                    key=f"radar_{candidate_id}"
                )

            with col2:
                skills_to_show = explanation['top_skills'] if show_all_skills else explanation['top_skills'][:5]
                st.plotly_chart(
                    create_contribution_chart(skills_to_show),
                    use_container_width=True,
                    key=f"bar_{candidate_id}"
                )

            # Skills breakdown
            st.markdown("#### 🎯 Skills Breakdown")

            skills_to_display = explanation['top_skills'] if show_all_skills else explanation['top_skills'][:5]

            if skills_to_display:
                skills_df = pd.DataFrame(skills_to_display)
                skills_df['proficiency'] = skills_df['proficiency'].apply(lambda x: f"{x*100:.2f}%")
                st.dataframe(skills_df, use_container_width=True, hide_index=True)
            else:
                st.info("ℹ️ This candidate has no skills matching this job's requirements.")

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