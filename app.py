import streamlit as st
import tempfile
import json
from pathlib import Path

from agent import build_graph, build_initial_state
from models import DegreePlan
from pdf_parser import parse_transcript
from main import configure_logging

# Page configuration
st.set_page_config(
    page_title="AI Degree Planning Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        margin-bottom: 20px;
    }
    .card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f9fafb;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🎓 AI Degree Planning Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload your transcript, set an academic goal, and let the AI instantly generate your optimal degree plan.</p>', unsafe_allow_html=True)

# Initialize logging for the agent
configure_logging()

# Sidebar for Inputs
with st.sidebar:
    st.header("1. Upload Transcript")
    uploaded_file = st.file_uploader("Select your transcript (PDF or TXT)", type=["pdf", "txt"], help="Upload your most recent transcript so the AI knows what you've completed.")
    
    st.header("2. Academic Goal")
    goal_input = st.text_area(
        "What do you want to achieve?", 
        value="Plan my next two semesters to finish the AI minor",
        help="Be specific! E.g., 'Plan my next 3 semesters for a CS major'."
    )
    
    st.divider()
    generate_button = st.button("Generate Degree Plan", type="primary")

# Main Content Area
if generate_button:
    if uploaded_file is None:
        st.error("⚠️ Please upload a transcript file first.")
    elif not goal_input.strip():
        st.error("⚠️ Please enter an academic goal.")
    else:
        with st.status("🤖 AI Agent is analyzing your record and planning...", expanded=True) as status:
            try:
                st.write("📄 Processing transcript...")
                
                # Save the uploaded file temporarily
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Parse transcript
                transcript = parse_transcript(tmp_path)
                st.write(f"✅ Identified Student ID: `{transcript.student_id}`")
                st.write(f"✅ Found **{len(transcript.completed_courses)}** completed courses.")
                
                degree_plan = DegreePlan(
                    student_id=transcript.student_id,
                    completed_courses=transcript.completed_courses,
                    planned_semesters=[],
                )

                st.write("⚙️ LLM is building your personalized degree plan (this may take a moment)...")
                state = build_initial_state(goal=goal_input, degree_plan=degree_plan)
                graph = build_graph()
                result = graph.invoke(state)

                final_plan = result["degree_plan"]
                plan_notes = result.get("planning_notes", "")
                final_response = result.get("final_response", "")
                
                status.update(label="Degree Plan Generated Successfully!", state="complete", expanded=False)
                
                # Clean up temporary file
                Path(tmp_path).unlink(missing_ok=True)
                
            except Exception as e:
                status.update(label="An error occurred during planning", state="error", expanded=True)
                st.error(f"Error details: {str(e)}")
                st.stop()
                
        # Display Results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🗓️ Your Personalized Degree Plan")
            if not final_plan.planned_semesters:
                st.info("No future semesters planned. You might have already completed all requirements!")
            else:
                for sem in final_plan.planned_semesters:
                    with st.container():
                        st.markdown(f"#### {sem.semester_name}")
                        course_data = []
                        for course in sem.courses:
                            course_data.append({
                                "Course Code": course.course_code,
                                "Title": course.title,
                                "Credits": course.credits
                            })
                        if course_data:
                            st.table(course_data)
                        else:
                            st.write("No courses scheduled.")
                        st.markdown("---")
        
        with col2:
            st.subheader("📝 Agent Planning Notes")
            with st.container():
                st.info(plan_notes if plan_notes else "No specific planning notes.")
            
            if final_response:
                st.subheader("💬 Final Message")
                st.warning(final_response)
                
            st.divider()
            
            st.subheader("📥 Export")
            st.write("Download your final degree plan in JSON format for your records or academic advisor.")
            
            # Serialize payload compatible with pydantic v1 & v2
            payload = final_plan.model_dump() if hasattr(final_plan, "model_dump") else final_plan.dict()
            plan_json = json.dumps(payload, indent=2)
            
            st.download_button(
                label="Download Degree Plan JSON",
                data=plan_json,
                file_name=f"degree_plan_{transcript.student_id}.json",
                mime="application/json",
                use_container_width=True
            )
else:
    # Landing page state
    st.info("👈 Upload your transcript and set your goals in the sidebar to get started.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card"><h3>📄 1. Upload</h3><p>Upload your current transcript. We support PDF and TXT files. All sensitive info stays local.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><h3>🎯 2. Describe</h3><p>Tell the AI what you want to achieve, like finishing a specific minor, major, or hitting 120 credits.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card"><h3>✨ 3. Generate</h3><p>Our autonomous agent validates prerequisites, checks availability, and maps out your future semesters.</p></div>', unsafe_allow_html=True)
