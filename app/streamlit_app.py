import streamlit as st
import pickle
import re
import pdfplumber

def clean_resume(text):
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'RT|cc', '  ', text)
    text = re.sub(r'#\S+', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_skills(resume_text):
    skills_list = [
        "Python", "Machine Learning", "Java", "SQL", "Deep Learning", "Data Analysis", "Data Visualization", "Pandas", "Numpy", "Scikit-learn", "TensorFlow", "Keras", "Matplotlib", "Seaborn", "Excel", "Power BI", "Tableau", "Statistics", "HTML", "CSS", "JavaScript", "React", "Node.js", "Django", "Flask", "Spring Boot", "My SQL", "MongoDB", "GIT", "GitHub", "AWS", "Docker", "Linux", "NLP", "Natural Language Processing", "Computer Vision" ]
    
    resume_text = resume_text.lower()
    found_skills = []

    for skills in skills_list:
        if skills.lower() in resume_text:
            found_skills.append(skills.title())

    return sorted(list(set(found_skills)))

def get_required_skills(category):
    required_skills = {
        "Data Science": ["Python", "SQL", "Machine Learning", "Data Analysis", "Pandas", "Numpy", "Statistics", "Data Visualization", "Scikit Learn", "Matplotlib" ],
        "Python Developer": ["Python", "Django", "Flask", "GIT", "SQL", "Github", "APIs", "Backend Development"],
        "Java Developer": ["Java", "Spring Boot", "MySQL", "SQL", "Git", "Backend Development", "OOP"],
        "Web Designing": ["HTML", "CSS", "Javascript", "React", "UI Design","Responsive Design"],
        "Database": ["SQL", "MySQL", "Database Design", "Data Modeling","Query Optimization"],
        "DevOps Engineer": ["Linux", "Docker", "AWS", "Git", "CI/CD","Cloud Computing"],
        "Testing": ["Manual Testing", "Automation Testing", "Selenium","Test Cases", "Bug Tracking"],
        "Business Analyst": ["Excel", "SQL", "Power BI", "Tableau","Data Analysis", "Requirement Analysis"],
        }        

    return required_skills.get(category, [])

def calculate_resume_score(matched_skills, required_skills):
    if len(required_skills) == 0:
        return 50
    
    score = (len(matched_skills) / len(required_skills)) * 100
    return round(score, 2)

def generate_suggestions(missing_skills, score):
    suggestions = []

    if score >= 80:
        suggestions.append("Your resume is strong for this role.")
        suggestions.append("Add more projects achievements and measurable results to make it better.")
    elif score >= 50:
        suggestions.append("Your resume has a decent match, but it needs improvements.")
        suggestions.append("Add the missing skills and include real projects related to this job role.")
    else:
        suggestions.append("Your resume match is weak for this role.")
        suggestions.append("Focus on learning the missing skills and add practical project experiences.")

    if missing_skills:
        suggestions.append("Important missing skills: " + ", ".join(missing_skills[:5]))

    return suggestions

model_path = "/Users/vikirthan17/Documents/Project_04_AI_Resume_Analyzer/models/resume_classifier_model.pkl"
vectorizer_path = "/Users/vikirthan17/Documents/Project_04_AI_Resume_Analyzer/models/tfidf_vectorizer.pkl"

with open(model_path, "rb") as file:
    classifier_model = pickle.load(file)

with open(vectorizer_path, "rb") as file:
    tfidf_vectorizer = pickle.load(file)

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="centered"
)

st.title("📄 AI Resume Analyzer")
st.write(
    "Upload your resume PDF or paste resume text to predict the job category, "
    "extract skills, calculate resume score, and get improvement suggestions."
)

st.sidebar.title("Project Info")
st.sidebar.write("Project: AI Resume Analyzer")
st.sidebar.write("Model: Logistic Regression")
st.sidebar.write("Feature Extraction: TF-IDF")
st.sidebar.write("Dataset: Updated Resume Dataset")

input_option = st.radio(
    "Choose input method:",
    ["Upload Resume PDF", "Paste Resume Text"]
)

resume_text = ""

if input_option == "Upload Resume PDF":
    uploaded_file = st.file_uploader("Upload your resume PDF", type=["pdf"])

    if uploaded_file is not None:
        resume_text = extract_text_from_pdf(uploaded_file)
        st.success("Resume text extracted successfully!")

        with st.expander("View Extracted Resume Text"):
            st.write(resume_text)

else:
    resume_text = st.text_area(
        "Paste your resume text here:",
        height=250,
        placeholder="Paste your resume text here..."
    )

if st.button("Analyze Resume"):
    if resume_text.strip() == "":
        st.warning("Please upload a resume or paste resume text first.")

    else:
        cleaned_text = clean_resume(resume_text)
        text_tfidf = tfidf_vectorizer.transform([cleaned_text])
        prediction = classifier_model.predict(text_tfidf)
        predicted_category = prediction[0]

        found_skills = extract_skills(cleaned_text)
        required_skills = get_required_skills(predicted_category)

        matched_skills = [
            skill for skill in required_skills
            if skill.lower() in [s.lower() for s in found_skills]
        ]

        missing_skills = [
            skill for skill in required_skills
            if skill.lower() not in [s.lower() for s in found_skills]
        ]

        resume_score = calculate_resume_score(matched_skills, required_skills)
        suggestions = generate_suggestions(missing_skills, resume_score)

        st.subheader("Prediction Results")
        st.success(f"Predicted Resume Category: {predicted_category}")

        st.subheader("Resume Match Score")
        st.progress(int(resume_score))
        st.write(f"Resume Score: **{resume_score}%**")

        st.subheader("Extracted Skills")
        if found_skills:
            st.write(", ".join(found_skills))
        else:
            st.warning("No major skills detected.")

        st.subheader("Matched Skills")
        if matched_skills:
            st.write(", ".join(matched_skills))
        else:
            st.warning("No matched skills found for the predicted category.")

        st.subheader("Missing Skills")
        if missing_skills:
            st.write(", ".join(missing_skills))
        else:
            st.success("No major missing skills found.")

        st.subheader("Improvement Suggestions")
        for suggestion in suggestions:
            st.write("- " + suggestion)

        with st.expander("View Cleaned Resume Text"):
            st.write(cleaned_text[:1500])        

