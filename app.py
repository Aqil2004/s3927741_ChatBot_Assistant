# Cyber Security Course Advisor via AWS Bedrock
# Author: Cyrus Gao, extended by Xiang Li
# Updated: May 2025

import streamlit as st
import json
import boto3
from datetime import datetime
from PyPDF2 import PdfReader

#custom imports
from data_cleaning import clean_course_data, remove_duplicate_courses, clean_pdf_text, retrieve_relevant_text, clean_user_query
import os

# === AWS Configuration === #
COGNITO_REGION = "ap-southeast-2"
BEDROCK_REGION = "ap-southeast-2"
MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
IDENTITY_POOL_ID = "ap-southeast-2:eaa059af-fd47-4692-941d-e314f2bd5a0e"
USER_POOL_ID = "ap-southeast-2_NfoZbDvjD"
APP_CLIENT_ID = "3p3lrenj17et3qfrnvu332dvka"
USERNAME = "s3927741@student.rmit.edu.au" # Replace with your username
PASSWORD = "OniiChanDaisuke<3"    # Replace with your password

kb_text = ""

#Custom helper functions
def load_local_kb(kb_path="KB"):
    json_files = []
    pdf_files = []
    for file in os.listdir(kb_path):
        full_path = os.path.join(kb_path, file)
        if file.endswith(".json"):
            json_files.append(full_path)
        elif file.endswith(".pdf"):
            pdf_files.append(full_path)
    return json_files, pdf_files

def load_and_process_kb(kb_path="KB", user_query=None):
    _, pdf_files = load_local_kb(kb_path)
    query = user_query if user_query and user_query.strip() else "general info"
    cleaning_prompt = "You are an answering assistant that gives precise answers. I have a query and a list of file names. I need you to " \
    "check which of these file names would match the query's needs. If you cannot find any close matches then only respond with 'None' " \
    "otherwise only respond with each file name seperated by a comma (e.g. 'Course_1.pdf,Course_233.pdf,Ligaments.json')." \
    "Query: " + str(query) + "" \
    "File names: " + str(pdf_files)
    response = invoke_bedrock(cleaning_prompt)
    if response != "None":
        pdf_files = [
            os.path.join(kb_path, f.strip().replace("'", "").replace('"', ""))
            for f in response.split(",")
            if f.strip()
        ]
        #st.text(pdf_files)
    kb_text = ""
    if pdf_files:
        extracted_text = extract_clean_pdf_text(pdf_files)
        chunks = extracted_text.split("\n\n")
        relevant_chunks = retrieve_relevant_text(query, chunks, top_k=5)
        kb_text = "\n\n".join(relevant_chunks)
    return kb_text

def extract_clean_pdf_text(pdf_files):
    """Extract and clean all PDF text."""
    all_text = []
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    all_text.append(text)
        except Exception as e:
            all_text.append(f"[Error reading {pdf}: {str(e)}]")
    combined_text = "\n\n".join(all_text)
    return clean_pdf_text(combined_text)

# === Helper: Get AWS Credentials === #
def get_credentials(username, password):
    idp_client = boto3.client("cognito-idp", region_name=COGNITO_REGION)
    response = idp_client.initiate_auth(
        AuthFlow="USER_PASSWORD_AUTH",
        AuthParameters={"USERNAME": username, "PASSWORD": password},
        ClientId=APP_CLIENT_ID,
    )
    id_token = response["AuthenticationResult"]["IdToken"]

    identity_client = boto3.client("cognito-identity", region_name=COGNITO_REGION)
    identity_response = identity_client.get_id(
        IdentityPoolId=IDENTITY_POOL_ID,
        Logins={f"cognito-idp.{COGNITO_REGION}.amazonaws.com/{USER_POOL_ID}": id_token},
    )

    creds_response = identity_client.get_credentials_for_identity(
        IdentityId=identity_response["IdentityId"],
        Logins={f"cognito-idp.{COGNITO_REGION}.amazonaws.com/{USER_POOL_ID}": id_token},
    )

    return creds_response["Credentials"]


# === Helper: Build Prompt from JSON + Structure === #
def build_prompt(courses, user_question, structure=None):
    course_dict = {c["title"]: c for c in courses}

    structure_text = ""
    if structure and "recommended_courses" in structure:
        structure_text += "### Recommended Study Plan by Year:\n"
        for year, course_titles in structure["recommended_courses"].items():
            structure_text += f"**{year.replace('_', ' ').title()}**:\n"
            for title in course_titles:
                course = course_dict.get(title)
                if course:
                    structure_text += f"- {title} ({course['course_code']})\n"
                else:
                    structure_text += f"- {title} (not found in course list)\n"
            structure_text += "\n"

    course_list = []
    for course in courses:
        title = course.get("title", "Untitled")
        code = course.get("course_code", "N/A")
        desc = course.get("description", "No description available.")
        course_type = course.get("course_type", "N/A")
        minor = course.get("minor_track", [])
        minor_info = f", Minor: {minor[0]}" if minor else ""
        course_text = f"- {title} ({code}): {desc}\n  Type: {course_type}{minor_info}"
        course_list.append(course_text)
    full_course_context = "\n".join(course_list)

    prompt = (
        "You are a helpful assistant that supports students in selecting courses from the "
        "Recommend only from the official course list. Each course is categorized as core, capstone, minor, or elective. "
        "Use the recommended structure to suggest suitable courses based on study year and interest.\n\n"
        + structure_text
        + "\n### All Available Courses:\n"
        + full_course_context
        + "\n\nUser:\n" + user_question
    )
    
    return prompt

# --- Combine chat history with new question ---
def build_conversational_prompt(history, new_question, courses=None, structure=None, extracted_text=None):
    """
    Builds a prompt that includes limited chat history for context.
    """
    history_text = ""
    for turn in history[-3:]:  # last 3 exchanges only
        history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

    # Depending on input mode, reuse existing logic
    if courses:
        context = build_prompt(courses, new_question, structure)
    elif extracted_text:
        context = (
            "You are a course advisor. The following is extracted from official course documents:\n\n"
            + extracted_text +
            "\n\nPlease answer the following question based on this information:\n"
            + new_question
        )
    else:
        context = new_question
   
    if not structure and use_local_docs:
        kb_text = load_and_process_kb("KB", new_question)
        if kb_text:
            context += "\n\n### Additional Reference from Local Knowledge Base:\n" + kb_text

    final_prompt = (
        "You are continuing a helpful conversation with a student about RMIT course selection.\n"
        "Here is the previous conversation:\n"
        f"{history_text}\n"
        "Now respond to the user's latest question using the context below:\n\n"
        f"{context}"
    )
    return final_prompt


# === Helper: Extract text from multiple PDFs === #
def extract_text_from_pdfs(pdf_files):
    all_text = []
    for pdf_file in pdf_files:
        try:
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    all_text.append(text.strip())
        except Exception as e:
            all_text.append(f"[Error reading file {pdf_file.name}: {str(e)}]")
    return "\n\n".join(all_text)


# === Helper: Invoke Claude via Bedrock === #
def invoke_bedrock(prompt_text, max_tokens=640, temperature=0.3, top_p=0.9):
    credentials = get_credentials(USERNAME, PASSWORD)

    bedrock_runtime = boto3.client(
        "bedrock-runtime",
        region_name=BEDROCK_REGION,
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretKey"],
        aws_session_token=credentials["SessionToken"],
    )

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "messages": [{"role": "user", "content": prompt_text}]
    }

    response = bedrock_runtime.invoke_model(
        body=json.dumps(payload),
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


# === Streamlit UI === #
st.set_page_config(page_title="RMIT Chatbot Assistant", layout="centered")

#LOGIN ACCESS
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_section = st.sidebar.container()
    login_section.header("🔐 Access Portal")
    username_input = login_section.text_input("Username (student email)", placeholder="(Enter any value)")
    password_input = login_section.text_input("Password", type="password", placeholder="(Enter any value)")

    #if username_input == USERNAME and password_input == PASSWORD:
    if username_input and password_input:
        st.session_state.logged_in = True
        st.rerun()
    else:
        login_section.warning("Enter your username and password to continue. (Any input value can be used)")
        st.stop()
else:
    # Only hide the login elements, not the whole sidebar
    st.sidebar.empty()  # clears just the content, keeps sidebar visible


#MEMORY
if "conversations" not in st.session_state:
    st.session_state.conversations = {"Conversation 1": []}  # dict: name -> list of turns
if "active_convo" not in st.session_state:
    st.session_state.active_convo = "Conversation 1"
chat_history = st.session_state.conversations[st.session_state.active_convo]


st.title("\U0001F393 RMIT Chatbot Advisor")
st.markdown("This assistant helps students in RMIT enrolment and course advice.")

st.subheader("Step 1: Choose your data input format")
upload_mode = st.radio("Select format:", ["Structured JSON files", "Unstructured PDF files"])

#Local documents in context
    


if upload_mode == "Structured JSON files":
    uploaded_courses_json = st.file_uploader("\U0001F4C1 Upload `courses_data.json`", type=["json"], key="courses")
    uploaded_structure_json = st.file_uploader("\U0001F4C1 Upload `cyber_security_program_structure.json`", type=["json"], key="structure")
    uploaded_pdfs = None
else:
    # === Toggle for Local Documentation === #
    use_local_docs = st.toggle("📁 Use locally saved PDF documentation (from KB/)", value=True)
    uploaded_pdfs = st.file_uploader("\U0001F4C4 Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)
    uploaded_courses_json = None
    uploaded_structure_json = None

st.subheader("Step 2: Ask a question")
user_question = st.text_input(
    "\U0001F4AC What would you like to ask?",
    placeholder="e.g., I'm a second-year student interested in digital forensics and blockchain."
)
user_question = clean_user_query(user_question) #make it easier to work with

if st.button("\U0001F4A1 Get Advice"):
    if not user_question:
        st.warning("Please enter a question.")
    elif upload_mode == "Structured JSON files" and (not uploaded_courses_json or not uploaded_structure_json):
        st.warning("Please upload both JSON files.")
    elif upload_mode == "Unstructured PDF files" and not (uploaded_pdfs or use_local_docs):
        st.warning("Please upload at least one PDF file.")
    else:
        try:
            if upload_mode == "Structured JSON files":
                courses = json.load(uploaded_courses_json)
                courses = clean_course_data(courses)
                courses = remove_duplicate_courses(courses)
                structure = json.load(uploaded_structure_json)
                prompt = build_conversational_prompt(chat_history, user_question, courses=courses, structure=structure)
            else:
                extracted_text = extract_text_from_pdfs(uploaded_pdfs)
                
                extracted_text = clean_pdf_text(extracted_text)
                text_chunks = extracted_text.split("\n\n")
                relevant_chunks = retrieve_relevant_text(user_question, text_chunks, top_k=5)
                relevant_context = "\n\n".join(relevant_chunks)
                prompt = build_conversational_prompt(chat_history, user_question, extracted_text=relevant_context)

            with st.spinner("\U0001F50D Generating advice..."):
                answer = invoke_bedrock(prompt)
                chat_history.append({"user": user_question, "assistant": answer})
                st.success("\u2705 Response received")
                st.text_area("\U0001F916 Advisor's Answer", answer, height=300)

        except Exception as e:
            st.error(f"\u274C Error: {str(e)}")

st.sidebar.header("💬 Conversation History")

# Show list of existing conversations as a selectbox
convo_names = list(st.session_state.conversations.keys())
selected = st.sidebar.selectbox("Select Conversation", convo_names, index=convo_names.index(st.session_state.active_convo))

# When user changes selection, update active convo
if selected != st.session_state.active_convo:
    st.session_state.active_convo = selected
    st.rerun()

# Add new conversation
if st.sidebar.button("➕ New Conversation"):
    new_name = f"Conversation {len(convo_names) + 1}"
    st.session_state.conversations[new_name] = []
    st.session_state.active_convo = new_name
    st.rerun()

# Delete current conversation (confirm if needed)
if st.sidebar.button("🗑️ Delete Current Conversation"):
    if st.session_state.active_convo in st.session_state.conversations:
        del st.session_state.conversations[st.session_state.active_convo]
        # Switch to first remaining convo or create new
        if st.session_state.conversations:
            st.session_state.active_convo = list(st.session_state.conversations.keys())[0]
        else:
            st.session_state.conversations = {"Conversation 1": []}
            st.session_state.active_convo = "Conversation 1"
        st.rerun()

st.subheader(f"🗂 Conversation History — {st.session_state.active_convo}")
for turn in chat_history:
    st.markdown(f"**User:** {turn['user']}")
    st.markdown(f"**Assistant:** {turn['assistant']}")
    st.markdown("---")

if st.sidebar.button("⬇️ Download Chat History (JSON)"):
    convo_data = st.session_state.conversations
    json_data = json.dumps(convo_data, indent=2, ensure_ascii=False)
    st.sidebar.download_button(
        label="Download Chat JSON",
        data=json_data,
        file_name="chat_history.json",
        mime="application/json"
    )

