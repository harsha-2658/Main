import streamlit as st
import base64
import io
import os
from PIL import Image
from typing import TypedDict, Optional, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain / LangGraph imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY_2")
GEMINI_MODEL = "gemini-2.5-flash-lite"


def resize_for_preview(image, size=(650, 420)):
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGB")

    try:
        return image.resize(size, Image.Resampling.LANCZOS)
    except AttributeError:
        return image.resize(size, Image.LANCZOS)


def prepare_image_payload(image, max_size=1200, quality=75):
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGB")

    width, height = image.size
    if max(width, height) > max_size:
        scale = max_size / max(width, height)
        image = image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffered.getvalue()).decode()


# --- 1. SCHEMAS ---
class AadhaarData(BaseModel):
    full_name: str = Field(description="The full name on the Aadhaar card.")
    dob: str = Field(description="Date of birth in DD/MM/YYYY format.")
    gender: str = Field(description="Gender (Male/Female/Transgender).")
    aadhaar_number: str = Field(description="The 12-digit Aadhaar number.")

class PANData(BaseModel):
    full_name: str = Field(description="The full name on the PAN card.")
    dob: str = Field(description="Date of birth in DD/MM/YYYY format.")
    pan_number: str = Field(description="The 10-digit alphanumeric PAN number.")

class DrivingLicenseData(BaseModel):
    full_name: str = Field(description="The full name on the driving licence.")
    dob: str = Field(description="Date of birth in DD/MM/YYYY format.")
    licence_number: str = Field(description="Driving licence number in the format AN01 20130003278.")
    validity: str = Field(description="Validity period of the driving licence.")

class PassportData(BaseModel):
    full_name: str = Field(description="The full name on the passport.")
    dob: str = Field(description="Date of birth in DD/MM/YYYY format.")
    passport_number: str = Field(description="The passport number.")
    date_of_expiry: str = Field(description="Date of expiry in DD/MM/YYYY format.")

class VoterIDData(BaseModel):
    full_name: str = Field(description="The full name on the voter ID.")
    dob: str = Field(description="Date of birth in DD/MM/YYYY format.")
    voter_id_number: str = Field(description="The voter ID number.")

# --- 2. AGENT LOGIC (STATE & NODES) ---
class AgentState(TypedDict):
    image_base64: str
    doc_type: Optional[Literal["Aadhaar", "PAN", "Driving License", "Passport", "Voter ID", "Unknown"]]
    extracted_results: Optional[dict]
    error: Optional[str]


# Reuse a single client and structured LLM instances to avoid repeated initialization overhead.
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GEMINI_API_KEY)
aadhaar_extractor = llm.with_structured_output(AadhaarData)
pan_extractor = llm.with_structured_output(PANData)
driving_license_extractor = llm.with_structured_output(DrivingLicenseData)
passport_extractor = llm.with_structured_output(PassportData)
voter_id_extractor = llm.with_structured_output(VoterIDData)


def to_serializable(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def supervisor_agent(state: AgentState):
    prompt = "Identify if this image is an 'Aadhaar' card, a 'PAN' card, a 'Driving License', a 'Passport', a 'Voter ID', or 'Unknown'. Respond with ONLY one short phrase."
    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state['image_base64']}"}}
    ])
    response = llm.invoke([message]).content.strip().lower()

    if "aadhaar" in response:
        doc_type = "Aadhaar"
    elif "pan" in response:
        doc_type = "PAN"
    elif "driving" in response or "license" in response or "licence" in response:
        doc_type = "Driving License"
    elif "passport" in response:
        doc_type = "Passport"
    elif "voter" in response or "elector" in response:
        doc_type = "Voter ID"
    else:
        doc_type = "Unknown"

    return {"doc_type": doc_type}


def aadhaar_agent(state: AgentState):
    message = HumanMessage(content=[
        {"type": "text", "text": "Extract Name, DOB, Gender, and 12-digit Aadhaar Number."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state['image_base64']}"}}
    ])
    try:
        res = aadhaar_extractor.invoke([message])
        return {"extracted_results": to_serializable(res)}
    except Exception as e:
        return {"error": f"Aadhaar Validation: {str(e)}"}


def pan_agent(state: AgentState):
    message = HumanMessage(content=[
        {"type": "text", "text": "Extract Full Name, DOB, and 10-digit PAN Number."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state['image_base64']}"}}
    ])
    try:
        res = pan_extractor.invoke([message])
        return {"extracted_results": to_serializable(res)}
    except Exception as e:
        return {"error": f"PAN Validation: {str(e)}"}


def driving_license_agent(state: AgentState):
    message = HumanMessage(content=[
        {"type": "text", "text": "Extract full name, DOB, driving licence number, and validity of the licence."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state['image_base64']}"}}
    ])
    try:
        res = driving_license_extractor.invoke([message])
        return {"extracted_results": to_serializable(res)}
    except Exception as e:
        return {"error": f"Driving License Validation: {str(e)}"}


def passport_agent(state: AgentState):
    message = HumanMessage(content=[
        {"type": "text", "text": "Extract full name, DOB, passport number, and date of expiry."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state['image_base64']}"}}
    ])
    try:
        res = passport_extractor.invoke([message])
        return {"extracted_results": to_serializable(res)}
    except Exception as e:
        return {"error": f"Passport Validation: {str(e)}"}


def voter_id_agent(state: AgentState):
    message = HumanMessage(content=[
        {"type": "text", "text": "Extract full name, DOB, and voter ID number."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state['image_base64']}"}}
    ])
    try:
        res = voter_id_extractor.invoke([message])
        return {"extracted_results": to_serializable(res)}
    except Exception as e:
        return {"error": f"Voter ID Validation: {str(e)}"}


def router(state: AgentState):
    if state["doc_type"] == "Aadhaar":
        return "aadhaar"
    if state["doc_type"] == "PAN":
        return "pan"
    if state["doc_type"] == "Driving License":
        return "driving_license"
    if state["doc_type"] == "Passport":
        return "passport"
    if state["doc_type"] == "Voter ID":
        return "voter_id"
    return "end"

builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor_agent)
builder.add_node("aadhaar", aadhaar_agent)
builder.add_node("pan", pan_agent)
builder.add_node("driving_license", driving_license_agent)
builder.add_node("passport", passport_agent)
builder.add_node("voter_id", voter_id_agent)
builder.set_entry_point("supervisor")
builder.add_conditional_edges("supervisor", router, {"aadhaar": "aadhaar", "pan": "pan", "driving_license": "driving_license", "passport": "passport", "voter_id": "voter_id", "end": END})
builder.add_edge("aadhaar", END)
builder.add_edge("pan", END)
builder.add_edge("driving_license", END)
builder.add_edge("passport", END)
builder.add_edge("voter_id", END)
graph = builder.compile()

# --- 3. UI IMPLEMENTATION ---

def main():
    st.set_page_config(page_title="iDReader Enterprise", page_icon="🛡️", layout="wide")

    # CUSTOM CSS: Font Size Increase and Bolding
    st.markdown("""
        <style>
        .main { background-color: #fcfcfc; }
        
        /* Global Font Adjustments */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Large Header Titles */
        h1 { font-size: 3rem !important; font-weight: 800 !important; }
        h3 { font-size: 1.8rem !important; font-weight: 700 !important; color: #333; }

        /* Button Styling */
        .stButton>button { 
            width: 100%; border-radius: 8px; height: 4em; 
            background-color: #002e63; color: white; 
            font-size: 1rem !important; font-weight: 800 !important;
            border: none;
        }

        /* Data Card Customization */
        .data-card {
            background-color: #ffffff; padding: 25px; border-radius: 12px;
            border: 2px solid #f0f2f6; margin-bottom: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }
        .field-label { 
            color: #555; font-size: 1rem !important; 
            text-transform: uppercase; font-weight: 800 !important; 
            letter-spacing: 0.05em; margin-bottom: 8px;
        }
        .field-value { 
            color: #002e63; font-size: 1.6rem !important; 
            font-weight: 700 !important; display: block;
        }

        /* Status Badge */
        .status-badge {
            font-size: 1.4rem !important; font-weight: 800 !important;
            color: #1a73e8; background-color: #e8f0fe;
            padding: 10px 20px; border-radius: 10px; display: inline-block;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    _, center_col, _ = st.columns([1, 6, 1])
    with center_col:
        st.markdown("<h1 style='text-align: center; color: #002e63; margin-bottom: 0;'>iDReader</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666; font-size: 1.2rem;'><strong>Details extracted for Aadhaar, PAN, Driving License, Passport, or Voter ID</strong></p>", unsafe_allow_html=True)
        st.divider()

    left_space, col1, col2, right_space = st.columns([1, 4, 4, 1], gap="large")

    if 'current_file' not in st.session_state:
        st.session_state['current_file'] = None
    if 'result' not in st.session_state:
        st.session_state['result'] = None

    with col1:
        st.markdown("### 📄 Input Document")
        
        if st.session_state['current_file'] is None:
            uploaded_file = st.file_uploader("Upload ID Document either PAN or Aadhar image", type=["jpg", "jpeg", "png"], key="uploader")
            if uploaded_file:
                st.session_state['current_file'] = uploaded_file
                st.rerun()
        else:
            image = Image.open(st.session_state['current_file'])
            preview_image = resize_for_preview(image)
            st.image(preview_image, use_container_width=False)
            
            st.markdown(f"**Current File:** `{st.session_state['current_file'].name}`")
            c1, c2 = st.columns(2)
            if c1.button("🗑️ Remove File",width="stretch"):
                st.session_state['current_file'] = None
                st.session_state['result'] = None
                st.rerun()
            
            if c2.button("🚀 Extract",width="stretch"):
                img_b64 = prepare_image_payload(image)
                
                with st.spinner("Executing multi-agent workflow..."):
                    result = graph.invoke({"image_base64": img_b64})
                    st.session_state['result'] = result

    with col2:
        st.markdown("### 📋 Extracted Profile Details")
        
        if st.session_state['result']:
            res = st.session_state['result']
            
            if res.get("error"):
                st.error(res["error"])
            elif res["doc_type"] == "Unknown":
                st.warning("Identification Failed")
            else:
                st.markdown(f'<div class="status-badge">DOCUMENT TYPE: {res["doc_type"].upper()}</div>', unsafe_allow_html=True)
                
                data = res["extracted_results"]
                for key, value in data.items():
                    label = key.replace('_', ' ').title()
                    st.markdown(f"""
                        <div class="data-card">
                            <div class="field-label">{label}</div>
                            <div class="field-value">{value}</div>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="border: 2px dashed #ddd; padding: 100px 20px; text-align: center; border-radius: 12px; color: #aaa; font-weight: 600; font-size: 1.2rem;">
                    SYSTEM IDLE: AWAITING INPUT
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

# c1, c2 = st.columns(2)
#             if c1.button("🗑️ Remove File"):
#                 st.session_state['current_file'] = None
#                 st.session_state['result'] = None
#                 st.rerun()
            
#             if c2.button("🚀 Run Analysis"):
#                 buffered = io.BytesIO()
#                 image.convert("RGB").save(buffered, format="JPEG")
#                 img_b64 = base64.b64encode(buffered.getvalue()).decode()
                
#                 with st.spinner("Executing multi-agent workflow..."):
#                     result = graph.invoke({"image_base64": img_b64})
#                     st.session_state['result'] = result