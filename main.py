
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import joblib
import plotly.graph_objects as go

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
from datetime import datetime
import zipfile

# page main configuration
st.set_page_config(page_title="Asthma Prediction Dashboard", layout="wide")

# css styles
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background-image: linear-gradient(135deg, #FFF9EF 0%, #C3DAF2 50%, #0A1F5A 100%) !important;
            background-attachment: fixed !important;
            background-size: cover !important;
        }
        
        /* Header */
        h2 {
            color: #0A1F5A;
            padding: 5px;
            border-radius: 10px;
        }

        /* Subheaders */
        .stSubheader {
            color: #0A1F5A;
        }

        /* Form container */
        div[data-testid="stForm"] {
            background-color: #C3DAF2;
            padding: 1.5rem 1.5rem;
            border-radius: 15px;
            box-shadow: 0px 3px 6px rgba(0,0,0,0.1);
            width: 95%;
            max-width: 900px;
            margin: 0 auto;
        }

        /* INPUT BOXES  */
        input[type="text"], input[type="number"] {
            background-color: #FFFFFF;
            color: #0A1F5A;
            border-radius: 6px;
            padding: 4px 6px;
            font-size: 14px;
            height: 40px;
        }

        /* SELECT BOX */
        div[data-baseweb="select"] > div {
            background-color: #FFFFFF;
            border-radius: 6px;
            height: 40px;
            font-size: 14px;
        }

        label, .stMarkdown {
            font-size: 14px ;
            color: #0A1F5A;
            margin-bottom: 2px;
        }

        div[role="radiogroup"] label, div[role="checkbox"] label {
            font-size: 14px;
        }
        
        div[data-testid="stRadio"] {
            margin-top: 10px;
            margin-bottom: -5px;
        }

        div[data-testid="column"] {
            padding-top: 0px;
            padding-bottom: 0px;
        }

        div.row-widget.stRadio label {
            font-size: 14px;
        }

        section[data-testid="stVerticalBlock"] {
            margin-top: 0px !important;
            margin-bottom: 0px !important;
        }

        /*  button */
        div.stButton > button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
            padding: 0.3rem 0.3rem;
            font-size: 14px;
        }

        .button-row {
            display: flex;
            justify-content: flex-end;
            gap: 0.5rem;
        }
        
        /* Dashboard containers */
        /* Card Container Style */
        div.css-1r6slb0, div[data-testid="stVerticalBlock"] > div[style*="background-color"] {
            border: 1px solid rgba(10,31,90,0.1);
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }
        .card-header {
            color: #0A1F5A;
            font-weight: 700;
            margin-bottom: 10px;
            font-size: 16px;
        }
        .small-muted {
            color: #64748B;
            font-size: 12px;
            margin-top: 5px;
            text-align: center;
        }
        /* Result Card */
        .result-card {
            background-color: #FFFFFF;
            padding: 20px;
            border-radius: 12px;
            border-left: 8px solid #0A1F5A;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        
        .st-key-styled_container, .st-key-styled_container2, .st-key-styled_container3, .st-key-styled_container4, .st-key-styled_container5, .st-key-styled_container6 {
            background-color: #FFFFFF;
            border-radius:1rem;
            padding:1rem;
            min-height:300px;
            box-shadow: 3px 5px 15px 0px rgba(128, 128, 128, 0.245);
        }
        
        .st-key-styled_container2, .st-key-styled_container4, .st-key-styled_container6 {  
            display: flex;
            justify-content: center;
            align-items: center; 
        }
    </style>
""", unsafe_allow_html=True)

# session
if "show_form" not in st.session_state:
    st.session_state.show_form = True

if 'form_counter' not in st.session_state:
    st.session_state.form_counter = 0
    
# header 
st.markdown("""
    <style>
    div.block-container {
        padding-top: 1rem;
    }

    /* Align title + button nicely */
    div[data-testid="column"]:has(button) {
        display: flex;
        align-items: center;
        justify-content: flex-end;
    }

    div[data-testid="stButton"] > button[kind="secondary"] {
        background-color: #0A1F5A !important;
        color: white !important;
        border-radius: 15px !important;
        font-weight: 500 !important;
        border: none !important;
        height: 38px !important;
        margin-top: 8px !important;
        max-width: 40px !important
        width: 70% !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"]:hover {
        background-color: #FFF9EF !important;
        color: #0A1F5A !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS---
def create_pdf(patient_info, prediction_res, figures, explanations):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()
    
    normal_style = styles['Normal']
    normal_style.fontName = 'Courier'
    
    # Custom Styles
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], alignment=TA_CENTER, fontSize=24, spaceAfter=20, textColor=colors.HexColor("#0A1F5A"), fontName='Courier-Bold')
    h2_style = ParagraphStyle('Heading2', parent=styles['Heading2'], fontSize=16, spaceAfter=10, textColor=colors.HexColor("#0A1F5A"), borderPadding=5, fontName='Courier-Bold')
    exp_style = ParagraphStyle('Exp', parent=normal_style, fontSize=10, leading=14, alignment=TA_JUSTIFY, fontName='Courier')
    
    # Title
    story.append(Paragraph("Prediction Analysis Report", title_style))
    story.append(Spacer(1, 10))
    
    # Patient Info Section
    story.append(Paragraph("Patient Information", h2_style))
    p_data = [
        ["Name:", patient_info['name'], "Date:", datetime.now().strftime("%Y-%m-%d")],
        ["Age:", f"{patient_info['age']} Years", "Gender:", patient_info['gender']],
        ["BMI:", f"{patient_info['bmi']:.2f}", "", ""]
    ]
    p_table = Table(p_data, colWidths=[1.2*inch, 2.5*inch, 1*inch, 2*inch])
    p_table.setStyle(TableStyle([
        ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor("#0A1F5A")),
        ('FONTNAME', (0,0), (-1, -1), 'Courier'),
        ('FONTNAME', (0,0), (0, -1), 'Courier-Bold'), 
        ('FONTNAME', (2,0), (2, -1), 'Courier-Bold'), 
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#E0E0E0")),
    ]))
    story.append(p_table)
    story.append(Spacer(1, 20))
    
    # Prediction Result Section
    story.append(Paragraph("Prediction Summary", h2_style))
    
    # Color based on result
    bg_color = colors.HexColor("#FADBD8") if prediction_res['is_positive'] else colors.HexColor("#D5F5E3")
    text_color = colors.HexColor("#C0392B") if prediction_res['is_positive'] else colors.HexColor("#27AE60")
    
    res_text = f"<b>{prediction_res['result']}</b>"
    conf_text = f"Predicted Risk Probability: {prediction_res['confidence']}"
    
    res_data = [
        [Paragraph(res_text, ParagraphStyle('Res', parent=normal_style, fontSize=14, textColor=text_color, alignment=TA_CENTER))],
        [Paragraph(conf_text, ParagraphStyle('Conf', parent=normal_style, fontSize=12, textColor=colors.HexColor("#0A1F5A"), alignment=TA_CENTER))]
    ]
    
    res_table = Table(res_data, colWidths=[7*inch])
    res_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg_color),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('BOX', (0,0), (-1,-1), 1, text_color),
    ]))
    story.append(res_table)
    story.append(Spacer(1, 25))
    
    # Visual Results Section
    story.append(Paragraph("Clinical Analysis Details", h2_style))
    
    for title, fig in figures.items():
        if fig:
            block_elements = []
            
            block_elements.append(Paragraph(title, ParagraphStyle('CTitle', parent=normal_style, fontSize=12, spaceAfter=5, textColor=colors.HexColor("#0A1F5A"), fontName='Courier-Bold')))
            
            if isinstance(fig, BytesIO):
                img_buffer = fig
            else:
                img_bytes = fig.to_image(format="png", width=800, height=400, scale=1)
                img_buffer = BytesIO(img_bytes)
            
            rl_img = Image(img_buffer, width=5*inch, height=2.5*inch, hAlign='CENTER')
            block_elements.append(rl_img)
            block_elements.append(Spacer(1, 5))
            
            exp_text = explanations.get(title, "No analysis available.")
            block_elements.append(Paragraph(f"<b>Clinical Note:</b> {exp_text}", exp_style))
            block_elements.append(Spacer(1, 20))
            
            story.append(KeepTogether(block_elements))
            
    # Disclaimer
    story.append(Spacer(1, 30))
    disclaimer_text = "<b>DISCLAIMER:</b> This report is generated by an AI-based predictive model for educational and research purposes only. It does not constitute a medical diagnosis. Please consult a qualified healthcare professional for clinical evaluation."
    story.append(Paragraph(disclaimer_text, ParagraphStyle('Disclaimer', parent=normal_style, fontSize=8, textColor=colors.gray, alignment=TA_JUSTIFY)))
            
    # Footer
    def add_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Courier', 9)
        canvas.setFillColor(colors.HexColor("#7F8C8D"))
        canvas.drawString(inch, 0.75 * inch, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        canvas.drawString(7 * inch, 0.75 * inch, f"Page {doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    buffer.seek(0)
    return buffer
       
def load_model():
    try:
        model = joblib.load('xgboost_best_final-new.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure model is in the working directory.")
        return None

def get_bmi_zones(age, gender):
    # Simplified CDC Percentile Data (Approximate)
    if age >= 20:
        return 18.5, 25.0, 30.0
    
    # Dictionary: Age -> [Male_5th, Male_85th, Male_95th, Female_5th, Female_85th, Female_95th]
    pediatric_data = {
        5: [13.0, 16.6, 18.3, 12.7, 16.7, 18.9], 6: [13.0, 17.0, 19.2, 12.7, 17.2, 19.9],
        7: [13.1, 17.4, 20.6, 12.7, 17.7, 21.0], 8: [13.2, 17.9, 22.0, 12.9, 18.3, 22.2],
        9: [13.4, 18.6, 23.4, 13.1, 19.1, 23.5], 10: [13.6, 19.4, 24.8, 13.5, 20.0, 24.6],
        11: [13.9, 20.3, 26.1, 13.9, 20.9, 25.8], 12: [14.2, 21.1, 27.2, 14.4, 21.9, 27.0],
        13: [14.6, 21.9, 28.2, 14.9, 22.8, 28.1], 14: [15.1, 22.7, 29.0, 15.4, 23.6, 29.0],
        15: [15.7, 23.5, 29.9, 15.9, 24.2, 29.6], 16: [16.2, 24.2, 30.6, 16.2, 24.7, 30.1],
        17: [16.7, 24.9, 31.3, 16.4, 25.2, 30.5], 18: [17.2, 25.6, 32.0, 16.4, 25.6, 30.8]
    }
    lookup_age = max(5, min(18, int(age)))
    d = pediatric_data.get(lookup_age)
    return (d[0], d[1], d[2]) if gender == "Male" else (d[3], d[4], d[5])

model = load_model()

# --- SIDEBAR MODE SELECTION ---
if "app_mode" not in st.session_state:
    st.session_state["app_mode"] = "Single Patient"

with st.sidebar:
    st.header("Input Method")
    
    if st.button("Single Patient", key="btn_mode_single", use_container_width=True, type="primary" if st.session_state["app_mode"] == "Single Patient" else "secondary"):
        st.session_state["app_mode"] = "Single Patient"
        st.rerun()

    if st.button("Import Data", key="btn_mode_bulk", use_container_width=True, type="primary" if st.session_state["app_mode"] == "Multiple Patients" else "secondary"):
        st.session_state["app_mode"] = "Multiple Patients"
        st.rerun()

    app_mode = st.session_state["app_mode"]
    st.markdown("---")
    if app_mode == "Multiple Patients":
        st.info("Import data expects a CSV file")

# --- SESSION STATE FOR ACTIVE PATIENT ---
if 'active_patient' not in st.session_state:
    st.session_state['active_patient'] = None

# Initialize variables to defaults to prevent NameError if dashboard is not running
name, age, bmi, gender, pollution, fev1, fvc = None, 0, 0.0, "Male", 0.0, 0.0, 0.0
pet, family_history, hist_allergies, eczema, hayfever, gerd = "No", "No", "No", "No", "No", "No"
wheezing, sob, chest_tightness, coughing, night_symptoms, exercise_induced = "No", "No", "No", "No", "No", "No"

if app_mode == "Single Patient":
    col_title, col_btn = st.columns([6, 1])

    with col_title:
        st.markdown("""
            <h3 style='
                display: flex;
                align-items: center;
            '>Asthma Prediction Dashboard</h3>
        """, unsafe_allow_html=True)

    with col_btn:
        add_clicked = st.button("Add New", key="add_new", use_container_width=False)
        if add_clicked:
            st.session_state.form_counter += 1
            st.query_params['reset'] = 'true'
            st.rerun()

    # form
    st.markdown("""
        <div class="form-wrapper">
    """, unsafe_allow_html=True)

    with st.form(f"asthma_form_{st.session_state.form_counter}", clear_on_submit=False):
        
        st.subheader("Patient Vitals and Lung Function")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input("Name", placeholder="Full name")
            age = st.number_input("Age", min_value=0, max_value=120, value=9)
            bmi = st.number_input("BMI", min_value=5.0, max_value=60.0, value=15.8, step=0.1)    
            
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            pollution = st.number_input("Pollution Exposure (Index)", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
            fev1 = st.number_input("Lung Function FEV1 (L)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
        
        with col3:
            fvc = st.number_input("Lung Function FVC (L)", min_value=0.5, max_value=8.0, value=2.5, step=0.1)
            st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Symptoms and Conditions")
        col4, col5 = st.columns(2)
        with col4:
            st.markdown("**Allergy/History**")
            pet = st.radio("Pet Allergy", ["Yes", "No"], index=1, key='pet_allergy')
            family_history = st.radio("Family History of Asthma", ["Yes", "No"], index=1, key='fam_history')
            hist_allergies = st.radio("History of Allergies", ["Yes", "No"], index=1, key='hist_allergies')
            eczema = st.radio("Eczema", ["Yes", "No"], index=1, key='eczema')
            hayfever = st.radio("Hay Fever", ["Yes", "No"], index=1, key='hayfever')
            gerd = st.radio("Gastroesophageal Reflux (GERD)", ["Yes", "No"], index=1, key='gerd')

        with col5:
            st.markdown("**Current Symptoms**")
            wheezing = st.radio("Wheezing", ["Yes", "No"], index=1, key='wheezing')
            sob = st.radio("Shortness of Breath", ["Yes", "No"], index=1, key='sob')
            chest_tightness = st.radio("Chest Tightness", ["Yes", "No"], index=1, key='chest_tightness')
            coughing = st.radio("Frequent Coughing", ["Yes", "No"], index=1, key='coughing')
            night_symptoms = st.radio("Nighttime Symptoms (e.g., waking up)", ["Yes", "No"], index=1, key='night_symptoms')
            exercise_induced = st.radio("Exercise Induced Symptoms", ["Yes", "No"], index=1, key='exercise_induced')
        
        spacer, col_cancel, col_predict = st.columns([4.0, 0.6, 0.7])
        
        st.markdown("""
            <style>
                /* make form buttons fill their column */
                div.stButton > button {
                    width: 100%;
                }
            </style>
        """, unsafe_allow_html=True)

        with col_cancel:
            cancel = st.form_submit_button("Cancel", type="secondary")
        with col_predict:
            predict = st.form_submit_button("Predict", type="primary")
            if predict:
                # Store form data in session state
                st.session_state['active_patient'] = {
                    'name': name, 'age': age, 'bmi': bmi, 'gender': gender, 'pollution': pollution, 'fev1': fev1, 'fvc': fvc,
                    'pet': pet, 'family_history': family_history, 'hist_allergies': hist_allergies, 
                    'eczema': eczema, 'hayfever': hayfever, 'gerd': gerd,
                    'wheezing': wheezing, 'sob': sob, 'chest_tightness': chest_tightness, 
                    'coughing': coughing, 'night_symptoms': night_symptoms, 'exercise_induced': exercise_induced
                }
    
    st.markdown("</div>", unsafe_allow_html=True)
    if cancel:
        st.info("Form cleared. You can edit the fields.")

elif app_mode == "Multiple Patients":
    st.header("Multiple Patient Upload")
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file:
        if st.sidebar.button("Submit"):
            try:
                df = pd.read_csv(uploaded_file)
                df.columns = [str(c).strip().lstrip('\ufeff') for c in df.columns]
                # Store raw data in session state
                st.session_state['bulk_data'] = df
                st.success(f"Successfully loaded {len(df)} patient records.")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    if 'bulk_data' in st.session_state:
        df = st.session_state['bulk_data']
        
        # convert to binary (0/1)
        def to_binary(val):
            if isinstance(val, str):
                return 1 if val.strip().lower() in ['1','yes','true'] else 0
            try: return int(val)
            except: return 0
            
        # find column values using multiple possible names (aliases)
        def get_val(row, keys, default):
            for k in keys:
                if k in row: return row[k]
                # Case insensitive check with stripping
                for col in row.index:
                    if str(k).strip().lower() == str(col).strip().lower(): return row[col]
            return default

        # --- COLUMN MAPPING ---
        all_cols = list(df.columns)
        def find_col(options):
            for col in all_cols:
                if str(col).strip().lower() in [o.lower() for o in options]: return col
            return None

        auto_name = find_col(['Name', 'Patient Name', 'Full Name', 'Patient', 'PatientName', 'First Name'])
        auto_name = find_col(['Name', 'Patient Name', 'Full Name', 'Patient', 'PatientName', 'First Name', 'patient_name', 'p_name', 'id', 'patient_id'])
        
        name_col = auto_name

        patient_options = df[name_col].astype(str).tolist() if name_col else [f"Patient {i+1}" for i in range(len(df))]
        
        selected_patient_name = st.selectbox("Select Patient to Analyse", patient_options)
        
        if selected_patient_name:
            # Find the row
            if name_col:
                row = df[df[name_col].astype(str) == selected_patient_name].iloc[0]
                name = row[name_col]
            else:
                idx = int(selected_patient_name.split(" ")[1]) - 1
                row = df.iloc[idx]
                name = f"Patient {idx+1}"
            
            
            age = get_val(row, ['Age', 'age', 'Age (Years)'], 10)
            bmi = get_val(row, ['BMI', 'bmi'], 18.0)
            gender = get_val(row, ['Gender', 'Sex', 'Gender_1'], 'Male')
            pollution = get_val(row, ['Pollution', 'Pollution Exposure', 'PollutionIndex'], 5.0)
            fev1 = get_val(row, ['FEV1', 'Lung Function FEV1', 'fev1'], 2.0)
            fvc = get_val(row, ['FVC', 'Lung Function FVC', 'fvc'], 2.5)
            
            # Map Yes/No fields
            pet = "Yes" if to_binary(get_val(row, ['PetAllergy_1', 'Pet Allergy', 'Pet', 'PetAllergy', 'pet_allergy'], 0)) else "No"
            family_history = "Yes" if to_binary(get_val(row, ['FamilyHistoryAsthma_1', 'Family History', 'FamilyHistory', 'Family History of Asthma', 'fam_history'], 0)) else "No"
            hist_allergies = "Yes" if to_binary(get_val(row, ['HistoryOfAllergies_1', 'History of Allergies', 'HistoryOfAllergies', 'hist_allergies', 'Other Allergies'], 0)) else "No"
            eczema = "Yes" if to_binary(get_val(row, ['Eczema_1', 'Eczema', 'eczema'], 0)) else "No"
            hayfever = "Yes" if to_binary(get_val(row, ['HayFever_1', 'Hay Fever', 'HayFever', 'hayfever'], 0)) else "No"
            gerd = "Yes" if to_binary(get_val(row, ['GastroesophagealReflux_1', 'GERD', 'Gastroesophageal Reflux', 'Acid Reflux', 'gerd'], 0)) else "No"
            wheezing = "Yes" if to_binary(get_val(row, ['Wheezing_1', 'Wheezing', 'wheezing'], 0)) else "No"
            sob = "Yes" if to_binary(get_val(row, ['ShortnessOfBreath_1', 'Shortness of Breath', 'ShortnessOfBreath', 'sob', 'SoB'], 0)) else "No"
            chest_tightness = "Yes" if to_binary(get_val(row, ['ChestTightness_1', 'Chest Tightness', 'ChestTightness', 'chest_tightness'], 0)) else "No"
            coughing = "Yes" if to_binary(get_val(row, ['Coughing_1', 'Coughing', 'Frequent Coughing', 'coughing'], 0)) else "No"
            night_symptoms = "Yes" if to_binary(get_val(row, ['NighttimeSymptoms_1', 'Night Symptoms', 'Nighttime Symptoms', 'night_symptoms'], 0)) else "No"
            exercise_induced = "Yes" if to_binary(get_val(row, ['ExerciseInduced_1', 'Exercise Induced', 'Exercise Induced Symptoms', 'exercise_induced'], 0)) else "No"
            
            # Store multiple data in session state
            st.session_state['active_patient'] = {
                'name': name, 'age': age, 'bmi': bmi, 'gender': gender, 'pollution': pollution, 'fev1': fev1, 'fvc': fvc,
                'pet': pet, 'family_history': family_history, 'hist_allergies': hist_allergies, 
                'eczema': eczema, 'hayfever': hayfever, 'gerd': gerd,
                'wheezing': wheezing, 'sob': sob, 'chest_tightness': chest_tightness, 
                'coughing': coughing, 'night_symptoms': night_symptoms, 'exercise_induced': exercise_induced
            }

        # --- ZIP EXPORT SECTION ---
        st.markdown("### Patient List Export")
        if st.button("Generate All Reports (ZIP)"):
            if model is None:
                st.error("Model not loaded.")
            else:
                progress_bar = st.progress(0)
                zip_buffer = BytesIO()
                
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    total_rows = len(df)
                    for i, row in df.iterrows():
                        # 1. Extract & Normalize Data
                        p_name = row[name_col] if name_col else f"Patient {i+1}"
                        p_age = get_val(row, ['Age', 'age'], 10)
                        p_bmi = get_val(row, ['BMI', 'bmi'], 18.0)
                        p_gender = get_val(row, ['Gender', 'Sex', 'Gender_1'], 'Male')
                        
                        # Normalize gender
                        if str(p_gender).strip().lower() in ['1', 'female', 'f']:
                            p_gender = "Female"
                        elif str(p_gender).strip().lower() in ['0', 'male', 'm']:
                            p_gender = "Male"
                            
                        p_pollution = get_val(row, ['Pollution', 'Pollution Exposure', 'PollutionIndex'], 5.0)
                        p_fev1 = get_val(row, ['FEV1', 'Lung Function FEV1', 'fev1'], 2.0)
                        p_fvc = get_val(row, ['FVC', 'Lung Function FVC', 'fvc'], 2.5)
                        
                        p_pet = to_binary(get_val(row, ['PetAllergy_1', 'Pet Allergy', 'Pet', 'PetAllergy', 'pet_allergy'], 0))
                        p_fam = to_binary(get_val(row, ['FamilyHistoryAsthma_1', 'Family History', 'FamilyHistory', 'Family History of Asthma', 'fam_history'], 0))
                        p_hist = to_binary(get_val(row, ['HistoryOfAllergies_1', 'History of Allergies', 'HistoryOfAllergies', 'hist_allergies', 'Other Allergies'], 0))
                        p_ecz = to_binary(get_val(row, ['Eczema_1', 'Eczema', 'eczema'], 0))
                        p_hay = to_binary(get_val(row, ['HayFever_1', 'Hay Fever', 'HayFever', 'hayfever'], 0))
                        p_gerd = to_binary(get_val(row, ['GastroesophagealReflux_1', 'GERD', 'Gastroesophageal Reflux', 'Acid Reflux', 'gerd'], 0))
                        p_wheez = to_binary(get_val(row, ['Wheezing_1', 'Wheezing', 'wheezing'], 0))
                        p_sob = to_binary(get_val(row, ['ShortnessOfBreath_1', 'Shortness of Breath', 'ShortnessOfBreath', 'sob', 'SoB'], 0))
                        p_chest = to_binary(get_val(row, ['ChestTightness_1', 'Chest Tightness', 'ChestTightness', 'chest_tightness'], 0))
                        p_cough = to_binary(get_val(row, ['Coughing_1', 'Coughing', 'Frequent Coughing', 'coughing'], 0))
                        p_night = to_binary(get_val(row, ['NighttimeSymptoms_1', 'Night Symptoms', 'Nighttime Symptoms', 'night_symptoms'], 0))
                        p_exer = to_binary(get_val(row, ['ExerciseInduced_1', 'Exercise Induced', 'Exercise Induced Symptoms', 'exercise_induced'], 0))

                        # 2. Prepare Input Data
                        input_row = pd.DataFrame([{ 
                            'Age': p_age, 'BMI': p_bmi, 'PollutionExposure': p_pollution,
                            'LungFunctionFEV1': p_fev1, 'LungFunctionFVC': p_fvc,
                            'Gender_1': 1 if str(p_gender).strip().lower() in ['female', 'f', '1'] else 0,
                            'PetAllergy_1': p_pet,
                            'FamilyHistoryAsthma_1': p_fam,
                            'HistoryOfAllergies_1': p_hist,
                            'Eczema_1': p_ecz,
                            'HayFever_1': p_hay,
                            'GastroesophagealReflux_1': p_gerd,
                            'Wheezing_1': p_wheez,
                            'ShortnessOfBreath_1': p_sob,
                            'ChestTightness_1': p_chest,
                            'Coughing_1': p_cough,
                            'NighttimeSymptoms_1': p_night,
                            'ExerciseInduced_1': p_exer,
                        }])
                        
                        if hasattr(model, "feature_names_in_"):
                            input_row = input_row[model.feature_names_in_]
                        
                        # 3. Predict
                        proba_raw = model.predict_proba(input_row)[0]
                        raw_p = proba_raw[1]
                        disp_p = min(max(raw_p, 0.05), 0.95)
                        
                        # Risk Category
                        if disp_p < 0.40:
                            risk_cat = "Low Risk"
                            res_str = "LOW ASTHMA RISK ESTIMATED"
                            is_pos = False
                        elif disp_p < 0.70:
                            risk_cat = "Moderate Risk"
                            res_str = "MODERATE ASTHMA RISK ESTIMATED"
                            is_pos = True
                        else:
                            risk_cat = "High Risk"
                            res_str = "HIGH ASTHMA RISK DETECTED (MODEL ESTIMATE)"
                            is_pos = True
                        
                        # 4. Generate Explanations
                        bmi_u, bmi_o, bmi_ob = get_bmi_zones(p_age, p_gender)
                        if p_bmi < bmi_u: b_stat = "Underweight"
                        elif p_bmi < bmi_o: b_stat = "Healthy Weight"
                        elif p_bmi < bmi_ob: b_stat = "Overweight"
                        else: b_stat = "Obese"
                        
                        sym_list = [k for k, v in {"Wheezing": p_wheez, "SoB": p_sob, "Tightness": p_chest, "Cough": p_cough}.items() if v == "Yes"]
                        sym_txt = f"Symptoms: {', '.join(sym_list)}." if sym_list else "No primary symptoms."
                        
                        exps = {
                            "Asthma Risk Probability": f"Estimated Risk: {disp_p*100:.1f}% ({risk_cat}).",
                            "Symptom Profile": f"{sym_txt}",
                            "BMI Status": f"BMI {p_bmi:.2f} ({b_stat}).",
                            "Environmental Risk": f"Pollution: {p_pollution}.",
                            "Risk Breakdown": "Risk factor analysis.",
                            "Feature Contribution": "Key influencing factors."
                        }
                        
                        # 5. Generate Figures (Matplotlib)
                        figs = {}
                        # Gauge
                        fig_g, ax_g = plt.subplots(figsize=(5, 2))
                        ax_g.barh(0, 100, color='#FADBD8', height=0.5)
                        ax_g.barh(0, 50, color='#D5F5E3', height=0.5)
                        ax_g.scatter([disp_p*100], [0], color='#0A1F5A', s=150, zorder=10, marker='|', linewidth=4)
                        ax_g.set_xlim(0, 100); ax_g.set_yticks([]); ax_g.set_xlabel("Risk Probability (%)")
                        buf_g = BytesIO(); fig_g.savefig(buf_g, format='png', bbox_inches='tight'); buf_g.seek(0); plt.close(fig_g)
                        figs["Asthma Risk Probability"] = buf_g
                        
                        # Radar
                        cats = ['Wheeze', 'SoB', 'Tightness', 'Cough', 'Night', 'Exercise']
                        vals = [p_wheez, p_sob, p_chest, p_cough, p_night, p_exer]
                        vals += vals[:1]
                        angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist(); angles += angles[:1]
                        fig_r, ax_r = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
                        ax_r.plot(angles, vals, color='#0A1F5A', linewidth=2)
                        ax_r.fill(angles, vals, color='#0A1F5A', alpha=0.25)
                        ax_r.set_xticks(angles[:-1]); ax_r.set_xticklabels(cats, size=8); ax_r.set_yticks([])
                        buf_r = BytesIO(); fig_r.savefig(buf_r, format='png', bbox_inches='tight'); buf_r.seek(0); plt.close(fig_r)
                        figs["Symptom Profile"] = buf_r
                        
                        # BMI
                        fig_b, ax_b = plt.subplots(figsize=(5, 2))
                        ax_b.barh(0, 60, color='#FADBD8'); ax_b.barh(0, bmi_ob, color='#FCF3CF'); 
                        ax_b.barh(0, bmi_o, color='#D5F5E3'); ax_b.barh(0, bmi_u, color='#EBF5FB')
                        ax_b.axvline(p_bmi, color='#0A1F5A', linewidth=3)
                        ax_b.set_xlim(10, 45); ax_b.set_yticks([]); ax_b.set_xlabel("BMI Value")
                        buf_b = BytesIO(); fig_b.savefig(buf_b, format='png', bbox_inches='tight'); buf_b.seek(0); plt.close(fig_b)
                        figs["BMI Status"] = buf_b
                        
                        # Env
                        fig_e, ax_e = plt.subplots(figsize=(5, 3))
                        env_v = [p_pollution, 10 if p_pet==1 else 0, 10 if p_hay==1 else 0, 10 if p_ecz==1 else 0]
                        env_c = ['#E74C3C' if v > 5 else '#2ECC71' for v in env_v]
                        ax_e.bar(['Pollution', 'Pet', 'Hay Fever', 'Eczema'], env_v, color=env_c)
                        buf_e = BytesIO(); fig_e.savefig(buf_e, format='png', bbox_inches='tight'); buf_e.seek(0); plt.close(fig_e)
                        figs["Environmental Risk"] = buf_e
                        
                        # Breakdown
                        c_hist = sum([p_fam, p_hist, p_ecz, p_hay, p_gerd, p_pet])
                        c_symp = sum([p_wheez, p_sob, p_chest, p_cough, p_night, p_exer])
                        c_env = 1 if p_pollution > 6 else 0
                        if p_bmi > 25: c_env += 1
                        fig_br, ax_br = plt.subplots(figsize=(5, 3))
                        ax_br.bar(['History', 'Symptoms', 'Vitals'], [c_hist, c_symp, c_env], color=['#6888A8', '#0A1F5A', '#F1C40F'])
                        buf_br = BytesIO(); fig_br.savefig(buf_br, format='png', bbox_inches='tight'); buf_br.seek(0); plt.close(fig_br)
                        figs["Risk Breakdown"] = buf_br

                        # 6. Create PDF
                        p_info = {"name": p_name, "age": p_age, "gender": p_gender, "bmi": p_bmi}
                        pr_info = {"result": res_str, 
                                   "confidence": f"{disp_p*100:.1f}%", "is_positive": is_pos}
                        
                        pdf_data = create_pdf(p_info, pr_info, figs, exps)
                        
                        # 7. Add to ZIP
                        safe_fname = str(p_name).replace(" ", "_")
                        zip_file.writestr(f"Asthma_Report_{safe_fname}.pdf", pdf_data.getvalue())
                        
                        progress_bar.progress((i + 1) / total_rows)
                
                zip_buffer.seek(0)
                st.session_state['bulk_zip_data'] = zip_buffer
                st.success("All reports generated successfully!")
        
        if 'bulk_zip_data' in st.session_state:
            st.download_button("Download All Reports (ZIP)", st.session_state['bulk_zip_data'], "Asthma_Reports_Batch.zip", "application/zip", type="primary")
        
        st.markdown("---")

# --- DASHBOARD LOGIC ---
if st.session_state['active_patient'] is not None:
    p = st.session_state['active_patient']
    name, age, bmi, gender = p['name'], p['age'], p['bmi'], p['gender']
    
    # Normalize gender for display and logic
    if str(gender).strip().lower() in ['1', 'female', 'f']:
        gender = "Female"
    elif str(gender).strip().lower() in ['0', 'male', 'm']:
        gender = "Male"
        
    pollution, fev1, fvc = p['pollution'], p['fev1'], p['fvc']
    pet, family_history, hist_allergies = p['pet'], p['family_history'], p['hist_allergies']
    eczema, hayfever, gerd = p['eczema'], p['hayfever'], p['gerd']
    wheezing, sob, chest_tightness = p['wheezing'], p['sob'], p['chest_tightness']
    coughing, night_symptoms, exercise_induced = p['coughing'], p['night_symptoms'], p['exercise_induced']

    if model: 
        input_data = pd.DataFrame([{ 
            'Age': age,
            'BMI': bmi,
            'PollutionExposure': pollution,
            'LungFunctionFEV1': fev1,
            'LungFunctionFVC': fvc,
            
            # Binary Mappings
            'Gender_1': 1 if str(gender).strip().lower() in ['female', 'f', '1'] else 0,
            'PetAllergy_1': 1 if pet == "Yes" else 0,
            'FamilyHistoryAsthma_1': 1 if family_history == "Yes" else 0,
            'HistoryOfAllergies_1': 1 if hist_allergies == "Yes" else 0,
            'Eczema_1': 1 if eczema == "Yes" else 0,
            'HayFever_1': 1 if hayfever == "Yes" else 0,
            'GastroesophagealReflux_1': 1 if gerd == "Yes" else 0,
            'Wheezing_1': 1 if wheezing == "Yes" else 0,
            'ShortnessOfBreath_1': 1 if sob == "Yes" else 0,
            'ChestTightness_1': 1 if chest_tightness == "Yes" else 0,
            'Coughing_1': 1 if coughing == "Yes" else 0,
            'NighttimeSymptoms_1': 1 if night_symptoms == "Yes" else 0,
            'ExerciseInduced_1': 1 if exercise_induced == "Yes" else 0,
        }])
        
        if hasattr(model, "feature_names_in_"):
            expected_features = model.feature_names_in_
            
            missing = [col for col in expected_features if col not in input_data.columns]
            if missing:
                st.error(f"Missing features for prediction: {missing}")
                st.stop()
                
            input_data = input_data[expected_features]
        else: 
            st.warning("Model does not have feature names information. Proceeding without alignment check.")
            
        try:
            prediction_proba = model.predict_proba(input_data)
            raw_prob = prediction_proba[0][1]
            # Probability Calibration (Display Only)
            disp_prob = min(max(raw_prob, 0.05), 0.95)
            
            # Risk Category Mapping
            if disp_prob < 0.40:
                risk_category = "Low Risk"
                status_text = "LOW ASTHMA RISK ESTIMATED"
                status_color = "#27AE60" # Green
            elif disp_prob < 0.70:
                risk_category = "Moderate Risk"
                status_text = "MODERATE ASTHMA RISK ESTIMATED"
                status_color = "#F39C12" # Orange
            else:
                risk_category = "High Risk"
                status_text = "HIGH ASTHMA RISK DETECTED (MODEL ESTIMATE)"
                status_color = "#C0392B" # Red
                
        except Exception as e:
            st.error(f"Error during model prediction: {e}")
            st.stop()
            
        # --- GENERATE CLINICAL EXPLANATIONS ---
        # 1. BMI Status
        bmi_under, bmi_over, bmi_obese = get_bmi_zones(age, gender)
        if bmi < bmi_under: bmi_status = "Underweight"
        elif bmi < bmi_over: bmi_status = "Healthy Weight"
        elif bmi < bmi_obese: bmi_status = "Overweight"
        else: bmi_status = "Obese"
        
        bmi_context = "adult standard" if age >= 20 else f"pediatric growth charts for a {int(age)}-year-old {gender}"
        
        # 2. Symptom Count
        symptoms_list = [k for k, v in {"Wheezing": wheezing, "Shortness of Breath": sob, "Chest Tightness": chest_tightness, "Coughing": coughing}.items() if v == "Yes"]
        symptom_text = f"The patient reports {len(symptoms_list)} primary respiratory symptoms: {', '.join(symptoms_list)}." if symptoms_list else "No primary respiratory symptoms reported."
        
        # Symptom Consistency Check
        symptom_score = sum([1 if x == "Yes" else 0 for x in [wheezing, sob, chest_tightness, coughing, night_symptoms, exercise_induced]])
        warning_msg = ""
        if symptom_score == 0 and disp_prob >= 0.80:
            warning_msg = "⚠️ <b>NOTE:</b> High risk is driven primarily by lung function indicators despite minimal reported symptoms."

        # 3. Explanations Dictionary
        explanations = {
            "Asthma Risk Probability": f"The machine learning model estimates a {disp_prob*100:.1f}% risk probability based on the provided clinical features. Risk Category: {risk_category}.",
            "Symptom Profile": f"{symptom_text} The radar chart visualizes the presence of key asthma indicators. A larger area indicates a higher symptomatic burden.",
            "BMI Status": f"Patient BMI is {bmi:.2f}. Based on {bmi_context}, this falls into the '{bmi_status}' category. Weight management is often a key factor in asthma control.",
            "Environmental Risk": f"Environmental exposure analysis: Pollution Index is {pollution}/10. Reported triggers include: {', '.join([k for k,v in {'Pets':pet, 'Hay Fever':hayfever, 'Eczema':eczema}.items() if v=='Yes'])}.",
            "Risk Breakdown": "This chart categorizes risk factors into Clinical History (genetics/allergies), Current Symptoms, and Vital/Environmental factors to highlight the primary drivers of the prediction.",
            "Feature Contribution": "This analysis (SHAP) highlights which specific variables pushed the model towards its decision. Longer bars indicate a stronger influence on the final prediction."
        }

        # --- PREPARE FIGURES (Generate before display) ---
        # 1. Risk Gauge
        gauge_fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = disp_prob*100,
            number = {'suffix': "%"},
            gauge = {
                'axis': {'range': [0, 100]}, 'bar': {'color': "#0A1F5A"},
                'steps' : [{'range': [0, 40], 'color': "#D5F5E3"}, {'range': [40, 70], 'color': "#F39C12"}, {'range': [70, 100], 'color': "#FADBD8"}],
                'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': disp_prob*100}}))
        gauge_fig.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=10), font={'family': "Arial", 'color': "#0A1F5A"})

        # 2. Feature Contribution
        shap_fig = None
        if hasattr(model, 'feature_importances_'):
            imps = model.feature_importances_
            names = model.feature_names_in_
            contribs = []
            for i, feature_name in enumerate(names):
                val = input_data.iloc[0][feature_name]
                if feature_name == 'Age': val = val / 100
                elif feature_name == 'BMI': val = (val - 10) / 40
                elif feature_name == 'PollutionExposure': val = val / 10
                elif 'Lung' in feature_name: val = (5 - val) / 5
                contrib = imps[i] * val
                contribs.append((feature_name.replace('_1','').replace('LungFunction',''), contrib))
            
            # Normalize contributions
            total_imp = sum(abs(c[1]) for c in contribs)
            if total_imp == 0: total_imp = 1
            
            norm_contribs = []
            for feat_name, val in contribs:
                n_val = val / total_imp
                # Visual Cap at 60%
                if n_val > 0.6: n_val = 0.6
                elif n_val < -0.6: n_val = -0.6
                if abs(n_val) > 0.001:
                    norm_contribs.append((feat_name, n_val))
            
            norm_contribs.sort(key=lambda x: x[1], reverse=False)
            shap_fig = go.Figure(go.Bar(
                x=[c[1] for c in norm_contribs[-8:]], y=[c[0] for c in norm_contribs[-8:]],
                orientation='h', marker_color='#0A1F5A'
            ))
            shap_fig.update_layout(
                height=300, margin=dict(l=10,r=10,t=10,b=20),
                xaxis=dict(showticklabels=False), font={'family': "Arial", 'color': "#0A1F5A"}
            )

        # 3. Symptom Radar
        cats = ['Wheeze', 'SoB', 'Tightness', 'Cough', 'Night', 'Exercise']
        vals = [1 if x=="Yes" else 0 for x in [wheezing, sob, chest_tightness, coughing, night_symptoms, exercise_induced]]
        radar_fig = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]], fill='toself',
            fillcolor='rgba(10, 31, 90, 0.2)', line=dict(color='#0A1F5A')
        ))
        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
            showlegend=False, height=300, margin=dict(l=30, r=30, t=10, b=10),
            font={'family': "Arial", 'color': "#0A1F5A"}
        )

        # 4. BMI Bar
        bmi_fig = go.Figure()
        zones = [("Under", 0, 18.5, "#EBF5FB"), ("Normal", 18.5, 25, "#D5F5E3"), ("Over", 25, 30, "#FCF3CF"), ("Obese", 30, 60, "#FADBD8")]
        for l, s, e, c in zones:
            bmi_fig.add_trace(go.Bar(x=[e-s], y=[''], base=s, orientation='h', marker_color=c, name=l, hoverinfo='none'))
        bmi_fig.add_vline(x=bmi, line_width=3, line_color="#0A1F5A")
        bmi_fig.add_annotation(x=bmi, y=0, text=f"{bmi:.2f}", showarrow=False, yshift=20, font=dict(color="#0A1F5A", weight="bold"))
        bmi_fig.update_layout(
            barmode='stack', height=300, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(range=[10,45], showgrid=False), yaxis=dict(showticklabels=False),
            showlegend=False, font={'family': "Arial", 'color': "#0A1F5A"}
        )

        # 5. Environmental Risk
        env_labels = ['Pollution', 'Pet Allergy', 'Hay Fever', 'Eczema']
        env_vals = [pollution, 10 if pet=="Yes" else 0, 10 if hayfever=="Yes" else 0, 10 if eczema=="Yes" else 0]
        env_colors = ['#E74C3C' if v > 5 else '#2ECC71' for v in env_vals]
        env_fig = go.Figure(go.Bar(
            x=env_labels, y=env_vals, marker_color=env_colors, text=env_vals, textposition='auto'
        ))
        env_fig.update_layout(
            height=300, margin=dict(l=20,r=20,t=20,b=20),
            yaxis=dict(range=[0, 12], showticklabels=False),
            font={'family': "Arial", 'color': "#0A1F5A"}
        )

        # 6. Risk Breakdown
        count_hist = sum([1 if x == "Yes" else 0 for x in [family_history, hist_allergies, eczema, hayfever, gerd, pet]])
        count_symp = sum([1 if x == "Yes" else 0 for x in [wheezing, sob, chest_tightness, coughing, night_symptoms, exercise_induced]])
        count_env = 1 if pollution > 6 else 0
        if bmi > 25: count_env += 1
        breakdown_fig = go.Figure(go.Bar(
            x=['History', 'Symptoms', 'Vitals'],
            y=[count_hist, count_symp, count_env],
            marker_color=['#6888A8', '#0A1F5A', '#F1C40F'],
            text=[count_hist, count_symp, count_env], textposition='auto'
        ))
        breakdown_fig.update_layout(
            height=300, margin=dict(l=20,r=20,t=20,b=20),
            yaxis=dict(showticklabels=False),
            font={'family': "Arial", 'color': "#0A1F5A"}
        )

        # --- GENERATE PDF ---
        
        # --- MATPLOTLIB GENERATION (No Flashing Windows) ---
        
        # 1. Gauge (Horizontal Bar)
        fig_g, ax_g = plt.subplots(figsize=(5, 2))
        ax_g.barh(0, 100, color='#FADBD8', height=0.5) # Red Zone
        ax_g.barh(0, 50, color='#D5F5E3', height=0.5)  # Green Zone
        ax_g.scatter([disp_prob*100], [0], color='#0A1F5A', s=150, zorder=10, marker='|', linewidth=4)
        ax_g.set_xlim(0, 100); ax_g.set_yticks([]); ax_g.set_xlabel("Risk Probability (%)")
        buf_g = BytesIO(); fig_g.savefig(buf_g, format='png', bbox_inches='tight'); buf_g.seek(0); plt.close(fig_g)

        # 2. Radar (Polar Plot)
        cats = ['Wheeze', 'SoB', 'Tightness', 'Cough', 'Night', 'Exercise']
        vals = [1 if x=="Yes" else 0 for x in [wheezing, sob, chest_tightness, coughing, night_symptoms, exercise_induced]]
        
        # Check if any symptoms present
        if sum(vals) == 0:
             # Create an empty plot with text "No Symptoms Reported"
             fig_r, ax_r = plt.subplots(figsize=(4, 4))
             ax_r.text(0.5, 0.5, "No Reported Symptoms", ha='center', va='center', fontsize=12, color='#64748B')
             ax_r.axis('off')
        else:
             vals += vals[:1] # Close the loop
             angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist(); angles += angles[:1]
             fig_r, ax_r = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
             ax_r.plot(angles, vals, color='#0A1F5A', linewidth=2)
             ax_r.fill(angles, vals, color='#0A1F5A', alpha=0.25)
             ax_r.set_xticks(angles[:-1]); ax_r.set_xticklabels(cats, size=8)
             ax_r.set_yticks([])
        
        buf_r = BytesIO(); fig_r.savefig(buf_r, format='png', bbox_inches='tight'); buf_r.seek(0); plt.close(fig_r)

        # 3. BMI (Stacked Bar)
        b_u, b_o, b_ob = get_bmi_zones(age, gender)
        fig_b, ax_b = plt.subplots(figsize=(5, 2))
        ax_b.barh(0, 60, color='#FADBD8', label='Obese'); ax_b.barh(0, b_ob, color='#FCF3CF', label='Over'); 
        ax_b.barh(0, b_o, color='#D5F5E3', label='Normal'); ax_b.barh(0, b_u, color='#EBF5FB', label='Under')
        ax_b.axvline(bmi, color='#0A1F5A', linewidth=3); ax_b.text(bmi, 0.6, f"{bmi:.2f}", ha='center', color='#0A1F5A', fontweight='bold')
        ax_b.set_xlim(10, 45); ax_b.set_yticks([]); ax_b.set_xlabel("BMI Value")
        buf_b = BytesIO(); fig_b.savefig(buf_b, format='png', bbox_inches='tight'); buf_b.seek(0); plt.close(fig_b)

        # 4. Environmental (Bar)
        fig_e, ax_e = plt.subplots(figsize=(5, 3))
        env_v = [pollution, 10 if pet=="Yes" else 0, 10 if hayfever=="Yes" else 0, 10 if eczema=="Yes" else 0]
        env_c = ['#E74C3C' if v > 5 else '#2ECC71' for v in env_v]
        ax_e.bar(['Pollution', 'Pet', 'Hay Fever', 'Eczema'], env_v, color=env_c)
        buf_e = BytesIO(); fig_e.savefig(buf_e, format='png', bbox_inches='tight'); buf_e.seek(0); plt.close(fig_e)

        # 5. Breakdown (Bar)
        fig_br, ax_br = plt.subplots(figsize=(5, 3))
        ax_br.bar(['History', 'Symptoms', 'Vitals'], [count_hist, count_symp, count_env], color=['#6888A8', '#0A1F5A', '#F1C40F'])
        buf_br = BytesIO(); fig_br.savefig(buf_br, format='png', bbox_inches='tight'); buf_br.seek(0); plt.close(fig_br)

        report_figures = {
            "Asthma Risk Probability": buf_g,
            "Symptom Profile": buf_r,
            "BMI Status": buf_b,
            "Environmental Risk": buf_e,
            "Risk Breakdown": buf_br
        }
        
        # 6. Feature Contribution (Optional)
        if 'contribs' in locals() and contribs:
            fig_s, ax_s = plt.subplots(figsize=(5, 3))
            ax_s.barh([c[0] for c in norm_contribs[-8:]], [c[1] for c in norm_contribs[-8:]], color='#0A1F5A')
            buf_s = BytesIO(); fig_s.savefig(buf_s, format='png', bbox_inches='tight'); buf_s.seek(0); plt.close(fig_s)
            report_figures["Feature Contribution"] = buf_s
            
        patient_info = {"name": name if name else "Unknown", "age": age, "gender": gender, "bmi": bmi}
        pred_info = {"result": status_text, "confidence": f"{disp_prob*100:.1f}%", "is_positive": disp_prob >= 0.4}
        
        pdf_bytes = None
        try:
            with st.spinner("Generating PDF Report..."):
                pdf_bytes = create_pdf(patient_info, pred_info, report_figures, explanations)
        except Exception as e:
            st.error(f"Error generating PDF: {e}")

        # --- DISPLAY DASHBOARD ---
        st.markdown("---")
        
        # Header with Download Button
        h_col1, h_col2 = st.columns([7, 2])
        with h_col1:
            st.markdown("## Prediction Analysis Result")
        with h_col2:
            safe_name = (name if name else "Patient").replace(" ", "_")
            if pdf_bytes:
                st.download_button(
                    label="Download Report",
                    data=pdf_bytes,
                    file_name=f"Asthma_Report_{safe_name}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )
        
        
        st.markdown(f"""
            <div class="result-card" style="border-left: 8px solid {status_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h3 style="color: {status_color}; margin:0;">{status_text}</h3>
                        <p style="color: #0A1F5A; margin: 5px 0 0 0; font-size: 18px;">
                            Predicted Risk Probability: <b>{disp_prob*100:.1f}%</b> <span style="font-size:0.8em; color:#666">({risk_category})</span>
                        </p>
                    </div>
                    <div style="text-align: right; color: #576574; border-left: 1px solid #ddd; padding-left: 10px;">
                        <small>PATIENT</small><br>
                        <b>{name if name else 'Unknown'}</b><br>
                        {age} Years | {gender}
                    </div>
                </div>
            </div>
            {f'<div style="color: #856404; background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 20px; border: 1px solid #ffeeba;">{warning_msg}</div>' if warning_msg else ''}
            <div style="font-size: 0.8em; color: #666; margin-top: 5px; margin-bottom: 20px; text-align: center;">
                <i>This system provides decision support only and is not intended to replace professional medical diagnosis.</i>
            </div>
        """, unsafe_allow_html=True)
            
        
        r1_col1, r1_col2, r1_col3 = st.columns(3)
        
        with r1_col1:
            # CHART 1: Risk Gauge
            with st.container(key="styled_container"):
                st.markdown('<div class="card-header">Asthma Risk Probability</div>', unsafe_allow_html=True)
                st.plotly_chart(gauge_fig, use_container_width=True)

        with r1_col2:
            # CHART 2: Feature Contribution
            with st.container(key="styled_container2"):
                st.markdown('<div class="card-header">Feature Contribution</div>', unsafe_allow_html=True)
                if shap_fig:
                    st.plotly_chart(shap_fig, use_container_width=True)
                else:
                    st.info("Unavailable")

        with r1_col3:
            # CHART 3: Symptom Radar
            with st.container(key="styled_container3"):
                st.markdown('<div class="card-header">Symptom Profile</div>', unsafe_allow_html=True)
                st.plotly_chart(radar_fig, use_container_width=True)

        # --- ROW 2 ---
        r2_col1, r2_col2, r2_col3 = st.columns(3)

        with r2_col1:
            # CHART 4: BMI Bar
            with st.container(key="styled_container4"):
                st.markdown('<div class="card-header">BMI Status</div>', unsafe_allow_html=True)
                st.plotly_chart(bmi_fig, use_container_width=True)

        with r2_col2:
            # CHART 5: Environmental Risk
            with st.container(key="styled_container5"):
                st.markdown('<div class="card-header">Environmental Risk</div>', unsafe_allow_html=True)
                st.plotly_chart(env_fig, use_container_width=True)

        with r2_col3:
            # CHART 6: Risk Breakdown
            with st.container(key="styled_container6"):
                st.markdown('<div class="card-header">Risk Breakdown</div>', unsafe_allow_html=True)
                st.plotly_chart(breakdown_fig, use_container_width=True)
    else:
        st.error("Model is not loaded. Cannot perform prediction.")