import streamlit as st
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import time
import base64  # Required for embedding local image in CSS
import io  # Required for embedding local image in CSS
import os  # To check if file exists

# --- Page Config ---
st.set_page_config(
    layout="wide",
    page_title="Concrete Bond AI Predictor",
    initial_sidebar_state="expanded"
)


# --- Function to convert local image to Base64 ---
def get_base64_image(image_path):
    if not os.path.exists(image_path):  # Check if background image exists
        st.error(
            f"Error: Background image file not found at {image_path}. Please ensure 'picture.png' is in the same directory as app.py.")
        return None
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        st.error(f"Error encoding background image {image_path}: {e}")
        return None


# --- Get Base64 for the background image ---
background_image_path = 'picture.png'
encoded_image = get_base64_image(background_image_path)

# --- Custom CSS (Reintroduced for background image and logo scaling) ---
if encoded_image:
    st.markdown(f"""
    <style>
        /* General Styling */
        .stApp {{
            background-color: #f0f2f6; /* Light gray background */
            font-family: 'Inter', sans-serif; /* Example Google Font */
        }}

        /* Header with Background Image */
        .header-background {{
            background-image: url("data:image/png;base64,{encoded_image}");
            background-size: cover; /* Cover the entire area */
            background-position: center; /* Center the image */
            padding: 40px 20px; /* Adjusted padding */
            text-align: center;
            color: white; /* Ensure title text is visible */
            border-radius: 10px; /* Rounded corners for the container */
            margin-bottom: 25px; /* Increased margin */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Subtle shadow */
            position: relative; /* Needed for overlay */
            overflow: hidden; /* Hide overflow from border-radius */
        }}

        .header-background::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.45); /* Slightly darker overlay */
            z-index: 1;
        }}

        .header-background h1 {{
            color: white !important; /* Force white color for title */
            font-size: 2.8em; /* Adjusted title size */
            font-weight: 700; /* Bold */
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.6); /* Text shadow for better contrast */
            z-index: 2; /* Bring text above overlay */
            position: relative; /* Needed for z-index to work */
            margin-bottom: 0; /* Remove default margin */
        }}

        /* Sidebar Styling (Simplified) */
        [data-testid="stSidebar"] {{
            background-color: #ffffff;
            padding: 1rem;
            border-right: 1px solid #e0e0e0;
        }}
         [data-testid="stSidebar"] h2 {{ /* Target sidebar header */
            color: #1e40af; /* Blue */
            font-size: 1.6em;
            font-weight: bold;
            border-bottom: 2px solid #3b82f6;
            padding-bottom: 5px;
            margin-bottom: 15px;
        }}
         [data-testid="stSidebar"] h3 {{ /* Target sidebar subheader */
            color: #3b82f6; /* Lighter Blue */
            font-size: 1.2em;
            font-weight: 600;
            margin-top: 20px;
        }}

        /* --- UPDATED: Sidebar Logo Scaling --- */
        .sidebar-logo-container img {{
            width: 90%; /* Set a fixed width or use a larger percentage */
            max-width: 260px; /* Override any max-width restrictions */
            height: auto;
            display: block;
            margin-left: auto;
            margin-right: auto;
            padding-bottom: 15px;
        }}

        /* Metric Styling */
        [data-testid="stMetric"] {{
            background-color: #ffffff;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border: 1px solid #e0e0e0;
            text-align: center; /* Center metric content */
        }}
        [data-testid="stMetricLabel"] {{
            font-size: 1.1em;
            color: #4a4a4a;
            font-weight: 600;
            margin-bottom: 5px; /* Add space below label */
        }}
        [data-testid="stMetricValue"] {{
            font-size: 2.2em;
            color: #1e40af; /* Blue accent */
            font-weight: bold;
        }}

        /* Button Styling */
        .stButton>button {{
            background-color: #3b82f6; /* Blue */
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px; /* Slightly larger padding */
            font-size: 1.1em;
            font-weight: bold;
            transition: background-color 0.3s ease, transform 0.1s ease;
            width: 100%;
            margin-top: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stButton>button:hover {{
            background-color: #1d4ed8; /* Darker blue */
            transform: translateY(-1px); /* Slight lift on hover */
        }}
        .stButton>button:active {{
             transform: translateY(0px); /* Reset lift on click */
        }}

        /* Expander Styling */
        .stExpander {{
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            background-color: #ffffff;
            margin-top: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .stExpander header {{
            color: #1e40af; /* Blue */
            font-weight: 600;
            font-size: 1.1em;
        }}

        /* Info/Success/Error messages */
        .stAlert {{
            border-radius: 8px;
        }}

        /* Footer */
        .footer-text {{
             text-align: center;
             margin-top: 40px;
             color: #6b7280; /* Gray text */
             font-size: 0.9em;
        }}
        .footer-text hr {{
             border-top: 1px solid #e0e0e0;
             margin-bottom: 10px;
        }}

    </style>
    """, unsafe_allow_html=True)


# --- 2. Model Loading Functions --- (Unchanged)
@st.cache_resource
def load_xgboost_model():
    """Loads the XGBoost (Load) model."""
    try:
        model = joblib.load('Force-XG-Final_Model.pkl'); return model
    except Exception as e:
        st.error(f"Error loading XGBoost (Load) model: {e}"); return None


@st.cache_resource
def load_xgboost_nbs_model():
    """Loads the XGBoost (NBS) model."""
    try:
        model = joblib.load('NBS-XG-Final_Model.pkl'); return model
    except Exception as e:
        st.error(f"Error loading XGBoost (NBS) model: {e}"); return None


@st.cache_resource
def load_xgb_failure_model():
    """Loads the XGBoost (Failure Mode) model."""
    try:
        model = joblib.load('Failure_Mode_XG_Final_Model.pkl'); return model
    except Exception as e:
        st.error(f"Error loading XGBoost (Failure Mode) model: {e}"); return None


# Load only XGBoost models
xgb_model = load_xgboost_model()
xgb_nbs_model = load_xgboost_nbs_model()
xgb_failure_model = load_xgb_failure_model()

# --- 3. Define Mappings and Column Order --- (Unchanged)
fibre_type_map = {'Without Fibre': 0, 'Steel Fibre': 1, 'Polypropylene Fibre': 2}
rebar_type_map = {'Plain Round Rebar': 0, 'Deformed Steel Rebar': 1}
stirrups_map = {'With Stirrups': 0, 'Without Stirrups': 1}
epoxy_map = {'With Epoxy Coat': 0, 'Without Epoxy Coat': 1}
failure_mode_map = {0: "Pullout Failure", 1: "Splitting Failure"}

FEATURE_COLUMNS_CLEAN = [
    'Fibre type', 'Volume fraction (%)', 'Age at testing (days)', 'fc (Mpa)',
    'bar diameter (cm)', 'Bar yield stress (Mpa)',
    'Rebar Type', 'Stirrups', 'Stirrups Bar Diameter (cm)', 'Epoxy Coat', 'embedment length (cm)',
    'Cover (cm)', 'Temperature (°C)', 'Time (Hour)', 'Heating Rate (C/min)',
    'Air Cooling', 'Water Cooling', 'Without Cooling', 'Ambient',
    'Corrosion Level (%)'
]

# --- 4. User Interface (Sidebar) ---
with st.sidebar:
    logo_path = "Logo.png"
    if os.path.exists(logo_path):
        # Wrap the image in a div with a custom class for CSS targeting
        st.markdown('<div class="sidebar-logo-container">', unsafe_allow_html=True)
        st.image(logo_path)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Logo.png not found. Displaying placeholder.")
        st.image("https://placehold.co/300x80/DDDDDD/000000?text=Logo+Missing", use_container_width=True)

    st.header("Model Input Parameters")
    st.write('Enter the values below for prediction:')
    st.divider()

    inputs = {}
    st.subheader("Categorical Features")
    fibre_selection_text = st.selectbox('Fibre type', list(fibre_type_map.keys()), help="Type of fibre reinforcement.")
    inputs['Fibre type'] = fibre_type_map[fibre_selection_text]
    inputs['Rebar Type'] = rebar_type_map[
        st.selectbox('Rebar Type', list(rebar_type_map.keys()), help="Type of rebar.")]
    stirrups_selection_text = st.selectbox('Stirrups', list(stirrups_map.keys()), help="Presence of stirrups.")
    inputs['Stirrups'] = stirrups_map[stirrups_selection_text]
    inputs['Epoxy Coat'] = epoxy_map[st.selectbox('Epoxy Coat', list(epoxy_map.keys()), help="Epoxy coating.")]
    cooling_method_text = st.selectbox('Cooling Method',
                                       ['Ambient (No high temperature)', 'Air-cooled', 'Water-cooled', 'No-cooled'],
                                       help="Cooling method.")
    inputs['Cooling Method'] = cooling_method_text

    st.subheader("Numerical Features")
    is_without_fibre = (fibre_selection_text == 'Without Fibre')
    vol_frac_value = 0.0 if is_without_fibre else 1.0
    inputs['Volume fraction (%)'] = st.number_input('Volume fraction (%)', 0.0, 5.0, vol_frac_value, 0.1,
                                                    disabled=is_without_fibre,
                                                    help="Volume of fibres (0 if 'Without Fibre').")
    inputs['Age at testing (days)'] = st.number_input('Age at testing (days)', 7, 120, 28, 1, help="Age of concrete.")
    inputs['fc (Mpa)'] = st.number_input('Concrete Strength (fc, Mpa)', 20, 100, 35, 1, help="Compressive strength.")
    inputs['bar diameter (cm)'] = st.number_input('Bar diameter (cm)', 0.75, 3.0, 1.0, 0.01, help="Rebar diameter.")
    inputs['Bar yield stress (Mpa)'] = st.number_input('Bar yield stress (Mpa)', 300, 700, 420, 10,
                                                       help="Rebar yield stress.")
    is_without_stirrups = (stirrups_selection_text == 'Without Stirrups')
    stirrup_dia_value = 0.0 if is_without_stirrups else 0.8
    inputs['Stirrups Bar Diameter (cm)'] = st.number_input('Stirrups Bar Diameter (cm)', 0.0, 5.0, stirrup_dia_value,
                                                           0.1, disabled=is_without_stirrups,
                                                           help="Stirrup diameter (0 if 'Without Stirrups').")
    inputs['embedment length (cm)'] = st.number_input('Embedment length (cm)', 3.0, 25.0, 10.0, 0.1,
                                                      help="Rebar embedment length.")
    inputs['Cover (cm)'] = st.number_input('Cover (cm)', 2.0, 10.0, 3.0, 0.1, help="Concrete cover.")
    inputs['Temperature (°C)'] = st.number_input('Temperature (°C)', 20, 1000, 20, 10, help="Max temperature.")
    is_ambient = (cooling_method_text == 'Ambient (No high temperature)')
    time_value = 0.0 if is_ambient else 1.0
    heating_rate_value = 0 if is_ambient else 10
    inputs['Time (Hour)'] = st.number_input('Time (Hour)', 0.0, 24.0, time_value, 0.5, disabled=is_ambient,
                                            help="Time at max temp (0 if Ambient).")
    inputs['Heating Rate (C/min)'] = st.number_input('Heating Rate (C/min)', 0, 30, heating_rate_value, 1,
                                                     disabled=is_ambient, help="Heating rate (0 if Ambient).")
    inputs['Corrosion Level (%)'] = st.number_input('Corrosion Level (%)', 0.0, 100.0, 0.0, 0.1,
                                                    help="Rebar corrosion level.")

    predict_button = st.button('Run Prediction', type="primary", key='predict_button_sidebar',
                               use_container_width=True)

# --- Main Area ---

# --- Header with Background Image ---
if encoded_image:
    st.markdown('<div class="header-background"><h1>Concrete Bond Strength Prediction</h1></div>',
                unsafe_allow_html=True)
else:
    st.title('Concrete Bond Strength Prediction')  # Fallback title
    # Fallback image placeholder
    st.image("https://placehold.co/1200x150/DDDDDD/000000?text=Concrete+Bond+Strength+Predictor",
             use_container_width=True)

# --- UPDATED: Improved Welcome Text ---
st.markdown("""
### Welcome to the Concrete Bond Strength Predictor

This interactive web tool uses advanced **XGBoost machine learning models** to estimate the **bond performance between concrete and reinforcing steel** under various thermal and environmental conditions.

You can adjust the material, geometric, and exposure parameters to obtain:
- The **predicted pull-out load** (kN)
- The **normalized bond strength (NBS%)**, defined as the bond strength at elevated temperature relative to the ambient (room temperature) bond strength
- The **likely failure mode** (Pullout or Splitting)

These insights can help engineers and researchers better understand the influence of temperature, corrosion, and material design on bond behavior.
""")

# --- ADDED: Help & Usage Guide ---
with st.expander("Usage Guide"):
    st.markdown("""
    #### How to Use
    1.  **Set Input Parameters:**
        Use the sidebar to specify the material properties, reinforcement details, and environmental conditions. Hover over the input labels for more details.
    2.  **Run the Prediction:**
        Click **Run Prediction** in the sidebar to compute the bond load, normalized bond strength, and failure mode.
    3.  **View Results:**
        The predicted values will appear in the main panel below. You can also expand the sections further down to view:
        - The **raw inputs** you entered.
        - The **processed data** used by the model.

    #### Model Information
    - The models are trained using experimental data on concrete–rebar bond performance under elevated temperature conditions.
    - Each model (Load, NBS, and Failure Mode) is based on the **optimized XGBoost algorithm**, which was selected after comparing four machine learning models and demonstrated the highest prediction accuracy and robustness.

    #### Important Notes
    - The predictions are intended for **research and educational purposes** only.
    - Always verify with experimental results or design codes before practical use.
    - Input values outside typical laboratory ranges may lead to unreliable predictions.
    """)

# --- ADDED: Model Validation Section ---
with st.expander(" Model Validation & Performance"):
    st.markdown("""
    The XGBoost models were trained and validated using an extensive experimental database on concrete–rebar bond strength.

    **Performance Highlights (on Test Set):**

    * **Bond Strength (Load):**
        * R² Score: 0.949
        * Mean Absolute Error (MAE): 4.074 kN
        * Root Mean Squared Error (RMSE): 6.231 kN
    * **NBS (%):**
        * R² Score: 0.907
        * Mean Absolute Error (MAE): 6.319 %
        * Root Mean Squared Error (RMSE): 9.993 %

    * **Failure Mode:**
       * Accuracy: 0.983


    * **Dataset Size:** 1000+ laboratory-tested samples

    These metrics indicate a high level of predictive accuracy for the studied parameter ranges.
    """)

# --- Prediction Logic and Display ---
st.header('Prediction Results')
st.markdown("---")
res_col1, res_col2, res_col3 = st.columns(3)

prediction_load = None
prediction_nbs = None
prediction_failure = None

if predict_button:
    if xgb_model and xgb_nbs_model and xgb_failure_model:
        with st.spinner('Processing XGBoost predictions... Please wait.'):
            time.sleep(1.5)
            try:
                processed_inputs = inputs.copy()
                selected_cooling = processed_inputs.pop('Cooling Method')
                processed_inputs['Air Cooling'] = 1 if selected_cooling == 'Air-cooled' else 0
                processed_inputs['Water Cooling'] = 1 if selected_cooling == 'Water-cooled' else 0
                processed_inputs['Without Cooling'] = 1 if selected_cooling == 'No-cooled' else 0
                processed_inputs['Ambient'] = 1 if selected_cooling == 'Ambient (No high temperature)' else 0

                if processed_inputs['Fibre type'] == fibre_type_map['Without Fibre']: processed_inputs[
                    'Volume fraction (%)'] = 0.0
                if processed_inputs['Stirrups'] == stirrups_map['Without Stirrups']: processed_inputs[
                    'Stirrups Bar Diameter (cm)'] = 0.0
                if selected_cooling == 'Ambient (No high temperature)':
                    processed_inputs['Time (Hour)'] = 0.0;
                    processed_inputs['Heating Rate (C/min)'] = 0.0

                input_df_clean = pd.DataFrame([processed_inputs])
                input_df_clean = input_df_clean[FEATURE_COLUMNS_CLEAN]

                prediction_load = xgb_model.predict(input_df_clean)[0]
                prediction_nbs = xgb_nbs_model.predict(input_df_clean)[0]
                fail_pred = xgb_failure_model.predict(input_df_clean)[0]
                prediction_failure = failure_mode_map.get(fail_pred, "Unknown")

                # --- Display Results ---
                with res_col1:
                    st.metric("Predicted Load (kN)", f"{prediction_load:.2f} kN")
                with res_col2:
                    st.metric("Predicted NBS (%)", f"{prediction_nbs:.2f} %")
                with res_col3:
                    st.metric("Failure Mode", prediction_failure)
                st.success("Predictions processed successfully!")

                # --- Expanders ---
                with st.expander("Show Raw User Inputs"):
                    display_inputs = inputs.copy()
                    try:
                        for key, val_map in [('Fibre type', fibre_type_map), ('Rebar Type', rebar_type_map),
                                             ('Stirrups', stirrups_map), ('Epoxy Coat', epoxy_map)]:
                            display_inputs[key] = [k for k, v in val_map.items() if v == inputs.get(key, None)][0]
                        st.dataframe(pd.DataFrame([display_inputs], index=[0]), use_container_width=True)
                    except (IndexError, KeyError):
                        st.dataframe(pd.DataFrame([inputs], index=[0]), use_container_width=True)
                with st.expander("Show Processed Data Sent to Models"):
                    st.dataframe(input_df_clean, use_container_width=True)

            except Exception as e:
                st.error(f"An unexpected error occurred during prediction: {e}");
                st.exception(e)
    else:
        st.error("One or more XGBoost models failed to load. Cannot run predictions.")

else:  # If button not pressed
    with res_col1:
        st.metric("Predicted Load (kN)", "--")
    with res_col2:
        st.metric("Predicted NBS (%)", "--")
    with res_col3:
        st.metric("Failure Mode", "--")
    st.info("Please fill parameters and click 'Run Prediction'.")

# --- ADDED: Contact Us Section ---
st.markdown("---")
with st.expander(" Contact Us"):
    st.markdown("""
    #### We'd love to hear from you!

    If you have any questions, suggestions, or feedback about this application or its models, feel free to reach out.

    **Contact Information:**
    -  **Email:** matin.rezvani11@sharif.edu, vahabtoufigh@gmail.com
    -  **Institution:** Department of Civil Engineering, Sharif University of Technology

    You can also contact us to:
    - Request collaboration or model access
    - Report issues or suggest improvements
    - Share relevant experimental data for model enhancement

    ---
    _Thank you for supporting data-driven research in structural engineering!_
    """)

# --- ADDED: Version Info Footer ---
st.markdown("""
<div class="footer-text">
    <hr>
    Version: 1.0.0 | Developed by: Material AI Research Team | Last Updated: October 2025
</div>
""", unsafe_allow_html=True)
