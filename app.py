import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.signal import savgol_filter, detrend
from PIL import Image

# Load your pre-trained model and pre-processing pipeline
model = joblib.load('multi_output_stacking_model.pkl')
scaler = joblib.load('preprocess_pipeline.pkl')

# Set page configuration
st.set_page_config(page_title="MIR Soil Spectral Analysis", layout="wide")
from PIL import Image


# Add custom CSS to hide the "View Full Screen" option
hide_streamlit_style = """
            <style>
            button[title="View fullscreen"] {
                visibility: hidden;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load and resize the images
icar_logo = Image.open("ICAR Logo.png")
iiss_logo = Image.open("IISS Logo.png")
icraf_logo = Image.open("ICRAF Logo.png")


# Resize the images to the same dimensions
icar_logo = icar_logo.resize((100, 120))
iiss_logo = iiss_logo.resize((100, 120))
icraf_logo = icraf_logo.resize((100, 120))

# Layout the title and logos using columns
col1, col2, col3, col4 = st.columns([1,1, 6, 1])
with col1:
    st.image(icar_logo, use_column_width=False)
with col2:
    st.image(iiss_logo, use_column_width=False)
with col3:
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@500&display=swap');
            h2 {
                font-family: 'Roboto', sans-serif;
                text-align: center;
                font-size: 32px;
                font-weight: bold;
                margin-bottom: 10px;
            }
        </style>
        <h2>MIR Spectroscopy Analysis System for Indian Soil</h2>
    """, unsafe_allow_html=True)
with col4:
    st.image(icraf_logo, use_column_width=False)


# Sidebar with instructions and data example
st.sidebar.markdown("### Instructions")
st.sidebar.markdown("""
1. Ensure the first column is the sample ID.
2. The following columns represent the spectra wavelengths (MIR), while rows represent the measurements.
3. If the number of columns differs, the model will preprocess the data accordingly.
""")
st.sidebar.image('Data_Format.png', caption='Example of the CSV format')

# Dropdown to select which property to predict
property_selection = st.selectbox(
    'Select Property to Predict:',
    ['Cu', 'Zn', 'Fe', 'Mn', 'All']
)

# File uploader for CSV
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Ensure the first column is treated as ID
    ids = data.iloc[:, 0]
    spectra = data.iloc[:, 1:]

    # Handle varied number of columns by truncating or padding
    expected_columns = 1714  # Change to the expected number of features
    if spectra.shape[1] > expected_columns:
        spectra = spectra.iloc[:, :expected_columns]
    elif spectra.shape[1] < expected_columns:
        # Padding with zeros (or any other value like mean) for missing columns
        padding = np.zeros((spectra.shape[0], expected_columns - spectra.shape[1]))
        spectra = np.hstack([spectra, padding])


    # Apply Savitzky-Golay filter and baseline correction
    X_smoothed = savgol_filter(spectra, window_length=11, polyorder=2, axis=1)
    X_corrected = detrend(X_smoothed, axis=1)

    # Scale the data using the pre-trained scaler
    spectra_scaled = scaler.transform(X_corrected)
    
    # Make predictions using the model
    predictions = model.predict(spectra_scaled)
    
    # Create a DataFrame for the results
    if property_selection == 'All':
        results = pd.DataFrame(predictions, columns=['Cu', 'Zn', 'Fe', 'Mn'])
    else:
        property_index = ['Cu', 'Zn', 'Fe', 'Mn'].index(property_selection)
        results = pd.DataFrame(predictions[:, property_index], columns=[property_selection])
    
    results.insert(0, 'ID', ids)
    
    # Display the results
    st.markdown("### Prediction Results")
    st.write(results)
    
    # Option to download the results
    csv = results.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download predictions as CSV", data=csv, file_name='predictions.csv', mime='text/csv')