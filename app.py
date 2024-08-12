import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objs as go
from scipy.signal import savgol_filter, detrend
from PIL import Image

# Load your pre-trained model and pre-processing pipeline
model = joblib.load('multi_output_stacking_model.pkl')
scaler = joblib.load('preprocess_pipeline.pkl')

# Set page configuration
st.set_page_config(page_title="MIR Soil Spectral Analysis", layout="wide")

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
icar_logo = icar_logo.resize((120, 120))
iiss_logo = iiss_logo.resize((120, 120))
icraf_logo = icraf_logo.resize((180, 120))

# Layout the title and logos using columns
col1, col2, col3, col4 = st.columns([1, 1, 6, 1])
with col1:
    st.image(icar_logo, use_column_width=False)
with col2:
    st.image(iiss_logo, use_column_width=False)
with col3:
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Edu VIC WA NT Beginner:wght@500&display=swap');
            h2 {
                font-family: 'Edu VIC WA NT Beginner', serif;
                text-align: center;
                font-size: 48px;
                text-shadow: 2px 2px 2px #aaa;
                font-weight: bold;
                margin-bottom: 10px;
                color: #092922;
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
    'Select soil property to  make predictions:',
    ['Cu', 'Zn', 'Fe', 'Mn', 'All']
)

# File uploader for CSV
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Ensure the first column is treated as ID
    ids = data.iloc[:, 0]
    spectra = data.iloc[:, 1:]
    
    # Retrieve the spectral ranges from the column headers
    x_values = spectra.columns.astype(str)  # Ensure headers are strings
    
    # Handle varied number of columns by truncating or padding
    expected_columns = 1714  # Change to the expected number of features
    if spectra.shape[1] > expected_columns:
        spectra = spectra.iloc[:, :expected_columns]
    elif spectra.shape[1] < expected_columns:
        # Padding with zeros (or any other value like mean) for missing columns
        padding = np.zeros((spectra.shape[0], expected_columns - spectra.shape[1]))
        spectra = np.hstack([spectra, padding])
        spectra = pd.DataFrame(spectra)  # Convert back to DataFrame for easier handling

    # Number input to select the number of rows to plot
    num_rows = st.number_input("Enter the number of rows to preview spectral data:", min_value=1, max_value=spectra.shape[0], value=5, step=1)

    # Layout the buttons in a single row
    st.markdown("""
        <style>
            .stButton {
                width: 150px;
                height: 40px;
                font-size: 26px;
                font-weight: bold;
                border-radius: 5px;
                margin: 0 10px;
            }
            .button-container {
                display: flex;
                justify-content: center;
                margin: 40px 0;  
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Container for buttons
    col1, blank_col, col2 = st.columns([1,3,1])
    
    with col1:
        if st.button("Preview Spectral Data"):
            plot_button = True
        else:
            plot_button = False
    
    with col2:
        if st.button("Run the Model"):
            run_model_button = True
        else:
            run_model_button = False

    if plot_button:
        selected_spectra = spectra.iloc[:num_rows, :]

        fig = go.Figure()

        for i in range(num_rows):
            fig.add_trace(go.Scatter(
                x=x_values,
                y=selected_spectra.iloc[i, :],
                mode='lines',
                name=f"{ids.iloc[i]}"
            ))

        # Define the interval for x-axis labels to avoid congestion
        display_interval = 30  # Display every 30th label; adjust as needed
        tick_indices = np.arange(0, len(x_values), display_interval)
        tick_vals = [x_values[i] for i in tick_indices]

        fig.update_layout(
            title="Spectral Plot",
            xaxis_title="Wavelength (cm⁻¹)",
            yaxis_title="Absorbance",
            legend_title="Sample ID",
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(
                tickvals=tick_indices,  # Indices of ticks to display
                ticktext=tick_vals,     # Labels for the ticks
                tickangle=-90           # Optional: Rotate labels if needed
            )
        )

        st.plotly_chart(fig)

    if run_model_button:
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
        st.download_button(label="Download the model predictions as CSV", data=csv, file_name='predictions.csv', mime='text/csv')
