import streamlit as st
import requests

st.set_page_config(page_title="MaterialAI", page_icon="🔬")

# 1. UI HEADER
st.title("🔬 Materials Band Gap Predictor")
st.markdown("---")

# 2. INPUT SECTION
formula = st.text_input("Enter Chemical Formula", value="TiO2", help="e.g., GaN, Si, NaCl, GaAs")

# 3. PREDICTION LOGIC
if st.button("Predict Property", type="primary"):
    with st.spinner("Querying XGBoost model on GCP..."):
        try:
            # REPLACE THIS URL with your live Cloud Run link
            api_url = f"https://bandgap-1007710367480.asia-south2.run.app/predict?formula="
            
            response = requests.get(api_url)
            
            if response.status_code == 200:
                data = response.json()
                bg_value = data['prediction']['band_gap_ev']
                
                # Big beautiful display
                st.balloons()
                st.metric(label="Predicted Band Gap", value=f"{bg_value} eV")
                
                # Logic for interpretation
                if bg_value == 0:
                    st.info("Status: **Metallic**")
                elif bg_value < 3.0:
                    st.success("Status: **Semiconductor**")
                else:
                    st.warning("Status: **Insulator**")
            else:
                st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error("Connection failed. Check if the GCP Service is active.")

st.sidebar.caption("System: FastAPI + XGBoost + Docker + GCP")
