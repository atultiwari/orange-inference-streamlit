import streamlit as st
import pickle
import os
import Orange

# Must be the first Streamlit command
st.set_page_config(
    page_title="Orange Predictor App",
    page_icon="🍊",
    layout="centered"
)

def load_orange_model(uploaded_file):
    """Loads a .pkcls file using pickle and extracts Orange UI metadata."""
    try:
        model = pickle.load(uploaded_file)
        
        # Extract features (attributes)
        features = []
        for attr in model.domain.attributes:
            feature_info = {
                "name": attr.name,
                "type": "continuous" if attr.is_continuous else ("discrete" if attr.is_discrete else "other")
            }
            if attr.is_discrete:
                feature_info["values"] = attr.values
            features.append(feature_info)
            
        # Extract class variable info
        class_var = model.domain.class_var
        class_info = {
            "name": class_var.name,
            "values": class_var.values
        }
        
        return {
            "model": model,
            "features": features,
            "class_variable": class_info
        }
    except Exception as e:
        st.error(f"Failed to parse Orange model: {str(e)}")
        return None

def main():
    st.title("🍊 Orange AI Inference")
    st.markdown("Upload your Orange Data Mining classification model (`.pkcls`) to dynamically generate an input form and get instant predictions.")
    
    st.divider()

    uploaded_file = st.file_uploader("Step 1: Upload Model", type=['pkcls'])
    
    if uploaded_file is not None:
        
        # Load and parse the model directly into memory
        with st.spinner("Analyzing model metadata..."):
            metadata = load_orange_model(uploaded_file)
            
        if metadata is not None:
            st.success(f"Model `{uploaded_file.name}` loaded successfully!")
            st.divider()
            
            st.subheader("Step 2: Enter Input Features")
            
            # Create dynamic form based on parsed features
            with st.form("prediction_form"):
                input_data = {}
                cols = st.columns(2)
                
                for idx, feature in enumerate(metadata["features"]):
                    col = cols[idx % 2]
                    
                    if feature["type"] == "discrete":
                        input_data[feature['name']] = col.selectbox(
                            label=f"{feature['name']} (discrete)",
                            options=feature["values"]
                        )
                    else:
                        input_data[feature['name']] = col.number_input(
                            label=f"{feature['name']} (continuous)",
                            value=0.0,
                            step=0.1
                        )
                        
                submit_button = st.form_submit_button("Generate Prediction")
                
            if submit_button:
                model = metadata["model"]
                
                # Construct data row based on domain attributes order
                data_row = []
                for attr in model.domain.attributes:
                    val = input_data[attr.name]
                    # Map discrete string values back to their index 
                    if attr.is_discrete and isinstance(val, str):
                        try:
                            val = attr.values.index(val)
                        except ValueError:
                            st.error(f"Invalid value '{val}' for discrete feature '{attr.name}'")
                            return
                    data_row.append(float(val))
                    
                X = [data_row]
                
                try:
                    # Predict Class
                    pred_idx = model(X)[0]
                    predicted_class = model.domain.class_var.values[int(pred_idx)]
                    
                    # Predict Probabilities
                    probs = model(X, model.Probs)[0]
                    prob_dict = {
                        model.domain.class_var.values[i]: float(probs[i])
                        for i in range(len(model.domain.class_var.values))
                    }
                    
                    st.divider()
                    st.subheader("Prediction Results")
                    
                    st.markdown(f"**Predicted Class (target: {metadata['class_variable']['name']})**")
                    st.success(f"### {predicted_class}")
                    
                    st.markdown("**Class Probabilities**")
                    
                    # Sort probabilities descending
                    sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))
                    
                    for class_name, prob in sorted_probs.items():
                        st.write(f"{class_name}: {prob*100:.1f}%")
                        st.progress(prob)
                        
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
