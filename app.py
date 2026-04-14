import streamlit as st
import pandas as pd
import tempfile
import os
from src.data_loader import DataLoader
from src.trainer import ModelTrainer
from src.predictor import Predictor

st.set_page_config(
    page_title="ML Simple Trainer",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ML Simple Trainer")
st.markdown("Train machine learning models and make predictions easily.")

# Sidebar untuk konfigurasi
with st.sidebar:
    st.header("⚙️ Configuration")
    model_type = st.selectbox(
        "Model Type",
        options=["random_forest", "logistic_regression"],
        help="Random Forest untuk akurasi lebih baik, Logistic Regression untuk interpretasi mudah"
    )
    test_size = st.slider(
        "Test Set Size (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        help="Persentase data untuk testing"
    ) / 100
    random_state = st.number_input(
        "Random State",
        min_value=0,
        max_value=100,
        value=42,
        help="Seed untuk reproducible results"
    )

# Tab untuk Train dan Predict
tab1, tab2 = st.tabs(["📊 Train Model", "🔮 Make Predictions"])

# ============================================
# TAB 1: TRAINING
# ============================================
with tab1:
    st.header("Train a New Model")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file for training",
        type=["csv"],
        help="File CSV harus memiliki kolom fitur dan satu kolom target (label)"
    )
    
    if uploaded_file is not None:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Display data preview
            df = pd.read_csv(uploaded_file)
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            # Display column info
            st.subheader("Column Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Features:**")
                st.write(list(df.columns[:-1]) if len(df.columns) > 1 else ["No feature columns"])
            with col2:
                st.write("**Target Column:**")
                st.write(df.columns[-1] if len(df.columns) > 0 else "Unknown")
            
            # Target column selection
            target_column = st.selectbox(
                "Select Target Column",
                options=df.columns.tolist(),
                index=len(df.columns) - 1,
                help="Kolom yang akan diprediksi"
            )
            
            if st.button("🚀 Start Training", type="primary"):
                with st.spinner("Training model... Please wait."):
                    try:
                        # Load and split data
                        X_train, X_test, y_train, y_test, problem_type = DataLoader.load_and_split(
                            file_path=tmp_path,
                            target_column=target_column,
                            test_size=test_size,
                            random_state=random_state
                        )
                        
                        # Train model
                        trainer = ModelTrainer(
                            model_type=model_type,
                            problem_type=problem_type
                        )
                        trainer.train(X_train, y_train)
                        
                        # Evaluate
                        metrics = trainer.evaluate(X_test, y_test)
                        
                        # Save model
                        model_path = trainer.save()
                        
                        # Display results
                        st.success("✅ Training completed successfully!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Problem Type", problem_type)
                        with col2:
                            if problem_type == "classification":
                                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                            else:
                                st.metric("MSE", f"{metrics['mse']:.4f}")
                        with col3:
                            st.metric("Model Saved", "models/model.pkl")
                        
                        # Display dataset info
                        st.subheader("Dataset Information")
                        st.write(f"Total samples: {len(X_train) + len(X_test)}")
                        st.write(f"Training samples: {len(X_train)}")
                        st.write(f"Testing samples: {len(X_test)}")
                        
                        # Feature importance (for Random Forest)
                        if model_type == "random_forest" and problem_type == "classification":
                            st.subheader("Feature Importance")
                            importance = trainer.model.feature_importances_
                            feature_names = X_train.columns.tolist()
                            importance_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': importance
                            }).sort_values('Importance', ascending=False)
                            st.dataframe(importance_df)
                        
                    except ValueError as e:
                        st.error(f"Error: {e}")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")
        
        finally:
            # Clean up temp file
            os.unlink(tmp_path)

# ============================================
# TAB 2: PREDICTION
# ============================================
with tab2:
    st.header("Make Predictions")
    
    # Check if model exists
    if not os.path.exists('models/model.pkl'):
        st.warning("⚠️ No trained model found. Please train a model first in the Train tab.")
    else:
        uploaded_pred_file = st.file_uploader(
            "Upload CSV file for prediction",
            type=["csv"],
            key="predict_uploader",
            help="File CSV harus memiliki kolom fitur yang sama dengan data training (tanpa kolom target)"
        )
        
        if uploaded_pred_file is not None:
            # Display prediction data preview
            pred_df = pd.read_csv(uploaded_pred_file)
            st.subheader("Prediction Data Preview")
            st.dataframe(pred_df.head(10))
            
            if st.button("🔮 Run Prediction", type="primary"):
                with st.spinner("Making predictions..."):
                    try:
                        # Save uploaded file to temp location
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                            tmp_file.write(uploaded_pred_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        try:
                            # Load model and predict
                            model = Predictor.load_model()
                            predictions = Predictor.predict(model, tmp_path)
                            
                            # Display results
                            st.success("✅ Predictions completed!")
                            
                            # Create results dataframe
                            results_df = pred_df.copy()
                            results_df['Prediction'] = predictions
                            
                            st.subheader("Prediction Results")
                            st.dataframe(results_df)
                            
                            # Download button for results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name="predictions.csv",
                                mime="text/csv"
                            )
                            
                            # Simple chart for regression predictions
                            if len(predictions) > 0 and isinstance(predictions[0], float):
                                st.subheader("Predictions Visualization")
                                st.line_chart(predictions)
                        
                        finally:
                            os.unlink(tmp_path)
                    
                    except ValueError as e:
                        st.error(f"Error: {e}")
                    except FileNotFoundError as e:
                        st.error(f"Model not found: {e}")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")

# Footer
st.markdown("---")
st.markdown(
    "ML Simple Trainer | Built with Streamlit | "
    "[Documentation](docs/blueprint.md)"
)
