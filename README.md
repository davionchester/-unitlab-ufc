# UnitLab UFC

UFC-only prediction research app for moneylines and props.

## Deploy

This is a Streamlit app. Main file: `app.py`.

1. Install dependencies from `requirements.txt`.
2. Run `streamlit run app.py` locally, or deploy the repository with Streamlit Community Cloud.
3. In the app, build/refresh historical data, then train/calibrate the model.

The model reports held-out accuracy and Brier score. Confidence is a probability-strength score, not a guarantee.

The default historical research dataset is from the public GitHub repository `larissapavan/ufc-historical-fight-dataset` and is downloaded at runtime. Verify licensing/terms before commercial use.
