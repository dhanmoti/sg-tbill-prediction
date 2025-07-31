# app.py
import pandas as pd
import gradio as gr
from pycaret.regression import load_model, predict_model

# Load your trained model
final_model = load_model("final-model")

def predict(
    auction_tenor,
    auction_amount,
    auction_month,
    auction_year,
    days_to_maturity
):
    # Prepare input without target column
    input_df = pd.DataFrame({
        'Auction Tenor (Month)': [auction_tenor],
        'Auction Amount (S$M)': [auction_amount],
        'Auction Month': [auction_month],
        'Auction Year': [auction_year],
        'Days to Maturity': [days_to_maturity]
    })

    preds = predict_model(final_model, data=input_df)

    return preds["prediction_label"][0]  # predicted Cut-off Yield (%)

inputs = [
    gr.Number(label="Auction Tenor (Month)", value=12, precision=0),
    gr.Number(label="Auction Amount (S$M)", value=100.0, precision=2),
    gr.Number(label="Auction Month", value=1, precision=0),
    gr.Number(label="Auction Year", value=2025, precision=0),
    gr.Number(label="Days to Maturity", value=365, precision=0),
]

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=gr.Number(label="Predicted Cut-off Yield (%)"),
    title="Cut-off Yield Prediction - Extreme Gradient Boosting",
    description="Disclaimer: This project is for educational and demonstration purposes only. It is not intended to provide investment advice or financial recommendations. The model predictions are based on historical data, and past performance does not guarantee future results. Always consult a licensed financial advisor before making investment decisions."
)

if __name__ == "__main__":
    demo.launch()