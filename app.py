from fastai.vision.all import *
import gradio as gr
import pandas as pd

# Load your trained model
learn = load_learner("deepfake_model.pkl")

# History to store results
history = []

# Prediction function
def predict_image(img):
    try:
        pil_img = PILImage.create(img)
        pred_class, pred_idx, probs = learn.predict(pil_img)
        confidence = float(probs[pred_idx])
        history.append({
            "Prediction": str(pred_class),
            "Confidence": round(confidence, 4)
        })
        df = pd.DataFrame(history)
        return str(pred_class), confidence, df
    except Exception as e:
        return "Error", 0.0, pd.DataFrame()

# Gradio app using Blocks
with gr.Blocks(title="Deepfake Detection") as app:
    # Custom header with logo and name
    gr.Markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="font-size:24pt; margin-bottom: 4px;">Deepfake Detection with Transfer Learning and FastAI</h1>
                <h2 style="font-size:18pt; margin-top: 0px;">ADS 564 - Deep Learning</h2>
                <p><b>Prepared by:</b> Şeyma Gülşen Akkuş</p>
            </div>
        </div>
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload a Face Image")
            predict_btn = gr.Button("Predict")
            clear_btn = gr.ClearButton([image_input])
        with gr.Column():
            pred_label = gr.Label(label="Predicted Class")
            conf_score = gr.Number(label="Confidence Score")

    # Prediction History Table
    results_table = gr.Dataframe(label="Prediction History", headers=["Prediction", "Confidence"], datatype=["str", "number"])

    # Connect buttons to functions
    predict_btn.click(fn=predict_image, inputs=image_input, outputs=[pred_label, conf_score, results_table])

# Run the app
if __name__ == "__main__":
    app.launch()