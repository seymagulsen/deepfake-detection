from fastai.vision.all import *
import gradio as gr

# Load your trained model
learn = load_learner("deepfake_detection.pkl")

# Define the prediction function
def predict_image(img):
    pred_class, pred_idx, probs = learn.predict(img)
    return {
        "Predicted Class": str(pred_class),
        "Confidence Score": float(probs[pred_idx])
    }

# Create the Gradio interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=[
        gr.Label(label="Predicted Class"),
        gr.Number(label="Confidence Score")
    ],
    title="Deepfake Detection App",
    description="Upload a face image to check if it’s manipulated or original."
)

# Run the app
if __name__ == "__main__":
    interface.launch()
    
# This code creates a simple web app using Gradio to classify images as deepfake or real.
# The model is loaded from a file named "deepfake_detection.pkl".