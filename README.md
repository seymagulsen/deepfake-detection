# Deepfake Detection with Transfer Learning and FastAI
A robust image classification project to detect manipulated media using deep learning, developed as a midterm project for the **ADS 564 - Deep Learning** course.

**Prepared by:** Şeyma Gülşen Akkuş  
**Live Demo:** [Gradio App on Hugging Face Spaces](https://huggingface.co/spaces/seyma9gulsen/deepfake_detection)  
**Blog Post:** [Detecting Deepfakes with Transfer Learning](https://medium.com/@seyma.gulsen/detecting-deepfakes-with-transfer-learning-building-a-robust-classifier-using-xception-and-fastai-b87384a753c7)

---

## Project Overview

Deepfakes — synthetic media created using artificial intelligence — are increasingly used in both creative and malicious contexts. This project presents a deep learning-based classifier to detect fake face images using:

- ✅ **Transfer Learning** with the Xception architecture
- ✅ **FastAI** for high-level training and model interpretation
- ✅ **Gradio** for interactive web deployment

The model is trained on a curated subset of the [FaceForensics++](https://github.com/ondyari/FaceForensics) dataset, distinguishing between original and manipulated (DeepFakes, FaceSwap, Face2Face, FaceShifter) frames.

---

## Model Highlights

- **Architecture**: Xception (`legacy_xception` from `timm`)
- **Training Strategy**:
  - Freeze → Train → Unfreeze → Fine-tune
  - Discriminative Learning Rates
  - Regularization: MixUp, Label Smoothing
  - Progressive Resizing
  - Test-Time Augmentation (TTA)

- **Performance (with TTA)**:
  - Accuracy: **97.06%**
  - Precision: **96.96%**
  - Recall: **97.27%**
  - F1 Score: **97.05%**

---

## Gradio App

Upload a face image and receive:
- A prediction (real or fake)
- Confidence score

👉 Try it now: [Live on Hugging Face](https://huggingface.co/spaces/seyma9gulsen/deepfake_detection)

---

## Training Pipeline Summary

- **Data Preparation**: Extracted up to 100 frames from selected videos for each class (`original` and `fake_sequences`).
- **DataBlock Setup**: Used FastAI’s modular pipeline to define image loading, augmentation, and labeling.
- **Model Training**: Fine-tuned an Xception model using cross-entropy loss and macro evaluation metrics (Precision, Recall, F1 Score).
- **Evaluation**: Assessed performance using a confusion matrix, top loss visualization, classification report, and Test-Time Augmentation (TTA).

---

## Techniques Used

| **Category**         | **Techniques**                                           |
|----------------------|----------------------------------------------------------|
| Transfer Learning     | Xception with ImageNet weights                          |
| Data Augmentation     | Flip, rotation, zoom, lighting, warp                    |
| Regularization        | MixUp, Label Smoothing                                  |
| Optimization          | Learning Rate Finder, Discriminative Learning Rates     |
| Evaluation            | TTA, Confusion Matrix, Top Losses                       |
| Deployment            | Gradio Blocks on Hugging Face                           |

---

## Future Work

- **Video-Level Detection**: Aggregate predictions across frames using temporal models.
- **Multi-Class Classification**: Identify specific types of manipulations (e.g., FaceSwap, DeepFakes).
- **Temporal Modeling**: Explore RNNs, 3D CNNs, or Transformer-based models to capture frame-to-frame inconsistencies.

---

## References

- [FaceForensics++ Dataset](https://github.com/ondyari/FaceForensics)
- [FastAI Library](https://github.com/fastai/fastai)
- [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models)
- [Gradio](https://gradio.app/)

---

## Acknowledgments

This project was completed as part of the **ADS 564 - Deep Learning** course at TED University.  
Special thanks to the instructors and peers for their guidance and feedback throughout the project.

---

## Contact

**Şeyma Gülşen Akkuş**  
📧 seyma.gulsen@tedu.edu.tr  
📝 [Medium Blog](https://medium.com/@seyma.gulsen)


