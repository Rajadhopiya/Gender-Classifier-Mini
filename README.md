
![2.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/gfvc6sCbh9saiVnczYH2c.png)

# **Gender-Classifier-Mini**

> **Gender-Classifier-Mini** is an image classification vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for a single-label classification task. It is designed to classify images based on gender using the **SiglipForImageClassification** architecture.  

```py
Accuracy: 0.9720
F1 Score: 0.9720

Classification Report:
              precision    recall  f1-score   support

    Female ♀     0.9660    0.9796    0.9727      2549
      Male ♂     0.9785    0.9641    0.9712      2451

    accuracy                         0.9720      5000
   macro avg     0.9722    0.9718    0.9720      5000
weighted avg     0.9721    0.9720    0.9720      5000
```

![Untitled.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/MNO7bk_1wr5lvfyTDnhjF.png)

The model categorizes images into two classes:
- **Class 0:** "Female ♀"
- **Class 1:** "Male ♂"

# **Run with Transformers🤗**

```python
!pip install -q transformers torch pillow gradio
```

```python
import gradio as gr
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Gender-Classifier-Mini"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def gender_classification(image):
    """Predicts gender category for an image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = {"0": "Female ♀", "1": "Male ♂"}
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=gender_classification,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Gender Classification",
    description="Upload an image to classify its gender."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
```

# **Intended Use:**  

The **Gender-Classifier-Mini** model is designed to classify images into gender categories. Potential use cases include:  

- **Demographic Analysis:** Assisting in understanding gender distribution in datasets.
- **Face Recognition Systems:** Enhancing identity verification processes.
- **Marketing & Advertising:** Personalizing content based on demographic insights.
- **Healthcare & Research:** Supporting gender-based analysis in medical imaging.
