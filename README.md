
# ** Text Detection System **

**Contributors:**  
- Harsh Kumar (220428)  
- **Manisha Kaushal (220623)**  
- Manvi Verma (220631)

---

## **Abstract**

Digitizing physical documents is often hindered by **motion blur**, **defocus blur**, and **perspective distortions**, which reduce OCR accuracy.  
This project introduces **Document Quality Restoration**, a multi-stage adaptive pipeline that:

- Detects blur using a **fine-tuned MobileNetV2 classifier**.  
- Applies **adaptive sharpening** only when blur is detected.  
- Performs **perspective correction** using a four-point transform.  
- Extracts and reconstructs clean text using **EasyOCR**.

This intelligent pre-processing system enhances OCR performance on real-world, camera-captured documents.

---

## **Key Features**

- **Adaptive Blur Classification** using MobileNetV2  
- **Conditional Sharpening** for degraded inputs  
- **Perspective Correction** for geometric distortions  
- **Contrast Enhancement** with CLAHE  
- **Accurate Text Extraction** using EasyOCR  
- **Automatic Text Assembly** in reading order  

---

## **Pipeline Overview**

1. **Document Quality Classification**
   - Uses fine-tuned **MobileNetV2** to classify images as _clean_ or _blurred_.
   - Achieved **87% validation accuracy**.

2. **Adaptive Sharpening**
   - Applies a sharpening kernel only to blurred images:

     ```python
     [[ 0, -1,  0],
      [-1,  5, -1],
      [ 0, -1,  0]]
     ```

3. **Perspective Correction**
   - Detects document boundaries using Canny edges and contour approximation.
   - Applies a **four-point perspective transform** and **CLAHE** for contrast adjustment.

4. **Text Recognition**
   - Utilizes **EasyOCR (CRNN + CTC loss)** for text extraction.

5. **Text Assembly**
   - Sorts text blocks by position and combines them into a coherent paragraph.

---

## **Results**

| Metric | Blur | Clean | Overall |
|:-------|:-----|:------|:--------|
| Precision | 0.90 | 0.80 | — |
| Recall | 0.90 | 0.80 | — |
| F1-Score | 0.90 | 0.80 | — |
| **Accuracy** | — | — | **87%** |

**High precision and recall** for blurred document detection  
**Improved OCR clarity** after adaptive pre-processing  

---

## ** Implementation Details**

- **Frameworks:** TensorFlow/Keras, OpenCV, EasyOCR  
- **Dataset:** [Blur Dataset (Kaggle)](https://www.kaggle.com/datasets/balabaskar/blur-dataset)  
- **Training Environment:** Kaggle Notebook  
- **Language:** Python  
- **Model:** MobileNetV2 (pre-trained on ImageNet)

---

## **Example Output**

| Stage | Description | Image |
|:------|:-------------|:------|
| Input | Skewed & blurred document | 🖼 Original Image |
| Corrected | Flattened & contrast-enhanced | 🖼 Processed Image |
| OCR Result | Accurate, readable text | 🖹 Extracted Text |

# Run the Jupyter Notebook
jupyter notebook text-detection.ipynb
