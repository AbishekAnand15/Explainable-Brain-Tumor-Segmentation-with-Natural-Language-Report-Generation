# Explainable Brain Tumor Segmentation with Natural Language Report Generation

XAI-Driven Brain Tumor Segmentation with Natural Language Interpretations

This project implements a complete workflow for brain tumor segmentation using deep learning, enriched with explainable AI techniques and automated natural-language clinical interpretations.

## 1. Overview  

The primary goal is to transform raw medical imaging data into actionable, transparent clinical insights.

| Step | Description | Purpose |
| :--- | :--- | :--- |
| **Input & Prep** | Take raw MRI scans and preprocess them. | Standardize data for model consumption. |
| **Segmentation** | Deep learning model identifies tumor regions. | Locate and quantify the pathology. |
| **Explanation (XAI)** | XAI techniques highlight the model's decision-making. | Ensure transparency and build clinical trust. |
| **Report Generation** | Findings are converted into clear, readable medical text. | Provide a final, useful output for clinicians. |

## 2. Methodology & Input Modalities  

Different MRI sequences (modalities) are crucial as each provides unique information about the tissue structure and pathology.



| Modality | Acronym | Information Revealed |
| :--- | :--- | :--- |
| **T1-weighted** | T1 | Basic brain anatomy (Gray/White Matter). |
| **T1 Contrast-Enhanced** | T1ce | Highlights **contrast-enhancing** (active) tumor parts. |
| **T2-weighted** | T2 | Reveals fluid-based structures and some pathologies. |
| **Fluid-Attenuated Inversion Recovery** | FLAIR | Suppresses normal fluid signal to reveal **edema** (swelling) clearly. |

**Goal:** Combining all sequences gives the model a complete, multi-dimensional understanding of the tumor’s shape, location, and internal structure.

## 3. Pre-processing Pipeline  

Before the MRI images are given to the model, they must be rigorously cleaned and standardized. This turns raw, messy MRI scans into uniform, machine-readable data.

## 1. Resampling: Ensures all scans have the same **voxel size**, preventing scale differences.
## 2. Skull-stripping: Removes non-brain tissue so the model focuses only on useful regions.
## 3. Normalization: Adjusts brightness/intensity levels (`Z-score` or `Min-Max`) so scans from different machines look similar.
## 4. Registration: Aligns all modalities (T1, T1ce, T2, FLAIR) slice-by-slice so they spatially match.
## 5. Patch Extraction: Breaks the large 3D scanned volume into smaller, manageable sections (patches) for the model to process efficiently.

## 4. Segmentation Model: The U-Net Architecture  

The core of the system is a deep convolutional neural network, typically a **U-Net** or a Transformer-based U-Net, designed for pixel-wise/voxel-wise segmentation.



* **Encoder (Contracting Path):** Observes the input image and progressively extracts important, high-level features (e.g., "is this a tumor border?", "is this fluid?").
* **Decoder (Expanding Path):** Uses the extracted features to reconstruct the tumor region at the original image resolution.
* **Skip Connections:** Directly connect corresponding layers of the encoder and decoder. This is crucial for retaining **fine details** (like sharp tumor edges) that would otherwise be lost in the compression of the encoder.

## 5. Training Strategy  

Training involves teaching the model how to recognize complex tumor patterns while ensuring it generalizes well to unseen data.

### Loss Functions
We combine loss functions to balance accuracy and consistency:
* **Dice Loss:** Helps maximize the overlap between the predicted tumor mask and the real tumor (ground truth). This is essential for highly imbalanced medical data.
* **Cross-Entropy Loss:** Improves per-pixel classification accuracy.

### Generalization Techniques
* **Data Augmentation:** Images are randomly flipped, rotated, or slightly distorted to make the model robust.
* **Learning Rate Schedules:** Adjust the model's learning speed smoothly, typically starting fast and slowing down near the end of training.
* **Mixed Precision:** Speeds up training and reduces memory usage by performing some calculations in a lower-precision format.

## 6. Inference Process  

Once trained, the model analyzes new, previously unseen MRI scans to produce the final segmentation mask.

### 1. Input Preparation: The new image is preprocessed identically to the training data.
### 2. Prediction: The model processes the data and predicts a probability value (0 to 1) for every voxel/pixel.
### 3. Thresholding: Voxel probability values above a set threshold (e.g., 0.5) are converted into the final tumor/non-tumor binary mask.
### 4. Post-processing: Small isolated errors or "speckles" are removed to clean the final mask.

### Output Tumor Regions
The final output mask distinguishes between clinically relevant tumor sub-regions:
* **Whole Tumor** (WT)
* **Tumor Core** (TC)
* **Enhancing Tumor Regions** (ET)

## 7. Evaluation Metrics  

System reliability is proven by robust evaluation metrics.

| Metric | Formula/Description | Focus |
| :--- | :--- | :--- |
| **Dice Score** | $2 \cdot \frac{|A \cap B|}{|A| + |B|}$ | Measures overlap accuracy (most common). |
| **Intersection over Union (IoU)** | $\frac{|A \cap B|}{|A \cup B|}$ | Checks matching between predicted ($A$) and real ($B$) regions. |
| **Hausdorff Distance** | Measures the maximum distance between the boundaries. | Evaluates edge sharpness and boundary quality. |
| **Sensitivity** | True Positives / (True Positives + False Negatives) | How well the system detects the tumor (recall). |
| **Volume Calculation** | Sum of predicted tumor voxels $\times$ Voxel Size | Estimates absolute tumor size for clinical tracking. |

## 8. Explainable AI (XAI)  

Deep learning models are often "black boxes." XAI is crucial to show **why** the model made a specific segmentation, aiding expert verification.



* **Grad-CAM (Gradient-weighted Class Activation Mapping):** Highlights the regions (a heatmap) that strongly influenced the model's decision for a specific class (e.g., "enhancing tumor").
* **Integrated Gradients:** Shows how much each input pixel's intensity contributes to the final prediction.
* **SHAP (SHapley Additive exPlanations):** Explains how local groups of pixels or features contribute to the final output, providing a more localized view of importance.

**Purpose:** XAI overlays help experts quickly verify that the model focused on meaningful, pathological tumor areas and not on background noise.

## 9. Natural Language Interpretation  

The final step converts the quantitative findings into a qualitative, readable report.

### 1. Extraction: Numerical metrics (Dice Score, Volume) and spatial details (e.g., "Right Frontal Lobe") are automatically extracted from the segmentation mask.
### 2. XAI Summary: The system summarizes which regions were highlighted by the XAI methods (e.g., "strong focus on T1ce contrast").
### 3. Generation: Findings are fed into a **template-based** or **transformer-based** (e.g., GPT) text generator.
### 4. Output: A short, factual, medical-style paragraph is produced, often including a measure of prediction uncertainty.

## 10. Complete Pipeline  

The full workflow operates as a transparent, end-to-end automated system:

`Load MRI Scan` $\rightarrow$ `Pre-process Data` $\rightarrow$ `Apply Segmentation Model` $\rightarrow$ `Clean Predicted Mask` $\rightarrow$ `Generate XAI Heatmaps` $\rightarrow$ `Compute Tumor Metrics` $\rightarrow$ `Extract Structured Findings` $\rightarrow$ `Generate NL Report` $\rightarrow$ **Display Results & Visuals**

## 11. Limitations  

* **Scanner Inconsistency:** Different MRI scanners/protocols produce inconsistent images, which can affect normalization and segmentation.
* **XAI Fidelity:** XAI heatmaps are an approximation and may not perfectly represent the *exact* underlying model reasoning.
* **NLG Constraints:** Natural Language Generation systems need strict constraints to avoid generating factually incorrect or misleading statements.
* **Clinical Verification:** Final results must always be treated as *support* and verified by certified medical professionals.

## 12. Example Model Output  

### Visuals
* **Image:** MRI slice with tumor region segmented and highlighted.
* **Heatmap:** Overlay showing areas most important to the model's decision.

### Example Report:
> **“A 45 mm mass is visible in the right frontal lobe with surrounding edema. XAI analysis indicates strong focus on contrast-enhancing regions (T1ce modality). The predicted Dice Score for the whole tumor is 0.89. Confidence is moderate. Clinical correlation recommended.”**

## 13. Applications  

This transparent and automated system can be used for:

* **Clinical Decision Support (CDS):** Providing fast, quantitative measurements to assist radiologists.
* **Brain Tumor Research:** Enabling large-scale analysis and comparison of tumor characteristics.
* **Automated Reporting Pipelines:** Reducing the manual burden of creating initial pathology reports.
* **Model Verification:** Offering a clear method to check model fairness and consistency.
