# **DeepFake Face Swap Using Diffusion Models**

## **Overview**
This project aims to develop a **DeepFake face swap model** that leverages **diffusion models** for realistic identity transformation. The model takes a **source face** and a **target face** as input and generates an output that seamlessly replaces the source identity with the target.  

Diffusion models offer stable training and high-quality synthesis, addressing common challenges in deepfake generation while ensuring robust identity preservation.  

## **Data Acquisition**
### **Primary Dataset**
We use the **CelebA/CelebA-HQ** dataset, a large collection of aligned face images with diverse identities, ideal for training a face swap model. More details on the dataset can be found [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).  

Since the full dataset consists of **200k images**, we will use a **small subset** depending on compute resources.  

### **Pre-processing**
- Face images are **aligned, cropped, and normalized** for consistency.  
- Standard **data augmentations** such as **random flips** and **slight rotations** are applied to improve model robustness.  

## **Features & Attributes**
### **Visual Features**
- Input consists of **source and target face images**.  
- **Identity features** are extracted from a pre-trained face recognition network (e.g., **FaceNet** or **ArcFace**) to capture essential identity details.  

### **Latent Representation**
- A **latent embedding** is derived from both the source and target images to guide the diffusion process, ensuring that the generated face maintains the target’s identity characteristics.  

## **Approach & Model Architecture**
### **Data Pre-processing**
- Utilize **CelebA/CelebA-HQ** to obtain aligned and normalized face images.  
- Extract **latent representations** from the target face using a face recognition model.  

### **Model Architecture & Training**
- **Diffusion Framework:**  
  - Implement a **Denoising Diffusion Probabilistic Model (DDPM)** that gradually refines noisy images into high-quality outputs.  
- **Conditional Integration:**  
  - Condition the **reverse diffusion process** on the target identity’s latent features, allowing the model to infuse the target’s characteristics during face swap generation.  

### **Loss Functions**
- **Denoising Loss:** Trains the model to predict the noise added at each timestep.  
- **Identity Preservation Loss:** Uses **cosine similarity** between the generated face and the target identity features to ensure core identity is maintained.  
- **Reconstruction Loss (Optional):** Applied when paired training samples are available.  

## **Evaluation Metrics**
- **Fréchet Inception Distance (FID):** Measures the overall realism of the generated images.  
- **Identity Preservation Score:** Computes **cosine similarity** between facial embeddings of the generated face and the target.  
- **Structural Similarity Index (SSIM) / LPIPS:** Assesses perceptual quality and consistency compared to the input images.  

## **Conclusion**
This project leverages **modern diffusion models** and **well-established datasets** to perform high-fidelity **DeepFake face swapping**. By emphasizing both **visual realism** and **identity preservation**, the model ensures high-quality, identity-aware face synthesis, making it a strong foundation for further exploration in **DeepFake research** and **AI-driven identity transformation**.
