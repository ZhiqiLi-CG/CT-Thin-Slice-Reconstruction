# Proposal for CT reconstruction

## (1) background and motivation
- Under normal conditions, there are only 2D medical images such as X-ray, CT, MRI to help doctor to know the patient's condition. This may lead to some mistakes or emergency.

- Our research topic mainly focus on 3D brain tumors segmentation.
This kind of research can help a lot in clinical practice. On the one hand, it can help doctors visualize the size and location(, especially the surroudings of the tumor) in the patient's brain, which can greatly reduce the blindness of tumor cutting. On the other hand, if we permit the patient to see the 3D modeling of his/her tumor, he/she may have more trust in the operation.

- There are pictures of 3D-MRI,
- What we want to do is to first make tumor segmentation on the existing 3D-MRI dataset.

- However, we also have some more realistic ideas, which is also have something to do with 3D reconstruction.
Nowadays, even though MRI (Magnetic Resonance Imaging) is more easier to tell whether the patient has a tumor and may have low radiation for some  more advanced MRI equipment, CT is the most widely used technology because its scaning takes much less time and is much cheaper than MRI.   
To reduce financial pressure on the patients, we think maybe we can use 3D reconstruction of the brain CT to get the model just similar to 3D-MRI and then do tumor segmentation on the 3D reconstruction model. 
## (2) existing work
- Intelligent image reconstruction based on Deep Learning is a research hotspot in the field of medical imaging. This section will briefly introduce the existing work of Deep Learning in low-dose CT imaging. 
- Normal dose CT (NDCT) used routinely in clinic is the result of combined multi angle X-ray measurement, and the radiation is large. In order to reduce the potential risk to patients, low-dose CT (LDCT) imaging by reducing ray intensity or sparse angle has attracted extensive attention. However, the smaller the X-ray flux, the more serious the reconstructed CT noise, which reduces the image quality, which will inevitably affect the accuracy of diagnosis. 
- At present, in addition to the traditional LDCT imaging methods such as total variation, model-based iterative reconstruction (MBIR) and dictionary learning, the research of LDCT reconstruction based on deep learning is a research hotspot. It mainly realizes the end-to-end LDCT image denoising task through CNN model, so as to complete the restoration from LDCT to NDCT image. In the following are four existing typical research methods.
### 2.1 encoding and decoding method based on encoder and decoder CNN
### 2.2 CNN method combined with wavelet transform
### 2.3 Generative Adversarial Networks，GAN 
### 2.4 Deeping learning method based on  sinogram domain
## (3) proposed method & (4) specifics of method
- First of all，the datasets we used are LoDoPaB-CT,3D intracranial Aneurysm Dataset for Deep Learning in github and so on, and we will focuse on the method named low-dose CT and the dababase of LoDoPaB-CT is for this.
**Why we focuse on this method?**
    - analytic methods such as fltered back-projection (FBP) or iterative reconstruction (IR) are accurate but they have a disadvantage that high doses of applied radiation are potentially harmful to the patients however low dose leads to several challenges like undersampling or increased noise levels.
Then the mothod for us to solve it is using more prior 
    - knowledge.However,analytical methods are only able to use very limited prior information. The it is natural for us to resort data driven methods by deep learning.
    - DL-based approaches benefit strongly from the availability of comprehensive datasets and that's why we use the LoDoPaB-CT.
- **We are considering two framework. The first frame work we call it total deffereialble frame work.**  That is to say, the framework is end-to-end and reconstruct the 3D model or produce insterested slices based on slices input.  However, this kind of model is difficult because of the  complex neural network, which is  responsible for a lot of work.This framework we will take it as the advanced edition of our work.  **And the second framework we consider is called pipline framework**.This framework divides the reconstruction process into 2 steps.The first step is to  denoise the images obtained by low-dose CT while keeping features and the second step is to reconstruc the 3D model, depending on the result of last step.  To be the inital edition, we will only combine the first step with deep learning method and second step is solved by conventioal methods. 
- **Now we will describe the first method roughly**
This method is inspired by the paper [1].  It not neccesary for us to reconstruct the total 3D model,because not all of the parts are insterested and doctors always cared about some slices nowadays.  We only need to produce the slice of required views, based on the input slices.And this is easier for us to do.
**Then the question is how to achieve this goal?**.  We could look ar the question from another angle.  Take it as a problem of interpolation(similar but not the same).  When we consider a pixel of the required slice,  this pixel is related to some pixels of the input slices.  And we could reconstruct the pixel by those pixels, like interpolation.  However, in interpolation, we always calculate the pixel by the mean values of those pixels or calculate the pixel by a B-spline surface based on those pixels.  And in this problem, we are not supposed to adopt it, because of the complex nature. There are always continuous structure in the tissue we cared about in the area of medical diagnosis, such as vessel, nerve and muscle. All of these should be extracted as features and be presearved in the final result.  So the neural network is needed to solve this problem.  Therefore in our network，we will first use the CNN to extract parts of vessel et al. as features.

- **Then, we will give a outline of the second method**.And in the first step, we will use the following network to denoise. 
    - Low-Dose CT with a Residual Encoder-Decoder Convolutional Neural Network:
The purpose of this article is to solve the problem that it is difficult to solve keep feature whilr denoising because of the difficulty of modeling the statistical characteristics in the image domain.And he combine the autoencoder, the deconvolution network, and shortcut connections into the residual encoder-decoder convolutional neural network (RED-CNN) for low-dose CT imaging. 
    - And the second step we will construct it by the conventional method called Marching Cube. This method could be applied to medical images, since in the original paper *Marching Cubes:A High Resolution 3D Surface Construction Algorithm*, E.Lorensen et al. have mentioned how to use apply method in medical images.




## (5) proposed outcomes
Generally speaking, we hope that we will be able to use 3D reconstruction of the brain CT to get the model just similar to 3D-MRI and then do tumor segmentation on the 3D reconstruction model. In that way, we can reduce financial pressure for the patients. Furthermore, LDCT reconstruction based on deep learning will be used beforehand to improve the image quality of low-dose CT which is safer than the normal one.

## References
[1] LoDoPaB-CT, a benchmark dataset for low-dose computed tomography reconstruction