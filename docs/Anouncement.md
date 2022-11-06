# Anouncement
Last week,we have made a mistake that the low-dose dataset we found is not for 3D reconstruction, and it is for denoising, namely 2D reconstructoin. And this week after read a paper about reconstruction of thin slices, we began to realise a question that **3D reconstruction of CT slice is related to 2D enhancement**. 

Last week, we have said that our goal is 3D reconstruction, and this week we have determined which kind of 3D reconstruction we will do and how could we do that.See as below.

## Our goal
- **Our goal is to reconstruct several thin slices based on the low-dosed thick slice, and the idea is from the article:**
*Stereo-Correlation and Noise-Distribution Aware ResVoxGAN for Dense Slices Reconstruction and Noise Reduction in Thick Low-Dose CT*
- **You may have a question: What is thick or thin slice?**
This question is related to the principles of CT.  Refer to the picture below.  Each pixel of the CT slice represent a voxel of the detected object, and each pixel is the mean value of the voxel on the dimension of z.  And if the voxel is thick, more information will be compressed into one pixel.  This is similar to this case, if a resolution of a pixture is low then a pixel is actrually mean value of a large space and the image is blurred.  That is to say, if a slice is  thick, the information it contains is less precise. 
![](https://codimd.s3.shivering-isles.com/demo/uploads/upload_5eaa0dda84855b1fca58aebb2b8ccb06.JPG)
- **Why we call it 3D reconstruction?**
This time we should make clear the conception of 3D reconstruction.  3D reconstruction is to restore the depth dimension, namely z dimension of the target object, based on several 2D images.  Then, we should note that the basic idea of 3D reconstrution is the restoration of the depth dimension. So back to our topic, we could find that our goul which is to reconstruct thin slice is about this idea indeed, that is we are restore the depth dimension and increase the resolution of the depth dimention.
- **Why this goal is important?**
**Remember that one of our goal is low-dose**
(1)Thick slices thichness is widely set in clinical imaging. It is characterized with the advantages of higher dose efficiency more reduced storage, faster iterative reconstruction speed and more clear imaging quality than the thin one
(2)Thick slices and Low-dose acquired CT image directly decrease the x-ray radiation flux mainly from lower operating current, voltage or exposure time, to reduces the risk of inducing genetic, cancerous, and disease conditions
(3)However,  thick LDCT that the sparse slices and low spatial resolution poorly demonstrate the coronal/sagittal anatomy , and the low-dose acquisition inherently compromises the signal-to-noise ratio. Then thin slice thickness is required.
- **How could we accomplish this goal?**
Firstly, we want to learn some methods from the thin-slice reconstrucion itself.  However, researches related to this topic are seemly insufficient and we would like to adopt some methods of 2D image  enhancement such as U-net,GAN and so on.
But why we chould use the methods of 2D image  enhancement? Because the intrinsic goal of this problems have much in common. They both aims at increase the resolution of the images, and 2D image enhancement is on x-y dimension and thin-slie reconstruction is on z dimension.
- What is our dataset?
(1)We will adopt the dataset which is adopted by *Stereo-Correlation and Noise-Distribution Aware ResVoxGAN for Dense Slices Reconstruction and Noise Reduction in Thick Low-Dose CT*. We have the same goal with this article and its success ensure the reliability of this dataset.
And this dataset can be downloaded from:https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758026
(2)The feature of the dataset is as follow.
![](https://codimd.s3.shivering-isles.com/demo/uploads/upload_ec988e1c3b8fb09bb80e3e285a7c513c.png)

# Premilinary
## 1.Self-Attention
https://zhuanlan.zhihu.com/p/130509584

# Methods
## 1.ResVoxGAN
This method is truly for thin-slice reconstruction. 
![](https://codimd.s3.shivering-isles.com/demo/uploads/upload_e945f5465c8f024131f7a0bd99b671ca.JPG)

ResVoxGAN promotes dense slice reconstruction and noise reduction for thick
LDCT, via 
- (1) MSVGenerator of consecutive 3D multi-scale residual blocks and Subpixel Convnet for fine-granted stereo feature extraction and latent spatial structure reconstruction;
- (2) Stereo-correlation & Image-expression Constraints for structural detail and scene content
- (3) Coupled discriminators for realistic anatomic structure distribution and valid noise-reduction distribution.

This article we only read it roughly and we will study it further.

-----------------------------------
And this week we also have read some papers about multiple work of deep learning methods in denoising and find that the the frontier of the denoising field, that is transformed network.  

The main problem is denoising problem is a ill-posed problem, so preserving features while denoising is hard and more information need to solve this problem, and therefore deep-learning is applied.  And then the question is how could we  enhancement imaged based on this information and there are several deep learning methods in denoising about this. 
## 2.Convolutional Encoder-Decoder Networks with Symmetric Skip Connections
### 2.1 feature
- Convoluation layer and deconvolution layer as encodes and decodes.
- skip connections to help restore details
### 2.2 contributions
To achieve better performance , this article propose **a very deep fully convolutional encoding-decoding framework for image restoration**.  The network is composed of multiple layers of convolution and deconvolution operators.  And **the convolution layer** act as a feature extractor, which capture the abstraction of image contents while eliminating noisies/corruptions and **deconvolutional layers** are then used to recover the image details.  However, very deep networks are hard to train, because of the problem of gradient vanishing,  and **skip-layer connections have been proposed**, which pass image details from convolutional layers to deconvolutional layers and is beneficial for restoration.  And **it can be used to handle different levels of noises using a single model.  Also this network is end-to-end**.

The **skip-layer connections** are very novel.  They can help to back-propagate the gradients to bottom layers and pass image details to top layers, making training of the end-to-end mapping easier and more effective, and thus achieving performance improvement while the network going deeper. ![](https://codimd.s3.shivering-isles.com/demo/uploads/upload_e119e88ca39d51c01d5bdfa9301c7b5e.JPG)

### 2.3 problem
## 3.RED-CNN
![](https://codimd.s3.shivering-isles.com/demo/uploads/upload_38d86f287922431363ee4b83819b23ab.JPG)

### 3.1 feature
### 3.2 contributions
### 3.3 problem
## 4. Transformer
**This moethods is novel and it is the first time to apply it into medical image denoising in 2021 and we would like to focus on it.**
This part we will focus on the basic transformer method and the next parts we will focus on the Eformer methods and Uformer methods.
- **What is transformer?**:
## 5.Eformer
Eformer: Edge Enhancement based Transformer for Medical Image Denoising
![](https://codimd.s3.shivering-isles.com/demo/uploads/upload_043c2678375ad22de75794d4384e085d.png)
### 5.1 Edge Enhancement part:Sobel-Feldman Operator
- Sobel Filter is specifically used in edge detection algorithms as it helps in emphasizing on the edges.Originally the operator had two variations - vertical and horizontal, but this particle also include diagonal versions similar to 
*Tengfei Liang, Yi Jin, Yidong Li, and Tao Wang. Edcnn: Edge enhancement-based densely connected network with compound loss for low-dose ct denoising. 2020 15th IEEE International Conference on Signal Processing (ICSP), Dec 2020.*
-  The set of image feature maps containing edge information are efficiently concatenated with the input projection and other parts of the network.(Which refer to the output of Sobel Convolution block in the above figure)


## 6.GAN-based

# Our Hesitation
Since we have read a lot of papers about image  enhancement, especially denoising, **we are confused whether we should change our topic from 3D reconstruction to image enhancement**.  Beacause dataset for image enhancement is more abundant and there are many mature methods we could use directly.  However, because the methods of the image enhancement are mature and  kind of complete, it is hard for us to innovate and the meaning of some modification of image enhancement methods is less valuable.And for 3D reconstruction, it is easier for us to innovate, we think and the reference about it may be less.
**Therefore we would like to ask for your help**