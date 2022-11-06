# Some Related Techniques
### 1.ResCNN
https://zhuanlan.zhihu.com/p/31852747
feature map:https://www.cnblogs.com/yh-blog/p/10052915.html
### 2.Trick to train network
https://zhuanlan.zhihu.com/p/110278004
### 3.pixel shuffling
https://blog.csdn.net/g11d111/article/details/82855946
https://blog.csdn.net/u014636245/article/details/98071626
#### 3.1 periodical shuffling
### 4.Super resolution
https://blog.csdn.net/sinat_39372048/article/details/81628945
This field is related to our topic closely, and following papers, I think, are required.
- *Enhanced Deep Residual Networks for Single Image Super-Resolution*https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf
Downlad here:

# Our recent plan
- This week, we have determined our topic as thin slices reconstruction based on thick CT slice.  We choose this topic for three reasons.  The first reason is that few researches have done about it and then it is easier to get some achievement.  The second reason is that this topic is meaningful, according to out last proposal.  The third reasion is that several method we could refer to, for this topic is close to field of super-resolution of graphics and video, and it seems that video method could be applied directively and we could develop them to apply into CT slice better.
- We have found that there is a successful application of super-resolution method to CT slice reconstruction, and we have recorded it in the following part.
- After reading several papers, from this week, we begin to get in familiar with Pytorch and read some code of these papers and execute these codes on Colab to learn how to write codes to implement novel idea. 
# Our schedule
## 1.Monday
### 1.1Fang
### 1.2Li
Today I have learned the concept of ResCNN, pixel shuffling, super resolution.

**An idea**: In 2D image reconstruction, edge-detection method could be used to enhance the network.  Let us consider this phenomena,why we use detect the edge? that's because edge has inconsistant pixels.  Then in the field of dense reconstruction, waht is similar to the edge? that is the pixel that has largely different color with the pixel at the same place of its neighbor slice.  I think this is one direction for us to modify others' network.

**An inspiration**: Similar to 1.2.1 and 1.2.2, we could find some super-resolution method and modify it to out topic.  Furtermore,we could modify more than one model and compare there performance.
#### 1.2.1 Enhanced Deep Residual Networks for Single Image Super-Resolution
https://github.com/LimBee/NTIRE2017 this is its source code by lua.
This paper is the preliminary of the 1.2.2 paper and several parts of the network below is based on this paper.
##### 1.2.1.1 Contribution
Two models are proposed, the first is enhanced deep super-resolution network (EDSR) and second is multi-scale
deep super-resolution system (MDSR) which can reconstruct high-resolution images of different upscaling factors in a single model.
- By removing unnecessary modules from convetional ResNet architecture, it achieve improved results while making the model compact.
- Employ residual scaling techniques to stably train large models.
- It develop a multi-scale super-resolution network to reduce the model size and training time and take the advantage of **inter-scale correlation**(this should be noticed for we might use).Along with it, a training method is found
##### 1.2.1.2 Details
###### 1.2.1.2.1 Residual blocks
![](https://codimd.s3.shivering-isles.com/demo/uploads/upload_9f74b1e3c39bfce700435d913aa34642.JPG)
**BN is removed, why?**
this is the inspiration from * S. Nah, T. H. Kim, and K. M. Lee. Deep multi-scale convolutional neural network for dynamic scene deblurring.*
-  Since batch normalization layers normalize the features, they get rid of range flexibility from networks by normalizing the features, it is better to remove them
-  GPU memory usage is also sufficiently reduced.  Consequently, we can build up a larger model that has better performance than conventional ResNet structure under limited computational resources.
###### 1.2.1.2.2 Single-scale model
- First, increase the number of parameters can exnhance the performance.  Consider CNN, there are two factor related to the number of parameters,depth (the number of layers) B and width (the number of feature channels) F, which occupies roughly O(BF) memory with O(BF^2) parameters.  That is to say, we prefer to increase the number of F.
- However, increasing the number of feature maps above a certain level would make the training procedure numerically unstable.  This paper solve this problem by adopting the residual scaling with factor 0.1, and constant scaling layers are placed after the last convolution layers.
![](https://codimd.s3.shivering-isles.com/demo/uploads/upload_c527c98c27177a1e4b2fc2c43e84d958.JPG)
this layer can be integrated into the previous convolution layer for the computational effificiency

*C. Szegedy, S. Ioffe, V. Vanhoucke, and A. Alemi. Inceptionv4, inception-resnet and the impact of residual connections on learning.* 
###### 1.2.1.2.3 Training method
It investigate the model training method that transfers knowledge from a model trained at other scales.To utilize scale-independent information during training, it train high-scale models from pre-trained low-scale models, for example, when training the model for upsampling factor ×3 and ×4, it initialize the model parameters with pre-trained ×2 network.
##### 1.2.1.3 Models
###### 1.2.1.3.1 EDSR
![](https://codimd.s3.shivering-isles.com/demo/uploads/upload_bcaa21eaa27be9fbe0e8d3a5b3daa438.png)
B = 32, F = 256 with a scaling factor 0.1
###### 1.2.1.3.2 Multi-scale mode
![](https://codimd.s3.shivering-isles.com/demo/uploads/upload_355a6ff9bdbf2408caf90585ac69652e.png)
Consider it as an iter-related task, similar to VDSR, * J. Kim, J. Kwon Lee, and K. M. Lee. Accurate image super-resolution using very deep convolutional networks*
-  Each of pre-processing module consists of two residual blocks with 5 × 5 kernels.  By adopting larger kernels for pre-processing modules, the scale-specific part can be kept shallow
-   At the end of the multi-scale model, scale-specific upsampling modules are located in parallel to handle multi-scale reconstruction
B = 80 and F = 64 ,for 3 different scales have about 1.5M parameters each, totaling 4.5M.

#### 1.2.2 Residual CNN-based Image Super-Resolution for CT Slice Thickness Reduction using Paired CT Scans :Preliminary Clinical Validation
In this paper, two main points should be focused on, the  first is the effective SR method for CT slice thickness reduction using CNN poposed and the second is the statement that the importance of using real data rather than image generated by simulation.
**This paper I will read in details tommorrow**
### 1.3Shi
### 1.4Xu
## 2.Tuesday
### 2.1Fang
### 2.2Li
**Idea:** Actually, our topic could be considered as the normal super-resolution problem.  Conventional super-resolution problem is to solve the problem of down-pooled graphics on the square grid and our topics is to solve the problem of down-pooled graphics on columns.
#### 2.2.1 Residual CNN-based Image Super-Resolution for CT Slice Thickness Reduction using Paired CT Scans :Preliminary Clinical Validation
![](https://codimd.s3.shivering-isles.com/demo/uploads/upload_c705d0396fd71a2e0d02c1aaf421c3d4.png)

**The main idea here is to solve the graphics on the cross section by super-resolution methods.**
#### 2.2.2 
### 2.3Shi
### 2.4Xu
## 3.Wednesday
### 3.1Fang
### 3.2Li
A review of image super-resolution, including network designs, learing principles, loss funciotns and different components.
*Refer to Deep Learning for Image Super-resolution: A Survey*
#### 3.2.1 evaluation metrics
Image quality assessment (IQA) methods include subjective methods based on humans’ perception (i.e., how realistic the image looks) and objective computational methods.  And  these methods aren’t necessarily consistent between each other.  The objective IQAmethods have three kinds, full-reference methods, reduced-reference ethos based on comparisons of extracted features and no-reference methods(for example, learning methods,that train a neural network to tell the quality of image). The subjective methos is not related to our topic and thus we will not introoduce it here.
##### Peak Signal-to-Noise Ratio
$$
\operatorname{PSNR}=10 \cdot \log _{10}\left(\frac{L^{2}}{\frac{1}{N} \sum_{i=1}^{N}(I(i)-\hat{I}(i))^{2}}\right)
$$
 Since the PSNR is only related to the pixel-level MSE, only caring about the differences between corresponding pixels instead of visual perception, it often leads to 
 
 poor performance in representing the reconstruction quality in real scenes.
##### Structural Similarity
This is called structural similarity index (SSIM)
luminance:$\mathcal{C}_{l}(I, \hat{I})=\frac{2 \mu_{I} \mu_{\hat{I}}+C_{1}}{\mu_{I}^{2}+\mu_{\hat{I}}^{2}+C_{1}}$
contrast:$\mathcal{C}_{c}(I, \hat{I})=\frac{2 \sigma_{I} \sigma_{\hat{I}}+C_{2}}{\sigma_{I}^{2}+\sigma_{\hat{I}}^{2}+C_{2}}$
structural similarity:
$$
\begin{aligned}
\sigma_{I \hat{I}} &=\frac{1}{N-1} \sum_{i=1}^{N}\left(I(i)-\mu_{I}\right)\left(\hat{I}(i)-\mu_{\hat{I}}\right) \\
\mathcal{C}_{s}(I, \hat{I}) &=\frac{\sigma_{I \hat{I}}+C_{3}}{\sigma_{I} \sigma_{\hat{I}}+C_{3}}
\end{aligned}
$$
Then the SSIM is given by:
$$
\operatorname{SSIM}(I, \hat{I})=\left[\mathcal{C}_{l}(I, \hat{I})\right]^{\alpha}\left[\mathcal{C}_{c}(I, \hat{I})\right]^{\beta}\left[\mathcal{C}_{s}(I, \hat{I})\right]^{\gamma}
$$
where $\alpha, \beta, \gamma$ are control parameters for adjusting the relative importance.
#### Task-based Evaluation
Evaluate the pictures based on different performance of models fed by them,for other topics.
#### Other IQA Methods
 The multi-scale structural similarity
(MS-SSIM) supplies more flexibility than single-scale SSIM in incorporating the variations of viewing conditions
#### 3.2.2 super-resolution frameworks
![](https://codimd.s3.shivering-isles.com/demo/uploads/upload_c4fdf29d4f0dbbe12bdcd3f0768ea418.png)
![](https://codimd.s3.shivering-isles.com/demo/uploads/upload_f2b85969a1df8d208197492d60d5dda4.png)
- Pre-upsampling Super-resolution: utilizing traditional upsampling algorithms to obtain higher resolution images and then refining them using deep neural networks is a straightforward solution.make
- Post-upsampling Super-resolution: it is aim to  improve the computational ef and mak full use of deep learning technologyowever,since most operations are performed in high-dimensional space, the cost of time and space is much higher than other frameworks.
- Progressive Upsampling Super-resolution
-  Iterative Up-and-down Sampling Super-resoluionxity ar
#### 3.2.2 main components
##### Interpolation-based Upsampling
##### Learning-based Upsampling
- Learning-based Upsampling
- Sub-pixel Layer
![](https://codimd.s3.shivering-isles.com/demo/uploads/upload_e04fe7caaa33cc506e419b43865b5728.png)


#### 3.2.3 networks
- Residual Learning
-  Local multi-path learning
-  ...
### 3.3Shi
### 3.4Xu
## 4.Thursday
### 4.1Fang
### 4.2Li
**We could consider the problem that how to preserve the feature at the same layer.**
This method could be refered to:*Recurrent back-projection network for video super-resolution*
** Very similar to our work! video super-resolution. ** 
Video super-resolution need to capture two kind of dependency, that is the spatial dependency in the same frame and the temporal depenency in different frames.This is very similar to our topic, for different slices have dependency of its adjant slices, which is similar to the temporal dependency in different frames of video.
Now here is some information about video super-resolution.
#### video super-resolution
For video super-resolution, multiple frames provide much
more scene information, and there are not only intra-frame
spatial dependency but also inter-frame temporal dependency. There are several kinds of methods, including explicit motion compensation, optical flow-based methods, recurrent methods and so on.  For motion compensation and optical flow-based methods can not be applied into our topic, we will focus on the recurrent methods
recurrent methos capture the spatial-temporal dependency without explicit motion compensation 
- *Deep video super-resolution network using dynamic upsampling filters without explicit motion compensation*
This paper generates dynamic upsampling filters and the HR residual image based on the local spatio-temporal neighborhoods of each pixel, and also avoid explicit motion compensation
- *Fast: A framework to accelerate super-resolution processing on compressed videos*
This paper exploits compact descriptions of the structure and pixel correlations extracted by compression algorithms, transfers the SR results from one frame to adjacent frames, and much accelerates the state-of-the-art SR algorithms with little performance loss
- *Frame-recurrent video super-resolution*
This paper uses previously inferred HR estimates to reconstruct the subsequent HR frames by two deep CNNs in a recurrent manner.
- *Recurrent back-projection network for video super-resolution*
This paper extracts spatial and temporal contexts by a recurrent encoder-decoder, and combines them with an iterative refinement framework based on the back-projection mechanism
- *Fast spatio-temporal residual network for video super-resolution*
This paper employs two much smaller 3D convolution filters to replace the original large filter, and thus enhances the performance through deeper CNNs while maintaining low computational cost.

Comment: Since there are more inforation supposed to considered in the video super-resolution,like motions, brightness and color change, which we will not consider. So some related method could be simplify to suit our topic and maybe we will adopt this idea. 
### 4.3Shi
### 4.4Xu
