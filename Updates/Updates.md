# Pending Updates

## 02/07/24

[I2U-Net: A dual-path U-Net with rich information interaction for medical image segmentation](https://www.sciencedirect.com/science/article/pii/S136184152400166X)

可以用来改进teacher-student

#### CLIP

在医学图像分析中，CLIP（Contrastive Language–Image Pre-Training）是OpenAI提出的一种多模态模型，能够处理文本和图像的联合理解。它通过同时训练文本和图像的表示，使其在共同的特征空间中具有相似的表示。这种特性使得CLIP在医学图像分析中的应用具有以下几个主要优势：

跨模态检索：通过CLIP，用户可以使用自然语言描述来检索相关的医学图像。例如，医生可以输入一个描述特定病症的文本，CLIP可以找到对应的医学影像。

标注效率：医学图像数据的标注通常需要大量的专业知识和时间。CLIP通过对图像和文本的联合表示，可以减少标注工作量，甚至可以在无标注数据的情况下进行初步的分析和分类。

多模态诊断：CLIP能够结合图像和文本信息，为复杂的医学诊断提供更丰富的参考。例如，结合病人的病历文本和影像资料，CLIP可以辅助医生做出更准确的诊断。

知识迁移：CLIP在大规模通用数据上预训练，可以将其获取的知识迁移到医学领域，提升模型在医学图像分析任务上的表现，即使医学数据集较小。

具体来说，CLIP的训练过程包括：

文本编码器：使用一个Transformer架构来处理文本数据，将文本转换为特征向量。

图像编码器：使用一个CNN（如ResNet）来处理图像数据，将图像转换为特征向量。

对比学习：通过一个对比学习任务，使同一对图像和文本的特征向量尽可能接近，而不同对的特征向量尽可能远离。

这种方法在多个领域展示了其强大的通用性和灵活性，在医学图像分析中，CLIP可以被用来辅助诊断、自动标注、信息检索等多个应用场景。

```python

1. 安装必要的库

2. 数据加载和预处理

3. 定义UNet模型

4. 使用CLIP进行推理增强

import clip
from PIL import Image
import numpy as np

# 加载CLIP模型
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# 定义辅助函数
def select_roi_with_clip(image, text_prompt):
    image_preprocessed = preprocess(image).unsqueeze(0).to(device)
    text_inputs = clip.tokenize([text_prompt]).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_preprocessed)
        text_features = clip_model.encode_text(text_inputs)
        similarity = (image_features @ text_features.T).softmax(dim=-1)
    return similarity

# 加载并预处理示例图像
clip_image = Image.open("path_to_your_image.jpg").convert("RGB")
text_prompt = "a photo of a brain with a tumor"

# 获取CLIP相似度
similarity = select_roi_with_clip(clip_image, text_prompt)
print(f"Similarity with the prompt: {similarity.item():.4f}")

5. 使用UNet进行推理

from monai.inferers import sliding_window_inference

model.eval()
with torch.no_grad():
    output = sliding_window_inference(image.unsqueeze(0).to(device), (96, 96, 96), 4, model)
segmentation = torch.argmax(output, dim=1).cpu().numpy()

# 显示结果
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(clip_image)

plt.subplot(1, 2, 2)
plt.title("Segmented Image")
plt.imshow(segmentation[0, 0], cmap="gray")
plt.show()

```


## 30/06/24

[Universal and extensible language-vision models for organ segmentation and tumor detection from abdominal computed tomography](https://www.sciencedirect.com/science/article/pii/S1361841524001518)

## 27/06/2024

[Universal and extensible language-vision models for organ segmentation and tumor detection from abdominal computed tomography](https://www.sciencedirect.com/science/article/abs/pii/S1361841524001518)

[Laplace neural operator for solving differential equations](https://www.nature.com/articles/s42256-024-00844-4#:~:text=The%20LNO%20is%20capable%20of,solving%20several%20ODEs%20and%20PDEs.)

## 24/06/2024

- [ ]  [A collective AI via lifelong learning and sharing at the edge](https://www.nature.com/articles/s42256-024-00800-2) `NatureMI`

- [ ] [Reconciling privacy and accuracy in AI for medical imaging](https://www.nature.com/articles/s42256-024-00858-y) `NatureMI`

- [ ] [A Bayesian network for simultaneous keyframe and landmark detection in ultrasonic cine](https://www.sciencedirect.com/science/article/pii/S1361841524001531) `MedIA` 

Bayesian network

`GaussianNB` 是一个用于分类任务的高斯朴素贝叶斯分类器，它是朴素贝叶斯（Naive Bayes）算法的一个具体实现。朴素贝叶斯算法基于贝叶斯定理，并假设特征之间相互独立。`GaussianNB` 假设每个特征的值服从高斯（正态）分布。

#### 高斯朴素贝叶斯分类器的原理 （最简单的）

#### (1) 朴素贝叶斯定理
对于一个给定的输入样本 $\mathbf{x} = (x_1, x_2, \ldots, x_n)$，朴素贝叶斯分类器根据贝叶斯定理计算后验概率 $P(C_k | \mathbf{x})$，其中 $C_k$ 是类别标签。贝叶斯定理表示为：
$$P(C_k | \mathbf{x}) = \frac{P(\mathbf{x} | C_k) \cdot P(C_k)}{P(\mathbf{x})}$$

由于对于所有类别 $C_k$ 的 $P(\mathbf{x})$ 是相同的，因此分类时只需比较分子部分 $P(\mathbf{x} | C_k) \cdot P(C_k)$。

#### (2) 独立性假设
朴素贝叶斯分类器假设特征 \( x_i \) 在给定类别 \( C_k \) 的条件下是相互独立的，因此：
$$P(\mathbf{x} | C_k) = \prod_{i=1}^{n} P(x_i | C_k)$$

#### (3) 高斯分布假设
对于高斯朴素贝叶斯分类器，假设每个特征 \( x_i \) 在给定类别 \( C_k \) 的条件下服从高斯分布：
$$P(x_i | C_k) = \frac{1}{\sqrt{2\pi \sigma_{C_k, i}^2}} \exp\left(-\frac{(x_i - \mu_{C_k, i})^2}{2\sigma_{C_k, i}^2}\right)$$

其中 $\mu_{C_k, i}$ 和 $\sigma_{C_k, i}$ 分别是类别 $C_k$ 的特征 $x_i$ 的均值和标准差。

#### 示例代码

下面是一个简单的例子，展示如何使用 `GaussianNB` 进行分类：

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.datasets import mnist

# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train = x_train.reshape((x_train.shape[0], -1)) / 255.0
x_test = x_test.reshape((x_test.shape[0], -1)) / 255.0

# 使用PCA进行降维
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

# 训练高斯朴素贝叶斯分类器
clf = GaussianNB()
clf.fit(x_train_pca, y_train)

# 进行预测
y_pred = clf.predict(x_test_pca)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
```

`GaussianNB` 是朴素贝叶斯分类器的一种实现，假设每个特征在各类别条件下服从高斯分布。通过这种假设，`GaussianNB` 可以在处理高维数据时保持计算上的高效性，并且在许多实际应用中表现良好。尽管对于像MNIST这样复杂的图像数据集，`GaussianNB` 可能不是最优的选择，但通过特征提取和降维等预处理步骤，仍然可以用来进行基本的分类任务。


**论文步骤**

检测一段超声影像中的心脏关键帧和心脏标志点，具体步骤如下：

编码器-解码器提取特征：
对每一帧影像进行编码，提取出高维特征向量。
解码这些特征向量，生成热图，用于粗定位标志点。

自适应贝叶斯超图细化标志点位置：
利用超图结构捕捉标志点之间的复杂空间关系，对初步定位的标志点进行细化调整。

双向门控循环单元捕捉时间信息：
将影像序列输入Bi-GRU，捕捉时间上的前后依赖关系。
结合时间信息和标志点的运动状态，检测关键帧。

多任务学习框架：
通过共享编码器提取的特征，同时进行关键帧检测和标志点检测。
利用关键帧检测结果优化标志点检测，反之亦然。
这种方法通过复杂的图结构和结合时间与空间信息，实现了对超声影像序列中关键帧和标志点的准确检测。


**贝叶斯网络**：适用于建模复杂依赖关系和不确定性的系统，能够处理多维和复杂数据，但计算复杂。

**贝叶斯网络示例**：假设我们有一个医疗诊断系统，贝叶斯网络可以建模多个症状和疾病之间的复杂关系。例如，一个节点表示“发热”，另一个节点表示“咳嗽”，还有一个节点表示“流感”。边表示这些症状和疾病之间的条件依赖关系。通过贝叶斯网络，可以进行推理，如在已知某些症状的情况下推断患某种疾病的概率。

**高斯朴素贝叶斯**：适用于简单分类任务，计算简单，适合处理特征相对独立的高维数据。

**高斯朴素贝叶斯示例**：高斯朴素贝叶斯用于图像分类，例如MNIST数据集的数字分类。每个像素点作为一个特征，假设每个特征在给定类别的条件下独立且服从高斯分布。通过计算每个类别的先验概率和每个特征的条件概率，可以快速进行分类决策。






## 23/06/2024

[Assessing the arrhythmogenic propensity of fibrotic substrate using digital twins to inform a mechanisms-based atrial fibrillation ablation strategy](https://www.nature.com/articles/s44161-024-00489-x)

## 20/06/2024

CVPR 2024 best paper

- [x] [Generative Image Dynamics](https://generative-dynamics.github.io/)

- [x] [Rich Human Feedback for Text-to-Image Generation](https://arxiv.org/html/2312.10240v1)

CVPR 2024 best student paper

- [ ] [BioCLIP: A Vision Foundation Model for the Tree of Life](https://imageomics.github.io/bioclip/)

- [ ] [Mip-Splatting: Alias-free 3D Gaussian Splatting](https://github.com/autonomousvision/mip-splatting?tab=readme-ov-file)


## 19/06/2024

Survey

- [ ] [Foundation Models for Biomedical Image Segmentation: A Survey](https://arxiv.org/abs/2401.07654)

- [ ] [Foundational Models in Medical Imaging: A Comprehensive Survey and Future Vision](https://arxiv.org/abs/2310.18689) - [github](https://github.com/xmindflow/Awesome-Foundation-Models-in-Medical-Imaging)

- [ ] [A Comprehensive Survey of Foundation Models in Medicine](https://arxiv.org/abs/2406.10729)

## 18/06/2024

medical image analysis content


## 17/06/2024
MedIA

- [ ] [A review of uncertainty quantification in medical image analysis: Probabilistic and non-probabilistic methods](https://www.sciencedirect.com/science/article/pii/S1361841524001488)

## 16/06/2024

IEEE-TMI

- [ ] [A Dual Enrichment Synergistic Strategy to Handle Data Heterogeneity for Domain Incremental Cardiac Segmentation](https://ieeexplore.ieee.org/document/10433413)

- [ ] [Synthetic Optical Coherence Tomography Angiographs for Detailed Retinal Vessel Segmentation Without Human Annotations](https://ieeexplore.ieee.org/document/10400503)

## 15/06/2024

- [ ] [3D Vascular Segmentation Supervised by 2D Annotation of Maximum Intensity Projection](https://ieeexplore.ieee.org/document/10423041)

IEEE-TMI 

Confident Learning and Uncertainty Estimation

MRA CTA Vascular Segmentation

- [ ] [Prediction of diagnosis and diastolic filling pressure by AI-enhanced cardiac MRI: a modelling study of hospital data](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(24)00063-3/fulltext)

The Lancet Digital Health 

### U-Net Family for medical image segmentation

#### Category 1: Improvement of backbone by introducing new structures in the encoder or decoder

1. U-Net
2. Residual U-Net
3. V-Net
4. 3D U-Net
5. H-DenseUNet
6. GP-Unet
7. nnU-Net
8. SUNet
9. MDU-Net
10. DUNet
11. RA-UNet
12. Dense Multi-path U-Net
13. Stacked Dense U-Nets
14. Prostate Segmentation U-net
15. LADDERNET
16. Glioma Segmentation with Cascaded Unet
17. Attention U-Net
18. R2U-Net
19. Squeeze & Excitation U-Net
20. AnatomyNet
21. 3D RoI-aware U-Net
22. Y-Net
23. Fully Dense UNet
24. U-NetPlus
25. CE-Net
26. Graph U-Net
27. ST-UNet
28. Connection Sensitive Attention U-NET
29. CIA-Net
30. W-Net
31. Coordination-guided Deep Neural Networks
32. U2-Net
33. ScleraSegNet
34. AHCNet
35. Recurrent U-Net
36. MFP-Unet
37. Partially Reversible U-Net
38. ResUNet-a
39. Multi-task U-net
40. RAUNet
41. 3D U2-Net
42. SegNAS3D
43. 3D Dilated Multi-Fiber Network
44. Unet-GAN
45. Siamese U-Net
46. U^2-Net
47. UNET 3+
48. BiO-Net
49. Projective Skip-Connections
50. BCDU-Net
51. CR-Unet

#### Category 2: Improvement of bottleneck, introduction of new operations on latent space

1. MA-Net
2. SA-UNet
3. RIC U-Net
4. FRCU-Net
5. SMU-Net
6. ASPPP-FC DenseNet
7. COPLE-Net
8. JCS

#### Category 3: Improvement of skip connection, introduce new operations in the skip connection process

1. U-Net++
2. U-Net 3+
3. Attention U-Net
4. RA-UNet
5. Attention UNet++
6. Projective Skip-Connections
7. BCDU-Net
8. CR-Unet

### Category 4: Improvement of overall structure, adding a new structure based the auto-encoder

1. Probabilistic U-Net
2. Hierarchical Probabilistic U-Net
3. MRF-UNet
4. Bayesian Skip Net
5. VAE U-Net
6. Probability Map Guided Bi-directional Recurrent UNet
7. Improved Attention U-Net
8. Focal Tversky Attention U-Net
9. Polar Transformation M-Net
10. Cascaded U-Net
11. 3D Attention U-Net





## 14/06/2024

[Geometry-Informed Neural Networks](https://arxiv.org/abs/2402.14009)

## 13/06/2024
IEEE-TPAMI

[Supervision by Denoising](https://ieeexplore.ieee.org/document/10197225)


## 12/06/2024:
Nature Machine Intelligence

[Unsupervised ensemble-based phenotyping enhances discoverability of genes related to left-ventricular morphology](https://www.nature.com/articles/s42256-024-00801-1)

## 11/06/2024:

European Heart Journal



## 10/06/2024:

paper summary


Journal

- [ ] Lancet series:

The Lancet Digital Health

[Deep learning models for thyroid nodules diagnosis of fine-needle aspiration biopsy: a retrospective, prospective, multicentre study in China](https://www.sciencedirect.com/science/article/pii/S2589750024000852?dgcid=rss_sd_all)
 
- [ ] Nature series:
      
Nature cardiovascular research
      
[Cardiometabolic and renal phenotypes and transitions in the United States population](https://www.nature.com/articles/s44161-023-00391-y)

- [ ] IEEE series:
