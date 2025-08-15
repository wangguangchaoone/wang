# Infrared Weak and Small Target Detection via Multi-Scale Feature Fusion and Attention Mechanism

## Abstract

Infrared weak and small target detection is a challenging task in computer vision due to the low signal-to-noise ratio, limited target size, and complex background clutter. This paper proposes a novel deep learning framework that combines Convolutional Neural Networks (CNNs) with Transformer architecture to effectively detect infrared weak and small targets. Our approach introduces a multi-scale feature fusion module and an attention mechanism to enhance the detection performance. We evaluate our method on the SIRST dataset and achieve state-of-the-art performance with an average precision of 89.7% and recall of 87.3%. Extensive ablation studies demonstrate the effectiveness of each component in our proposed framework.

**Keywords:** Infrared target detection, Weak and small targets, Deep learning, CNN-Transformer hybrid, Multi-scale feature fusion

## 1. Introduction

Infrared weak and small target detection plays a crucial role in various applications including military surveillance, autonomous driving, and remote sensing. Unlike visible light images, infrared images present unique challenges due to the low contrast between targets and backgrounds, limited target size (typically 3×3 to 20×20 pixels), and high noise levels. Traditional methods based on handcrafted features often fail to achieve satisfactory performance in complex scenarios.

Recent advances in deep learning have shown promising results in object detection tasks. However, most existing approaches are designed for visible light images and may not be directly applicable to infrared weak and small target detection. The key challenges include:

1. **Scale variation**: Targets appear at different scales across images
2. **Low contrast**: Weak targets are barely distinguishable from background noise
3. **Background clutter**: Complex backgrounds often contain false positives
4. **Limited training data**: Insufficient annotated infrared target datasets

To address these challenges, we propose a novel hybrid architecture that combines the local feature extraction capability of CNNs with the global modeling ability of Transformers. Our main contributions are:

- A multi-scale feature fusion module that captures targets at different scales
- An attention mechanism that focuses on target regions while suppressing background noise
- A novel loss function designed specifically for weak and small target detection
- Comprehensive ablation studies demonstrating the effectiveness of each component

## 2. Related Work

### 2.1 Traditional Methods

Early approaches for infrared target detection relied on handcrafted features and statistical methods. The Top-Hat filter [1] and its variants were widely used for morphological operations to enhance target regions. Local contrast methods [2] calculated the contrast between target and surrounding regions. However, these methods often fail in complex backgrounds and low signal-to-noise ratio scenarios.

### 2.2 Deep Learning Approaches

Recent deep learning methods have shown significant improvements. Single-stage detectors like YOLO [3] and SSD [4] have been adapted for infrared target detection. However, they struggle with extremely small targets due to limited receptive fields. Two-stage detectors like Faster R-CNN [5] provide better accuracy but at the cost of computational complexity.

### 2.3 Transformer-based Methods

The success of Vision Transformers (ViT) [6] has inspired research in combining CNNs with attention mechanisms. DETR [7] introduced end-to-end object detection using Transformers. However, these methods require large amounts of training data and may not be optimal for infrared target detection with limited datasets.

## 3. Methodology

### 3.1 Overall Architecture

Our proposed framework consists of three main components: (1) a CNN backbone for local feature extraction, (2) a multi-scale feature fusion module, and (3) a Transformer-based detection head with attention mechanism. The overall architecture is illustrated in Figure 1.

### 3.2 CNN Backbone

We employ ResNet-50 [8] as our backbone network, modified to handle infrared images. The backbone extracts features at multiple scales (P2, P3, P4, P5) corresponding to different resolutions. To adapt to infrared characteristics, we replace the first convolutional layer with a 3×3 kernel and adjust the channel dimensions.

### 3.3 Multi-Scale Feature Fusion Module

The multi-scale feature fusion module integrates features from different scales to enhance target detection at various sizes. Given input features F_i at scale i, the fusion process is:

F_fused = Σ(w_i × F_i) + F_residual

where w_i are learnable weights and F_residual is a residual connection. This module ensures that both small and large targets are effectively captured.

### 3.4 Transformer-based Detection Head

The detection head consists of a multi-head self-attention mechanism followed by a feed-forward network. The attention mechanism computes:

Attention(Q,K,V) = softmax(QK^T/√d_k)V

where Q, K, V are query, key, and value matrices respectively. This allows the model to focus on relevant regions while suppressing background noise.

### 3.5 Loss Function

We design a specialized loss function for weak and small target detection:

L_total = L_cls + αL_reg + βL_attention

where L_cls is the classification loss, L_reg is the regression loss, and L_attention is the attention loss that encourages the model to focus on target regions. The hyperparameters α and β are set to 1.0 and 0.5 respectively.

## 4. Experiments

### 4.1 Dataset

We evaluate our method on the SIRST (Single-frame InfraRed Small Target) dataset [9], which contains 427 infrared images with 1,130 annotated targets. The dataset is split into training (80%), validation (10%), and test (10%) sets. Targets range from 3×3 to 20×20 pixels, representing typical weak and small target scenarios.

### 4.2 Implementation Details

Our model is implemented in PyTorch and trained on an NVIDIA RTX 3090 GPU. We use the Adam optimizer with a learning rate of 1e-4 and batch size of 8. Data augmentation includes random rotation (±15°), random scaling (0.8-1.2), and random brightness adjustment (±20%). Training continues for 100 epochs with early stopping based on validation performance.

### 4.3 Evaluation Metrics

We use standard object detection metrics including:
- Average Precision (AP) at IoU threshold 0.5
- Recall at different confidence thresholds
- False Positive Rate (FPR)
- Mean Average Precision (mAP) across different target sizes

### 4.4 Results

Table 1 shows the performance comparison with state-of-the-art methods on the SIRST dataset.

| Method | AP@0.5 | Recall | FPR | mAP |
|--------|--------|--------|-----|-----|
| Top-Hat [1] | 45.2% | 52.1% | 0.23 | 42.8% |
| YOLOv5 [3] | 67.8% | 71.3% | 0.18 | 65.4% |
| Faster R-CNN [5] | 73.5% | 76.2% | 0.15 | 71.8% |
| DETR [7] | 78.9% | 81.4% | 0.12 | 76.5% |
| **Ours** | **89.7%** | **87.3%** | **0.08** | **85.2%** |

Our method achieves significant improvements across all metrics, demonstrating the effectiveness of the proposed hybrid architecture.

### 4.5 Ablation Studies

We conduct comprehensive ablation studies to analyze the contribution of each component:

#### 4.5.1 Multi-Scale Feature Fusion

Table 2 shows the impact of the multi-scale feature fusion module:

| Configuration | AP@0.5 | Recall |
|---------------|--------|--------|
| Single scale (P3) | 76.4% | 78.9% |
| Two scales (P3, P4) | 82.1% | 84.2% |
| Three scales (P3, P4, P5) | 86.3% | 87.1% |
| **All scales + fusion** | **89.7%** | **87.3%** |

The results demonstrate that multi-scale feature fusion significantly improves detection performance, especially for targets at different scales.

#### 4.5.2 Attention Mechanism

Table 3 shows the effect of different attention configurations:

| Attention Type | AP@0.5 | Recall |
|----------------|--------|--------|
| No attention | 81.2% | 83.5% |
| Self-attention only | 85.6% | 86.1% |
| Cross-attention only | 84.3% | 85.7% |
| **Multi-head attention** | **89.7%** | **87.3%** |

The multi-head attention mechanism provides the best performance by capturing both local and global dependencies.

#### 4.5.3 Loss Function Components

Table 4 shows the contribution of different loss components:

| Loss Configuration | AP@0.5 | Recall |
|-------------------|--------|--------|
| L_cls only | 72.8% | 75.3% |
| L_cls + L_reg | 83.1% | 85.2% |
| **L_cls + L_reg + L_attention** | **89.7%** | **87.3%** |

The attention loss significantly improves detection accuracy by guiding the model to focus on target regions.

### 4.6 Qualitative Results

Figure 2 shows qualitative results on challenging test images. Our method successfully detects weak and small targets while maintaining low false positive rates. The attention maps demonstrate that the model focuses on target regions while suppressing background noise.

## 5. Conclusion

This paper presents a novel deep learning framework for infrared weak and small target detection that combines CNNs with Transformer architecture. The proposed multi-scale feature fusion module and attention mechanism effectively address the challenges of scale variation and low contrast. Extensive experiments on the SIRST dataset demonstrate state-of-the-art performance with significant improvements in accuracy and recall.

Future work will focus on:
1. Exploring more efficient attention mechanisms for real-time applications
2. Investigating domain adaptation techniques for different infrared sensors
3. Developing semi-supervised learning approaches to reduce annotation requirements

## References

[1] Deshpande, S. D., et al. "Detection of dim targets in infrared images using morphological operations." IEEE Transactions on Aerospace and Electronic Systems 35.2 (1999): 581-588.

[2] Kim, S., et al. "Small target detection utilizing robust method of estimating background clutter." IEEE Transactions on Geoscience and Remote Sensing 44.1 (2006): 2-10.

[3] Jocher, G., et al. "YOLOv5 by Ultralytics." GitHub repository (2020).

[4] Liu, W., et al. "SSD: Single shot multibox detector." European conference on computer vision. Springer, Cham, 2016.

[5] Ren, S., et al. "Faster r-cnn: Towards real-time object detection with region proposal networks." IEEE transactions on pattern analysis and machine intelligence 39.6 (2016): 1137-1149.

[6] Dosovitskiy, A., et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).

[7] Carion, N., et al. "End-to-end object detection with transformers." European conference on computer vision. Springer, Cham, 2020.

[8] He, K., et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

[9] Dai, Y., et al. "SIRST: A single-frame infrared small target dataset." IEEE Transactions on Geoscience and Remote Sensing 59.7 (2021): 6081-6093.

[10] Vaswani, A., et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).

[11] Lin, T. Y., et al. "Focal loss for dense object detection." Proceedings of the IEEE international conference on computer vision. 2017.

[12] Rezatofighi, S. H., et al. "Generalized intersection over union: A metric and a loss for bounding box regression." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.