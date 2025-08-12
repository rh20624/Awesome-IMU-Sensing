# Awesome-IMU-Sensing

A collection of datasets, papers, and other resources for IMU-based human activity recognition (HAR).

The organization of this repository refers to our survey **"Towards Generalizable Human Activity Recognition: A Survey"**.

For more related information, feel free to check out our website [http://www.zhiqinghong.one/](http://www.zhiqinghong.one/).

## Table of Contents

- [Generalization-Oriented Training Settings](#generalization-oriented-training-settings)
- [Datasets and Benchmarks](#datasets-and-benchmarks)
- [Model-Centric Methodology](#model-centric-methodology)
- [Data-Centric Methodology](#data-centric-methodology)
- [Applications](#applications)

## Generalization-Oriented Training Settings

### Within-Group

2. **"Ensembles of deep lstm learners for activity recognition using wearables"**. *Guan et al.* IMWUT 2017. [[Paper](https://dl.acm.org/doi/abs/10.1145/3090076)]
3. **"Label Propagation: An Unsupervised Similarity Based Method for Integrating New Sensors in Activity Recognition Systems"**. *Rey et al.* IMWUT 2017. [[Paper](https://dl.acm.org/doi/abs/10.1145/3130959)]
4. **"Unleashing the Power of Shared Label Structures for Human Activity Recognition"**. *Zhang et al.* CIKM 2023. [[Paper](https://dl.acm.org/doi/abs/10.1145/3583780.3615101)]
5. **"Decomposing and Fusing Intra-and Inter-Sensor Spatio-Temporal Signal for Multi-Sensor Wearable Human Activity Recognition"**. *Xie et al.* AAAI 2025. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/33582)]

### Cross-Person

1. **"XHAR: Deep Domain Adaptation for Human Activity Recognition with Smart Devices"**. *Zhou et al.* SECON 2020. [[Paper](https://ieeexplore.ieee.org/abstract/document/9158431)]
2. **"Latent Independent Excitation for Generalizable Sensor-based Cross-Person Activity Recognition"**. *Qian et al.* AAAI 2021. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/17416)]
3. **"Domain Generalization for Activity Recognition via Adaptive Feature Fusion"**. *Qin et al.* TIST 2022. [[Paper](https://dl.acm.org/doi/full/10.1145/3552434)]
4. **"SWL-Adapt: An Unsupervised Domain Adaptation Model with Sample Weight Learning for Cross-User Wearable Human Activity Recognition"**. *Hu et al.* AAAI 2023. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/25743)]

### Cross-Device

1. **"XHAR: Deep Domain Adaptation for Human Activity Recognition with Smart Devices"**. *Zhou et al.* SECON 2020. [[Paper](https://ieeexplore.ieee.org/abstract/document/9158431)]
2. **"ColloSSL: Collaborative Self-Supervised Learning for Human Activity Recognition"**. *Jain et al.* IMWUT 2022. [[Paper](https://dl.acm.org/doi/abs/10.1145/3517246)]
3. **"HyperHAR: Inter-sensing Device Bilateral Correlations and Hyper-correlations Learning Approach for Wearable Sensing Device Based Human Activity Recognition"**. *Ahmad et al.* IMWUT 2024. [[Paper](https://dl.acm.org/doi/abs/10.1145/3643511)]
4. **"UniMTS: Unified Pre-training for Motion Time Series"**. *Zhang et al.* NeurIPS 2024 [[Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c290d4373c495b2cad0625d6288260f0-Abstract-Conference.html)]

### Cross-Position

1. **"Cross-position activity recognition with stratified transfer learning"**. *Chen et al.* PMC 2019. [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S1574119218303432)]
2. **"A Systematic Study of Unsupervised Domain Adaptation for Robust Human-Activity Recognition"**. *Chang et al.* IMWUT 2020. [[Paper](https://dl.acm.org/doi/abs/10.1145/3380985)]
3. **"Semantic-Discriminative Mixup for Generalizable Sensor-based Cross-domain Activity Recognition"**. *Lu et al.* IMWUT 2022. [[Paper](https://dl.acm.org/doi/abs/10.1145/3534589)]
4. **"Cross-Domain HAR: Few-Shot Transfer Learning for Human Activity Recognition"**. *Thukral et al.* TIST 2025. [[Paper](https://dl.acm.org/doi/full/10.1145/3704921)]

### Cross-Activity

1. **"NuActiv: recognizing unseen new activities using semantic attribute-based learning"**. *Cheng et al.* MobiSys 2013 [[Paper](https://dl.acm.org/doi/abs/10.1145/2462456.2464438)]
2. **"UniMTS: Unified Pre-training for Motion Time Series"**. *Zhang et al.* NeurIPS 2024 [[Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c290d4373c495b2cad0625d6288260f0-Abstract-Conference.html)]
3. **"ZeroHAR: Sensor Context Augments Zero-Shot Wearable Action Recognition"**. *Chowdhury et al.* AAAI 2025 [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/33762)]

### Cross-Dataset

1. **"Cross-Dataset Activity Recognition via Adaptive Spatial-Temporal Transfer Learning"**. *Qin et al.* IMWUT 2020 [[Paper](https://dl.acm.org/doi/abs/10.1145/3369818)]
2. **"Semantic-Discriminative Mixup for Generalizable Sensor-based Cross-domain Activity Recognition"**. *Lu et al.* IMWUT 2022. [[Paper](https://dl.acm.org/doi/abs/10.1145/3534589)]
3. **"GLOBEM: Cross-Dataset Generalization of Longitudinal Human Behavior Modeling"**. *Xu et al.* IMWUT 2023. [[Paper](https://dl.acm.org/doi/abs/10.1145/3569485)]
4. **"CrossHAR: Generalizing Cross-dataset Human Activity Recognition via Hierarchical Self-Supervised Pretraining"**. *Hong et al.* IMWUT 2024 [[Paper](https://dl.acm.org/doi/abs/10.1145/3659597)]

## Datasets and Benchmarks

### IMU-Only Dataset

| Dataset | Subjects | Sensors | Activities | datasize | Year |
| --- | --- | --- | --- | --- | --- |
| CAPTURE-24 | 151 | acc | 200 unique labels | 3883 h | 2024 |
| TNDA-HAR | 23 | acc, gyro | 8 daily activities | 5.7 h | 2021 |
| HAR70+ | 18 | acc | 8 daily activities | 12.6 h | 2020 |
| WISDM | 51 | acc, gyro | 18 daily activities | 91.8 h | 2019 |
| MotionSense | 24 | acc, gyro | 6 daily activities | - | 2019 
|SHL Challenge | 3 | acc, gyro, mag | 8 transportation modes | 2812 h | 2018 | 
| MobiAct | 57 | acc, gyro | 9 daily activities and 4 falls | - | 2016 |  
| Shoaib | 10 | acc, gyro, mag | 13 daily activities | 6.5 h | 2016 |  
| HHAR | 9 | acc, gyro | 6 daily activities | - | 2015 |  
| WHARF | 16 | acc | 8 motion primitives | - | 2013 |  
| DSADS | 8 | acc, gyro, mag | 19 | daily, sports activities | 12.7 h | 2013 | 
| UCI-HAR | 30 | acc, gyro | 6 daily activities | - | 2012 | 
| USC-HAD | 14 | acc, gyro, mag | 12 daily activities | - | 2012  
| Daphnet FoG | 10 | acc | 3 walking activities | 8.3 h | 2009 | 
| Skoda Mini  Checkpoint | 1 | acc, 3D acc sensor | 10 assembly-line activities | - | 2008 |

### Multimodal Dataset

| Dataset | Subjects | Sensors | Activities | datasize | Year |
| --- | --- | --- | --- | --- | --- |
| WEAR | 22 | acc, video | 18 sports activities | 19 h | 2024 |
| HARTH | 22 | acc, video | 12 daily activities | 35.9 h | 2021 |
| w-HAR | 22 | acc, gyro, stretch sensor | 7 daily activities | 3 h | 2020 | 
| MMAct | 40 | RGB-video, keypoints, acc,  gyro, ori, Wi-Fi, pressure | 37 daily, abnormal, desk work activities | - | 2019 |  
| RealWorld HAR | 15 | acc, gyro, mag, GPS, light,  sound level | 8 daily activities | 124.3h | 2016 | 
| UTD-MHAD | 8 | RGB video, depth video, skeleton positions, acc, gyro | 27 daily, sports activities,  gestures | - | 2015 |  
| MHEALTH | 10 | acc, gyro, mag, ECG | 12 daily activities | - | 2014 |  
| Berkeley MHAD | 12 | acc, optical capture system, video, depth sensor, audio | 11 daily activities | 1.37 h | 2013 | 
| PAMAP2 | 9 | acc, gyro, mag, heart rate | 18 daily activities | 10 h | 2012 | 
| Opportunity | 4 | acc, gyro, mag, ambient sensors | 9 kitchen activities, 9 gestures | 25 h | 2011 |

## Model-Centric Methodology

### Transformation Recognition

1. **"Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding"**. *Tonekaboni et al.* ICLR 2021 [[Paper](https://arxiv.org/abs/2106.00750)]
2. **"Self-supervised Learning for Reading Activity Classification"**. *Islam et al.* IMWUT 2021
3. **"SelfHAR: Improving Human Activity Recognition through Self-training with Unlabeled Data"**. *Tang et al.* IMWUT 2021

### Reconstruction

1. **"LIMU-BERT: Unleashing the Potential of Unlabeled Data for IMU Sensing Applications"**. *Xu et al.* SenSys 2021 [[Paper](https://dl.acm.org/doi/abs/10.1145/3485730.3485937)]
2. **"Scaling wearable foundation models"**. *Narayanswamy et al.* ICLR 2025
3. **"LSM-2: Learning from Incomplete Wearable Sensor Data"**. *Xu et al.* arXiv 2025

### Contrastive & Predictive Learning

1. **"Contrastive Predictive Coding for Human Activity Recognition"**. *Haresamudram et al.* IMWUT 2021 [[Paper](https://dl.acm.org/doi/abs/10.1145/3463506)]

### Feature Disentanglement

1. **"Adversarial Multi-view Networks for Activity Recognition"**. *Bai et al.* IMWUT 2020 [[Paper](https://dl.acm.org/doi/abs/10.1145/3397323)]

### Multi-task Learning

1. **"CrossHAR: Generalizing Cross-dataset Human Activity Recognition via Hierarchical Self-Supervised Pretraining"**. *Hong et al.* IMWUT 2024 [[Paper](https://dl.acm.org/doi/abs/10.1145/3659597)]

### Federated Learning

1. **"Practically Adopting Human Activity Recognition"**. *Xu et al.* MobiCom 2023 [[Paper](https://dl.acm.org/doi/abs/10.1145/3570361.3613299)]

### Active Learning

1. **"NuActiv: recognizing unseen new activities using semantic attribute-based learning"**. *Cheng et al.* MobiSys 2013 [[Paper](https://dl.acm.org/doi/abs/10.1145/2462456.2464438)]

### LLM-based Learning

1. **"Sensor2Text: Enabling Natural Language Interactions for Daily Activity Tracking Using Wearable Sensors"**. *Chen et al.* IMWUT 2024 [[Paper](https://dl.acm.org/doi/abs/10.1145/3699747)]

## Data-Centric Methodology

### Multi-Modal Fusion

1. **"MESEN: Exploit Multimodal Data to Design Unimodal Human Activity Recognition with Few Labels"**. *Xu et al.* SenSys 2023 [[Paper](https://dl.acm.org/doi/abs/10.1145/3625687.3625782)]

### Cross-Modal Learning

1. **"Zero-Shot Learning for IMU-Based Activity Recognition Using Video Embeddings"**. *Tong et al.* IMWUT 2021 [[Paper](https://dl.acm.org/doi/abs/10.1145/3494995)]

### Data Augmentation

1. **"Learning IMU Bias with Diffusion Model"**. *Zhou et al.* arXiv 2025 [[Paper](https://arxiv.org/abs/2505.11763)]

## Applications

### Healthcare & Rehabilitation

1. **"Deep Learning-Based Near-Fall Detection Algorithm for Fall Risk Monitoring System Using a Single Inertial Measurement Unit"**. *Choi et al.* TNSRE 2022 [[Paper](https://ieeexplore.ieee.org/abstract/document/9857937)]

### Sports & Fitness Monitoring

1. **"The Positive Impact of Push vs Pull Progress Feedback: A 6-week Activity Tracking Study in the Wild"**. *Cauchard et al.* IMWUT 2019 [[Paper](https://dl.acm.org/doi/abs/10.1145/3351234)]

### Work Assessment

1. **"Unsupervised Factory Activity Recognition with Wearable Sensors Using Process Instruction Information"**. *Xia et al.* IMWUT 2019 [[Paper](https://dl.acm.org/doi/abs/10.1145/3328931)]

### Smart Home & Assisted Living

1. **"ThumbUp: Secure Smartwatch Controller for Smart Homes Using Simple Hand Gestures"**. *Yu et al.* TMC 2022 [[Paper](https://ieeexplore.ieee.org/abstract/document/9928390)]

### Transportation & Mobility

1. **"Benchmarking the SHL Recognition Challenge with Classical and Deep-Learning Pipelines"**. *Wang et al.* UbiComp 2018 [[Paper](https://dl.acm.org/doi/abs/10.1145/3267305.3267531)]

### Human-Robot Interaction

1. **"Human–robot interaction based on wearable IMU sensor and laser range finder"**. *Cifuentes et al.* RAS 2014 [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0921889014001122)]

### AR & VR Interaction

1. **"MobilePoser: Real-Time Full-Body Pose Estimation and 3D Human Translation from IMUs in Mobile Consumer Devices"**. *Xu et al.* UIST 2024 [[Paper](https://dl.acm.org/doi/10.1145/3654777.3676461)]

### Embodied Agents 

1. **"HandCept: A Visual-Inertial Fusion Framework for Accurate Proprioception in Dexterous Hands"**. *Huang et al.* arXiv 2025 [[Paper](https://arxiv.org/abs/2505.08213)]













