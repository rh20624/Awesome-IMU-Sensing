<img width="1221" alt="image" src="https://github.com/rh20624/Awesome-IMU-Sensing/assets/15633495/ca236d63-91bd-4b41-b1a4-0ede41e46545"># Awesome-IMU-Sensing
A collection of datasets, papers, directions, and other resources for IMU-based mobile sensing.

**Popular venues**
- IMWUT/Ubicomp, Sensys, Mobicom
- AAAI, IJCAI, KDD, ICLR, TKDE

## Public Datasets

## Reviews
[2024] Machine Learning Techniques for Sensor-based Human Activity Recognition with Data Heterogeneity - A Review

[2024] A Survey on Multimodal Wearable Sensor-based Human Action Recognition

[2024] A Survey of IMU Based Cross-Modal Transfer Learning in Human Activity Recognition

【Ubicomp 2021】11 Years with Wearables: Quantitative Analysis of Social Media, Academia, News Agencies, and Lead User Community from 2009-2020 on Wearable Technologie
novelty: market analysis, wearable technology, sentiment analysis


## Human Activity Recognition

### Sels-supervised learning
[2024] Comparing Self-Supervised Learning Techniques for Wearable Human Activity Recognition

[2024] HARMamba: Efficient Wearable Sensor Human Activity Recognition Based on Bidirectional Selective SSM
- Mamba-based HAR model

【Ubicomp 2022】colloSSL: Collaborative Self-Supervised Learning for Human Activity Recognition
- multi-device, contrastive learning

【Ubicomp 2022】Assessing the State of Self-Supervised Human Activity Recognition Using Wearables
- benchmark study

[Sensys 2021] LIMU-BERT: Unleashing the Potential of Unlabeled Data for IMU Sensing Applications

【Ubicomp 2021】Self-supervised Learning for Reading Activity Classification

【Ubicomp 2021】Contrastive Predictive Coding for Human Activity Recognition

【Ubicomp 2021】SelfHAR: Improving Human Activity Recognition through Self-training with Unlabeled Data

【Ubicomp 2019】Multi-task Self-Supervised Learning for human Activity Detection

### Weak-supervised learning
【Ubicomp 2019】Leveraging Active Learning and Conditional Mutual Information to Minimize Data Annotation in Human Activity Recognition
- few label
【Ubicomp 2020】Weakly Supervised Multi-Task Representation Learning for Human Activity Analysis Using Wearables

### Multi-modal HAR
[Sensys 2023]MESEN: Exploit Multimodal Data to Design Unimodal Human Activity Recognition with Few Labels

[Sensys21] UniTS: Short-Time Fourier Inspired Neural Networks for Sensory Time Series Classification

[IMWUT 2023] Synthetic Smartwatch IMU Data Generation from In-the-wild ASL Videos
- cross-modal data synthesis

[IMWUT 2020] IMUTube: Automatic Extraction of Virtual on-body Accelerometry from Video for Human Activity Recognition
- data synthesis

[Ubicomp 2021] Zero-Shot Learning for IMU-Based Activity Recognition Using Video Embeddings
- video + IMU

【Ubicomp 2021】IMU2Doppler: Cross-Modal Domain adaptation for Doppler-based Activity Recognition Using IMU Data
- transfer IMU to mmwave
【Ubicomp 2019】Vision2Sensor: Knowledge Transfer Across Sensing Modalities for Human Activity Recognition
- transfer knowledge from video to IMU
【Ubicomp 2019】Exploring the Efficacy of Sparse, General-Purpose Sensor Constellations for Wide-Area Activity Sensing
- multi-sensor fusion

### Federated learning


### Privacy-enhanced HAR
[IMWUT 2023] SeRaNDiP - Leveraging Inherent Sensor Random Noise for Differential Privacy Preservation in Wearable Community Sensing Applications

[Mobicom 23] Practically Adopting Human Activity Recognition
- federated learning + data augmentation

[IMWUT 2023] Hierarchical Clustering-based Personalized Federated Learning for Robust and Fair Human Activity Recognition
- federated learning + fairness
- 
[Sensys 21] FedDL: Federated Learning via Dynamic Layer Sharing for Human Activity Recognition
- federated learning



### Generalization HAR
[IMWUT 2023] GLOBEM: Cross-Dataset Generalization of Longitudinal Human Behavior Modeling
- cross-dataset depression detection

[IMWUT 2022] Semantic-Discriminative Mixup for Generalizable Sensor-based

[ICASAP 2022] Local and global alignments for generalizable sensor-based human activity recognition

[AAAI 2021] Latent Independent Excitation for Generalizable Sensor-based Cross-Person Activity Recognition

[Ubicomp 2020] Incremental Real-Time Personalization in Human Activity Recognition Using Domain Adaptive Batch Normalization
- domain adaptation, cross user

【Ubicomp 2020】A Systematic Study of Unsupervised Domain Adaptation for Robust Human-Activity Recognition
- cross device locations
【Ubicomp 2019】Cross-Dataset Activity Recognition via Adaptive Spatial-Temporal Transfer Learning

### Explainable HAR

[IMWUT 2023] X-CHAR: A Concept-based Explainable Complex Human Activity
Recognition Model

### Special application scenarios

【Ubicomp 2020】Robust Unsupervised Factory Activity Recognition with Body-worn Accelerometer Using Temporal Structure of Multiple Sensor Data Motifs
【Ubicomp 2021】Leveraging Activity Recognition to Enable Protective Behavior Detection in Continuous Data
【Ubicomp 2019】The Wearables Development Toolkit: An Integrated Development Environment for Activity Recognition Applications
【Ubicomp 2019】Integrating Activity Recognition and Nursing Care Records: The System, Deployment, and a Verification Study.
- medical application
【Ubicomp 2019】Unsupervised Factory Activity Recognition with Wearable Sensors Using Process Instruction Information

### Other HAR
[IMWUT 2023] ConvBoost: Boosting ConvNets for Sensor-based Activity Recognition

[imwut 2023] DAPPER: Label-Free Performance Estimation after Personalization for Heterogeneous Mobile Sensing
- predict the performance of a model on new data

【Ubicomp 2022】Learning Disentangled Behaviour Patterns for Wearable-based Human Activity Recognition

【Ubicomp 2020】Deriving Effective Human Activity Recognition Systems through Objective Task Complexity Assessment

【Ubicomp 2020】Adversarial Multi-view Networks for Activity Recognition

[IMWUT 2020] KATN: Key Activity Detection via Inexact Supervised Learning

【Ubicomp 2021】CrowdAct: Achieving High-Quality Crowdsourced Datasets in Mobile Activity Recognition

【Ubicomp 2021】Attend and Discriminate: Beyond the State-of-the-Art for Human Activity Recognition Using Wearable Sensors
- exploit the latent relationships between multi-channel sensor modalities and specific activities, data-agnostic augmentation
【Ubicomp 2019】The Positive Impact of Push vs Pull Progress Feedback: A 6-week Activity Tracking Study in the Wild


## Gesture Recognition
[IMWUt 2023] DRG-Keyboard: Enabling Subtle Gesture Typing on the Fingertip with Dual IMU Rings

【Ubicomp 2021】A CNN-based Human Activity Recognition System Combining a Laser Feedback Interferometry Eye Movement Sensor and an IMU for Context-aware Smart Glasses
novelty: combine Eye Movement Sensor and IMU, HAR,Smart glasses,transfer learning to personalize the classification


## Gait Recognition

## Other modality data
[IMWUT 2023] RF-CM: Cross-Modal Framework for RF-enabled Few-Shot Human Activity Recognition
- wifi + mmwave
[Sensys 2021] OneFi: One-Shot Recognition for Unseen Gesture via COTS WiFi
[Sensys 2020] RF-net: a unified meta-learning framework for RF-enabled one-shot human activity recognition
【Ubicomp 2020】DeepMV: Multi-View Deep Learning for Device-Free Human Activity Recognition
- wifi
【Ubicomp 2020】CARIN: Wireless CSI-based Driver Activity Recognition under the Interference of Passengers
【Ubicomp 2019】Personalized Context-aware Collaborative Online Activity Prediction
- a user-location-time-activity 4D-tensor and a location-time-POI 3D-tensor
[Sensys 19] RFID based real-time recognition of ongoing gesture with adversarial learning
【Ubicomp 2021】Fall Detection via Inaudible Acoustic Sensing
【Ubicomp 2019】Towards a Diffraction-based Sensing Approach on Human Activity Recognition
- wifi
## Smart city applications
[2024] Transportation mode recognition based on low-rate acceleration and location signals with an attention-based multiple-instance learning network
