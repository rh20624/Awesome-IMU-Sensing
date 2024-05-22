# Awesome-IMU-Sensing
A collection of datasets, papers, directions, and other resources for IMU-based mobile sensing.

**Main sensing tasks**
- Human activity recognition (HAR) / human activity sensing
- Gesture recognition
- Gait recognition
- Localization and navigation

**Popular venues**
- IMWUT/Ubicomp, Sensys, Mobicom, PerCom
- AAAI, IJCAI, KDD, ICLR, TKDE

## Public Datasets
[2024] CAPTURE-24: A large dataset of wrist-worn activity tracker data collected in the wild for human activity recognition
- 2,562 hours of annotated data
- [https://github.com/OxWearables/capture24](https://github.com/OxWearables/capture24)

[2016] RealWorld
- Downstairs, Upstairs, Lying, Sitting, Standing, Jumping, Walking, Running
- [https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/](https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/)

[2012] UCI dataset
- Acc and Gyro
- varying ages (19 to 48 years), genders, heights, and weights

[2012] WISDM dataset
- Acc only
- walking,jogging,upstairs,downstairs,sitting,standing
- [https://www.cis.fordham.edu/wisdm/dataset.php](https://www.cis.fordham.edu/wisdm/dataset.php)


| Dataset      | User | # of Activity | # of Sample | Device                  | Placement                                      | Sampling Rate | Time |
|--------------|------|---------------|-------------|------------------------------|------------------------------------------------|---------------|---------------|
| CAPTURE-24   | 151   | >200             | 2,562 hours    | Bracelet                   | waist                                         | 100Hz  | 2024 |
| MobiAct   |  -  | 9              | -    | -                   | -                                         | 50~100Hz          | 2017 |
| RealWorld |  -  | 8              | -    | -                   | -                                         | 50Hz          | 2016 |
| WISDM |  36  | 6              | 1,098,207    | Smartphone                   | front leg pocket                                         | 20Hz          | 2012 |
| UCI      | 30   | 6             | 1,687        | Smartphone                   | waist                                         | 50Hz          | 2012 |
| HHAR     | 9    | 6             | 7,968        | Phone, watch       | -                                              | 50~200Hz      | 2015 | 
| Shoaib   | 10   | 7             | 7,500        | Smartphone                   | pockets, wrist, arm, belt | 50Hz        | - |
| Motion   | 24   | 6             | 4,108        | Smartphone                   | front pocket                                  | 50Hz          | 2017 |


## Reviews

[2024] Transfer Learning in Human Activity Recognition: A Survey

[2024] Machine Learning Techniques for Sensor-based Human Activity Recognition with Data Heterogeneity - A Review

[2024] A Survey on Multimodal Wearable Sensor-based Human Action Recognition

[2024] A Survey of IMU Based Cross-Modal Transfer Learning in Human Activity Recognition

[IJCNN 2024] A Survey on Wearable Human Activity Recognition: Innovative Pipeline Development for Enhanced Research and Practice

[ICASSPW 2023] A Survey of Datasets, Applications, and Models for IMU Sensor Signals

[ACM Computing Survey 2022] [A Survey of Privacy Vulnerabilities of Mobile Device Sensors](https://dl.acm.org/doi/10.1145/3510579)
- wearable sensors, privacy concern

【IMWUT 2021】11 Years with Wearables: Quantitative Analysis of Social Media, Academia, News Agencies, and Lead User Community from 2009-2020 on Wearable Technologie
- market analysis, wearable technology, sentiment analysis

[ACM computing survey 2014] A tutorial on human activity recognition using body-worn inertial sensors


## App1: Human Activity Recognition

### Self-supervised learning

[IMWUT 2024] [CrossHAR: Generalizing Cross-dataset Human Activity Recognition via Hierarchical Self-Supervised Pretraining](https://dl.acm.org/doi/10.1145/3659597)
- hierarchical pretraining

[2024] [Comparing Self-Supervised Learning Techniques for Wearable Human Activity Recognition](https://arxiv.org/abs/2404.15331)

[2024] HARMamba: Efficient Wearable Sensor Human Activity Recognition Based on Bidirectional Selective SSM
- Mamba-based HAR model

[Percom 2023] Investigating Enhancements to Contrastive Predictive Coding for Human Activity Recognition


【IMWUT 2022】colloSSL: Collaborative Self-Supervised Learning for Human Activity Recognition
- multi-device, contrastive learning

【IMWUT 2022】Assessing the State of Self-Supervised Human Activity Recognition Using Wearables
- benchmark study

[Sensys 2021] [LIMU-BERT: Unleashing the Potential of Unlabeled Data for IMU Sensing Applications](https://dl.acm.org/doi/10.1145/3485730.3485937)

【IMWUT 2021】Self-supervised Learning for Reading Activity Classification

【IMWUT 2021】Contrastive Predictive Coding for Human Activity Recognition

【IMWUT 2021】SelfHAR: Improving Human Activity Recognition through Self-training with Unlabeled Data

【IMWUT 2019】Multi-task Self-Supervised Learning for human Activity Detection

### Weak-supervised learning
【IMWUT 2019】Leveraging Active Learning and Conditional Mutual Information to Minimize Data Annotation in Human Activity Recognition
- few label
【IMWUT 2020】Weakly Supervised Multi-Task Representation Learning for Human Activity Analysis Using Wearables


### LLM + HAR
[2024] HARGPT: Are LLMs Zero-Shot Human Activity Recognizers?
- prompt design; CoT

[2024] IMUGPT 2.0: Language-Based Cross Modality Transfer for Sensor-Based Human Activity Recognition

[2023] On the Benefit of Generative Foundation Models for Human Activity Recognition


### Multi-modal HAR
[PerCom 2024] iMove: Exploring Bio-impedance Sensing for Fitness Activity Recognition
- bio signal + IMU

[EMNLP 2023] IMU2CLIP: Language-grounded Motion Sensor Translation with Multimodal Contrastive Learning
- video + NLP + IMU
- from Meta Reality Lab

[Sensys 2023]MESEN: Exploit Multimodal Data to Design Unimodal Human Activity Recognition with Few Labels

[Sensys21] UniTS: Short-Time Fourier Inspired Neural Networks for Sensory Time Series Classification

[IMWUT 2023] Synthetic Smartwatch IMU Data Generation from In-the-wild ASL Videos
- cross-modal data synthesis

[IMWUT 2020] IMUTube: Automatic Extraction of Virtual on-body Accelerometry from Video for Human Activity Recognition
- data synthesis

[IMWUT 2021] Zero-Shot Learning for IMU-Based Activity Recognition Using Video Embeddings
- video + IMU

【IMWUT 2021】IMU2Doppler: Cross-Modal Domain adaptation for Doppler-based Activity Recognition Using IMU Data
- transfer IMU to mmwave

【IMWUT 2019】Vision2Sensor: Knowledge Transfer Across Sensing Modalities for Human Activity Recognition
- transfer knowledge from video to IMU

【IMWUT 2019】Exploring the Efficacy of Sparse, General-Purpose Sensor Constellations for Wide-Area Activity Sensing
- multi-sensor fusion

### Tools or benchmarks
[PerCom 2024] Evaluation of Video-Assisted Annotation of Human IMU Data
- a tool for IMU annotation


### Privacy-enhanced HAR
[IMWUT 2023] SeRaNDiP - Leveraging Inherent Sensor Random Noise for Differential Privacy Preservation in Wearable Community Sensing Applications

[Mobicom 23] Practically Adopting Human Activity Recognition
- federated learning + data augmentation

[IMWUT 2023] Hierarchical Clustering-based Personalized Federated Learning for Robust and Fair Human Activity Recognition
- federated learning + fairness

[Sensys 21] FedDL: Federated Learning via Dynamic Layer Sharing for Human Activity Recognition
- federated learning



### Generalization HAR
[IMWUT 2024] AutoAugHAR: Automated Data Augmentation for Sensor-based Human Activity Recognition
- cross-user; data augmentation; 

[IMWUT 2024] Optimization-Free Test-Time Adaptation for Cross-Person Activity Recognition
- cross-user, test-time adaptation

[IMWUT 2023] GLOBEM: Cross-Dataset Generalization of Longitudinal Human Behavior Modeling
- cross-dataset depression detection

[IMWUT 2022] Semantic-Discriminative Mixup for Generalizable Sensor-based

[ICASAP 2022] Local and global alignments for generalizable sensor-based human activity recognition

[AAAI 2021] Latent Independent Excitation for Generalizable Sensor-based Cross-Person Activity Recognition

[IMWUT 2020] Incremental Real-Time Personalization in Human Activity Recognition Using Domain Adaptive Batch Normalization
- domain adaptation, cross user

【IMWUT 2020】A Systematic Study of Unsupervised Domain Adaptation for Robust Human-Activity Recognition
- cross device locations

【Ubicomp 2019】Cross-Dataset Activity Recognition via Adaptive Spatial-Temporal Transfer Learning

### Explainable HAR

[IMWUT 2023] X-CHAR: A Concept-based Explainable Complex Human Activity
Recognition Model

### Special application scenarios

【IMWUT 2020】Robust Unsupervised Factory Activity Recognition with Body-worn Accelerometer Using Temporal Structure of Multiple Sensor Data Motifs

【IMWUT 2021】Leveraging Activity Recognition to Enable Protective Behavior Detection in Continuous Data

【IMWUT 2019】The Wearables Development Toolkit: An Integrated Development Environment for Activity Recognition Applications

【IMWUT 2019】Integrating Activity Recognition and Nursing Care Records: The System, Deployment, and a Verification Study.
- medical application

【IMWUT 2019】Unsupervised Factory Activity Recognition with Wearable Sensors Using Process Instruction Information

### Novelty / anomaly detection
[2024] CLAN: A Contrastive Learning based Novelty Detection Framework for Human Activity Recognition




### Other HAR

[PerCom 2023] ALAE-TAE-CutMix+: Beyond the State-of-the-Art for Human Activity Recognition Using Wearable Sensors
- data augmentation; cross-channel interaction;

[IMWUT 2023] ConvBoost: Boosting ConvNets for Sensor-based Activity Recognition

[imwut 2023] DAPPER: Label-Free Performance Estimation after Personalization for Heterogeneous Mobile Sensing
- predict the performance of a model on new data

[CIKM 23] Unleashing the Power of Shared Label Structures for Human Activity Recognition

[IMWUT 23] TAO: Context Detection from Daily Activity Patterns Using Temporal Analysis and Ontology
- context detection

[IMWUT 23] MMTSA: Multi-Modal Temporal Segment Attention Network for Efficient Human Activity Recognition
- activity segmentation

[2023] Temporal Action Localization for Inertial-based Human Activity Recognition
- activity segmentation

【IMWUT 2022】Learning Disentangled Behaviour Patterns for Wearable-based Human Activity Recognition

【IMWUT 2020】Deriving Effective Human Activity Recognition Systems through Objective Task Complexity Assessment

【IMWUT 2020】Adversarial Multi-view Networks for Activity Recognition

[IMWUT 2020] KATN: Key Activity Detection via Inexact Supervised Learning

【IMWUT 2021】CrowdAct: Achieving High-Quality Crowdsourced Datasets in Mobile Activity Recognition

【IMWUT 2021】Attend and Discriminate: Beyond the State-of-the-Art for Human Activity Recognition Using Wearable Sensors
- exploit the latent relationships between multi-channel sensor modalities and specific activities, data-agnostic augmentation

【IMWUT 2019】The Positive Impact of Push vs Pull Progress Feedback: A 6-week Activity Tracking Study in the Wild


## App2: Gesture Recognition
[IMWUt 2023] DRG-Keyboard: Enabling Subtle Gesture Typing on the Fingertip with Dual IMU Rings

[TMC 2023] Fine-Grained and Real-Time Gesture Recognition by Using IMU Sensors

[PerCom 2023] CHAR: Composite Head-body Activities Recognition with A Single Earable Device
- earable device

[IMWUT 22] The OnHW Dataset: Online Handwriting Recognition from IMU-Enhanced Ballpoint Pens with Machine Learning


【IMWUT 2021】A CNN-based Human Activity Recognition System Combining a Laser Feedback Interferometry Eye Movement Sensor and an IMU for Context-aware Smart Glasses
novelty: combine Eye Movement Sensor and IMU, HAR,Smart glasses,transfer learning to personalize the classification


## App3: Gait Recognition

[2023] Domain Adaptation for Inertial Measurement Unit-based Human Activity Recognition: A Survey

IMU Sensing Data-Based Kinetic Tremor Detection in Parkinson's Disease Patients

Detecting Parkinsonian Tremor from IMU DataCollected In-The-Wild using Deep Multiple-Instance Learning
- dataset: https://zenodo.org/record/3519213 

## App4: Indoor localization / navigation
[TITS 2024] Deep Learning for Inertial Positioning: A Survey


## App5: Smart city applications
[2024] Transportation mode recognition based on low-rate acceleration and location signals with an attention-based multiple-instance learning network

[Ubicomp/ISWC 2023] Enhancing Transportation Mode Detection using Multi-scale Sensor Fusion and Spatial-topological Attention
- IMU + GNSS data

## App6: Device mode classification
[2023] IMU Dataset For Motion and Device Mode Classification
- pocket, fixed hand, swing hand, backpack

## App7: Other applications
[NDSS 2023] StealthyIMU: Stealing Permission-protected Private Information From Smartphone Voice Assistant Using Zero-Permission Sensors
- IMU, voice
- privacy attack

## Other modality data
[IMWUT 2023] RF-CM: Cross-Modal Framework for RF-enabled Few-Shot Human Activity Recognition
- wifi + mmwave

[PerCom 2023] Exposing the CSI: A Systematic Investigation of CSI-based Wi-Fi Sensing Capabilities and Limitations
- benchmark study of WiFi

[PerCom 24 workshop] Text me the data: Generating Ground Pressure Sequence from Textual Descriptions for HAR
- GPT generates pressure data

[PerCom 2023] hEARt: Motion-resilient Heart Rate Monitoring with In-ear Microphones
- heart rate sensing via earphone

[PerCom 2023] Joint Estimation of the Distance and Relative Velocity of Obstacles via Smartphone Active Sound Sensing for Pedestrian Safety
- Sound sensing

[PerCom 2023] EMGSense: A Low-Effort Self-Supervised Domain Adaptation Framework for EMG Sensing
- EMG signal; pretraining + augmentation

[Sensys 2021] OneFi: One-Shot Recognition for Unseen Gesture via COTS WiFi

[Sensys 2020] RF-net: a unified meta-learning framework for RF-enabled one-shot human activity recognition

[IMWUT 2020] DeepMV: Multi-View Deep Learning for Device-Free Human Activity Recognition
- wifi

[IMWUT 2020] CARIN: Wireless CSI-based Driver Activity Recognition under the Interference of Passengers

[IMWUT 2019] Personalized Context-aware Collaborative Online Activity Prediction
- a user-location-time-activity 4D-tensor and a location-time-POI 3D-tensor

[Sensys 19] RFID based real-time recognition of ongoing gesture with adversarial learning

[IMWUT 2021] Fall Detection via Inaudible Acoustic Sensing[Ubicomp 2019] Towards a Diffraction-based Sensing Approach on Human Activity Recognition
- wifi


