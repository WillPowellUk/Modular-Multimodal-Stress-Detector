# Truly Modular Multimodal Stress Detection from Biosensor Measurements

Welcome to the **Modular Multimodal Stress Detection System**! 
This software accomanies my MsC project titled: Rethinking Stress Monitoring: Convenient Modular Early-Onset Multimodal Stress Detection with Attention Score Caching available [here](https://willpowell.uk/files/William_Powell_MSc_Thesis.pdf).

## 🪄 The Magic

We implement an architecture that has a:

- **Modular BCSA Mechanism**: Combines bidirectional cross-modal and self-attention networks to align multimodal sensor data and reduce noise, transforming each modality pair into an intermediate representation that enables any number and combination of modality to be used.  
- **Novel Bidirectional Encoder-Decoder Attention**: Enhances temporal context by allowing embeddings from past signals to re-attend to new data, improving prediction accuracy by dynamically updating the significance of previous information.
- **Sliding Attention Score Caching Mechanism**: Caches attention scores and projections to enable efficient real-time sequence-to-sequence predictions without recomputing past data.  
- **Predictor**: Fuses modality-specific outputs using ensemble methods, including hard voting, soft voting, and Kalman filter for real-time temporal updates.

The model supports both unimodal and multimodal inputs, adapting dynamically by inferring only relevant cross-attention blocks, optimizing latency for real-time applications.

## Key Features

### 🎧 **Mastoid-based sEMG & EEG Integration**
- Discreet, headphone-based stress monitoring.
- Measures muscle and brain activity through the mastoid area.
- 🧠 sEMG and EEG biomarkers help identify stress but are subject to noise challenges.

### 💡 **Pre-frontal Cortex fNIRS (Functional Near-Infrared Spectroscopy)**
- A portable, consumer-friendly modality for monitoring brain oxygenation.
- Highly predictive features but sensitive to individual variability.
  
### 📈 **Modular, Multimodal Architecture**
- Adaptable to various device combinations: sEMG, EEG, and fNIRS.
- Ensemble of cross-attention and self-attention blocks ensure temporal alignment of signals from different devices.
- Minimal performance drop (-2%) when switching between modalities or using only specific ones.

### 🚀 **High Efficiency with Sequence-to-Sequence Attention**
- Reduced stress detection window size by 5x using attention score caching.
- Optimized for faster processing with minimal computational complexity.

### 🧠 **Personalization Capabilities**
- Personalization techniques to mitigate participant variability.
- Tailored learning for improved accuracy and reduced computational overhead.

### 🛠 **Dataset & Testing**
- **MUSED Dataset**: Novel dataset for stress detection with innovative device integration.
- Tested on well-known datasets like **WESAD** and **UBFC-Phys**, ensuring robust performance across various stress monitoring modalities.

### 📊 **Performance Highlights**
- Outperforms state-of-the-art methods with minimal computation.
- True modularity between modalities with high generalization performance across different sensor configurations.

---

## Getting Started

### Installation
To start, create a new virtual enviroment or conda environment and install the required packages:
```
pip install -r requirements.txt
```
You may have to download and setup CUDA related packages (such as those linked with PyTorch) specifically for your device.

### Demo
To run the demo, view the [wesad_deom.ipynb](wesad_demo.ipynb) which takes you through downloading the WESAD dataset and running the whole pipeline. 


## Repository Structure

The structure of this repository is designed for easy modular implementation in new pipelines. 
It is structured as follows:

```
├── config_files
│   ├── dataset
│   └── model_training
│       ├── deep
│       └── traditional
├── data_collection
│   ├── crop_times
│   ├── data_visualization
│   ├── recordings
├── experiments
│   ├── ablation_studies
│   ├── manual_fe
│   ├── parameter_comp
│   └── results
│       ├── analysis
│       ├── data
│       └── graphs
├── src
│   ├── ml_pipeline
│   │   ├── analysis
│   │   ├── data_fusion
│   │   ├── data_loader
│   │   ├── decision_fusion
│   │   ├── feature_extraction
│   │   ├── feature_fusion
│   │   ├── feature_selection
│   │   ├── losses
│   │   ├── models
│   │   ├── preprocessing
│   │   ├── train
│   │   └── utils
│   ├── mused
│   │   ├── data_augmentation
│   │   ├── dataloader
│   │   ├── data_preprocessing
│   │   ├── data_processing
│   │   ├── dataset
│   │   ├── dataset_pkl
│   │   ├── datasets_cleaned
│   │   └── feature_analysis
│   ├── ubfc_phys
│   ├── utils
│   └── wesad
│       ├── data_preprocessing
│       ├── signal_preprocessing
│       └── WESAD
└── submodules
    ├── bleakheart
    └── pyEDA
```

## Colaboration & Problems
We welcome any improvements or use to our system 
We welcome any questions or fixes (please a new raise issue).

# License ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## MIT License

```text
MIT License

Copyright (c) [YEAR] [OWNER]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
