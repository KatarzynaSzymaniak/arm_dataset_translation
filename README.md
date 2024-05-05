## It’s GREAT: Gesture REcognition for Arm Translation.

Myoelectric control has emerged as a promising approach for a wide range of applications, including controlling limb prosthetics, teleoperating robots and enabling immersive interactions in the metaverse. However, the accuracy and robustness of MEC systems are often affected by various factors, including muscle fatigue, perspiration, drifts in electrode positions and changes in arm position. The latter has received less attention despite its significant impact on signal quality and decoding accuracy. To address this gap, we present GREAT, a novel dataset of surface electromyographic (EMG) signals captured from multiple arm positions. This dataset, comprising EMG and hand kinematics data from 8 participants performing 6 different hand gestures, provides a comprehensive resource for investigating position-invariant Myoelectric control decoding algorithms. We envision this dataset to serve as a valuable resource for both training and benchmarking arm position-invariant Myoelectric control algorithms. Additionally, to further expand the publicly available data capturing the variability of EMG signals across diverse arm positions, we propose a novel data acquisition protocol that can be utilized for future data collection.


Data collection protocol: https://github.com/MoveR-Digital-Health-and-Care-Hub/posture_dataset_collection 

#### Repository includes benchmarking and initial analysis of the dataset from paper.
 In the at_source/exp you will find the example runs of the experiemnt. 


##### Venv Packages included:
    Package                  Version   
    ------------------------ ----------
    cmake                    3.27.0    
    contourpy                1.1.0     
    cycler                   0.11.0    
    filelock                 3.12.2    
    fonttools                4.42.0    
    glob2                    0.7       
    h5py                     3.9.0     
    importlib-resources      6.0.1     
    Jinja2                   3.1.2     
    joblib                   1.3.1     
    kiwisolver               1.4.4     
    lit                      16.0.6    
    MarkupSafe               2.1.3     
    matplotlib               3.7.2     
    mpmath                   1.3.0     
    natsort                  8.4.0     
    networkx                 3.1       
    numpy                    1.24.4    
    nvidia-cublas-cu11       11.10.3.66
    nvidia-cuda-cupti-cu11   11.7.101  
    nvidia-cuda-nvrtc-cu11   11.7.99   
    nvidia-cuda-runtime-cu11 11.7.99   
    nvidia-cudnn-cu11        8.5.0.96  
    nvidia-cufft-cu11        10.9.0.58 
    nvidia-curand-cu11       10.2.10.91
    nvidia-cusolver-cu11     11.4.0.1  
    nvidia-cusparse-cu11     11.7.4.91 
    nvidia-nccl-cu11         2.14.3    
    nvidia-nvtx-cu11         11.7.91   
    packaging                23.1      
    pandas                   2.0.3     
    Pillow                   10.0.0    
    pip                      20.0.2    
    pkg-resources            0.0.0     
    pyparsing                3.0.9     
    python-dateutil          2.8.2     
    pytz                     2023.3    
    PyYAML                   6.0.1     
    scikit-learn             1.3.0     
    scipy                    1.10.1    
    setuptools               44.0.0    
    six                      1.16.0    
    sympy                    1.12      
    threadpoolctl            3.2.0     
    torch                    2.0.1     
    torchaudio               2.0.2     
    triton                   2.0.0     
    typing-extensions        4.7.1     
    tzdata                   2023.3    
    wheel                    0.41.0    
    zipp                     3.16.2 
