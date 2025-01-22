# Research on Chest X-rays

This research focuses on **Multi-label classification** of Chest X-ray images with the help of various **Deep Learning models** like DenseNet121, MobileNetV2, etc; to predict multiple lung diseases which include Emphysema, Hernia, Pneumonia, Cardiomegaly, and many more.

It also addresses the problem of **Class Imbalance** with the help of techniques like **Undersampling** and **Oversampling** in the dataset which itself is **challenging** as undersampling removes instances from the majority class, and oversampling adds copies of instances from the minority class to the dataset. Thus, the lack of a clear definition of minority and majority class instances prevents a straightforward application of these two methods. [Paper](https://www.sciencedirect.com/science/article/pii/S0950705124009894)

Also, **CLAHE Enhancement** is applied to reduce noise and improve image quality. Finally after training, `DenseNet121` achieved the best performance among all the models evaluated using various metrics.

## Dataset
The ChestX-ray14 dataset is **one of the largest** collections of anterior-view pulmonary X-ray scan data. For this research, a total of **112,120** bilateral thoracic X-ray data were obtained from the dataset. The ChestX-ray14 dataset contains data of **14 disease classes** and a **no finding class** from **30805 unique patients**. These disease classes are Emphysema, Infiltration, Mass, Pleural Thickening, Pneumonia, Pneumothorax, Atelectasis, Edema, Effusion, Hernia, Cardiomegaly, Pulmonary Fibrosis, Nodule, and Consolidation. `No finding` means the 14 listed disease patterns are not found in the image.

## Preprocessing Steps
The following steps were taken for preprocessing:

### CLAHE Enhancement

| Without CLAHE | With CLAHE |
| --- | --- |
| ![](https://github.com/user-attachments/assets/95d15c98-1185-44a0-b579-5b8c7aafaf02) | ![](https://github.com/user-attachments/assets/3e6c2ad8-62fc-4887-ba3d-cfd830056343) |

Contrast-limited adaptive histogram equalization (CLAHE) applied on all images to **reduce noise** and **improving image quality**. The tile size and clip limit are critical hyper-parameters for this method. An incorrect selection of hyper-parameters could have a big influence on the image quality. The optimal ones `tileGridSize (10, 10)` and the `clip limit (3)` were chosen here. 

### Data Balancing

| Before| After|
| --- | --- |
| ![before preprocessing](https://github.com/user-attachments/assets/0504325e-b125-4704-908b-e9cdf945dd6a) | ![after preprocessing](https://github.com/user-attachments/assets/1eda7c99-6045-4d46-b8f3-69dc9bb491f5) |



As you can see in the above bar chart that the dataset is highly imbalanced as the `No Finding` class contains 60361 images which is about **53% of the whole dataset** whereas `Hernia` contains only 227 images. And this imbalanced dataset further results in a high biased model predictions and poor performance for minority classes. So it is very important to first balance the dataset with the help of techniques like **Undersampling** and **Oversampling** to make it balanced then proceed for training. 

The following steps were taken to balance the dataset:
- As the dataset was highly imbalanced, an **Undersampling** technique was applied by **Randomly Removing** examples from over-represented classes to balance the dataset. This was done for each class individually, ensuring fairness in a multi-label setting.
- After undersampling , **Oversampling** of dataset was done by using **Data Augmention** with the help of [Albumentations](https://github.com/albumentations-team/albumentations) library to increase diversity. The following transformations were applied to the images:
   - HorizontalFlip
   - RandomRotate90
   - VerticalFlip
   - Sharpen (with different probability of transform after each iteration)

    ![](https://github.com/user-attachments/assets/c741c793-4ead-43f7-99ab-5e8ebd604fb2)


- Finally, all images were resized to **224x224** pixels to ensure uniform input dimensions for the model, optimizing performance and compatibility with the chosen architecture.

These steps helped both balance the dataset and also resizing it for further training of our model.
 
 ## Model Training
 The model training was done on **Kaggle's TPU VM v3-8** with 330GB of RAM.
   - Trained several deep learning models with a `70:15:15` split as the `train:validation:test` set which includes:
     - AlexNet
     - DenseNet121
     - InceptionResNetv2
     - MobileNetV2
   - Used `binary_crossentropy` as a multi-label loss function, with `sigmoid` as activation and `Adam` as its optimizer.
   
   - Various **Evaluation Metrics** were calculated besides `Accuracy` and `Loss` to assess overall performance which includes:
      - AUC
      - F1-score
      - Precision
      - Recall
   
## Results
Finally, after training, `DenseNet121` achieved the **best performance** among all the models evaluated using various metrics.

## Problems
When finalizing this work, I read this [Kaggle's discussion](https://www.kaggle.com/datasets/nih-chest-xrays/data/discussion/300917) which redirects to a [Blog post](https://laurenoakdenrayner.com/2017/12/18/the-chestxray14-dataset-problems/) by a radiologist stating that this dataset is in fact has some problems like:

- Compared to human visual assessment, the labels in the ChestXray14 dataset are inaccurate, unclear, and often describe medically unimportant findings.

- These label problems are internally consistent within the data, meaning models can show good test-set performance, while still producing predictions that don’t make medical sense.

- The above combination of problems mean the dataset as defined currently is not fit for training medical systems, and research on the dataset cannot generate valid medical claims without significant additional justification.

## Acknowledgments
I read several research papers to gain insights into different methodologies used in medical research involving deep learning:

- [ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases, Xiaosong Wang et al](https://arxiv.org/abs/1705.02315)
- [High-precision multiclass classification of lung disease through customized MobileNetV2 from chest X-ray images, F.J.M. Shamrat et al](https://www.sciencedirect.com/science/article/pii/S0010482523001117)
- [SynthEnsemble: A Fusion of CNN, Vision Transformer, and Hybrid Models for Multi-Label Chest X-Ray Classification, S.M. Nabil Ashraf et al](https://arxiv.org/abs/2311.07750)
- [Automated Classification of Lung Cancer Subtypes Using Deep Learning and CT-Scan Based Radiomic Analysis, Bryce Dunn et al](https://www.researchgate.net/publication/371388434_Automated_Classification_of_Lung_Cancer_Subtypes_Using_Deep_Learning_and_CT-Scan_Based_Radiomic_Analysis)
- [Improving Brain Tumor Classification: An Approach Integrating Pre-Trained CNN Models and Machine Learning Algorithm, M.R. Shoaib, J. Zhao, H.M. Emara et al](https://www.sciencedirect.com/science/article/pii/S2405844024095021)
- [Multi-Class Classification of Brain Disease using Machine Learning-Deep Learning approaches and Ranking based Similar Image Retrieval from Large Dataset, Dr. Dhanraj R.Dhotre et al](https://www.ijisae.org/index.php/IJISAE/article/view/3550)
- [LungNet22: A Fine-Tuned Model for Multiclass Classification and Prediction of Lung Disease Using X-ray Images, F. M. Javed Mehedi Shamrat et al](https://www.mdpi.com/2075-4426/12/5/680)
- [Enhanced diagnostic accuracy for multiple lung diseases using a fine-tuned MobileNetV2 model with advanced pre-processing techniques, Deepak Thakur et al](https://www.sciencedirect.com/science/article/abs/pii/S0957417424021390)

## License
MIT

