## FLEX-Faithful-Linguistic-Explanations-for-Neural-Net-Based-Model-Decisions
Tensorflow implementation of [FLEX: Faithful Linguistic Explanations for Neural Net Based Model Decisions](https://www.aaai.org/ojs/index.php/AAAI/article/view/4100) published in AAAI 2019

## Abstract

Explaining the decisions of a Deep Learning Network is imperative to safeguard end-user trust. Such explanations must be intuitive, descriptive, and faithfully explain why a model makes its decisions. In this work, we propose a framework called FLEX (Faithful Linguistic EXplanations) that generates post-hoc linguistic justifications to rationalize the decision of a Convolutional Neural Network. FLEX explains a model’s decision in terms of features that are responsible for the decision. We derive a novel way to associate such features to words, and introduce a new decision-relevance metric that measures the faithfulness of an explanation to a model’s reasoning. Experiment results on two benchmark datasets demonstrate that the proposed framework can generate discriminative and faithful explanations compared to state-of-the-art explanation generators. We also show how FLEX can generate explanations for images of unseen classes as well as automatically annotate objects in images. 

## Getting Started

1. Clone the repo.

2. Set up the enviornment. This repo uses ``` python 3.5 ```. Install required packages, 

   ```pip install -r requirements.txt```

   This code demonstrates the flex model developed to explain decisions of Compact-Bilinear Pooling classifier by [Yang Gao](https://arxiv.org/abs/1511.06062). The classifier is trained with Caffe. Thus, if you want generate image features or derive decision relevant words using this code by yourself, you need to set up Caffe too. Please use the version specified in [Yang Gao's repo](https://github.com/gy20073/compact_bilinear_pooling/tree/master/caffe-20160312). 
   Download Compact-Bilinear Pooling classifier trained weights from [here](https://drive.google.com/file/d/1fFNu1h3okT4K5KPWSPD4jOVtOly6Utab/view?usp=sharing) and place inside ```classifiers/cmpt-bilinear``` folder.

3. Download data.
   Download all the required to train or use a flex model on [CUB dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) from [here](https://drive.google.com/file/d/1Ft9zz__7L_MUMxDlEttLaqqMTIqhLZqg/view?usp=sharing). Unzip to ```data/cub``` folder. 
   
   ```unzip flex_cub_data.zip -d data/cub/```
   
   This zip file contains decision relevance vectors for train and val datasets as well as features for train, validation and test cub images obtained from compact-bilinear classifer.
   
4. Download trained flex model.
   Download trained flex model for cub dataset from [here](https://drive.google.com/file/d/1RzFiRFpk8sZW5KBEpdSgVGJlFJvEIDQN/view?usp=sharing) and unzip.
   
   ```unzip flex_cub_trained_model.zip -d trained_models/cub/```

## Generate Explanations

   To generate explanations using the trained model run following code.
   
   ```python generate_explanations.py --model_version flex_v1```
   
   Generated explanations will be written to a text file in ```trained_models/cub/<<model_version>>``` folder.
  
  
 ## Train FLEX for CUB Dataset
 
   If you want to change hyper parameters, please edit ```params_cup.py``` file. To train FLEX for cub run,
   
   ```python train_flex_cub.py```
   
   The trained weights will be saved to ```trained_models/cub/<<model_version>>```.
   
   
  ## Train FLEX for Another Dataset
  
  1. Create a ``` params.py ``` with the information of new dataset and the classifier
  2. ``` $ python prepare_data.py ``` to create text data dictionary.
  3. ``` $ python identify_important_words.py ``` to indentify important words for each image. 
  4. ``` $ python relevant_score_calculator.py ``` to create train and val relevance score vectors.
  5. ``` $ python extract_visual_features.py ``` to create train, val and test image features.
  6. Create a ``` train_flex.py ``` script and import ``` params ```.
  7. ``` $ python train_flex.py ```


## Citation
```@inproceedings{wickramanayake2019flex,
  title={FLEX: Faithful Linguistic Explanations for Neural Net Based Model Decisions},
  author={Wickramanayake, Sandareka and Hsu, Wynne and Lee, Mong Li},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={2539--2546},
  year={2019}
}
```
