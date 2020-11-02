# ChatterNet:
Code and Sample Data for ChatterNet: KDD 2020

	Deep Exogenous and Endogenous Influence Combination for Social Chatter Intensity Prediction,
    -Subhabrata Dutta, Sarah Masud, Soumen Chakrabarti, Tanmoy Chakraborty
    ACM KDD 2020
[Paper Link](https://arxiv.org/abs/2006.07812)    

## Dataset: 
* Download the Reddit submissions dumps for the months of [October](https://files.pushshift.io/reddit/submissions/RS_2018-10.xz), [November](https://files.pushshift.io/reddit/submissions/RS_2018-11.zst), and [December](https://files.pushshift.io/reddit/submissions/RS_2018-12.zst), 2018, and put them under the directory `ChatterNet/Reddit_dumps/Submissions/` 
* Download the Reddit comments for the months of [October](https://files.pushshift.io/reddit/comments/RC_2018-10.xz),  [November](https://files.pushshift.io/reddit/comments/RC_2018-11.zxt), and [December](https://files.pushshift.io/reddit/comments/RC_2018-12.zxt), 2018, and put them under the directory `ChatterNet/Reddit_dumps/Comments/` 
* Download the [news articles](https://drive.google.com/open?id=1huDMFg6PKr8Q5Dz9Ms5Hp--NQ34VlO-L), extract, and put
them under the directory `ChatterNet/News_articles/`.

## Corpus preprocessing: 
* Codes under the directory `ChatterNet/Data_processing/’`preprocess the raw data into intermediate representations and finally numpy arrays to train the model. 

## Training and testing ChatterNet:
Codes under the directory `ChatterNet/Model_codes/` are used for training and testing the model.

## Baselines:
The instructions for running the baselines is in the `README` of the baseline folder.