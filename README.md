# Diabetic Retinopathy Detection

This repository showcases the work I did for the [APTOS 2019 Blindness Detection Challenge](https://www.kaggle.com/c/aptos2019-blindness-detection/) on Kaggle. My final submission won a silver medal on Kaggle and the National Runners Up for the Philips Code to Care Challenge 2019.

The aim was to build a model to detect Diabetic Retinopathy from Fundus Photographs of the retina. The ouputs are integers between 0 to 4, with 0 indicating no disease and 4 indicating Proliferative DR. The evaluation criterion was quadratic weighted Cohen's Kappa Score. More detailed description can be found on Kaggle(linked above).


# Description

I tried several approaches. Relevant notebooks for these, along with their descriptions are in separate directories. The notebooks present on the root directory of this repository were used to train the best submission. The Kaggle kernel used for the best submission is [here](https://www.kaggle.com/atmadeepb/best-submission).

The best submission used regression two regression models. One was trained with smooth L1 loss and the other was trained using MSE(L2 loss). The final output was produced by averaging the output of these models. The real valued ouputs were converted to class predictions using optimized boundaries. These optimized boundaries were first obtained using Nelder Mead optimization of the Kappa Score metric on validation set. Later they were hand tuned to perform optimally on the public test set.
