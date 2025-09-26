# DE-GAN -- Document Enhancement using Generative Adversial Nets using Pytorch  

This work is an extension and reimplementation of whole DE-GAN in pytorch and All DIBCO datasets are used to train and tested on ISI Bengali dataset. As the Ground Truths are not available for ISI Bengali Dataset, we used Pytesseract for OCR for comparsion of predicted and degraded ones. By using this GAN, Character Error Rate has improved by 40% and Word Error Rate has improved by 20%.

# citation 
if this work is useful, please cite it as:

@ARTICLE{Souibgui2020,
  author={Mohamed Ali Souibgui and Yousri Kessentini},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={DE-GAN: A Conditional Generative Adversarial Network for Document Enhancement},
  year={2020},
  doi={10.1109/TPAMI.2020.[#######]}
}
