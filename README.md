# MetaAdv
### Platform
* Python: 3.7
* PyTorch: 1.5.0
### Dataset
We use the benchmark dataset MiniImageNet, which can be download [here](https://drive.google.com/file/d/1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk/view)
### Model
We use a four-layer-conv NN model
## Standard MAML training
Run MAML_TrainStd.ipynb, associate files include MAMLMeta.py, attack.py, learner.py
* Attack power level has to be changed in MAMLMeta.py
* The device in MAML_TrainStd.ipynb and attack.py should set to be the same. (same in the following adversarial training)
## MAML + FGSM-RS (random start)
Run trainfgsmrs.ipynb, associate files include metafgsm.py, attack.py, learner.py. To incorporate adversarial training in the inner-loop, please replace metafgsm.py with metafgsminout.py
## MAML + TRADES-RS
Run train_trade.ipynb, associate files include MetaFT.py, LoadUnlableData.py. The unlabled data can be downloaded from [here](https://drive.google.com/file/d/1QpEQFDC8SGoek6k20YFCksKbWEh6j5ei/view?usp=sharing)
## Standard training + few-short fine-tuning (Meta-tesing)
Run StandardTransNew.ipynb, associate files include LoadDataST.py, StandardTrans.py. StandardTransAdv.ipynb contains adversarial training in the model training process.
## Unlabeled data selection
Run figureselection.ipynb, associate files include 

## Visualization
Run robust_vis_neuron.ipynb, associate files include Visualization.py, vis_tool.py, MODELMETA.py.
* By maximizing the output of a nueron with a perturbation in th input, the feature is shown in the input under a robust model, while "random noise" is shown in the input under a standard MAML model.
* The fine-tuned model has the similar feature to the original model in the same neuron. This suggests that the robustness is kept in the fine-tuned model even without adding the adversarial training in the fine-tuning.
