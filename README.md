# MetaAdv
### Platform
* Python: 3.7
* PyTorch: 1.5.0
### Dataset
We use the benchmark dataset MiniImageNet, which can be download [here](https://drive.google.com/file/d/1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk/view)
### Model
We use a four-layer-conv NN model
## Standard MAML training
Run MAML_TrainStd.ipynb, associate files include MAMLMeta.py, attack.py, learner.py  (Attack power level has to be changed in MAMLMeta.py)
## MAML + FGSM-RS (random start)
Run trainfgsmrs.ipynb, associate files include metafgsm.py, attack.py, learner.py. To incorporate advasarial training in the inner-loop, please replace metafgsm.py with metafgsminout.py
## MAML + TRADES-RS
Run
