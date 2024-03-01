# Exploiting Model Intellectual Property Protection From a Training-Free Fingerprinting Strategy for Universal Tasks
We propose an efficient fingerprinting sample generation strategy, Meta-Fingerprint, which can be more comprehensively modeled as the fingerprint of the deep model. Moreover, considering both the category distribution and the confidence distribution, the generated samples are distributed within the stable region and near the decision boundary of the model. Finally, we introduce a heuristic sample perturbation algorithm, which generates a perturbed fingerprint with solid stability and generalization across multiple domains.

# Preparation
```
pip install -r requirements.txt
```

# How to run the code
## Dataset
```
1. First download the original data and put it in the ./data directory.

from ./util/data_adapter import SplitDataConverter
from ./util/data_adapter import defend_attack_split

2. Load the partitioned data set.
train, dev, test = SplitDataConverter().split("cifar10")

3. Split the trainset to defend and attack.
defenad, attack = defend_attack_split(train)
```

## Prepare source model
Our core code is placed in the ./src directory, where ./src/cv, ./src/nlp and ./src/bci correspond to computer vision, natural language processing and brain-computer interface codes respectively. It includes various methods of attacking fingerprint models.
```
python ./src/cv/source.py
python ./src/nlp/source.py --model_name bert --model_type source --epochs 3 --learning_rate 5e-5 --batch_size 128 --dataset_name thucnews
python ./src/bci/source.py --model_name ccnn --label valence --epochs 50 --lr 1e-4 --batch_size 64 --weight_decay 1e-4 --model_type source --dataset_name deap
```
## Attack the source model
The purpose of attacking the source model means to steal the performance of the source model and obtain a pirated model.
```
# cv examples
# IP removal attack
python ./src/cv/fine_tune.py
python ./src/cv/fine_prune.py
python ./src/cv/irrelevant.py
python ./src/cv/model_extract_l.py
python ./src/cv/model_extract_p.py
python ./src/cv/model_extract_adv.py
python ./src/cv/surrogate.py
python ./src/transfer_learning.py
# IP detection&removal attack
python ./src/cv/query_attack.py
```
## Extract Meta-Fingerprint.
The source model to be protected and its training set need to be provided.
```
cd ./model_ip/
python our.py

mf = MetaFingerprint(field="cv", model=source_model, dataset=train_dataset, device=device)
mf.generate_meta_fingerprint_point(20)
```

## Extract Perturbed-Fingerprint.
Add perturbations based on Meta-Fingerprint samples.
```
cd ./model_ip/
python our.py

pf = PerturbedFingerprint(field="cv", iters=20, lr=0.01)
pf.pfa_helper(source_model, 20)
```
## Compute AUC
```
cd ./model_ip/
# Meta-Fingerprint AUC without queryattack
fm = FingerprintMatch("cv", meta=True, device=device, ip_erase="original", n=20, lr=0.01, iters=20)
fm.dump_feature()
fm.fingerprint_recognition(verbose=True)
## Meta-Fingerprint AUC with queryattack
fm = FingerprintMatch("cv", meta=True, device=device, ip_erase="queryattack", n=20, lr=0.01, iters=20)
fm.dump_feature()
fm.fingerprint_recognition(verbose=True)

# Perturbed-Fingerprint AUC without queryattack
fm = FingerprintMatch("cv", meta=False, device=device, ip_erase="original", n=20, lr=0.01, iters=20)
fm.dump_feature()
fm.fingerprint_recognition(verbose=True)
# Perturbed-Fingerprint AUC with queryattack
fm = FingerprintMatch("cv", meta=False, device=device, ip_erase="queryattack", n=20, lr=0.01, iters=20)
fm.dump_feature()
fm.fingerprint_recognition(verbose=True)
```

# For more details, please focus our paper.
```
@inproceedings{xu2024intellectual,
    title={Exploiting Model Intellectual Property Protection From a Training-Free Fingerprinting Strategy for Universal Tasks},
    author={Tianhua Xu, Sheng-hua Zhong, Zhi Zhang, and Yan Liu},
    year={2024},
    note={ICML Reviewing},
}
```
