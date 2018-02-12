# Neural Collaborative Filtering

This is our implementation for the paper:

Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017). [Neural Collaborative Filtering.](http://dl.acm.org/citation.cfm?id=3052569) In Proceedings of WWW '17, Perth, Australia, April 03-07, 2017.

Three collaborative filtering models: Generalized Matrix Factorization (GMF), Multi-Layer Perceptron (MLP), and Neural Matrix Factorization (NeuMF). To target the models for implicit feedback and ranking task, we optimize them using log loss with negative sampling. 

**Please cite our WWW'17 paper if you use our codes. Thanks!** 

Author: Dr. Xiangnan He (http://www.comp.nus.edu.sg/~xiangnan/)


Run GMF:
```
python GMF.py --dataset vtc_cab --epochs 20 --batch_size 64 --num_factors 10 --regs [1e-5,1e-5] --num_neg 5 --lr 0.00001 --learner rmsprop --verbose 1 --out 1
```

Run MLP:
```
python MLP.py --dataset  vtc_cab  --epochs 20 --batch_size 256 --layers [64,32,16,8] --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1
```

Run NeuMF
```
python NeuMF.py --dataset  vtc_cab  --epochs 20 --batch_size 256 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1
```

Run NeuMF (with pre-training):
```
python NeuMF.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 8 --layers [64,32,16,8] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1 --mf_pretrain Pretrain/ml-1m_GMF_8_1501651698.h5 --mlp_pretrain Pretrain/ml-1m_MLP_[64,32,16,8]_1501652038.h5
```

Note on tuning NeuMF: our experience is that for small predictive factors, running NeuMF without pre-training can achieve better performance than GMF and MLP. For large predictive factors, pre-training NeuMF can yield better performance (may need tune regularization for GMF and MLP). 

### Dataset
The data contains 3 file
vtc_cab.train.rating contains pair of user-item and number of purchase
vtc_cab.test.rating contain test pair of user-item
vtc_cab.test.rating contain test pair of user0item and negative samples
###Note
NeuMF is overfited with this data so just use GMF the number of factor is 10 and hit-rate is about 0.6
to see sample prediction run python neuMF_predict.py