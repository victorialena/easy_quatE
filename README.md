## Heterogenous Label Prediction: OGB-biokg

This experiment is based on stanford OGB (1.2.1) benchmark. The description of 《Implementing & Extending Quaternion Knowledge Graph Embeddings》 is [avaiable here]().

### Note!
We propose the following changes to further improve our model's performance:

To_do list:
- [x] QuadE De-regularization
- [ ] QuatE-L1Reg
- [x] QuatE-STAGED
- [x] QuatE-RotatE

### Install environment:
``` 
    git clone https://github.com/PaddlePaddle/PGL.git
    pip install ogb    
```

Requirements: OGB the version is 1.3.0. and python 3.6+

### OGB dataset:
For this project, we selected the [ogbl-biokg]{https://ogb.stanford.edu/docs/linkprop/\#ogbl-biokg} biomedical knowledge graph (KG). This knowledge graph, which consists of 93K+ entities and 5M+ relations, was generated from a large number of biomedical data repositories. The dataset includes 5 entity types -- diseases (10,687 entities), drugs (10,533 entities), side effects (9,969 entities), proteins (17,499 entities), and protein functions (45,085 entities). The dataset also features 51 directed relation types, including 39 drug-drug relations and 8 protein-protein relations, along with drug-protein, drug-side effect, drug-protein, and function-function relations. Relations connecting the same entity types (e.g., protein-protein, drug-drug, function-function) are always symmetric. 
  
### The **detailed hyperparameter** is:

```
Arxiv_dataset(Batch): 
--batch_size        1024
--neg_smaple_size   512
--hidden_size       500
--lr                1e-5  
--n_epoch           5 
--lambda_e          [0.01, 0.0]
--lambda_r          [0.05, 0.0]
--gamma             12.
```
Hardware : GeForce RTX 208

### Reference performance for OGB:
|           |                            Test                    |                                                    |
| Model     | MRR        | Accuracy   | Precision    | Recall    |   MRR      | Accuracy   | Precision    | Recall    |
| ----------|----------- | -----------| -------------|-----------|----------- | -----------| -------------|-----------|
| QuatE     | 0.7652     |  0.5690    |   0.6126     |  0.4585   | 0.7995     |  0.5990    |   0.6391     |  0.5982   |
|     -DeReg| 0.8996     |  0.8626    |   0.6663     |  0.9959   | 0.9396     |  0.8793    |   0.8189     |  0.9980   |
|    -Staged| 0.8954     |  0.8345    |   0.6852     |  0.9885   | 0.9382     |  0.8684    |   0.8265     |  0.9912   |   
