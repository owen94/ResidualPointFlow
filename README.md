# Residual PointCloud Flow Learning 
Residual Flow Based generative model for point cloud generation 

- Motivation: 
  - Not too much work on point cloud generation which is useful for multiple downstream tasks 
  - Want better-quality pc than current state-of-the-art: PointFlow 
- Flow-base generative Model:
  - Main reference: 
    - Residual Flow Learning by Ricky, 2019 NeurIPS Spotlight 
    - Invertible ResNets, ICLR 2019 Oral 
    - FFJORD, ICLR 2019 Oral
    - Point Flow, ICCV Oral 2019

- Idea:
  - Use residual flow for better generative model 
  - Why:
    - Powerful resnet with conv1d  **v.s.** 3-layer linear model in PointFlow 
    - Unbiased log-determinant estimator v.s. biased ones iResnet
    - Hybrid training to leverage label info (TBD)
    - ... 
    
  
- Sanity check procedure 
    - conv1d block: code in layers.base.lipschitz1D, tested in main function.  -- done 
        - Spectral normalization inside the implementation, using power iteration to compute the weights before feeforward 
        - Compute weights for 1x1 or kxk kernals. 
    - linear1d block   --done 
        - removed the domains and codomains as they are not necessary for better performance 
    - invertible residual block with conv1d or linear1d  -- done 
        - initial_size = [n, c, l]
    - reflow with stacked resflow 
    
- TO DO List
    1. experiments with vanilla residual point flow 
    2. implement the conditional residual point flow model: concatenate the latent representation learnt by the encoder to the second conv-layer in iResBlock 
      - A typical iResBlock taks as input (n, c, l) -> 1x1 conv1d -> 3x3 conv1d **(concatenate latent features here)** -> 1x1 conv1d 
      - **Question**: only concatenate in the first block or every block? 
    
    3. extensive experiments 
    4. writeup 



### How to run the code 

####Dataset 

The point clouds are uniformly sampled from meshes from ShapeNetCore dataset (version 2) and use the official split. Download through this [link](https://drive.google.com/drive/folders/1G0rf-6HSHoTll6aH7voh-dXj6hCRhSAQ?usp=sharing)  - credit to Yant et al, PointFlow. 

```markdown
cd ResidualPointFlow/
mkdir data 
mv ShapeNetCore.v2.PC15k.zip data/
cd data
unzip ShapeNetCore.v2.PC15k.zip
```

#### Dependencies 

```markdown
pytorch = 1.2 
tqdm 
numpy
TensorboardX
cuda = 10.0 
```

#### Training

```python
python train.py 
```








