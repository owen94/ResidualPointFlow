# ResFlowPC
Residual Flow Based generative model for point cloud generation 

- Motivation: 
  - Not too much work on point cloud generation 
  - Want higher resolution and quality pc than current state-of-the-art: PointFlow 
- Flow-base generative Model:
  - Main reference: 
    - Residual Flow Learning by Ricky, 2019 NeurIPS Spotlight 
    - Invertible ResNets, ICLR 2019 Oral 
    - FFJORD, ICLR 2019 Oral
    - Point Flow, ICCV Oral 2019

- Idea:
  - Use residual flow for better generative model 
  - Why:
    - Powerful resnet 
    - Unbiased log-determinant estimator 
    - Memory efficiency 
    - Hybrid training 
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
    - add args for latent resflow 
    - data loader and training loop 
    - understand some parts: actnorm, squeeze, moving batchnorm 
    - conditional flow learning 