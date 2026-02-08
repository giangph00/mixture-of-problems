Receptive field theory
- Understanding the Effective Receptive Field in Deep CNNs (Luo, Li, Urtasun). Shows that the effective receptive field is smaller and Gaussian-like, with implications for depth and kernel choices. 
https://arxiv.org/abs/1701.04128
 

- Feature Visualization (Olah, Mordvintsev, Schubert). Visual/interactive tour of what conv nets attend to, connecting receptive fields to emergent features. 
https://distill.pub/2017/feature-visualization/

- CS231n Convolutional Neural Networks for Visual Recognition (Stanford). Lecture notes explaining receptive fields through kernels, stride, and pooling. 
https://cs231n.github.io/convolutional-networks/#conv
 

Saliency maps: math & theory
- Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps (Simonyan et al.). Derives gradient-based saliency as ∂score/∂pixels and shows class-sensitivity. 
https://arxiv.org/abs/1312.6034

- Attribution Baselines (Distill). Explains baseline choices and their mathematical consequences for attributions.
https://distill.pub/2020/attribution-baselines

Week 5: Saliency Mapping (lecture video). Conceptual walkthrough of deconvnet, guided backprop, and Grad-CAM rationale.
https://www.youtube.com/watch?v=pYaAMx_GfH0 

XAI Methods — Saliency (blog). Discusses vanilla gradient saliency foundations and interpretive caveats clearly.
https://erdem.pl/2022/02/xai-methods-saliency

Class Activation Maps & Grad-CAM
- Learning Deep Features for Discriminative Localization (Zhou et al., CAM). Introduces CAMs and their limitation to global-avg-pool architectures. 
https://arxiv.org/abs/1512.04150

Other interpretability techniques (vision)
- Distill: The Building Blocks of Interpretability. Conceptual primitives for interpreting deep visual representations. 
https://distill.pub/2018/building-blocks/

Network Dissection (project page). Quantifies unit–concept alignment for objects, parts, textures in CNNs.
[ Link netdissect.csail.mit.edu](https://netdissect.csail.mit.edu/)

