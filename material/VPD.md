**An overview of the proposed VPD framework.** A standard network for pedestrian detection employs a classification and a regression branch to obtain the confidence and localization of each proposal. Based on this framework, **VPD** develops a **reparameterization module** to perform **variational inference** on each proposal and a **statistic decoder** bridging the classification and regression branches to enhance the confidence estimation.

VPD approximates the true posterior of the proposal under two constraints:

1. **Maximizing the similarity** between the proposal distribution and its ground-truth prior measured by **JS divergence**.
2. **Maximizing the detection likelihood** of the proposal generated from the variational inferred distribution measured by common regression metrics (e.g., $smooth_{L1}$, $IoU$, $GIoU$, etc.).

These constraints encourage the detector to learn a robust and accurate bounding box of the object.