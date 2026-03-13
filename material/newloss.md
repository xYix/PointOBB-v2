由于任务是点监督目标检测（point-supervised object detection），因此训练中只能使用 gt bbox 的中心点信息。
目前的 JS loss 需要 gt bbox 的整体信息，因此不能使用，那么应当如何设计 loss 来训练回归分布