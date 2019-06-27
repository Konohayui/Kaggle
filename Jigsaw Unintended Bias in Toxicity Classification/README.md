**Final Model**

Average of LSTM + Bert Small

**Loss Function**
```
def custom_loss(data, targets):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:,1:2])(data[:,:1],targets[:,:1])
    bce_loss_2 = nn.BCEWithLogitsLoss()(data[:,1:],targets[:,2:])
    
    return bce_loss_1 + bce_loss_2
```

**PAPERS**

* [ADAPTIVE GRADIENT METHODS WITH DYNAMIC BOUND OF LEARNING RATE](https://arxiv.org/pdf/1902.09843.pdf)
* [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/pdf/1801.06146.pdf)
* [Probing the Need for Visual Context in Multimodal Machine Translation](https://arxiv.org/pdf/1903.08678.pdf)
* [Accelerating Recurrent Neural Network Training using Sequence Bucketing and Multi-GPU Data Parallelization](https://arxiv.org/ftp/arxiv/papers/1708/1708.05604.pdf)
* [Temporal Convolutional Networks: A Unified Approach to Action Segmentation](https://arxiv.org/pdf/1608.08242.pdf)
* [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf)
* [Multi-Task Deep Neural Networks for Natural Language Understanding](https://arxiv.org/pdf/1901.11504.pdf)
* [BI-DIRECTIONAL ATTENTION FLOW FOR MACHINE COMPREHENSION](https://arxiv.org/pdf/1611.01603.pdf)


**BASELINE PERFORMANCES**
![Train loss](https://github.com/shenmemingzine/Kaggle/blob/master/Jigsaw%20Unintended%20Bias%20in%20Toxicity%20Classification/baseline/Train_Loss.png)
![Valid loss](https://github.com/shenmemingzine/Kaggle/blob/master/Jigsaw%20Unintended%20Bias%20in%20Toxicity%20Classification/baseline/Val_Loss.png)

