### Text Classification

All kinds of neural text classififers implemented by Keras (tensorflow backend).

### Models

- [TextCNN, EMNLP2014](https://www.aclweb.org/anthology/D14-1181)  
Kim et al. Convolutional Neural Networks for Sentence Classification.

- [DCNN, ACL2014](http://www.aclweb.org/anthology/P14-1062)  
Kalchbrenner et al. A Convolutional Neural Network for Modelling Sentences

- [RCNN, AAAI2015](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552)  
Lai et al. Recurrent Convolutional Neural Networks for Text Classification.

- [HAN, NAACL-HLT2016](http://www.aclweb.org/anthology/N16-1174)  
Yang et al. Hierarchical Attention Networks for Document Classification.

- [DPCNN, ACL2017](https://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf)  
 Johnson et al. Deep Pyramid Convolutional Neural Networks for Text Categorization.

- [VDCNN, EACL2017](http://www.aclweb.org/anthology/E17-1104)  
Conneau et al. Very Deep Convolutional Networks for Text Classification.

- MultiTextCNN  
Extension of textcnn, stacking multiple cnns with the same filter size.

- BiLSTM  
Bidirectional lstm + max pooling over time.

- RNNCNN  
Bidirectional gru + conv + max pooling & avg pooling.

- CNNRNN  
conv + max pooling + Bidirectional gru + max pooling over time.
