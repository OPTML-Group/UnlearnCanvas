# Arbitrary Video Style Transfer via Multi-Channel Correlation
Yingying Deng, Fan Tang, Weiming Dong, Haibin Huang, Chongyang Ma, Changsheng Xu  <br>

## Results presentation 
<p align="center">
<img src="https://github.com/diyiiyiii/MCCNet/blob/main/results.png" width="100%" height="100%">
</p>
Visual comparisons of video style transfer results. The first row shows the video frame stylized results. The second row
shows the heat maps which are used to visualize the differences between two adjacent video frame.  <br>


## Framework
<p align="center">
<img src="https://github.com/diyiiyiii/MCCNet/blob/main/network.png" width="80%" height="80%">
</p> 
Overall structure of MCCNet. <br>




## Experiment
### Requirements
* python 3.6
* pytorch 1.4.0
* PIL, numpy, scipy
* tqdm  <br> 

### Testing 
Pretrained models: [vgg-model](https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view?usp=sharing),  [decoder],  [MCC_module](see above)   <br> 
Please download them and put them into the floder  ./experiments/  <br> 
```
python test_video.py  --content_dir input/content/ --style_dir input/style/    --output out
```
### Training  
Traing set is WikiArt collected from [WIKIART](https://www.kaggle.com/c/painter-by-numbers )  <br>  
Testing set is COCO2014  <br>  
```
python train.py --style_dir ../../datasets/Images --content_dir ../../datasets/train2014 --save_dir models/ --batch_size 4
```
### Reference
If you use our work in your research, please cite us using the following BibTeX entry ~ Thank you ^ . ^. Paper Link [pdf](coming soon)<br> 
```
@inproceedings{deng:2020:arbitrary,
  title={Arbitrary Video Style Transfer via Multi-Channel Correlation},
  author={Deng, Yingying and Tang, Fan and Dong, Weiming and Huang, haibin and Ma chongyang and Xu, Changsheng},
  booktitle={AAAI},
  year={2021},
 
}
@ARTICLE{10008203,
  author={Kong, Xiaoyu and Deng, Yingying and Tang, Fan and Dong, Weiming and Ma, Chongyang and Chen, Yongyong and He, Zhenyu and Xu, Changsheng},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Exploring the Temporal Consistency of Arbitrary Style Transfer: A Channelwise Perspective}, 
  year={2023},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TNNLS.2022.3230084}}
```
