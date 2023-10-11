# Seafloor_mapping_PointNet2
A seafloor mapping model based on PointNet++ using Pytorch

### Usage
For model training and test, firstly run:
```commandline
preprocessing_script.sh
```
Then, annotate the data in the 'split_data' folder.

When your data is ready, run:
```commandline
train_script.sh
test_script.sh
```

For model prediction, run:
```commandline
preprocessing_script.sh
predict_script.sh
```

### Acknowlegements

@article{Pytorch_Pointnet_Pointnet2,
      Author = {Xu Yan},
      Title = {Pointnet/Pointnet++ Pytorch},
      Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
      Year = {2019}
}
