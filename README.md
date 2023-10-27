# Seafloor_mapping_PointNet2
A seafloor mapping model based on PointNet++ using Pytorch

### Usage
#### Model training
Firstly, make sure you have at least one ICESat-2 h5 file in you data directory, then change mode to 'train', and run:
```commandline
preprocessing_script.sh
```

Then, annotate the data in the 'split_data' folder. Remember to put your annotated data in a folder called 'input_data'.

Lastly, when your data is ready, run:
```commandline
train_script.sh
test_script.sh
```

#### Model prediction
Make sure you have at least one ICESat-2 h5 file in you data directory, then change mode to 'test', and run:
```commandline
preprocessing_script.sh
predict_script.sh
```

#### Data
We have provided the data in the link [https://uottawa-my.sharepoint.com/personal/ylin152_uottawa_ca/_layouts/15/guestaccess.aspx?share=Ebt5C3gFwcJIlZU6z0qlPVEBdllTVC34v_8OwbugYX6xsg&e=A6X9mQ](https://www.dropbox.com/scl/fi/nhjs243nd4m0s7hymp78g/data_8192.zip?rlkey=oize826zgoetm9u3gaa7p1h3h&dl=0).

And we've also provided the trained model "model.pth" in the "trained_model" folder.

### Acknowlegements

@article{Pytorch_Pointnet_Pointnet2,
      Author = {Xu Yan},
      Title = {Pointnet/Pointnet++ Pytorch},
      Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
      Year = {2019}
}
