# Model SSD_mobileDet_cpu_coco converted to TFLite model graph

## Install TFOD(Tensorflow Object Detection) api

### Prerequiste

* Anaconda
* Tensorfow version 1.x
* pip

Open up your anaconda terminal in a local directory wherever you want to clone Tensorflow model repository and type in:

```
git clone https://github.com/tensorflow/models.git
```

This will add `model` folder to the current directory. Move over to `research` directory with this command:

```
cd models/research
```

Under research directory, there is one folder named `object_detection`. Under `object_detection` folder, there is a folder named `protos`. Under `protos` folder, there are `.proto` files which needs to be compiled. It can be done with this command.

```
protoc object_detection/protos/*.proto --python_out=.
```

Lastly, we need to install the packages present in `object_detection` folder.

```
cp object_detection/packages/tf1/setup.py .
python -m pip install --use-feature=2020-resolver .
```

## Fetching model SSD_mobileDet_cpu_coco_checkpoint

Download this file [checkpoints.tar.gz](http://download.tensorflow.org/models/object_detection/ssdlite_mobiledet_cpu_320x320_coco_2020_05_19.tar.gz) and untar it:

```
mobiledet_checkpoint_name = "ssd_mobiledet_cpu_coco" 

checkpoint_dict = {
    "ssd_mobiledet_cpu_coco": "http://download.tensorflow.org/models/object_detection/ssdlite_mobiledet_cpu_320x320_coco_2020_05_19.tar.gz"
}

folder_name_dict = {
    "ssd_mobiledet_cpu_coco": "ssdlite_mobiledet_cpu_320x320_coco_2020_05_19"
}

checkpoint_selected = checkpoint_dict[mobiledet_checkpoint_name]
folder_name = folder_name_dict[mobiledet_checkpoint_name]

# Get the pre-trained MobileDet checkpoints
!rm -rf folder_name
!wget -q $checkpoint_selected -O checkpoints.tar.gz
!tar -xvf checkpoints.tar.gz
```

This will get us 7 files, namely under `ssdlite_mobiledet_cpu_320x320_coco_2020_05_19` directory:

* model.ckpt-400000.data-00000-of-00001
* model.ckpt-400000.index
* model.ckpt-400000.meta
* model.tflite
* pipeline.config
* tflite_graph.pb
* tflite_graph.pbtxt

## Convert model checkpoints to TFLite compatible model graph

```
!python /content/models/research/object_detection/export_tflite_ssd_graph.py \
        --pipeline_config_path=$folder_name/pipeline.config \
        --trained_checkpoint_prefix=$folder_name/model.ckpt-400000 \
        --output_directory=$folder_name \
        --add_postprocessing_op=true
print("======Graph generated========")
!ls -lh $folder_name/*.pb
```

This will generate the TFLite compatible model graph `tflite_graph.pb`. This graph file generated will be converted to TFLite model.
