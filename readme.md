> 感谢：
>
> - [dog-qiuqiu/Yolo-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest)
>
> - [david8862/keras-YOLOv3-model-set](https://github.com/david8862/keras-YOLOv3-model-set)

# 0x01 yolo-fastest keras/tflite 推理

## 1. 准备 yolo-fastest 模型

下面所列文件均已经上传至项目

- [yolo-fastest.weight](./model/yolo-fastest.weight)| [模型源文件下载地址](https://github.com/dog-qiuqiu/Yolo-Fastest/tree/v0.1/Yolo-Fastest/VOC)

- 训练数据集：VOC2007 + 2012

- 类别：20类，文件：[voc_classes.txt](./configs/voc_classes.txt)
- keras 模型：[yolo-fastest.h5](./model/yolo-fastest.h5)
- tflite 模型：[yolo-fastest.tflite](model/yolo-fastest.tflite)

| Model                | mAP(%)           | Input     | BFLOPS | Size  | Data  |
| -------------------- | ---------------- | --------- | ------ | ----- | ----- |
| yolo-fastest.weights | 56.19            | 320x320x3 | 0.238  | 1.20M | 01/09 |
| yolo-fastest.h5      | ~~代码运行失败~~ | 320x320x3 | 0.238  | 1.92M | 05/19 |
| yolo-fastest.tflite  | 48.26            | 320x320x3 | 0.238  | 1.17M | 05/19 |

> yolo-fastest.tflite mAP caculate:
>
> ​	mAP@IoU=0.50 result: 48.264888
> ​	mPrec@IoU=0.50 result: 9.296873
> ​	mRec@IoU=0.50 result: 67.471812

模型 mAP 计算和转换成 tflite 格式的代码在这个仓库：[lebhoryi/keras-YOLOv3-model-set](https://github.com/Lebhoryi/keras-YOLOv3-model-set)

```shell
# darknet to keras
$ python tools/model_converter/convert.py ../Yolo-Fastest/Yolo-Fastest/VOC/yolo-fastest.cfg ../Yolo-Fastest/Yolo-Fastest/VOC/yolo-fastest.weights weights/yolo-fastest.h5  -f

# keras to tflite without quantize
$ python tools/model_converter/custom_tflite_convert.py --keras_model_file ./weights/yolo-fastest.h5 --output_file ./weights/yolo-fastest.tflite

# eval tflite model map
$ python eval.py --model_path weights/yolo-fastest.tflite --anchors_path configs/yolo_fastest_anchors.txt --classes_path configs/voc_classes.txt --model_image_size=320x320 --eval_type=VOC --iou_threshold=0.5 --conf_threshold=0.001 --annotation_file=2007_test.txt --save_result

# keras to tflite with quantize
$ python tools/model_converter/post_train_quant_convert.py --keras_model_file ./weights/yolo-fastest.h5 --annotation_file ~/Data/VOC/2007_test.txt --model_input_shape 320x320 --sample_num 30 --output_file ./weights/yolo-fastest.tflite
```

自己写的 [全网最最最轻量级检测网络 yolo-fastest 快速上手](https://blog.csdn.net/weixin_37598106/article/details/112544854)，也可以参考一下

## 2. 推理 keras 模型

代码：[inference_one_picture.ipynb](./inference_one_picture.ipynb)

代码中的注释已经很详细了，这里不展开叙述

结果：

![image-20210519174644413](https://gitee.com/lebhoryi/PicGoPictureBed/raw/master/img/20210519174706.png)

## 3. 模型后处理代码简要

> 均来自于参考资料：[david8862/keras-YOLOv3-model-set](https://github.com/david8862/keras-YOLOv3-model-set)

1. yolo 模型解码

   模型的输出结果是偏移量，并不是真正的 xywh，所以需要处理一下

   1. 先sigmoid，然后恢复 xy，归一化处理，和每个格子的左上角坐标相关
   2. 先sigmoid，然后恢复 wh，归一化处理，和每个 anchor 的大小相关
   3. 对 scores 和 objectness 做 sigmoid 处理

2. 计算真正置信度，然后根据置信度的阈值，保留高置信度的数据，第一次筛选

   true_scores = scores * objectness

3. `NMS` 处理，nms 代码来源：[yolo_postprocess_np.py](https://github.com/david8862/keras-YOLOv3-model-set/blob/master/common/yolo_postprocess_np.py)

   1. 获取 xywh
   2. 按照置信度大小排序，获得索引
   3. 计算所有面积
   4. 计算交并比
   5. 筛选，保留单类别最大概率索引
   6. 重复 3、4 步骤，直到索引为空

4. 画图

---

待完成：

- [ ] NMS 扩展
  - [ ] Fast/Cluster NMS
  - [ ] Weighted-Boxes-Fusion
  - [ ] soft nms

- [x] tflite 模型推理（在代码中已经实现 [inference_one_picture.ipynb](./inference_one_picture.ipynb)）
- [ ] 摄像头输入 + tflite 模型推理
- [ ] k210 嵌入式目标检测 + RT-AK