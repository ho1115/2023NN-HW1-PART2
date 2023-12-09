# 2023 類神經網路 作業一 第二部分 Uni-Perceptron by 王浩宇

## 1. github檔案說明

### 1-1 dataPreprocess


```
*使用前請先將mnist的csv dataset下載下來，並存放至路徑"your folder"/data/
可將csv dataset轉為方便使用的json格式，json架構如下:

json {
    'trainVectors' : 2d array of size 60000 * 784, 紀錄patterns的784維座標資訊
    'trainLabels' : int array of size 60000, 紀錄patterns的label資訊
    'sortByLabel' : array of size 10, stores 10 sub int array, 紀錄class 0 ~ 9分別有哪些patterns
}
```


### 1-2 uni-Perceptron
```
執行作業說明的step 1 到 step 4，其中一部分寫入或讀取Json檔案的code已註解，可依個人需求使用或修改。
```

### 1-3 LEGACY
```
被淘汰的舊版functions
```

## 2. 現成Json Zip下載說明

### 2-1 下載連結 :

### 2-2 Json檔案說明 :

* BaseJson : 紀錄train dataset資訊, 結構同1-1所述
* Pn1 : array of size 60000 * 784, 紀錄每個patterns的pn1 set中的784個pattern index
* Pn2 : 2d array , 紀錄每個patterns的pn2 set中的pattern index