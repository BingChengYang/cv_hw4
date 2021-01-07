# cv_hw4

* 1.Training
* 2.Testing


## Training model
使用以下指令訓練模型:
```
python train.py --lr=0.1 --epoches=2000 --mini_batch_size=64 --load_model=False --adjust_lr=True --step=500 --crop_size=41 --model="model.pkl"
```
lr代表learning rate的大小，default = 0.1</br>
epoches代表總共訓練幾個epoch，default = 2000</br>
mini_batch_size代表會使用mini_batch的大小，default = 64</br>
load_model代表是否要接續之前的model繼續訓練，default = False</br>
adjust_lr代表是否要使用learning rate scheduler使learning rate慢慢下降，default = True</br>
step代表經過多少epoch使lr下降一次，default = 500</br>
crop_size代表要將圖片裁剪成多小訓練，default = 41</br>
model代表所要接續訓練的模型，default = "model.pkl"</br>

## Testing model
使用下列指令來產生預測結果:
```
python test.py --test_model="model.pkl"
```
test_model代表所要選擇測試的model，default = "model.pkl"</br>
結束後會在0756545資料夾產生super resolution後的圖片
