import hashlib  # 用于计算文件哈希，用于检验文件是否完整/正确
import os       # 用于操作文件路径，创建文件夹
import tarfile  # 用于解压缩包
import zipfile  # 用于解压缩包
import requests # 用于从网络中下载文件
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader

from deep_learring import linear_regression

#@save 是 D2L 书籍的特殊标记，表示将这段代码保存到工具库中
DATA_HUB = dict()  # 存储数据集名称 → (下载地址, 校验哈希)
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'  # D2L 官方数据仓库地址

def download(name,cache_dir = os.path.join('..','data')): #@save
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} not in {DATA_HUB}"
    url, sha1_hash  = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])

    # 如果文件已经存在
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            print(f"✅ {name} 已存在，无需重新下载")
            return fname

    # 需要下载
    print(f"⬇️  正在下载 {name} → {fname}")
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    print(f"✅ {name} 下载完成！")
    return fname

def download_extract(name,folder = None):
    """下载并解压zip/tar文件"""
    print(f"\n===== 开始处理数据集：{name} =====")
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir , ext = os.path.splitext(fname)

    # 修复你的BUG：把 == 改成 in
    if ext == '.zip':
        fp = zipfile.ZipFile(fname,'r')
    elif ext in ('.tar','.gz'):
        fp = tarfile.open(fname,'r')
    else:
        assert False, "只有zip,tar文件可以被解压缩"

    print(f"📂 正在解压 → {data_dir}")
    fp.extractall(base_dir)
    fp.close()
    print(f"✅ {name} 解压完成！")
    return os.path.join(base_dir,folder) if folder else data_dir

def download_all():
    """下载DATA_HUB中的所有文件"""
    print("\n===== 开始批量下载所有数据集 =====")
    for name in DATA_HUB:
        download_extract(name)
    print("\n🎉 所有数据集已全部下载并解压完成！")


DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
#
all_features  = pd.concat([train_data.iloc[:, 1:-1],test_data.iloc[:, 1]])

## 筛选出不是字符串/类别的特征，取出这些特征的列名，存到numeric_features 中
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
## 对数据特征做标准化，这样使得每一列的均值都会变成0，标准差变成 1。
all_features[numeric_features] = all_features[numeric_features].apply(lambda x : (x - x.mean()) / (x.std()))
## 用0填充缺失值，因为标准化之后均值为0.填0之后，就相当于填了平均值，
all_features[numeric_features]  =all_features[numeric_features].fillna(0)
## 对数据中的离散值采用独热编码
## # “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features,dummy_na=True)
print(all_features.shape)

n_train = train_data.shape[0] ## 拿到测试集的行数
## 将数据中的转化成浮点数张量
all_features = all_features.astype(float)
train_features = torch.tensor(all_features[:n_train].values,dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values,dtype=torch.float32)
## 将数据中的saleprice先转成二维矩阵，之后转化成张量
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1 , 1),dtype=torch.float32)

## 训练
loss = nn.MSELoss() ## 定义损失函数，均方误差
in_features = train_features.shape[1] ## 拿到特征的数量

def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1),) ## 定义模型，单层神经网络
    return net

## 采用对数均方根误差
def log_rmse(net,features,labels):
    clipped_preds = torch.clamp(net(features),1,float('inf'))
    rmse  = torch.sqrt(loss(torch.log(clipped_preds),torch.log(labels)))
    return rmse.item()

def train(net,train_features,train_labels,test_features,test_labels,num_epochs,learning_rate,weight_decay,batch_size):
    ## 记录测试集和训练集的 log RMSE
    train_ls,test_ls = [],[]

    train_dataset = TensorDataset(train_features,train_labels)
    train_iter = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

    ## 使用 Adam 优化器
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay
                                 )
    ## 4.训练循环
    for epoch in range(num_epochs):
        for x,y in train_iter:
            optimizer.zero_grad()
            l = loss(net(x),y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net,train_features,train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net,test_features,test_labels))
    return train_ls,test_ls

def get_k_fold_data(k ,i ,x, y):
    assert k > 1
    fold_size = x.shape[0] // k
    x_train , y_train = None ,None
    for j in range(k):
        idx = slice(j * fold_size , (j + 1) * fold_size)
        x_part , y_part = x[idx,:], y[idx]
        if j == i:
            x_valid , y_valid = x_part, y_part
        elif x_train is None:
            x_train , y_train = x_part, y_part
        else:
            x_train = torch.cat([x_train,x_part],0)
            y_train = torch.cat([y_train,y_part],0)
    return x_train , y_train,x_valid,y_valid

def k_fold(k,x_train,y_train,num_epochs,learning_rate,weight_decay,batch_size):
    train_l_sum , valid_l_sum = 0 , 0
    for  i  in range(k):
        data = get_k_fold_data(k ,i ,x_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net,*data,num_epochs,learning_rate,weight_decay,batch_size)
        ## 取最后一轮的误差
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            print("第一折训练曲线：")
            print(f"训练误差: {train_ls}")
            print(f"验证误差: {valid_ls}")
        print(f'折{i + 1}，训练log rmse {train_ls[-1]:.6f}, 验证log rmse {valid_ls[-1]:.6f}')
    return train_l_sum / k, valid_l_sum / k

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')

def train_and_pred(train_features,test_features,train_labels,test_labels,num_epochs,learning_rate,weight_decay,batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    print(f'训练log rmse: {float(train_ls[-1]):.6f}')
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
    print("✅ 提交文件已生成：submission.csv")
    return submission
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)










