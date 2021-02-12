#coding:utf-8
# 路径置顶
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.getcwd())
sys.path.append("./")
from torch.nn.modules.distance import PairwiseDistance
from tqdm import tqdm
import torch
import time
# 导入文件
from train_dataset import TrainDataset
from triplet_loss import TripletLoss
import torchvision.transforms as transforms
from model import ResnetCBAMTriplet

BATCH_SIZE = 4
EMBEDDING_DIMENSION = 128
EPOCHES = 2
NUM_TRIPLETS = 10
MODEL_VERSION = '18'

# 训练数据的变换
train_data_transforms = transforms.Compose([
    # transforms.Resize([config['image_size'], config['image_size']]), # resize
    #transforms.RandomHorizontalFlip(), # 随机翻转
    transforms.ToTensor(), # 变成tensor
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# 训练数据生成器
train_dataloader = torch.utils.data.DataLoader(
    dataset=TrainDataset(
        face_dir="./Datasets/vggface2_train_face_mask",
        csv_name='./Datasets/vggface2_train_face_mask.csv',
        num_triplets=NUM_TRIPLETS,
        training_triplets_path='./Datasets/training_triplets_1000_mask.npy',
        transform=train_data_transforms,
        predicter_path='shape_predictor_68_face_landmarks.dat',
        img_size=256
    ),
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=False
)


pwd = os.path.abspath('./')
start_epoch = 0
model = ResnetCBAMTriplet(model_version = MODEL_VERSION, pretrained=False,embedding_dimension = EMBEDDING_DIMENSION)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

def adjust_learning_rate(optimizer, epoch):
    if epoch<30:
        lr =  0.125
    elif (epoch>=30) and (epoch<60):
        lr = 0.0625
    elif (epoch >= 60) and (epoch < 90):
        lr = 0.0155
    elif (epoch >= 90) and (epoch < 120):
        lr = 0.003
    elif (epoch>=120) and (epoch<160):
        lr = 0.0001
    else:
        lr = 0.00006
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

lr = 0.125
optimizer_model = torch.optim.Adagrad(model.parameters(), lr = lr,lr_decay=1e-4,weight_decay=0)

# 打卡时间、epoch
total_time_start = time.time()
start_epoch = start_epoch
end_epoch = start_epoch + EPOCHES
# 导入l2计算的
l2_distance = PairwiseDistance(2)
# 为了打日志先预制个最低auc和最佳acc在前头
best_roc_auc = -1
best_accuracy = -1


# epoch大循环
for epoch in range(start_epoch, end_epoch):
    print("\ntraining on TrainDataset! epoch: %s/%s " % (epoch+1,EPOCHES))
    epoch_time_start = time.time()
    triplet_loss_sum = 0
    sample_num = 0

    model.train()  # 训练模式
    # step小循环
    progress_bar = enumerate(tqdm(train_dataloader))
    for batch_idx, (batch_sample) in progress_bar:     
        # 获取本批次的数据
        # 取出三张人脸图(batch*图)
        anc_img = batch_sample['anc_img'].to(device)
        pos_img = batch_sample['pos_img'].to(device)
        neg_img = batch_sample['neg_img'].to(device)
        
        # 模型运算
        # 前向传播过程-拿模型分别跑三张图，生成embedding和loss（在训练阶段的输入是两张图，输出带loss，而验证阶段输入一张图，输出只有embedding）
        anc_embedding = model(anc_img)
        pos_embedding = model(pos_img)
        neg_embedding = model(neg_img)
        
        anc_embedding = torch.div(anc_embedding, torch.norm(anc_embedding))
        pos_embedding = torch.div(pos_embedding, torch.norm(pos_embedding))
        neg_embedding = torch.div(neg_embedding, torch.norm(neg_embedding))
      
        # 损失计算
        # 计算这个批次困难样本的三元损失

        triplet_loss = TripletLoss(0.1)
        loss = triplet_loss(anc_embedding, pos_embedding, neg_embedding, reduction='mean')


        # 反向传播过程
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        # update the optimizer learning rate
        adjust_learning_rate(optimizer_model, epoch)

        # 计算这个epoch内的总三元损失和计算损失所用的样本个数
        triplet_loss_sum += loss.item()
        sample_num += anc_embedding.shape[0]
    
    # 计算这个epoch里的平均损失
    avg_triplet_loss = triplet_loss_sum/sample_num
    print("avg_triplet_loss= %s" % avg_triplet_loss)
    epoch_time_end = time.time()

        # 每个epoch 后保存一次模型
    state = {
            'epoch': epoch + 1,
            'embedding_dimension': EMBEDDING_DIMENSION,
            'batch_size_training': BATCH_SIZE,
            'model_state_dict': model.state_dict(),
            'model_architecture': 'resnet{}CBAMTriplet'.format(MODEL_VERSION),
            'optimizer_model_state_dict': optimizer_model.state_dict()
    }
    if not os.path.isdir('./Model_training_checkpoints'):
        os.mkdir('./Model_training_checkpoints')
    if (epoch+1)%1 == 0:
        torch.save(state, 'Model_training_checkpoints/model_resnet{}cbam_triplet_epoch_{}.pt'.format(MODEL_VERSION,epoch + 1))