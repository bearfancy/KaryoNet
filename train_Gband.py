import os
import time
import torch
import torchvision
import random
from PIL import Image
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from MODELS.resnet_256d import resnet50
from MODELS.resnet import resnet50 as resnet50pretrain
from MODELS.resnet_MFIM import resnet50 as resnet50_MFIM
from MODELS.resnet_DAM import resnet50 as resnet50_DAM
from MODELS.resnet_MFIM_DAM import resnet50 as resnet50_MFIM_DAM
import argparse
import numpy as np
import copy
from toolkit.dataread import read_data
import logging
from datetime import datetime

# Device configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def setup_logging_and_directories(args):
    """设置日志记录和创建必要的目录"""
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建workdir下的实验文件夹
    experiment_name = f"{args.model_name}_{timestamp}"
    workdir = os.path.join(args.workdir, experiment_name)
    
    # 创建目录
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.join(workdir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(workdir, 'models'), exist_ok=True)
    
    # 设置日志记录
    log_file = os.path.join(workdir, 'logs', 'training.log')
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    # 更新模型保存路径
    args.model_path = os.path.join(workdir, 'models')
    
    return workdir, log_file

def adjust_learning_rate(optimizer, epoch, reducetime, reduced):
    if epoch==reducetime and reduced==0:
        lr = args.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def pretrain(model):
    modelpretrain = resnet50pretrain(num_classes=1000)
    modelpretrain.load_state_dict(torch.load(args.pretrained_model))
    model.conv1 = modelpretrain.conv1
    model.bn1 = modelpretrain.bn1
    model.relu = modelpretrain.relu
    model.maxpool = modelpretrain.maxpool
    model.layer1 = modelpretrain.layer1
    model.layer2 = modelpretrain.layer2
    model.layer3 = modelpretrain.layer3
    model.layer4 = modelpretrain.layer4
    model.avgpool = modelpretrain.avgpool

    return model


def main(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.model=='resnet50_MFIM' or args.model=='resnet50_MFIM_DAM':
        detr_checkpoint = torch.load('/home/wy/projects/染色体极性分类/KaryoNet/model/pretrain/detr-r50-e632da11.pth', map_location='cpu')
        names={'transformer.encoder.layers.0.self_attn.in_proj_weight', 'transformer.encoder.layers.0.self_attn.in_proj_bias', 'transformer.encoder.layers.0.self_attn.out_proj.weight', 'transformer.encoder.layers.0.self_attn.out_proj.bias', 'transformer.encoder.layers.0.linear1.weight', 'transformer.encoder.layers.0.linear1.bias', 'transformer.encoder.layers.0.linear2.weight', 'transformer.encoder.layers.0.linear2.bias', 'transformer.encoder.layers.0.norm1.weight', 'transformer.encoder.layers.0.norm1.bias', 'transformer.encoder.layers.0.norm2.weight', 'transformer.encoder.layers.0.norm2.bias', 'transformer.encoder.layers.1.self_attn.in_proj_weight', 'transformer.encoder.layers.1.self_attn.in_proj_bias', 'transformer.encoder.layers.1.self_attn.out_proj.weight', 'transformer.encoder.layers.1.self_attn.out_proj.bias', 'transformer.encoder.layers.1.linear1.weight', 'transformer.encoder.layers.1.linear1.bias', 'transformer.encoder.layers.1.linear2.weight', 'transformer.encoder.layers.1.linear2.bias', 'transformer.encoder.layers.1.norm1.weight', 'transformer.encoder.layers.1.norm1.bias', 'transformer.encoder.layers.1.norm2.weight', 'transformer.encoder.layers.1.norm2.bias', 'transformer.encoder.layers.2.self_attn.in_proj_weight', 'transformer.encoder.layers.2.self_attn.in_proj_bias', 'transformer.encoder.layers.2.self_attn.out_proj.weight', 'transformer.encoder.layers.2.self_attn.out_proj.bias', 'transformer.encoder.layers.2.linear1.weight', 'transformer.encoder.layers.2.linear1.bias', 'transformer.encoder.layers.2.linear2.weight', 'transformer.encoder.layers.2.linear2.bias', 'transformer.encoder.layers.2.norm1.weight', 'transformer.encoder.layers.2.norm1.bias', 'transformer.encoder.layers.2.norm2.weight', 'transformer.encoder.layers.2.norm2.bias'}
        transformer_checkpoint = {key:detr_checkpoint['model'][key] for key in detr_checkpoint['model'].keys() & names}
        if args.model=='resnet50_MFIM':
            model = resnet50_MFIM(args,num_classes=args.num_class)
        else:
            model = resnet50_MFIM_DAM(args,num_classes=args.num_class)
        if args.pretrained:
            model=pretrain(model)
        model_state_dict = model.state_dict()
        model_state_dict.update(transformer_checkpoint)
        model.load_state_dict(model_state_dict)
    elif args.model== 'resnet50' or args.model=='resnet50_DAM':
        if args.model== 'resnet50' : 
            model = resnet50(num_classes=args.num_class)
        else:
            model = resnet50_DAM(num_classes=args.num_class)
        if args.pretrained:
            model=pretrain(model)

    model = model.to(device)
    logging.info(f"模型结构:\n{model}")
    cost = nn.CrossEntropyLoss().to(device)
    nllcost = nn.NLLLoss().to(device)
    BCEcost = nn.BCELoss(reduction='mean').to(device)
    logsoftmax_func=nn.LogSoftmax(dim=1)
    softmax_func=nn.Softmax(dim=1)
    bestacc=0

    # Optimization
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
    mytransforms = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)), 
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
    rootdir=args.rootdir
    rootdirval=args.rootdirval
    valcasenames, lists, dirlist, normalboy, normalgirl = read_data(rootdir=rootdir,rootdirval=rootdirval)
    reduced=0
    label2list=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
    
    # 打印训练配置信息
    logging.info(f"\n{'='*80}")
    logging.info(f"🚀 开始训练 - {args.model}")
    logging.info(f"{'='*80}")
    logging.info(f"📊 数据集信息:")
    logging.info(f"  训练集路径: {rootdir}")
    logging.info(f"  验证集路径: {rootdirval}")
    logging.info(f"  验证集细胞数: {len(valcasenames)}")
    logging.info(f"  训练集染色体类型数: {len(lists)}")
    total_train_images = sum(len(file_list) for file_list in lists)
    logging.info(f"  训练集总图片数: {total_train_images:,}")
    logging.info(f"\n🔧 训练配置:")
    logging.info(f"  模型: {args.model}")
    logging.info(f"  染色体类别数: {args.num_class}")
    logging.info(f"  极性类别数: 4 (0°, 90°, 180°, 270°)")
    logging.info(f"  学习率: {args.lr}")
    logging.info(f"  总训练轮数: {args.iteration}")
    logging.info(f"  学习率衰减轮数: {args.lr_reduce_time}")
    logging.info(f"  设备: {device}")
    logging.info(f"  模型保存路径: {args.model_path}")
    logging.info(f"{'='*80}\n")
    for epoch in range(1, args.iteration + 1):
        model.train()
        start = time.time()
        batchnames=[]
        if epoch==args.lr_reduce_time:
            old_lr = optimizer.state_dict()['param_groups'][0]['lr']
            adjust_learning_rate(optimizer, epoch, args.lr_reduce_time, reduced)
            new_lr = optimizer.state_dict()['param_groups'][0]['lr']
            logging.info(f"\n📉 学习率调整: {old_lr:.6f} → {new_lr:.6f} (Epoch {epoch})")
            reduced=1
            
        # 随机选择性别
        sex=random.randint(0,100)%2
        variationed=0
        variation=random.randint(0,100)
        if sex==0:
            namelist=copy.copy(normalboy)
        else:
            namelist=copy.copy(normalgirl)
            
        # 数据增强策略: 随机添加或删除某些染色体类型来增加数据多样性
        if variationed==0 and variation%3==0:# +8
            namelist.append(7)
            variationed=1
        if variationed==0 and variation%10==0:# -7
            del namelist[12]
            variationed=1
        if variationed==0 and variation%5==0 and sex==0:# -y
            del namelist[-1]
            variationed=1
        if variationed==0 and variation%10==0:# +11
            namelist.append(10)
            variationed=1
        if variationed==0 and variation%10==0:# +12
            namelist.append(11)
            variationed=1
        if variationed==0 and variation%10==0:# +21
            namelist.append(20)
            variationed=1
        
        # 构建批次图片路径
        for i in namelist:
            batchnames.append(os.path.join(rootdir,dirlist[i],random.choice(lists[i]))) ##load the data to form a batch
        random.shuffle(batchnames)
        
        # 预分配tensor空间
        img_tensors = torch.empty(len(namelist), 3, 224, 224)  # 图片数据
        labelpair = torch.empty(len(namelist),len(namelist))   # 配对标签
        pola_tensors = torch.empty(len(namelist))              # 极性标签
        labellist=[]                                           # 类别标签列表
        labelgrouplist=[]                                      # 组别标签列表
        
        i=0
        for img_path in batchnames:
            # 新的文件命名格式：ID_核型_角度_细胞名_Karyotype.jpg
            filename = img_path.split('/')[-1]
            parts = filename.split('_')
            
            # 提取极性角度信息 (0, 90, 180, 270)
            angle = int(parts[2])  # 第3个字段是角度
            # 将角度映射到极性标签：0->0, 90->1, 180->2, 270->3
            pola_tensors[i] = angle // 90
            
            # 加载和预处理图片
            data = Image.open(img_path) # 使用PIL加载图片
            data = mytransforms(data) # 应用数据变换
            img_tensors[i,:,:,:] = data # 存储到tensor中
            
            # 提取染色体类别标签 (核型在第2个字段)
            filelabel = int(parts[1]) - 1  # 染色体类别标签 (1-24 -> 0-23)
            labellist.append(float(filelabel))
            labelgrouplist.append(float(label2list[filelabel]))
            i=i+1
        
        # 生成染色体配对关系矩阵
        for i in range(len(labellist)):
            for j in range(len(labellist)):
                if label2list[int(labellist[i])]==label2list[int(labellist[j])]:
                    labelpair[i,j]=1 # 同组染色体
                else:
                    labelpair[i,j]=0 # 不同组染色体
        
        # 转换为PyTorch tensor: 类别标签、配对标签、极性标签
        label_tensors = torch.from_numpy(np.array(labellist)).long()
        #labelgroup_tensors = torch.from_numpy(np.array(labelgrouplist)).long()
        images = img_tensors.to(device)
        labels = label_tensors.to(device)
        labelpairs = labelpair.to(device)
        polalabels = pola_tensors.long().to(device)
        #labelgroups = labelgroup_tensors.to(device)
        
        if args.model=='resnet50' or args.model=='resnet50_DAM':
            outputs,polaout = model(images)
            lossmain = cost(outputs, labels) # 染色体分类损失
            losspola = cost(polaout, polalabels) # 极性分类损失
            loss = lossmain+losspola
        elif args.model=='resnet50_MFIM' or args.model=='resnet50_MFIM_DAM':
            outputs, polaout, outputspair = model(images)
            losspair = BCEcost(outputspair[:,:,0],labelpairs)
            lossmain = cost(outputs, labels)
            losspola = cost(polaout, polalabels)
            loss = lossmain+0.5*losspair+losspola

        if epoch % 1 == 0:
            # 计算各个损失分量
            if args.model=='resnet50_DAM' or args.model=='resnet50_256d':
                loss_main = lossmain.item()
                loss_pola = losspola.item()
                logging.info(f"Epoch {epoch:6d} | Total Loss: {loss.data.item():.4f} | Main Loss: {loss_main:.4f} | Pola Loss: {loss_pola:.4f} | LR: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}")
            elif args.model=='resnet50_MFIM' or args.model=='resnet50_MFIM_DAM':
                loss_main = lossmain.item()
                loss_pola = losspola.item()
                loss_pair = losspair.item()
                logging.info(f"Epoch {epoch:6d} | Total Loss: {loss.data.item():.4f} | Main Loss: {loss_main:.4f} | Pola Loss: {loss_pola:.4f} | Pair Loss: {loss_pair:.4f} | LR: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}")
            else:
                logging.info(f"Epoch {epoch:6d} | Loss: {loss.data.item():.4f} | LR: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}")
        
        # 每1000个epoch显示进度
        if epoch % 1000 == 0:
            progress = epoch / args.iteration * 100
            logging.info(f"\n📊 训练进度: {epoch}/{args.iteration} ({progress:.1f}%) | 当前最佳准确率: {bestacc:.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 4000 == 0:
            logging.info(f"\n{'='*80}")
            logging.info(f"Epoch {epoch} - 开始验证...")
            logging.info(f"{'='*80}")
            
            torch.save(model, os.path.join(args.model_path, '%s-%d.pth' % (args.model_name, epoch)))
            model.eval()
            num=0
            correctnum=0
            correctpolanum=0
            total_cases = len(valcasenames)
            for case_idx, case in enumerate(valcasenames):
                if case_idx % 50 == 0:  # 每50个细胞显示一次进度
                    logging.info(f"验证进度: {case_idx+1}/{total_cases} ({100*(case_idx+1)/total_cases:.1f}%)")
                
                root1=os.path.join(rootdirval,case)
                img_list1=os.listdir(root1)
                allprob=[]
                labels=[]
                polalabels=[]
                
                # 动态创建tensor，根据实际图片数量
                num_images = len(img_list1)
                img_tensors = torch.empty(num_images, 3, 224, 224)
                
                casei=0
                for img1 in img_list1:
                    data = Image.open(os.path.join(root1, img1))
                    data = mytransforms(data)
                    img_tensors[casei,:,:,:] = data
                    
                    # 新的文件命名格式：ID_核型_角度_细胞名_Karyotype.jpg
                    parts = img1.split('_')
                    label = int(parts[1]) - 1  # 核型在第2个字段 (1-24 -> 0-23)
                    angle = int(parts[2])  # 角度在第3个字段
                    
                    labels.append(label)
                    polalabels.append(angle // 90)  # 将角度映射到极性标签
                    casei=casei+1
                    num=num+1
                images = img_tensors.to(device)
                if 'MFIM' in args.model:
                    outputs, _, _ = model(images)
                else:
                    outputs, _ = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                # # 调试信息：检查长度匹配
                # logging.info(f"  预测结果长度: {predicted.shape[0]}, 标签长度: {len(labels)}")
                # logging.info(f"  预测值范围: {predicted.min().item()}-{predicted.max().item()}")
                # logging.info(f"  标签值范围: {min(labels)}-{max(labels)}")
                
                # 确保长度匹配
                min_len = min(predicted.shape[0], len(labels))
                for pred in range(min_len):
                    if predicted[pred] == labels[pred]:  # 现在都是0-23范围，直接比较
                        correctnum=correctnum+1
            
            acc=correctnum / num
            logging.info(f"\n验证结果:")
            logging.info(f"  总图片数: {num}")
            logging.info(f"  正确分类数: {correctnum}")
            logging.info(f"  染色体分类准确率: {acc:.4f} ({acc*100:.2f}%)")
            logging.info(f"  当前最佳准确率: {bestacc:.4f} ({bestacc*100:.2f}%)")
            
            if acc>=bestacc:
                logging.info(f"  🎉 新的最佳模型！准确率提升: {bestacc:.4f} → {acc:.4f}")
                torch.save(model, os.path.join(args.model_path, '%s-best.pth' % args.model_name))
                bestacc=acc
            else:
                logging.info(f"  📈 准确率未提升，当前最佳: {bestacc:.4f}")
            
            logging.info(f"{'='*80}\n")
            model.train()
    
    # 训练结束
    logging.info(f"\n{'='*80}")
    logging.info(f"🏁 训练完成！")
    logging.info(f"{'='*80}")
    logging.info(f"📈 最终结果:")
    logging.info(f"  最佳准确率: {bestacc:.4f} ({bestacc*100:.2f}%)")
    logging.info(f"  最佳模型已保存: {args.model_path}/{args.model_name}-best.pth")
    logging.info(f"  最终模型已保存: {args.model_path}/{args.model_name}-{args.iteration}.pth")
    logging.info(f"{'='*80}\n")


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='train hyper-parameter')
    parser.add_argument("--num_class", default=24, type=int)
    parser.add_argument("--iteration", default=40000, type=int)
    parser.add_argument("--lr_reduce_time", default=20000, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=46, type=int)
    parser.add_argument("--model_name", default='model-resnet50_MFIM_DAM', type=str)
    parser.add_argument("--model_path", default='./model', type=str)
    parser.add_argument("--workdir", default='./experiments', type=str, help='工作目录，用于保存日志和模型')
    parser.add_argument("--pretrained", default=True, type=bool)
    parser.add_argument("--pretrained_model", default='./model/pretrain/resnet50-19c8e357.pth', type=str)
    parser.add_argument("--rootdir", default='dataset/JBM_G_category_dataset250922/train/', type=str)
    parser.add_argument("--rootdirval", default='dataset/JBM_G_category_dataset250922/val/', type=str)
    parser.add_argument("--model", default='resnet50_MFIM_DAM', type=str)
    parser.add_argument('--enc_layers', default=3, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    args = parser.parse_args()
    
    # 设置日志记录和创建目录
    workdir, log_file = setup_logging_and_directories(args)
    
    # 打印实验信息
    print(f"🔧 实验配置:")
    print(f"  实验文件夹: {workdir}")
    print(f"  日志文件: {log_file}")
    print(f"  模型保存路径: {args.model_path}")
    print(f"{'='*80}\n")
    
    main(args)
