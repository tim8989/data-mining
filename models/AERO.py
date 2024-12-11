# -*-coding:utf-8 -*-  
import torch
import torch.nn as nn
from models.temporal import Temporal
from models.concurrent import Concurrent

class AERO(nn.Module):
    def __init__(self, dim, embed_time, slide_win, small_win):
        super(AERO, self).__init__()
        self.name = 'AERO'
        self.slide_win = slide_win
        self.small_win = small_win
        self.trans = Temporal(embed_time, slide_win, small_win, 1)  # 1表示每个维度单独进入
        self.reslayer = Concurrent(dim, small_win, small_win, small_win)
        
        # 添加 Dropout 层来减少过拟合
        self.dropout = nn.Dropout(0.5)  # 50% 概率禁用神经元

    def forward(self, inputW, stage):
        # 冻结不同阶段的参数
        if self.train:
            if stage == 1:
                for name, param in self.reslayer.named_parameters():
                    param.requires_grad = False
                for name, param in self.trans.named_parameters():
                    param.requires_grad = True
            else:
                for name, param in self.trans.named_parameters():
                    param.requires_grad = False
                for name, param in self.reslayer.named_parameters():
                    param.requires_grad = True

        recon, res, origin, memory_list = [], [], [], []
        
        # 遍历输入数据，处理每一个时间步
        for i in range(1, inputW.shape[-1]):
            input_trans = inputW.permute(1, 0, 2)

            src_time = input_trans[:, :, 0].view(input_trans.shape[0], input_trans.shape[1], 1)
            src = input_trans[:, :, i].view(input_trans.shape[0], input_trans.shape[1], 1)

            tgt = input_trans[-self.small_win:, :, i].view(self.small_win, input_trans.shape[1], 1)

            # 通过Temporal层
            trans, memory = self.trans(src, tgt, src_time)
            recon.append(trans)
            origin.append(tgt)
            res.append(tgt - trans)
            memory_list.append(memory)

        # 合并所有的重构结果
        recon1 = torch.cat(recon, 2).permute(1, 0, 2)
        origin_full = torch.cat(origin, 2).permute(1, 2, 0)
        res_full = torch.cat(res, 2).permute(1, 2, 0)

        # 使用 Dropout 防止过拟合
        recon2 = self.reslayer(res_full, origin_full)
        recon2 = self.dropout(recon2)  # Apply dropout to output

        return recon1, recon2