import torch
import torch.nn as nn
from myparser import color
from time import time

def test(model, data_loader, env_config, model_config):
    print('{}Testing {} on {}{}'.format(color.HEADER, env_config['model_name'], env_config['dataset_name'], color.ENDC))
    start = time()

    # 获取设备信息（确保模型和数据在同一设备上）
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)  # 将模型迁移到正确的设备

    if 'AERO' in env_config['model_name']:
        lossFunction = nn.MSELoss(reduction='none')
        model.eval()
        pred12, loss12_list, pred1, loss1_list = [], [], [], []
        stage = 2

        for d in data_loader:
            # 确保输入数据迁移到正确的设备
            d = d.to(device)  # 将数据迁移到设备上

            with torch.no_grad():
                recon1, recon2 = model(d, stage)
            
            pred1.append(recon1)
            pred12.append(recon1 + recon2)
            short_data = d[:, -recon2.shape[1]:, 1:]
            
            loss1_last_time = lossFunction(recon1, short_data)[:, -1, :]
            loss12_last_time = lossFunction(recon1 + recon2, short_data)[:, -1, :]
            loss1_list.append(loss1_last_time)
            loss12_list.append(loss12_last_time)

        # 将结果从设备中转移到 CPU
        loss1_tensor = torch.cat(loss1_list, 0).cpu()
        loss12_tensor = torch.cat(loss12_list, 0).cpu()

        pred12_tensor = torch.cat(pred12, 0)
        pred1_tensor = torch.cat(pred1, 0)
        pred12_wo_window = pred12_tensor[:, -1, :]
        pred1_wo_window = pred1_tensor[:, -1, :]

        print('Testing time: ' + "{:10.4f}".format(time() - start) + ' s')
        return {
            'loss12': loss12_tensor.cpu().detach().numpy(),
            'loss1': loss1_tensor.cpu().detach().numpy(),
            'pred12': pred12_wo_window.cpu().detach().numpy(),
            'pred1': pred1_wo_window.cpu().detach().numpy()
        }