import torch
import torch.nn as nn
from myparser import color

def valid(model, data_loader, env_config, model_config):
    print('{}Validating {} on {}{}'.format(color.HEADER, env_config['model_name'], env_config['dataset_name'], color.ENDC))

    # 获取设备信息（确保模型和数据在同一设备上）
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)  # 将模型迁移到正确的设备

    if 'AERO' in env_config['model_name']:
        lossFunction = nn.MSELoss(reduction='none')
        model.eval()
        loss12_list, loss1_list = [], []
        stage = 2

        for d in data_loader:
            # 确保输入数据迁移到相同的设备
            d = d.to(device)

            with torch.no_grad():
                recon1, recon2 = model(d, stage)

            short_data = d[:, -recon2.shape[1]:, 1:]
            loss12 = torch.mean(lossFunction(recon1 + recon2, short_data)).cpu().detach().numpy()
            loss12_list.append(loss12)

        return loss12_list