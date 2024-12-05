import os
import torch
 
def save_model(model_d, model_g, args, file_name):
    filter_val = ['noise_dim', 
                  'epoch', 'batch_size', 
                  'hidden_dim_g', 'num_layers_g', 'lr_g',
                  'hidden_dim_d', 'num_layers_d', 'lr_d',
                  'd_iter', 'gp_lambda']
    args = {key:value for key, value in vars(args).items() if key in filter_val}
    torch.save({
        'args': args,
        'model_d': model_d.state_dict(),
        'model_g': model_g.state_dict()
    }, file_name)

def clear_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"資料夾 {folder_path} 不存在。")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                clear_folder(file_path)
                os.rmdir(file_path)
        except Exception as e:
            print(f"when delete {file_path} has error: {e}")

