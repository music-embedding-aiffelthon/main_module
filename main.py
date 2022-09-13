# 2022-09-10 13:54 Incheon

# --- import dataset ---
from utils.dataloader import mel_dataset
from torch.utils.data import DataLoader, random_split

# --- import train module ---
from train_module import TrainerModule

# --- import etc ---
from tqdm import tqdm
import os
import wandb
from utils.config_hook import yaml_config_hook
import numpy as np

# --- Define config ---
config_dir = os.path.join(os.path.expanduser('~'),'trainer_module/config')     
config = yaml_config_hook(os.path.join(config_dir, 'config.yaml'))


# --- collate batch for dataloader ---
def collate_batch(batch):
    x_train = [x for x, _ in batch]
    y_train = [y for _, y in batch]                  
        
    return np.array(x_train), np.array(y_train)


if __name__ == '__main__':    
    dataset_dir = os.path.join(os.path.expanduser('~'), config['dataset_dir'])            

    print("Loading dataset...")    
    data = mel_dataset(dataset_dir, config['top_genre'])
    print(f'Loaded data : {len(data)}\n')
    
    dataset_size = len(data)
    train_size = int(dataset_size * 0.8)    
    test_size = dataset_size - train_size
    
    linear_train_size = int(train_size * 0.1)
    linear_test_size = int((train_size * 0.1) * 0.25)
    dummy_data = train_size - (linear_train_size + linear_test_size)
    
    train_dataset, test_dataset, = random_split(data, [train_size, test_size])

    linear_train_dataset, linear_test_dataset, _ = random_split(train_dataset, [linear_train_size, linear_test_size, dummy_data])
    
    
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size']*8, shuffle=True, num_workers=0, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=int(config['batch_size']/4)*8, shuffle=True, num_workers=0, collate_fn=collate_batch)
    
    linear_train_dataloader = DataLoader(linear_train_dataset, batch_size=config['batch_size']*8, shuffle=True, num_workers=0, collate_fn=collate_batch)
    
    linear_test_dataloader = DataLoader(linear_test_dataset, batch_size=int(config['batch_size']/4)*8, shuffle=True, num_workers=0, collate_fn=collate_batch)


    print(f"batch_size = {config['batch_size']}")
    print(f"learning rate = {config['learning_rate']}")
    print(f"train_size = {train_size}")
    print(f"test_size = {test_size}")
    print(f"linear_train_size = {linear_train_size}")
    print(f"linear_test_size = {linear_test_size}")
    

    
    print('Data load complete!\n')

    trainer = TrainerModule(seed=42, config=config)
    
    trainer.train_model(train_dataloader, test_dataloader, num_epochs=config['pretrain_epoch'])
    trainer.save_model('freeze', trainer.state.step)
    
    
    trainer.load_model('freeze', pretrained=True)
    trainer.linear_evaluation(supervised=False)
    trainer.linear_train_model(linear_train_dataloader, linear_test_dataloader, 'semi-supervised',num_epochs=config['linear_evaluation_epoch'])
    trainer.save_model('unfreeze', trainer.state.step)
    
    trainer.linear_evaluation(supervised=True)
    trainer.linear_train_model(linear_train_dataloader, linear_test_dataloader, 'supervised', num_epochs=config['linear_evaluation_epoch'])
    trainer.save_model('supervised', trainer.state.step)
