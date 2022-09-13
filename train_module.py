# 2022-09-02 16:23 Seoul

# --- import dataset ---
import utils.losses 
from torch.utils.data import DataLoader, random_split

# --- import model ---
from model.supervised_model import *
from model.Conv2d_model import Conv2d_VAE, Encoder

# --- import framework ---
import flax 
from flax import jax_utils
from flax.training import train_state, common_utils, checkpoints
from flax.core.frozen_dict import unfreeze, freeze
import jax
import numpy as np
import jax.numpy as jnp
import optax

from tqdm import tqdm
import os
import wandb
from utils.config_hook import yaml_config_hook
import matplotlib.pyplot as plt

from functools import partial

# ---- top_k ----
@partial(jax.jit, static_argnames=['k'])
def top_k(logits, y,k):
    top_k = jax.lax.top_k(logits, k)[1]
    ts = jnp.argmax(y, axis=1)
    correct = 0
    for i in range(ts.shape[0]):
        b = (jnp.where(top_k[i,:] == ts[i], jnp.ones((top_k[i,:].shape)), 0)).sum()
        correct += b
    correct /= ts.shape[0]
    return correct 

# --- Define config ---
config_dir = os.path.join(os.path.expanduser('~'),'trainer_module/config')     
config = yaml_config_hook(os.path.join(config_dir, 'config.yaml'))

class TrainerModule:

    def __init__(self, 
                 seed,
                 config):
        
        super().__init__()
        self.config = config
        self.seed = seed
        self.exmp = jnp.ones((self.config['batch_size'], 48, 1876))
        # Create empty model. Note: no parameters yet
        self.model = Conv2d_VAE(dilation=self.config['dilation'], latent_size=self.config['latent_size'])
        self.Encoder = Encoder(dilation=self.config['dilation'], linear=True, n_features=self.config['top_genre'])
        # Prepare logging
        self.log_dir = self.config['checkpoints_path']
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model()
        wandb.init(
        project=config['model_type'],
        entity='aiffelthon',
        config=config
        )
        
    def create_functions(self):
        # Training function
        def train_step(state, batch):
            
            def loss_fn(params):
                mel, _ = batch
                mel = (mel/100) + 1
                recon_x = self.model.apply(params, mel)
                loss = ((recon_x - mel) ** 2).mean()
                return loss
            
            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params)
            grads = jax.lax.pmean(grads, axis_name='batch')
            state = state.apply_gradients(grads=grads)  # Optimizer update step
            
            return state, loss
        
        self.train_step = jax.pmap(partial(train_step), axis_name='batch')
        
        # Eval function        
        def eval_step(state, batch):
            mel, _ = batch
            mel = (mel/100) + 1
            recon_x = self.model.apply(state.params, mel)
            loss = ((recon_x - mel) ** 2).mean()
            
            return loss, recon_x        
        self.eval_step = jax.pmap(partial(eval_step), axis_name='batch')
        
        # linear train function 
        def linear_train_step(state, batch):    
      
            def loss_fn(params):
                mel, label = batch
                mel = (mel/100) + 1
                logits = Encoder(dilation=self.config['dilation'],
                            linear=True, n_features=self.config['top_genre']).apply(params, mel)        
                loss = jnp.mean(optax.softmax_cross_entropy(logits, label))
                return loss, logits

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, logits), grads = grad_fn(state.params)
            grads = jax.lax.pmean(grads, axis_name='batch')
            _, label = batch
            accuracy = top_k(logits, label, 1)
            top_k_accuracy = top_k(logits, label, self.config['top_k'])
            return state.apply_gradients(grads=grads), loss, accuracy, top_k_accuracy
        self.linear_train_step = jax.pmap(partial(linear_train_step), axis_name='batch')
        
        # linear eval function
        def linear_eval_step(state, batch):              
            mel, label = batch
            mel = (mel/100) + 1
            logits = Encoder(dilation=self.config['dilation'],
                        linear=True, n_features=self.config['top_genre']).apply(state.params, mel)        
            loss = jnp.mean(optax.softmax_cross_entropy(logits, label)) 
            accuracy = top_k(logits, label, 1)
            top_k_accuracy = top_k(logits, label, self.config['top_k'])
            return loss, accuracy, top_k_accuracy
        self.linear_eval_step = jax.pmap(partial(linear_eval_step), axis_name='batch')
        
        
    def init_model(self):
        # Initialize model
        rng = jax.random.PRNGKey(self.seed)
        rng, init_rng = jax.random.split(rng)
        params = self.model.init(init_rng, self.exmp)
        # Initialize optimizer
        optimizer = optax.adam(self.config['learning_rate'])
        
        # Initialize training state
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=optimizer)

    
    
    # ---- pretrain model ----
    
    def train_model(self, train_dataloader, test_dataloader, num_epochs=5):
        # Train model for defined number of epochs
        
        self.state = jax_utils.replicate(self.state)
        for epoch_idx in range(1, num_epochs+1):
            self.train_epoch(epoch_idx, train_dataloader, test_dataloader)

        self.state = jax_utils.unreplicate(self.state)
        
    def train_epoch(self, epoch, train_dataloader, test_dataloader):
        train_dataiter = iter(train_dataloader)
        test_dataiter = iter(test_dataloader)
        
        for batch in tqdm(range(len(train_dataloader)-1), desc=f'Epoch {epoch}'):
            train_batch = common_utils.shard(jax.tree_util.tree_map(np.asarray, next(train_dataiter)))
            test_batch = common_utils.shard(jax.tree_util.tree_map(np.asarray, next(test_dataiter)))
            
            self.state, train_loss = self.train_step(self.state, train_batch)
            test_loss, recon_x = self.eval_step(self.state, test_batch)
            
            wandb.log({'train_loss': jax.device_get(train_loss.mean()), 'test_loss': jax.device_get(test_loss.mean())})
            
            if self.state.step[0] % 100 == 0:
                recon_x = jax_utils.unreplicate(recon_x)
                fig1, ax1 = plt.subplots()
                im1 = ax1.imshow(recon_x[0], aspect='auto', origin='lower', interpolation='none')
                fig1.colorbar(im1)
                fig1.savefig('recon.png')
                plt.close(fig1)
                
                test_x = jax_utils.unreplicate(test_batch[0])
                fig2, ax2 = plt.subplots()
                im2 = ax2.imshow(test_x[0], aspect='auto', origin='lower', interpolation='none')
                fig2.colorbar(im2)
                fig2.savefig('x.png')
                plt.close(fig2)
                
                wandb.log({'reconstruction' : [
                            wandb.Image('recon.png')
                            ], 
                           'original image' : [
                            wandb.Image('x.png')
                            ]})
                
    # ---- linear evaluation ---- 
    
    # change self state
    def linear_evaluation(self, supervised=False):
        if supervised:
            rng = jax.random.PRNGKey(self.seed)
            params = Encoder(dilation=config['dilation'], linear=True, n_features=self.config['top_genre']).init(rng, self.exmp)            
            optimizer = optax.adam(learning_rate=self.config['learning_rate'])
            self.state = train_state.TrainState.create(
                apply_fn=Encoder(dilation=config['dilation'], linear=True, n_features=self.config['top_genre']).apply,
                    tx=optimizer,
                    params=params
            )
        else:
            rng = jax.random.PRNGKey(self.seed)
            linear_init = linear_evaluation(hidden_layer=self.config['hidden_layer'], n_features=self.config['top_genre']).init(rng, jnp.ones((self.config['batch_size'], self.config['latent_size'])))

            enc_unfreeze_variable = unfreeze(self.state.params)
            enc_state = enc_unfreeze_variable['params']['encoder']
            enc_batch = enc_unfreeze_variable['batch_stats']['encoder']
            params = {}
            params['params'] = enc_state
            params['batch_stats'] = enc_batch

            params['params']['linear_hidden_layer'] = linear_init['params']['linear_hidden_layer']
            params['params']['linear_classification'] = linear_init['params']['linear_classification']


            params = freeze(params)
            optimizer = optax.adam(learning_rate=self.config['learning_rate'])

            self.state = train_state.TrainState.create(
                    apply_fn=Encoder(dilation=config['dilation'], linear=True, n_features=self.config['top_genre']).apply,
                    tx=optimizer,
                    params=params)
        
    def linear_train_model(self, train_dataloader, test_dataloader, train_type='semi-supervised', num_epochs=5):
                                                                                       
        self.state = jax_utils.replicate(self.state)
        for epoch_idx in range(1, num_epochs+1):
            self.linear_train_epoch(epoch_idx, train_dataloader, test_dataloader, train_type=train_type)

        self.state = jax_utils.unreplicate(self.state)

                                                                                       
    def linear_train_epoch(self, epoch, train_dataloader, test_dataloader, train_type):
        train_dataiter = iter(train_dataloader)
        test_dataiter = iter(test_dataloader)
        
        for batch in tqdm(range(len(train_dataloader)-1), desc=f'Epoch {epoch}'):
            train_batch = common_utils.shard(jax.tree_util.tree_map(np.asarray, next(train_dataiter)))
            test_batch = common_utils.shard(jax.tree_util.tree_map(np.asarray, next(test_dataiter)))
            
            self.state, train_loss, train_accuarcy, train_top_k_accuarcy = self.linear_train_step(self.state, train_batch)
            test_loss, test_accuarcy, test_top_k_accuarcy = self.linear_eval_step(self.state, test_batch)
            
            wandb.log({f'{train_type}_linear_train_loss' : jax.device_get(train_loss.mean()), 
                       f'{train_type}_linear_test_loss' : jax.device_get(test_loss.mean()), 
                       f'{train_type}_linear_train_accuarcy': jax.device_get(train_accuarcy.mean()),
                       f'{train_type}_linear_test_accuarcy': jax.device_get(test_accuarcy.mean()),
                       f'{train_type}_linear_train_top_k_accuarcy': jax.device_get(train_top_k_accuarcy.mean()),
                       f'{train_type}_linear_test_top_k_accuarcy': jax.device_get(test_top_k_accuarcy.mean())})     
    
    

    
    # ---- etc ----
    
    def save_model(self, checkpoint_name, step=0):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir, target=self.state.params, prefix=f"{checkpoint_name}_{config['latent_size']}_", step=step)

    def load_model(self, checkpoint_name, pretrained=False):
        # Load model. We use different checkpoint for pretrained models
        if not pretrained:
            params = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=self.state.params, prefix=f"{checkpoint_name}_{config['latent_size']}")
        else:
            params = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(self.log_dir, f"{checkpoint_name}_{config['latent_size']}"), target=self.state.params)
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=self.state.tx)

    def checkpoint_exists(self):
        # Check whether a pretrained model exist for this autoencoder
        return os.path.isfile(os.path.join(CHECKPOINT_PATH, f"{config['projector_target']}_{config['latent_size']}.ckpt"))
    
    
          






