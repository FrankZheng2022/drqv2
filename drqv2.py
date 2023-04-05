# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())
        
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h

class CLIP(nn.Module):
    """
    Constrastive loss
    """

    def __init__(self, repr_dim, feature_dim, action_shape, hidden_dim, encoder, encoder_target, device):
        super(CLIP, self).__init__()

        self.encoder = encoder
        self.encoder_target = encoder_target
        self.device = device
        
        a_dim = action_shape[0]
        latent_a_dim = a_dim
        self.proj_sa = nn.Sequential(
            nn.Linear(feature_dim + latent_a_dim, hidden_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        self.proj_ss = nn.Sequential(
            nn.Linear(feature_dim*2, hidden_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, a_dim)
        )
        
        self.proj_a = nn.Sequential(
            nn.Linear(a_dim, latent_a_dim), 
            nn.LayerNorm(latent_a_dim), nn.Tanh()
        )
        
        self.proj_s = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
        
        self.W = nn.Parameter(torch.rand(feature_dim, feature_dim))
        self.inv_W = nn.Parameter(torch.rand(latent_a_dim, latent_a_dim))
        self.apply(utils.weight_init)
    
    def encode(self, x, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.proj_s(self.encoder_target(x))
        else:
            z_out = self.proj_s(self.encoder(x))
        return z_out
    
    def project_sa(self, s, a):
        x = torch.concat([s,a], dim=-1)
        return self.proj_sa(x)
    
    def project_ss(self, s, next_s):
        x = torch.concat([s, next_s], dim=-1)
        return self.proj_ss(x)
    
    ### barlow twins loss
    def compute_bt(self, z_a, z_b, lambda_param=5e-3):
        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD

        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
        # loss
        c_diff = (c - torch.eye(D,device=self.device)).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= lambda_param
        loss = c_diff.sum()

        return loss
        
    def compute_logits(self, z_a, z_pos, inv=False):
        """
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        
        W  = self.inv_W if inv else self.W
        Wz = torch.matmul(W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits
    
    
class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                    nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)
        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                    nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2
    

class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, encoder_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb, ema, loss_type, inv):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.encoder_tau = encoder_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.ema = ema
        self.loss_type = loss_type
        self.inv = inv

        # models
        self.encoder = Encoder(obs_shape, feature_dim).to(device)
        self.encoder_target = Encoder(obs_shape, feature_dim).to(device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.CLIP = CLIP(self.encoder.repr_dim, feature_dim, action_shape, hidden_dim, self.encoder, self.encoder_target, device).to(device)
        
        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.clip_opt = torch.optim.Adam(self.CLIP.parameters(), lr=lr)
        
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.CLIP.train()

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics
    
    def update_clip(self, obs, action, next_obs):
        metrics = dict()
        
        obs_anchor = self.aug(obs.float())
        obs_pos = self.aug(obs.float())
        z_a = self.CLIP.encode(obs_anchor)
        z_pos = self.CLIP.encode(obs_pos, ema=self.ema)
        if self.loss_type == 'bt':
            curl_loss = self.CLIP.compute_bt(z_a, z_pos)
        else:
            ### Compute the original loss for CURL
            logits = self.CLIP.compute_logits(z_a, z_pos)
            labels = torch.arange(logits.shape[0]).long().to(self.device)
            curl_loss = self.cross_entropy_loss(logits, labels)
        
        ### Compute loss for consistency
        with torch.no_grad():
            next_obs = self.aug(next_obs.float())
            next_z = self.CLIP.encode(next_obs, ema=self.ema)
        
        action_en = self.CLIP.proj_a(action)
        curr_za = self.CLIP.project_sa(z_a, action_en)
        if self.loss_type == 'bt':
            consistency_loss = self.CLIP.compute_bt(curr_za, next_z)
        else:
            logits = self.CLIP.compute_logits(curr_za, next_z)
            labels = torch.arange(logits.shape[0]).long().to(self.device)
            consistency_loss = self.cross_entropy_loss(logits, labels)
        
        ### Compute loss for inverse_prediction
        if self.inv:
            pred_action = self.CLIP.project_ss(z_a, next_z)
            inv_consistency_loss = self.mse_loss(pred_action, action)
        else:
            inv_consistency_loss = torch.tensor(0.)
        
        self.encoder_opt.zero_grad()
        self.clip_opt.zero_grad()
        (inv_consistency_loss + consistency_loss + curl_loss).backward()
        
        self.encoder_opt.step()
        self.clip_opt.step()
        if self.use_tb:
            metrics['curl_loss'] = curl_loss.item()
            metrics['fwd_loss']  = consistency_loss.item()
            metrics['inv_loss']  = inv_consistency_loss.item()
        return metrics
        
        
    
    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, r_next_obs = utils.to_torch(
            batch, self.device)

        # augment
        obs_en = self.aug(obs.float())
        next_obs_en = self.aug(next_obs.float())
        # encode
        obs_en = self.encoder(obs_en)
        with torch.no_grad():
            if self.ema:
                next_obs_en = self.encoder_target(next_obs_en)
            else:
                next_obs_en = self.encoder(next_obs_en)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs_en, action, reward, discount, next_obs_en, step))

        # update actor
        metrics.update(self.update_actor(obs_en.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)
        # update critic target
        utils.soft_update_params(self.encoder, self.encoder_target,
                                 self.encoder_tau)
        if self.loss_type in ['bt', 'cpc']:
            metrics.update(self.update_clip(obs, action, r_next_obs))
        
        return metrics
