import torch
from torch import nn
from torch.nn import functional as F

def CUDA(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x

def temp_sigmoid(x, temp=1.0):
    return torch.sigmoid(x/temp) 

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32, num_hidden=1, output_activation=None):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = nn.ReLU()
        self.output_activation = output_activation

        self.fc_list = nn.ModuleList()
        self.fc_list.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_hidden):
            self.fc_list.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        assert x.shape[-1] == self.input_dim
        for i in range(len(self.fc_list)):
            x = self.activation(self.fc_list[i](x))
        
        # no activation for the last layer
        if self.output_activation is not None:
            return self.output_activation(self.output_fc(x))
        return self.output_fc(x)


class CausalSim2Real(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim=10, hidden_dim=64, causal_dim=32, num_hidden=4, sparse_weight=0.01, sparse_norm=1.0, use_full=False, action_size=4):
        super(CausalSim2Real, self).__init__()
        self.input_dim = input_dim         # S
        self.output_dim = output_dim       # A
        self.causal_dim = causal_dim       # C
        self.emb_dim = emb_dim             # E

        # sharable encoder
        self.encoder = MLP(self.emb_dim+1, causal_dim, hidden_dim, num_hidden)
        self.encoder_idx_emb = CUDA(nn.Embedding(self.input_dim, self.emb_dim))

        # sharable decoder
        # self.decoder = MLP(causal_dim+self.emb_dim, 1, hidden_dim, num_hidden, output_activation=torch.sigmoid)
        self.decoder = MLP(causal_dim+self.emb_dim, 1, hidden_dim, num_hidden)
        self.decoder_idx_emb = CUDA(nn.Embedding(self.output_dim, self.emb_dim))

        self.use_full = use_full
        self.sparse_weight = sparse_weight
        self.sparse_norm = sparse_norm
        self.tau = 1
        self.mask_prob = nn.Parameter(3*torch.ones(input_dim, output_dim), requires_grad=True)
        self.mask = CUDA(torch.ones_like(self.mask_prob))
        self.action_size = action_size

    def forward(self, inputs, threshold=None):
        # inputs - [B, S+A, 1]
        assert len(inputs.shape) == 2
        inputs = inputs.unsqueeze(-1)

        # obtain state-action idx embedding
        # [S+A, E] -> [B, S+A, E+1]
        encoder_idx = self.encoder_idx_emb(CUDA(torch.arange(0, self.input_dim).long()))
        batch_encoder_idx = encoder_idx.repeat(inputs.shape[0], 1, 1).detach()
        inputs_feature = torch.cat([inputs, batch_encoder_idx], dim=-1)

        # encoder: [B, S+A, E+1] -> [B, S+A, C]
        latent_feature = self.encoder(inputs_feature) # [B, S+A, C]

        # prepare mask
        if not self.use_full:
            _, self.mask = self.get_mask(threshold)

        # feature mask: [B, S+A, C] x [S+A, S] -> [B, S, C]
        masked_feature = torch.einsum('bnc, ns -> bsc', latent_feature,  self.mask)

        # obtain state idx embedding
        # [S, E] -> [B, S, E+1]
        decoder_idx = self.decoder_idx_emb(CUDA(torch.arange(0, self.output_dim).long()))
        batch_decoder_idx = decoder_idx.repeat(inputs.shape[0], 1, 1).detach()
        masked_feature = torch.cat([masked_feature, batch_decoder_idx], dim=-1)

        # decoder: [B, S, E+C] -> [B, S]
        next_state = self.decoder(masked_feature).squeeze(-1)
        return next_state

    def loss_function(self, s_next_pred, s_next):
        mse = F.mse_loss(s_next_pred, s_next)
        sparse = self.sparse_weight * torch.mean(torch.sigmoid(self.mask_prob[:-self.action_size, :])**self.sparse_norm)
        info = {'mse': mse.item(), 'sparsity': sparse.item()}
        loss = mse + sparse
        return loss, info

    def get_mask(self, threshold):
        # prepare maskf
        mask_prob = temp_sigmoid(self.mask_prob, temp=1.0) 

        if threshold is not None:
            # use threshold to sample a mask
            mask = (mask_prob > threshold).float()
        else:
            # use gumbel softmax to sample a mask
            # [S+A, S] -> [S+A, S, 2]
            # build a bernoulli distribution with p=1-mask_prob and q=mask_prob
            mask_bernoulli = torch.cat([(1-mask_prob).unsqueeze(-1), mask_prob.unsqueeze(-1)], dim=-1).log()

            # (S+A) x S x 2 -> (S+A) x S x 2 - (S+A) x S
            mask = F.gumbel_softmax(mask_bernoulli, tau=self.tau, hard=True, dim=-1) 
            mask = mask[:, :, 1] # just keep the second channel since the bernoulli distribution is [0, 1]
        
        return mask_prob, mask

    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
