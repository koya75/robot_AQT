import torch
from torch import nn
import pfrl

use_cuda = torch.cuda.is_available()

class DQN(nn.Module):
    def __init__(self, n_actions, head_dim=32, num_encoder_layers=1, num_decoder_layers=1, device="cpu"):
        super(DQN, self).__init__()
        """
        DETR input (cnn) -> Transformer Encoder -> Transformer Decoder -> Q 
                            action query        ->
        """
        print("DQN tegaki")
        
        self.device = device

        self.n_actions = n_actions
        hidden_dim = 128
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        """encoder_layer = TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu")
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu")
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)        

        # output positional encodings (object queries)
        #self.query_action = nn.Parameter(torch.rand(self.n_actions, hidden_dim))
        
        #self.query_action = torch.zeros(self.n_actions, self.n_actions, device=torch.device(self.device))
        #for i in range(self.n_actions):
        #    self.query_action[i][i] = 1.0
        #query_As = []
        
        #for i in range(self.n_actions):
        #    query_As.append(torch.ones(1, hidden_dim, device=torch.device(device)) * (i+1))
        #self.query_action = torch.cat(query_As, dim=0)
        
        self.act_list = torch.zeros(self.n_actions, self.n_actions, device=torch.device(device))
        for i in range(self.n_actions):
            self.act_list[i][i] = 1.0
        self.action_encoder = nn.Linear(self.n_actions, hidden_dim)

        # spatial positional encodings (note that in baseline DETR we use sine positional encodings)
        self.row_embed = nn.Parameter(torch.rand(7, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(7, hidden_dim // 2))"""
        
        self.Q_linear = nn.Linear(32768, self.n_actions)
        self.q_f = pfrl.q_functions.DiscreteActionValueHead()
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, recurrent_state=None, item=None):
        # construct positional encodings
        x = self.conv(x).flatten(2)
        x = x.reshape(x.size(0), -1)
        
        """bs, c, h, w = x.shape
        pos = torch.cat([
            self.col_embed[:w].unsqueeze(0).repeat(h, 1, 1),
            self.row_embed[:h].unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        pos = pos.expand(pos.shape[0], bs, pos.shape[2])
        src = x.flatten(2).permute(2, 0, 1)
        memory = self.transformer_encoder(pos + 0.1 * src)
        
        query_embed = []
        for action in self.act_list:
            action_query = self.action_encoder(action).unsqueeze(0)
            query_embed.append(action_query)
        query_embed = torch.cat(query_embed, dim=0)
        
        tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)
        #tgt = torch.zeros_like(query_embed)
        hs = self.transformer_decoder(tgt, memory)[0] # 6,32,192
        Qs = self.Q_linear(hs).permute(2, 1, 0).squeeze(0) # 6,32,1 -> 1,32,6 -> 32,6"""

        Qs = self.Q_linear(x)
        Qs = self.q_f(Qs)
        return Qs