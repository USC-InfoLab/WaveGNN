import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import numpy as np
from tqdm import tqdm

from model.PE import PositionalEncodingTF, Time2VecEncoding


class SparseObsSeqEncoder(nn.Module):
    """
    Encoder module for encoding sparse observation sequences that utilizes a decay-based mechanism for aggregation.

    Args:
        obs_dim (int): The dimensionality of the observation.
        hidden_dim (int): The dimensionality of the hidden state.
        n_layers (int): The number of layers in the encoder.
        dropout (float, optional): The dropout probability. Default is 0.5.
    """

    def __init__(
        self,
        obs_dim,
        hidden_dim,
        n_layers,
        time_dim,
        dropout=0.5,
        device="cpu",
        positional_encoding="absolute_transformer",
    ):
        super(SparseObsSeqEncoder, self).__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.time_dim = time_dim
        self.device = device
        self.positional_encoding = positional_encoding

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=obs_dim, nhead=8, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=n_layers
        )
        self.projection = nn.Linear(obs_dim, hidden_dim)

        # define a vector ts_token to represent the entire sequence
        self.ts_token = nn.Parameter(torch.randn(1, 1, obs_dim), requires_grad=True)

        # learnable parameters for the decay-based aggregation
        self.decay_rate = nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def forward(self, obs_seq, mask, timestamps, raw_timestamps):
        """Encode the sparse observation sequence window of each node

        Args:
            obs_seq (Tensor): A tensor of shape (#batch_size, #window_size, #n_sensors, #self.obs_dim) containing the observation sequence.
            mask (Tensor): A tensor of shape (#batch_size, #window_size, #n_sensors) containing the mask for the observation sequence (i.e., if the observation is missing in that timestamp or not).
            timestamps (Tensor): A tensor of shape (#batch_size, #window_size, #self.obs_dim) containing the encoded timestamps of the observations.

        Returns:
            encoded_seq (Tensor): A tensor of shape (#batch_size, #n_sensors, #self.hidden_dim) containing the encoded observation sequence.

        """
        # Apply positional encoding that considers time deltas
        obs_seq = obs_seq + timestamps

        # Apply mask to the sequence
        obs_seq = obs_seq * mask.unsqueeze(
            -1
        )  # Ensures that missing values do not contribute to the attention calculation

        # reshape the sequence for the transformer encoder so that the shape is (batch_size * n_sensors, window_size, obs_dim)
        reshaped_obs_seq = obs_seq.permute(0, 2, 1, 3).reshape(
            -1, obs_seq.shape[1], obs_seq.shape[3]
        )

        # add the ts_token to the start of the sequence
        reshaped_obs_seq = torch.cat(
            [self.ts_token.repeat(reshaped_obs_seq.shape[0], 1, 1), reshaped_obs_seq],
            dim=1,
        )

        # handle the transformer mask
        transformer_mask = mask.permute(0, 2, 1).reshape(-1, obs_seq.shape[1])

        # add the ts_token to the mask so that all tokens can attend to the ts_token
        transformer_mask = torch.cat(
            [
                torch.ones(
                    transformer_mask.shape[0], 1, device=transformer_mask.device
                ),
                transformer_mask,
            ],
            dim=1,
        )
        transformer_mask = (
            ~transformer_mask.bool()
        )  # 0 where there is observation, 1 where there is no observation

        # Pass through the Transformer encoder
        encoded_seq = self.transformer_encoder(
            reshaped_obs_seq, src_key_padding_mask=transformer_mask
        )

        # reshape the encoded sequence back to the original shape
        encoded_seq = encoded_seq.reshape(
            obs_seq.shape[0], obs_seq.shape[2], obs_seq.shape[1] + 1, self.obs_dim
        )

        # get some weights that have higher weights for the recent observations based on timestamps
        agg_weights = self._get_decay_weights(
            raw_timestamps.squeeze(), mask, self.decay_rate
        )
        # aggregate the hidden states based on the weights
        selected_hidden_states = torch.einsum(
            "bws,bswh->bsh", agg_weights, encoded_seq[:, :, 1:, :]
        )

        # Project the encoded sequence to the hidden dimension
        encoded_seq = self.projection(selected_hidden_states)
        return encoded_seq

    def _get_decay_weights(self, timestamps, mask, decay_rate=0.1):
        """Get the decay-based weights for the observations

        Args:
            timestamps (Tensor): A tensor of shape (#batch_size, #window_size) containing the encoded timestamps of the observations.
            mask (Tensor): A tensor of shape (#batch_size, #window_size, #n_sensors) containing the mask for the observation sequence (i.e., if the observation is missing in that timestamp or not).
            decay_rate (float): The decay rate for the exponential decay function.

        Returns:
            agg_weights (Tensor): A tensor of shape (#batch_size, #window_size, #n_sensors) containing the decay-based weights for the observations.
        """
        if (
            timestamps.dim() == 1
        ):  # this happens sometimes that batch size is 1, the batch dimension is removed in this case
            timestamps = timestamps.unsqueeze(0)

        # Calculate time differences between the most recent timestamp and each observation
        if timestamps.dim() == 1:
            print("timestamps shape: ", timestamps.shape)
            timestamps = timestamps.unsqueeze(0)
        most_recent_timestamp = timestamps.max(dim=1, keepdim=True)[0]
        time_diffs = most_recent_timestamp - timestamps

        # Apply the exponential decay function
        decay_weights = torch.exp(-decay_rate * time_diffs)

        # Repeat decay weights for each sensor
        decay_weights = decay_weights.unsqueeze(-1).repeat(1, 1, mask.shape[-1])

        # Apply the mask to ignore missing observations
        decay_weights = decay_weights * mask

        # Normalize the weights
        agg_weights = decay_weights / (decay_weights.sum(dim=1, keepdim=True) + 1e-8)

        return agg_weights


class GNN_Module(nn.Module):
    """
    Graph Neural Network (GNN) module.

    Args:
        node_hidden_dim (int): The dimension of the hidden node features.
        device (str, optional): The device to run the module on (default: 'cpu').
        conv_num_layers (int, optional): The number of convolutional layers in the GNN (default: 3).
    """

    def __init__(self, node_hidden_dim, device="cpu", conv_num_layers=3):
        super(GNN_Module, self).__init__()
        self.node_feature_dim = node_hidden_dim
        self.conv_layers_num = conv_num_layers
        self.conv_hidden_dim = node_hidden_dim

        self.convs = nn.ModuleList()
        for _ in range(self.conv_layers_num):
            self.convs.append(
                pyg_nn.GCNConv(
                    self.conv_hidden_dim,
                    self.conv_hidden_dim,
                    add_self_loops=False,
                    normalize=False,
                )
            )
        self.layer_norms = nn.ModuleList(
            nn.LayerNorm(self.conv_hidden_dim) for _ in range(self.conv_layers_num)
        )

        # initialize weights using xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
        self.to(device)

    def forward(self, X, adj_mat):
        """
        Forward pass of the GNN module.

        Args:
            X (torch.Tensor): The input node features of shape (#batch_size, #self.n_sensors, #self.hidden_dim).
            adj_mat (torch.Tensor): The adjacency matrix of shape (#batch_size, #self.n_sensors, #self.n_sensors).

        Returns:
            torch.Tensor: The output node features after applying the GNN of shape (#batch_size, #self.n_sensors, node_hidden_dim).
        """
        X_gnn_list = []
        for batch_i in range(X.shape[0]):
            edge_indices, edge_attrs = pyg_utils.dense_to_sparse(adj_mat[batch_i])
            X_gnn = X[batch_i]
            for stack_i in range(self.conv_layers_num):
                X_res = X_gnn  # Store the current state for the residual connection
                X_gnn = self.convs[stack_i](X_gnn, edge_indices, edge_attrs)
                X_gnn = F.relu(
                    self.layer_norms[stack_i](X_gnn + X_res)
                )  # Add the residual connection
            X_gnn_list.append(X_gnn)
        X_gnn = torch.stack(X_gnn_list, dim=0)
        return X_gnn


class GraphEmbedding(nn.Module):
    """
    A class representing the graph embedding module.

    Args:
        n_sensors (int): The number of sensors in the graph.
        nodes_hidden_dim (int): The dimension of the hidden states of the nodes.
        graph_embedding_dim (int): The dimension of the graph embedding.
    """

    def __init__(self, n_sensors, nodes_hidden_dim, graph_embedding_dim):
        super(GraphEmbedding, self).__init__()
        self.n_sensors = n_sensors
        self.nodes_hidden_dim = nodes_hidden_dim
        self.graph_embedding_dim = graph_embedding_dim

        # Modules for embedding the graph
        self.mlp = nn.Sequential(
            nn.Linear(
                self.n_sensors * self.nodes_hidden_dim,
                (graph_embedding_dim * self.n_sensors // 2),
            ),
            nn.ReLU(),
            nn.Linear((graph_embedding_dim * self.n_sensors // 2), graph_embedding_dim),
            nn.ReLU(),
            nn.Linear(graph_embedding_dim, graph_embedding_dim),
            nn.ReLU(),
        )

        # pooling layer for max-pooling the node states feature-wise
        self.max_pooling = nn.MaxPool1d(self.n_sensors)

        # Final MLP to get the final graph embedding
        self.final_mlp = nn.Sequential(
            nn.Linear(graph_embedding_dim + nodes_hidden_dim, graph_embedding_dim),
            nn.ReLU(),
            nn.Linear(graph_embedding_dim, graph_embedding_dim),
            nn.ReLU(),
        )

    def forward(self, node_states):
        """
        Embed the graph using the node states.

        Args:
            node_states (Tensor): A tensor of shape (#batch_size, #n_sensors, #nodes_hidden_dim) containing the node states.

        Returns:
            graph_embedding (Tensor): A tensor of shape (#batch_size, #graph_embedding_dim) containing the graph embedding.
        """
        # Flatten the node states
        flattened_node_states = node_states.view(
            -1, self.n_sensors * self.nodes_hidden_dim
        )

        # Embed the graph
        graph_embedding = self.mlp(flattened_node_states)

        # Pool the node states feature-wise
        pooled_node_states = self.max_pooling(node_states.permute(0, 2, 1)).squeeze(-1)

        # Concatenate the graph embedding with the pooled node states
        graph_embedding = torch.cat([graph_embedding, pooled_node_states], dim=-1)

        # Apply a final MLP to get the final graph embedding
        graph_embedding = self.final_mlp(graph_embedding)

        return graph_embedding


class WaveGNN(nn.Module):
    """
    WaveGNN is a graph neural network model for Multivariate Irregularly Sample Time Series (MISTS) classification.

    Args:
        n_sensors (int): The number of sensors in the graph.
        static_features_len (int): The length of the static features.
        hidden_dim (int): The dimension of the hidden states of the nodes.
        num_attention_heads (int): The number of attention heads for the multi-head self-attention layer.
        args (argparse.Namespace): The arguments for the model.
    """

    def __init__(self, n_sensors, static_features_len, hidden_dim, args):
        super(WaveGNN, self).__init__()
        self.n_sensors = n_sensors
        self.n_classes = args.n_classes
        self.window_size = args.window_size
        self.dropout = args.dropout
        self.static_features_len = static_features_len
        self.hidden_dim = hidden_dim
        self.num_attention_heads = args.num_attention_heads
        self.observation_dim = args.observation_dim
        self.device = args.device
        self.positional_encoding = args.positional_encoding
        self.dataset = args.dataset
        self.args = args

        # Modules for initialize the node states
        # self.state_init_mlp = nn.Linear(static_features_len, n_sensors*hidden_dim)

        if not self.dataset.startswith("MIMIC3"):
            self.state_init_mlp = nn.Sequential(
                nn.Linear(static_features_len, 64),
                nn.Tanh(),
                nn.Linear(64, 128),
                nn.Tanh(),
                nn.Linear(128, n_sensors * hidden_dim // 4),
                nn.Tanh(),
                nn.Linear(n_sensors * hidden_dim // 4, n_sensors * hidden_dim),
            )

        # Modules for embedding the observations
        self.obs_embedding_module = self._create_observation_embedding_layer(
            self.observation_dim, hidden_dim // 2
        )

        # Modules for updating the node states
        self.obs_encoder = SparseObsSeqEncoder(
            hidden_dim // 2,
            hidden_dim,
            2,
            self.dropout,
            device=self.device,
            positional_encoding=self.positional_encoding,
        )

        # A weight matrix containing learned node embeddings for each sensor
        # This will be used to get long-term dependencies between sensors
        # (Similar to BysGNN's static adjacency matrices, but learned here)
        self.global_node_embeddings = nn.Parameter(torch.randn(n_sensors, hidden_dim))

        # A multi-head self-attention layer to get the attention weights for dynamic short-term dependencies
        # I.e., dynamic adjacency matrix for each window
        self.mhead_attn_layer = nn.MultiheadAttention(
            self.hidden_dim, self.num_attention_heads, batch_first=True
        )

        # alpha is a trainable parameter for the laplacian matrix (how much multihead attention is important, how much the global node embeddings are important.)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        # Message passing layers
        self.gnn_block = GNN_Module(hidden_dim)

        # Graph embedding layer
        self.graph_embedding = GraphEmbedding(n_sensors, hidden_dim, hidden_dim)

        # Final MLP for classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_classes),
        )

        # Timestamp encoding layer
        if self.positional_encoding == "absolute_transformer":
            self.timestamp_encoder = PositionalEncodingTF(
                d_model=hidden_dim // 2, max_len=500, MAX=10000
            )
        else:
            self.timestamp_encoder = Time2VecEncoding(
                in_feats=1, out_feats=hidden_dim // 2
            )

    def forward(
        self,
        x,
        mask,
        timestamps,
        static_features,
        relative_timestamps=None,
    ):
        """
        Forward pass of the WaveGNN model.

        Args:
            x (Tensor): Input tensor of shape (#batch_size, #window_size, #n_sensors, #observation_dim).
            mask (Tensor): Mask tensor of shape (#batch_size, #window_size, #n_sensors) indicating valid observations.
            timestamps (Tensor): Tensor of shape (#batch_size, #window_size) containing the timestamps of the observations.
            static_features (Tensor): A tensor of shape (#batch_size, #features) containing the static features.

        Returns:
            output (Tensor): Output tensor of shape (#batch_size, #n_classes) containing the predicted class probabilities.
        """
        raw_timestamps = timestamps
        # encode timestamps
        if self.positional_encoding == "absolute_transformer":
            timestamps = self.timestamp_encoder(timestamps)
            # repeat the timestamps for each sensor
            timestamps = timestamps.unsqueeze(2).repeat(1, 1, x.shape[2], 1)
        else:
            timestamps = self.timestamp_encoder(relative_timestamps)

        # initialize the node states based on the static features (e.g., patient demographics)
        node_states = 0
        if not self.dataset.startswith(
            "MIMIC3"
        ):  # for MIMIC3 dataset, we don't have static features
            node_states = self._init_node_states(static_features)

        # Embed the observations, encode the observation sequence, and get the delta update for the node states
        delta_update = self._get_delta_update(
            x,
            mask,
            timestamps,
            raw_timestamps,
            self.obs_embedding_module,
            self.obs_encoder,
        )

        # Update the node states
        node_states += delta_update

        # Now we need to calculate the attention weights for the dynamic short-term dependencies between sensors
        _, attn_weights = self.mhead_attn_layer(
            node_states, node_states, node_states, need_weights=True
        )
        # attn_laplacian = torch.softmax(attn_weights, dim = -1)
        attn_laplacian = attn_weights

        # Get the long-term dependencies between sensors based on the global node embeddings
        scores = torch.matmul(
            self.global_node_embeddings, self.global_node_embeddings.T
        )

        # mask the diagonal elements of the laplacian matrix
        mask = torch.eye(scores.size(0), device=scores.device).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        # apply softmax to get the long-term similarity scores
        base_laplacian = F.softmax(scores, dim=-1)

        # Combine the long-term and short-term dependencies
        laplacian = self.alpha * attn_laplacian + (1 - self.alpha) * base_laplacian

        # threshold the laplacian matrix
        laplacian = self._get_sparse_laplacian(
            laplacian, method="percentile", threshold=0.3
        )

        node_states = self.gnn_block(node_states, laplacian)

        # Get the graph embedding
        graph_embedding = self.graph_embedding(node_states)

        # Get the final class probabilities
        logits = self.classifier(graph_embedding)

        return logits

    def _init_node_states(self, static_features):
        """
        Initialize the node states of the graph.

        Args:
            static_features (Tensor): A tensor of shape (#batch_size, #features) containing the static features.

        Returns:
            node_states (Tensor): A tensor of shape (#batch_size, #self.n_sensors, #self.hidden_dim) containing the node states.
        """
        node_states = self.state_init_mlp(static_features).view(
            -1, self.n_sensors, self.hidden_dim
        )
        return node_states

    def _create_observation_embedding_layer(self, observation_dim, embedding_dim):
        """
        Create the observation embedding layer.

         Args:
             observation_dim (int): The dimension of the observation.
             embedding_dim (int): The dimension of the observation embedding.

         Returns:
             observation_embedding (nn.Sequential): A sequential module that embeds the observations.
        """
        obs_embedding_module = nn.Sequential(
            nn.Linear(observation_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, embedding_dim),
            nn.Tanh(),
        )
        return obs_embedding_module

    def _get_delta_update(
        self,
        obs_seq,
        mask,
        timestamps,
        raw_timestamps,
        obs_embedding_module,
        obs_seq_encoder,
    ):
        """Get a delta update for the node states based on the observation sequence.

        Args:
            obs_seq (Tensor): A tensor of shape (#batch_size, #window_size, #n_sensors, #observation_dim) containing the observation sequence.
            mask (Tensor): A tensor of shape (#batch_size, #window_size, #n_sensors) containing the mask for the observation sequence (i.e., which timestamps the observation for the corresponding sensor is present).
            timestamps (Tensor): A tensor of shape (#batch_size, #window_size) containing the timestamps of the observations in the sequence.
            obs_embedding_module (nn.Sequential): A module that embeds each observation.
            obs_seq_encoder (SparseObsSeqEncoder): The encoder module for encoding the sequence of embedded observations.

        Returns:
            delta_update (Tensor): A tensor of shape (#batch_size, #n_sensors, #hidden_dim) containing the delta update for the node states.
        """
        embedded_obs_seq = obs_embedding_module(obs_seq)
        delta_update = obs_seq_encoder(
            embedded_obs_seq, mask, timestamps, raw_timestamps
        )
        return delta_update

    def _get_sparse_laplacian(
        self, laplacian_matrix, method="percentile", threshold=0.3
    ):
        """
        Get a sparse laplacian matrix by thresholding the given laplacian matrix.

        Args:
            laplacian_matrix (Tensor): A tensor of shape (batch_size, n_sensors, n_sensors) containing the laplacian matrix.
            method (str): The method to use for thresholding (choices: 'percentile' and 'value').
            threshold (float): The threshold value for thresholding.

        Returns:
            sparse_laplacian (Tensor): A tensor of shape (batch_size, n_sensors, n_sensors) containing the sparse laplacian matrix.
        """
        if method == "percentile":
            batch_size, n_sensors, _ = laplacian_matrix.shape
            sparse_laplacian = torch.zeros_like(laplacian_matrix)

            for i in range(batch_size):
                threshold_value = torch.kthvalue(
                    laplacian_matrix[i].flatten(),
                    int(n_sensors * n_sensors * threshold),
                ).values
                sparse_laplacian[i] = torch.where(
                    laplacian_matrix[i] > threshold_value,
                    laplacian_matrix[i],
                    torch.zeros_like(laplacian_matrix[i]),
                )

        elif method == "value":
            sparse_laplacian = torch.where(
                laplacian_matrix > threshold,
                laplacian_matrix,
                torch.zeros_like(laplacian_matrix),
            )

        return sparse_laplacian
