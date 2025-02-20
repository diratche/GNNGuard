import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops


class GNNGuardConv(MessagePassing):
    """ Graph Convolutional Network (GCN) layer with GNNGuard modifications. """

    def __init__(self, in_channels, out_channels, improved=False, cached=False, 
                 bias=True, normalize=True, **kwargs):
        super(GNNGuardConv, self).__init__(aggr='add', **kwargs)  # Use sum aggregation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = Parameter(torch.Tensor(out_channels)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        """ Propagate messages and normalize. """
        x = torch.matmul(x, self.weight)
        edge_index, norm = self.norm(edge_index, x.size(0), edge_weight)
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        """ Message function that applies normalization. """
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        """ Apply bias after aggregation. """
        return aggr_out + self.bias if self.bias is not None else aggr_out

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        """ Compute normalization for message passing. """
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
