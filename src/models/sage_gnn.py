import torch
import torch.nn.functional as F
from torch_geometric.nn import Linear, SAGEConv


class SAGEGNNEncoder(torch.nn.Module):
    def __init__(
        self,
        data,
        hidden_channels,
        out_channels,
        dropout=0.0,
        normalize=False,
        skip_connections=False,
        aggr2="max",
    ):
        super().__init__()

        self.skip_connections = skip_connections

        self.dropout_fn = torch.nn.Dropout(dropout)

        self.convs = torch.nn.ModuleList()
        if skip_connections:
            self.lins = torch.nn.ModuleList()

        for hidden_channel in hidden_channels:
            self.convs.append(
                SAGEConv(
                    (-1, -1),
                    hidden_channel,
                    normalize=normalize,
                    aggr=aggr2,
                )
            )
            if skip_connections:
                self.lins.append(Linear(-1, hidden_channel))

        self.conv2 = SAGEConv(
            (-1, -1), out_channels, normalize=normalize, aggr=aggr2
        )

        if skip_connections:
            self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        if self.skip_connections:
            for lin, conv in zip(self.lins, self.convs):
                x = self.dropout_fn(conv(x, edge_index) + lin(x))
                x = x.relu()

            x = self.dropout_fn(self.conv2(x, edge_index) + self.lin2(x))
            x = x.relu()
            return x
        else:
            for conv in self.convs:
                x = self.dropout_fn(conv(x, edge_index))
                x = x.relu()

            x = self.dropout_fn(self.conv2(x, edge_index))
            x = x.relu()
            return x
