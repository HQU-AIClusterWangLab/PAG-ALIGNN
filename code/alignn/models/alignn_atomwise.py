"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""

from typing import Tuple, Union
import dgl
import dgl.function as fn
import numpy as np
from dgl.nn import AvgPooling
import torch

# from dgl.nn.functional import edge_softmax
from typing import Literal
from torch import nn
from torch.nn import functional as F
from alignn.models.utils import (
    RBFExpansion,
    compute_cartesian_coordinates,
    compute_pair_vector_and_distance,
    MLPLayer,
)
from alignn.graphs import compute_bond_cosines
from alignn.utils import BaseSettings


class ALIGNNAtomWiseConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.alignn."""

    name: Literal["alignn_atomwise"]
    alignn_layers: int = 2
    gcn_layers: int = 2
    atom_input_features: int = 1
    # atom_input_features: int = 92
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 64
    # hidden_features: int = 256
    # fc_layers: int = 1
    # fc_features: int = 64
    output_features: int = 1
    grad_multiplier: int = -1
    calculate_gradient: bool = True
    atomwise_output_features: int = 0
    graphwise_weight: float = 1.0
    gradwise_weight: float = 1.0
    stresswise_weight: float = 0.0
    atomwise_weight: float = 0.0
    # if link == log, apply `exp` to final outputs
    # to constrain predictions to be positive
    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    classification: bool = False
    force_mult_natoms: bool = False
    energy_mult_natoms: bool = True
    include_pos_deriv: bool = False
    use_cutoff_function: bool = False
    inner_cutoff: float = 3  # Ansgtrom
    stress_multiplier: float = 1
    add_reverse_forces: bool = True  # will make True as default soon
    lg_on_fly: bool = True  # will make True as default soon
    batch_stress: bool = True
    multiply_cutoff: bool = False
    use_penalty: bool = True
    extra_features: int = 0
    use_extra_feature_gated_fusion: bool = False
    extra_feature_fusion_gamma_init: float = 0.05
    extra_feature_fusion_gamma_max: float = 0.5
    use_magpie_node_features: bool = False
    magpie_node_features: int = 5
    magpie_gamma_init: float = 0.05
    magpie_gamma_max: float = 0.5
    use_site_symmetry_node_features: bool = False
    site_symmetry_node_features: int = 80
    site_symmetry_symprec: float = 1e-3
    site_symmetry_angle_tolerance: float = 5.0
    site_symmetry_gamma_init: float = 0.03
    site_symmetry_gamma_max: float = 0.2
    use_inplane_direction_attention: bool = False
    inplane_direction_attention_mode: Literal[
        "direction", "element_direction_prior"
    ] = "direction"
    inplane_direction_attention_gamma_init: float = 0.05
    inplane_direction_attention_gamma_max: float = 0.3
    use_inplane_angle_direction_gate: bool = False
    inplane_angle_direction_gate_gamma_init: float = 0.03
    inplane_angle_direction_gate_gamma_max: float = 0.2
    use_readout_attention: bool = False
    readout_attention_beta_init: float = 0.05
    readout_attention_beta_max: float = 0.5
    use_component_readout_attention: bool = False
    component_readout_attention_beta_init: float = 0.01
    component_readout_attention_beta_max: float = 0.1
    use_xy_readout_attention: bool = False
    xy_readout_attention_beta_init: float = 0.01
    xy_readout_attention_beta_max: float = 0.1
    exponent: int = 5
    penalty_factor: float = 0.1
    penalty_threshold: float = 1
    additional_output_features: int = 0
    additional_output_weight: float = 0

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553.

    see also arxiv:2003.0098.

    This is similar to CGCNN, but edge features only go into
    the soft attention / edge gating function, and the primary
    node update function is W cat(u, v) + b
    """

    def __init__(
        self, input_features: int, output_features: int, residual: bool = True
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.residual = residual
        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.LayerNorm(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.LayerNorm(output_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Edge-gated graph convolution.

        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        g = g.local_var()

        # instead of concatenating (u || v || e) and applying one weight matrix
        # split the weight matrix into three, apply, then sum
        # see https://docs.dgl.ai/guide/message-efficient.html
        # but split them on feature dimensions to update u, v, e separately
        # m = BatchNorm(Linear(cat(u, v, e)))

        # compute edge updates, equivalent to:
        # Softplus(Linear(u || v || e))
        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        # softmax version seems to perform slightly worse
        # that the sigmoid-gated version
        # compute node updates
        # Linear(u) + edge_gates ⊙ Linear(v)
        # g.edata["gate"] = edge_softmax(g, y)
        # g.ndata["h_dst"] = self.dst_update(node_feats)
        # g.update_all(fn.u_mul_e("h_dst", "gate", "m"), fn.sum("m", "h"))
        # x = self.src_update(node_feats) + g.ndata.pop("h")

        # node and edge updates
        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(m))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        """Set up ALIGNN parameters."""
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ):
        """Node and Edge updates for ALIGNN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        """
        g = g.local_var()
        lg = lg.local_var()
        # Edge-gated graph convolution update on crystal graph
        x, m = self.node_update(g, x, y)

        # Edge-gated graph convolution update on crystal graph
        y, z = self.edge_update(lg, m, z)

        return x, y, z


class ExtraFeatureGatedFusion(nn.Module):
    """Small gated correction on top of the extra-feature MLP head."""

    def __init__(
        self,
        head_features: int,
        extra_features: int,
        gamma_init: float = 0.05,
        gamma_max: float = 0.5,
    ):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(extra_features),
            MLPLayer(extra_features, head_features),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(head_features + extra_features),
            MLPLayer(head_features + extra_features, head_features),
            nn.Linear(head_features, head_features),
            nn.Sigmoid(),
        )
        self.gamma_max = float(gamma_max)
        if self.gamma_max <= 0:
            raise ValueError("extra_feature_fusion_gamma_max must be > 0.")
        init_ratio = float(gamma_init) / self.gamma_max
        init_ratio = float(np.clip(init_ratio, 1e-6, 1.0 - 1e-6))
        init_raw = np.log(init_ratio / (1.0 - init_ratio))
        self.gamma_raw = nn.Parameter(
            torch.tensor(init_raw, dtype=torch.float)
        )

    def gamma_value(self):
        """Return the bounded residual scale for extra-feature fusion."""
        return self.gamma_max * torch.sigmoid(self.gamma_raw)

    def forward(self, h_head: torch.Tensor, h_feat: torch.Tensor):
        """Apply a small descriptor correction after the main MLP head."""
        h_feat_proj = self.feature_proj(h_feat)
        gate = self.gate(torch.cat((h_head, h_feat), dim=1))
        return h_head + self.gamma_value() * gate * h_feat_proj


class InPlaneDirectionalEdgeAttention(nn.Module):
    """Bounded residual edge correction from in-plane bond direction."""

    directional_features = 8

    def __init__(
        self,
        hidden_features: int,
        atom_input_features: int,
        gamma_init: float = 0.05,
        gamma_max: float = 0.3,
        mode: str = "direction",
    ):
        super().__init__()
        valid_modes = {"direction", "element_direction_prior"}
        if mode not in valid_modes:
            raise ValueError(
                "inplane_direction_attention_mode must be one of "
                f"{sorted(valid_modes)}."
            )
        self.mode = mode
        self.hidden_features = int(hidden_features)
        self.atom_input_features = int(atom_input_features)
        self.direction_proj = nn.Sequential(
            nn.LayerNorm(self.directional_features),
            MLPLayer(self.directional_features, hidden_features),
            nn.Linear(hidden_features, hidden_features),
        )
        self.direction_gate = nn.Sequential(
            nn.LayerNorm(self.directional_features),
            MLPLayer(self.directional_features, hidden_features),
            nn.Linear(hidden_features, hidden_features),
            nn.Sigmoid(),
        )
        if self.mode == "element_direction_prior":
            element_pair_features = self.atom_input_features * 4
            self.element_pair_proj = nn.Sequential(
                nn.LayerNorm(element_pair_features),
                MLPLayer(element_pair_features, hidden_features),
            )
            interaction_features = hidden_features * 3
            self.element_direction_proj = nn.Sequential(
                nn.LayerNorm(interaction_features),
                MLPLayer(interaction_features, hidden_features),
                nn.Linear(hidden_features, hidden_features),
            )
            self.element_direction_gate = nn.Sequential(
                nn.LayerNorm(hidden_features * 4),
                MLPLayer(hidden_features * 4, hidden_features),
                nn.Linear(hidden_features, hidden_features),
                nn.Sigmoid(),
            )
        self.gamma_max = float(gamma_max)
        if self.gamma_max <= 0:
            raise ValueError("inplane_direction_attention_gamma_max must be > 0.")
        init_ratio = float(gamma_init) / self.gamma_max
        init_ratio = float(np.clip(init_ratio, 1e-6, 1.0 - 1e-6))
        init_raw = np.log(init_ratio / (1.0 - init_ratio))
        self.gamma_raw = nn.Parameter(
            torch.tensor(init_raw, dtype=torch.float)
        )

    def gamma_value(self):
        """Return the bounded residual scale for direction attention."""
        return self.gamma_max * torch.sigmoid(self.gamma_raw)

    def direction_features(self, r: torch.Tensor) -> torch.Tensor:
        """Build sign-stable in-plane direction descriptors for each bond."""
        eps = torch.tensor(1e-12, dtype=r.dtype, device=r.device)
        rx = r[:, 0:1]
        ry = r[:, 1:2]
        rz = r[:, 2:3]
        length = torch.linalg.norm(r, dim=1, keepdim=True).clamp_min(eps)
        ux_abs = torch.abs(rx) / length
        uy_abs = torch.abs(ry) / length
        uz_abs = torch.abs(rz) / length

        inplane_sq = (rx.pow(2) + ry.pow(2)).clamp_min(eps)
        inplane_norm = torch.sqrt(inplane_sq)
        inplane_ratio = inplane_norm / length
        cos2 = (rx.pow(2) - ry.pow(2)) / inplane_sq
        sin2 = (2.0 * rx * ry) / inplane_sq
        cos4 = cos2.pow(2) - sin2.pow(2)
        sin4 = 2.0 * cos2 * sin2
        return torch.cat(
            [
                ux_abs,
                uy_abs,
                uz_abs,
                inplane_ratio,
                cos2,
                sin2,
                cos4,
                sin4,
            ],
            dim=1,
        )

    @staticmethod
    def element_pair_features(
        g: dgl.DGLGraph, atom_features: torch.Tensor
    ) -> torch.Tensor:
        """Build directed source-target fixed element descriptors."""
        if g is None or atom_features is None:
            raise ValueError(
                "element_direction_prior mode requires both graph and atom "
                "features."
            )
        src, dst = g.edges()
        src_features = atom_features[src]
        dst_features = atom_features[dst]
        return torch.cat(
            [
                src_features,
                dst_features,
                torch.abs(src_features - dst_features),
                src_features * dst_features,
            ],
            dim=1,
        )

    def forward(
        self,
        edge_features: torch.Tensor,
        r: torch.Tensor,
        g: dgl.DGLGraph = None,
        atom_features: torch.Tensor = None,
    ):
        """Apply a small direction-aware residual correction to edge features."""
        direction = self.direction_features(r.to(edge_features.dtype))
        direction_embedding = self.direction_proj(direction)
        if self.mode == "element_direction_prior":
            element_pair = self.element_pair_features(g, atom_features).to(
                edge_features.dtype
            )
            element_embedding = self.element_pair_proj(element_pair)
            element_direction = direction_embedding * element_embedding
            interaction = torch.cat(
                [direction_embedding, element_embedding, element_direction],
                dim=1,
            )
            correction = self.element_direction_proj(interaction)
            gate = self.element_direction_gate(
                torch.cat(
                    [
                        edge_features,
                        direction_embedding,
                        element_embedding,
                        element_direction,
                    ],
                    dim=1,
                )
            )
        else:
            correction = direction_embedding
            gate = self.direction_gate(direction)
        return edge_features + self.gamma_value() * gate * correction


class ALIGNNAtomWise(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(
        self,
        config: ALIGNNAtomWiseConfig = ALIGNNAtomWiseConfig(
            name="alignn_atomwise"
        ),
    ):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        # print(config)
        self.classification = config.classification
        self.config = config
        self.atom_embedding = MLPLayer(
            config.atom_input_features, config.hidden_features
        )
        if config.use_site_symmetry_node_features:
            if config.site_symmetry_node_features <= 0:
                raise ValueError(
                    "site_symmetry_node_features must be > 0 when "
                    "use_site_symmetry_node_features=True."
                )
            self.site_symmetry_norm = nn.LayerNorm(
                config.site_symmetry_node_features
            )
            self.site_symmetry_embedding = MLPLayer(
                config.site_symmetry_node_features, config.hidden_features
            )
            self.site_symmetry_gamma_max = float(
                config.site_symmetry_gamma_max
            )
            if self.site_symmetry_gamma_max <= 0:
                raise ValueError("site_symmetry_gamma_max must be > 0.")
            init_ratio = (
                float(config.site_symmetry_gamma_init)
                / self.site_symmetry_gamma_max
            )
            init_ratio = float(np.clip(init_ratio, 1e-6, 1.0 - 1e-6))
            init_raw = np.log(init_ratio / (1.0 - init_ratio))
            self.site_symmetry_gamma_raw = nn.Parameter(
                torch.tensor(init_raw, dtype=torch.float)
            )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_input_features,
            ),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1,
                vmax=1.0,
                bins=config.triplet_input_features,
            ),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        if config.use_inplane_direction_attention:
            self.inplane_direction_attention = InPlaneDirectionalEdgeAttention(
                hidden_features=config.hidden_features,
                atom_input_features=config.atom_input_features,
                gamma_init=config.inplane_direction_attention_gamma_init,
                gamma_max=config.inplane_direction_attention_gamma_max,
                mode=config.inplane_direction_attention_mode,
            )
        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(
                    config.hidden_features,
                    config.hidden_features,
                )
                for idx in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(
                    config.hidden_features, config.hidden_features
                )
                for idx in range(config.gcn_layers)
            ]
        )

        self.readout = AvgPooling()
        if config.use_extra_feature_gated_fusion:
            if config.extra_features == 0:
                raise ValueError(
                    "use_extra_feature_gated_fusion=True requires "
                    "extra_features > 0."
                )
        if config.extra_features != 0:
            self.readout_feat = AvgPooling()
            # Credit for extra_features work:
            # Gong et al., https://doi.org/10.48550/arXiv.2208.05039
            self.extra_feature_embedding = MLPLayer(
                config.extra_features, config.extra_features
            )
            if config.use_extra_feature_gated_fusion:
                self.extra_feature_gated_fusion = ExtraFeatureGatedFusion(
                    head_features=config.hidden_features
                    + config.extra_features,
                    extra_features=config.extra_features,
                    gamma_init=config.extra_feature_fusion_gamma_init,
                    gamma_max=config.extra_feature_fusion_gamma_max,
                )
            # print('config.output_features',config.output_features)
            self.fc3 = nn.Linear(
                config.hidden_features + config.extra_features,
                config.output_features,
            )
            self.fc1 = MLPLayer(
                config.extra_features + config.hidden_features,
                config.extra_features + config.hidden_features,
            )
            self.fc2 = MLPLayer(
                config.extra_features + config.hidden_features,
                config.extra_features + config.hidden_features,
            )

        if self.classification:
            self.fc = nn.Linear(config.hidden_features, 1)
            self.softmax = nn.Sigmoid()
            # self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc = nn.Linear(config.hidden_features, config.output_features)
        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            self.fc.bias.data = torch.tensor(
                np.log(avg_gap), dtype=torch.float
            )
        elif config.link == "logit":
            self.link = torch.sigmoid

    def site_symmetry_gamma_value(self):
        """Return the bounded residual scale for site-symmetry features."""
        return self.site_symmetry_gamma_max * torch.sigmoid(
            self.site_symmetry_gamma_raw
        )

    def extra_feature_fusion_gamma_value(self):
        """Return the bounded residual scale for extra-feature fusion."""
        return self.extra_feature_gated_fusion.gamma_value()

    def inplane_direction_attention_gamma_value(self):
        """Return the bounded residual scale for direction attention."""
        return self.inplane_direction_attention.gamma_value()

    def forward(
        self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]
    ):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        if len(self.alignn_layers) > 0:
            if len(g) == 3:
                g, lg, lat = g
                lg = lg.local_var()
                # z = self.angle_embedding(lg.edata.pop("h"))
                z = self.angle_embedding(lg.edata["h"])
            else:
                g, lat = g
                g.ndata["cart_coords"] = compute_cartesian_coordinates(g, lat)
                r, bondlength = compute_pair_vector_and_distance(g)
                lg = g.line_graph(shared=True)
                lg.ndata["r"] = r
                lg.apply_edges(compute_bond_cosines)
                # print('lg',lg)
                # angle features (fixed)
        else:
            g, lat = g
        if self.config.extra_features != 0:
            features = g.ndata["extra_features"]
            # print('features',features,features.shape)
            features = self.extra_feature_embedding(features)
        # g = g.local_var()
        result = {}
        # initial node features: atom feature network...
        atom_features = g.ndata["atom_features"]
        x = atom_features
        # x = g.ndata.pop("atom_features")
        # print('x1',x,x.shape)

        x = self.atom_embedding(x)
        if self.config.use_site_symmetry_node_features:
            if "site_symmetry_features" not in g.ndata:
                raise KeyError(
                    "use_site_symmetry_node_features=True requires "
                    "g.ndata['site_symmetry_features']; enable it during "
                    "dataset graph construction."
                )
            site_symmetry = g.ndata["site_symmetry_features"].to(
                x.device
            ).type_as(x)
            site_symmetry = self.site_symmetry_norm(site_symmetry)
            x = x + self.site_symmetry_gamma_value() * (
                self.site_symmetry_embedding(site_symmetry)
            )
        # print('x2',x,x.shape)
        r = g.edata["r"]
        bondlength = torch.norm(r, dim=1)
        if self.config.lg_on_fly and len(self.alignn_layers) > 0:
            # re-compute bond angle cosines here to ensure
            # the three-body interactions are fully included
            # in the autograd graph. don't rely on dataloader/caching.

            lg.ndata["r"] = r  # overwrites precomputed r values
            lg.apply_edges(compute_bond_cosines)  # overwrites precomputed h
            z = self.angle_embedding(lg.edata["h"])
            # z = self.angle_embedding(lg.edata.pop("h"))

        y = self.edge_embedding(bondlength)
        if self.config.use_inplane_direction_attention:
            y = self.inplane_direction_attention(y, r, g, atom_features)
        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)
        # norm-activation-pool-classify
        out = torch.empty(1)
        additional_out = torch.empty(1)
        graph_embedding = torch.empty(1)
        if self.config.output_features is not None:
            h = self.readout(g, x)
            graph_embedding = h
            out = self.fc(h)
            if (
                self.config.extra_features != 0
            ):
                h_feat = self.readout_feat(g, features)
                if self.config.use_extra_feature_gated_fusion:
                    h = torch.cat((h, h_feat), 1)
                    h = self.fc1(h)
                    h = self.fc2(h)
                    graph_embedding = h
                    h = self.extra_feature_gated_fusion(h, h_feat)
                    out = self.fc3(h)
                else:
                    # print('h_feat',h_feat)
                    h = torch.cat((h, h_feat), 1)
                    h = self.fc1(h)
                    h = self.fc2(h)
                    graph_embedding = h
                    out = self.fc3(h)
                # print('out',out)
            out = torch.squeeze(out)

        forces = torch.empty(1)
        # The routed prediction-model-04 regressors do not train atomwise,
        # gradient, stress, or additional-output heads.  Keep the placeholder
        # fields for the shared trainer's result contract.
        atomwise_pred = torch.empty(1)
        # gradient = torch.empty(1)
        stress = torch.empty(1)
        if self.link:
            out = self.link(out)

        if self.classification:
            # out = torch.max(out,dim=1)
            out = self.softmax(out)
        result["out"] = out
        result["graph_embedding"] = graph_embedding
        result["additional"] = additional_out
        result["grad"] = forces
        result["stresses"] = stress
        result["atomwise_pred"] = atomwise_pred
        # print(result)
        return result


"""
if __name__ == "__main__":
    from jarvis.core.atoms import Atoms
    from alignn.graphs import Graph

    FIXTURES = {
        "lattice_mat": [
            [2.715, 2.715, 0],
            [0, 2.715, 2.715],
            [2.715, 0, 2.715],
        ],
        "coords": [[0, 0, 0], [0.25, 0.25, 0.25]],
        "elements": ["Si", "Si"],
    }
    Si = Atoms(
        lattice_mat=FIXTURES["lattice_mat"],
        coords=FIXTURES["coords"],
        elements=FIXTURES["elements"],
    )
    g, lg = Graph.atom_dgl_multi_graph(
        atoms=Si, neighbor_strategy="radius_graph", cutoff=5
    )
    lat = torch.tensor(atoms.lattice_mat)
    model = ALIGNNAtomWise(ALIGNNAtomWiseConfig(name="alignn_atomwise"))
    out = model([g, lg, lat])
    print(out)
"""
