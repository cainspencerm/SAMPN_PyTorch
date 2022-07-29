import utils  # local import

from torch import nn
import rdkit.Chem as Chem
import torch.nn.functional as F
import torch
import numpy as np
from typing import List, Tuple, Union
from typing import List, Union


class QSAR(nn.Module):
    def __init__(self, features_size=None, features_dim=None, hidden_size=384, depth=4, features_only=False, use_input_features=None, dropout=0., activation='ReLU', ffn_num_layers=2, ffn_hidden_size=300):
        super(QSAR, self).__init__()

        self.hidden_size = hidden_size
        self.depth = depth

        self._create_encoder()
        self._create_ffn(features_only, features_size, use_input_features, features_dim, dropout, activation, ffn_num_layers, ffn_hidden_size)

    def _create_encoder(self):
        """
        Creates the message passing encoder for the model.
        """
        self.encoder = MPN()

    def _create_ffn(self, features_only, features_size, use_input_features, features_dim, dropout, activation, ffn_num_layers, ffn_hidden_size):
        """
        Creates the feed-forward network for the model.

        :param features_only: Whether to use only the features or also the hidden states.
        :param features_size: The size of the features.
        :param use_input_features: Whether to use the input features or not.
        :param features_dim: The dimension of the features.
        :param dropout: The dropout probability.
        :param activation: The activation function.
        :param ffn_num_layers: The number of layers in the feed-forward network.
        :param ffn_hidden_size: The size of the hidden layers in the feed-forward network.
        """

        # If we use only the features, we don't need to add the input features.
        if features_only:
            first_linear_dim = features_size
        else:
            first_linear_dim = self.hidden_size
            if use_input_features:
                first_linear_dim += features_dim

        dropout = nn.Dropout(0.25)
        activation = nn.ReLU()

        if ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, 1)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, ffn_hidden_size)
            ]
            for _ in range(ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(ffn_hidden_size, ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(ffn_hidden_size, 1),
            ])

        self.ffn = nn.Sequential(*ffn)

    def forward(self, mol_batch):
        """
        Forward pass of the model.
        :param mol_batch: A batch of molecules.
        :return: The logits of the model.
        """
        # Get the features of the molecules.
        features = self.encoder(mol_batch)

        # Get the logits of the model.
        logits = self.ffn(features)

        return logits

ELEM_LIST=list(range(1,119))
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 +5+1

BOND_FDIM = 5 + 6
MAX_NB = 6
SMILES_TO_GRAPH={}

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom):

    return onek_encoding_unk(atom.GetAtomicNum() , ELEM_LIST) + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])+ onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]) + onek_encoding_unk(int(atom.GetChiralTag()), [0, 1, 2, 3])+onek_encoding_unk(int(atom.GetHybridization()),[
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ])+[1 if atom.GetIsAromatic() else 0]

def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
    fbond=fbond + fstereo
    return fbond


class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:
    - smiles: Smiles string.
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    """

    def __init__(self, smiles: str, atom_messages):
        """
        Computes the graph structure and featurization of a molecule.

        :param smiles: A smiles string.
        :param args: Arguments.
        """
        self.smiles = smiles
        self.n_atoms = 0
        self.n_bonds = 0
        self.f_atoms = []
        self.f_bonds = []
        self.a2b = []
        self.b2a = []
        self.b2revb = []

        mol = Chem.MolFromSmiles(smiles)

        self.n_atoms = mol.GetNumAtoms()
        for i, atom in enumerate(mol.GetAtoms()):
            self.f_atoms.append(atom_features(atom))
        self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]

        for _ in range(self.n_atoms):
            self.a2b.append([])

        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)
                if bond is None:
                    continue
                f_bond = bond_features(bond)
                if atom_messages:
                    self.f_bonds.append(f_bond)
                    self.f_bonds.append(f_bond)
                else:
                    self.f_bonds.append(self.f_atoms[a1] + f_bond)
                    self.f_bonds.append(self.f_atoms[a2] + f_bond)

                b1 = self.n_bonds
                b2 = b1 + 1

                self.a2b[a2].append(b1)
                self.b2a.append(a1)
                self.a2b[a1].append(b2)
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2


class BatchMolGraph:
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a MolGraph plus:
    - smiles_batch: A list of smiles strings.
    - n_mols: The number of molecules in the batch.
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - b_scope: A list of tuples indicating the start and end bond indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs: List[MolGraph], atom_messages):
        self.smiles_batch = [mol_graph.smiles for mol_graph in mol_graphs]
        self.n_mols = len(self.smiles_batch)

        self.atom_fdim = ATOM_FDIM
        self.bond_fdim = BOND_FDIM + (not atom_messages) * self.atom_fdim

        self.n_atoms = 1
        self.n_bonds = 1

        f_atoms = [[0] * self.atom_fdim]
        f_bonds = [[0] * self.bond_fdim]
        self.a_scope = []
        self.b_scope = []

        a2b = [[]]
        b2a = [0]
        b2revb = [0]
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        self.max_num_bonds = max(len(in_bonds) for in_bonds in a2b)

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])

        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None
        self.a2a = None


    def get_components(self) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                      torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                      List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the BatchMolGraph.

        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """
        if self.b2b is None:
            b2b = self.a2b[self.b2a]

            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
        """
        if self.a2a is None:
            a2neia=[]
            for incoming_bondIdList in self.a2b:
                neia=[]
                for incoming_bondId in incoming_bondIdList:
                    neia.append(self.b2a[incoming_bondId])
                a2neia.append(neia)
            self.a2a=a2neia

        return self.a2a

def mol2graph(smiles_batch: List[str], atom_messages) -> BatchMolGraph:
    """
    Converts a list of SMILES strings to a BatchMolGraph containing the batch of molecular graphs.

    :param smiles_batch: A list of SMILES strings.
    :param args: Arguments.
    :return: A BatchMolGraph containing the combined molecular graph for the molecules
    """
    mol_graphs = []
    for smiles in smiles_batch:
        if smiles in SMILES_TO_GRAPH:
            mol_graph = SMILES_TO_GRAPH[smiles]
        else:
            mol_graph = MolGraph(smiles, atom_messages)
        mol_graphs.append(mol_graph)

    return BatchMolGraph(mol_graphs, atom_messages)

class MPNEncoder(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, atom_fdim: int, bond_fdim: int, hidden_size=384, depth=4, dropout=.0, layers_per_message=1, undirected=False, atom_messages=False, features_only=False, use_input_features=None, normalize_messages=False, diff_depth_weights=True, layer_norm=False, attention=True, sumstyle=True):
        """Initializes the MPNEncoder.

        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = dropout
        self.layers_per_message = layers_per_message
        self.undirected = undirected
        self.atom_messages = atom_messages
        self.features_only = features_only
        self.use_input_features = use_input_features
        self.normalize_messages=normalize_messages
        self.diff_depth_weights=diff_depth_weights
        self.attention = attention
        self.sumstyle=sumstyle

        if self.features_only:
            return

        self.layer_norm = nn.LayerNorm(self.hidden_size) if layer_norm else None

        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.act_func = nn.ReLU()

        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        if diff_depth_weights:
            modulList=[nn.Linear(w_h_input_size, self.hidden_size)]
            modulList.extend(
                nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.depth - 1)])
            )
            self.W_h=nn.Sequential(*modulList)
        else:
            self.W_h = nn.Linear(w_h_input_size, self.hidden_size)

        if self.sumstyle==True:
            self.W_ah= nn.Linear(self.atom_fdim, self.hidden_size)
            self.W_o = nn.Linear(self.hidden_size, self.hidden_size)
        else:
            self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        if self.attention:
            self.W_a = nn.Linear(self.hidden_size, self.hidden_size)
            self.W_b = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, mol_graph: BatchMolGraph, features_batch: List[np.ndarray] = None, viz_dir: str = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()

            if self.features_only:
                return features_batch

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()

        if self.atom_messages:
            a2a = mol_graph.get_a2a()

        if self.atom_messages:
            input = self.W_i(f_atoms)
        else:
            input = self.W_i(f_bonds)

        message = self.act_func(input)

        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                nei_a_message = utils.index_select_ND(message, a2a)
                nei_f_bonds = utils.index_select_ND(f_bonds, a2b)
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)
                message = nei_message.sum(dim=1)
            else:
                nei_a_message = utils.index_select_ND(message, a2b)
                a_message = nei_a_message.sum(dim=1)
                rev_message = message[b2revb]
                message = a_message[b2a] - rev_message
            message = self.W_h(message)
            message = self.act_func(input + message)
            if self.normalize_messages:
                message = message / message.norm(dim=1, keepdim=True)
            
            if self.layer_norm:
                message = self.layer_norm(message)

            message = self.dropout_layer(message)


        a2x = a2a if self.atom_messages else a2b

        nei_a_message = utils.index_select_ND(message, a2x)
        a_message = nei_a_message.sum(dim=1)

        if self.sumstyle==True:
            a_input =self.W_ah(f_atoms) + a_message
        else:
            a_input = torch.cat([f_atoms, a_message], dim=1)
        atom_hiddens = self.act_func(self.W_o(a_input))
        atom_hiddens = self.dropout_layer(atom_hiddens)

        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                if self.attention:
                        att_w = torch.matmul(self.W_a(cur_hiddens), cur_hiddens.t())
                        att_w = F.softmax(att_w, dim=1)
                        att_hiddens = torch.matmul(att_w, cur_hiddens)
                        att_hiddens = self.act_func(self.W_b(att_hiddens))
                        att_hiddens = self.dropout_layer(att_hiddens)
                        mol_vec = (cur_hiddens + att_hiddens)
                else:
                    mol_vec = cur_hiddens
                mol_vec = mol_vec.sum(dim=0) / a_size
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)

        if self.use_input_features:
            features_batch = features_batch.to(mol_vecs)
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1,features_batch.shape[0]])
            mol_vecs = torch.cat([mol_vecs, features_batch], dim=1)

        return mol_vecs

class MPN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, atom_messages=False, atom_fdim=ATOM_FDIM, bond_fdim=BOND_FDIM, graph_input: bool = False):
        """
        Initializes the MPN.

        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param graph_input: If true, expects BatchMolGraph as input. Otherwise expects a list of smiles strings as input.
        """
        super(MPN, self).__init__()

        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim + (not atom_messages) * self.atom_fdim
        self.atom_messages = atom_messages

        self.graph_input = graph_input
        self.encoder = MPNEncoder(self.atom_fdim, self.bond_fdim, atom_messages=atom_messages)

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular SMILES strings.

        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if not self.graph_input:
            batch = mol2graph(batch, self.atom_messages)

        output = self.encoder.forward(batch, features_batch)

        return output
