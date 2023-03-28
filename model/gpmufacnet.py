import numpy as np
import torch
import torch.nn as nn
import gpytorch
from tqdm import tqdm
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import RBFKernel
from torch.utils.data import Dataset, DataLoader
from gpytorch.distributions import MultivariateNormal

from utils.seq_utils import sequences_to_mutation_sets

from . import register_model, torch_model


class GPMuFacNet(ExactGP):
    """Mutation Factorization Network (MuFacNet)"""

    def __init__(self, input_dim,likelihood,train_x=None,train_y=None,latent_dim=32, num_filters=32, hidden_dim=128, kernel_size=5):
        super().__init__(train_x, train_y, likelihood)
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()
        self.likelihood = likelihood
        self.mutation_context_encoder = nn.Sequential(
            nn.Conv1d(input_dim, num_filters, kernel_size),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters, kernel_size),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(1),
            nn.Linear(num_filters, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.joint_effect_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, mutation_sets, mutation_sets_mask):
        # Input:  - mutation_sets:      [batch_size, max_mutation_num, input_dim, context_width]
        #         - mutation_sets_mask: [batch_size, max_mutation_num]
        # Output: - predictions:        [batch_size, 1]

        batch_size, max_mutation_num, input_dim, context_width = list(mutation_sets.size())
        element_embeddings = self.mutation_context_encoder(
            mutation_sets.view(batch_size * max_mutation_num, input_dim, context_width)
        )
        element_embeddings = element_embeddings.view(
            batch_size, max_mutation_num, -1
        ) * torch.unsqueeze(mutation_sets_mask, dim=-1)
        set_embeddings = torch.sum(element_embeddings, dim=1)
        predictions = self.joint_effect_decoder(set_embeddings)
        mean_predictions = self.mean_module(predictions)
        covar_predictions = self.covar_module(predictions)
        
        return MultivariateNormal(mean_predictions, covar_predictions)


@register_model("GPmufacnet")
class MutationFactorizationModel(torch_model.TorchModel):
    def __init__(self, args, alphabet, starting_sequence, **kwargs):
        super().__init__(
            args, alphabet, net=GPMuFacNet(input_dim=len(alphabet),likelihood=GaussianLikelihood(),latent_dim=args.latent_dim)
        )
        
        self.wt_sequence = starting_sequence
        self.context_radius = args.context_radius
        self.loss_func = gpytorch.mlls.ExactMarginalLogLikelihood(self.net.likelihood, self.net)
        
    def get_data_loader(self, sequences, labels):
        # Input:  - sequences:    [dataset_size, sequence_length]
        #         - labels:       [dataset_size]
        # Output: - loader_train: torch.utils.data.DataLoader

        mutation_sets, mutation_sets_mask = sequences_to_mutation_sets(
            sequences, self.alphabet, self.wt_sequence, self.context_radius
        )
        labels = torch.from_numpy(labels).float()
        dataset_train = torch.utils.data.TensorDataset(mutation_sets, mutation_sets_mask, labels)
        loader_train = torch.utils.data.DataLoader(
            dataset=dataset_train, batch_size=self.args.batch_size, shuffle=True
        )
        return loader_train
    
    

    def compute_loss(self, data):
        # Input:  - mutation_sets:      [batch_size, max_mutation_num, alphabet_size, context_width]
        #         - mutation_sets_mask: [batch_size, max_mutation_num]
        #         - labels:             [batch_size]
        # Output: - loss:               [1]
        
        mutation_sets, mutation_sets_mask, labels = data
       
        self.net.eval()
        outputs = self.net(mutation_sets.to(self.device), mutation_sets_mask.to(self.device))
        loss = -self.loss_func(outputs, labels.to(self.device))
        return loss
    

    def get_fitness(self, sequences):
        # Input:  - sequences:   [batch_size, sequence_length]
        # Output: - predictions: [batch_size]

        self.net.eval()
        self.net.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mutation_sets, mutation_sets_mask = sequences_to_mutation_sets(
                sequences, self.alphabet, self.wt_sequence, self.context_radius
            )
            predictions = (
                self.net(mutation_sets.to(self.device), mutation_sets_mask.to(self.device)))
            
            observation = self.net.likelihood(predictions).sample()
        
        # predictions.mean
        # predictions.variance
        observation = observation.detach().numpy()  
        # predictions = np.squeeze(predictions, axis=-1)
        
        return observation