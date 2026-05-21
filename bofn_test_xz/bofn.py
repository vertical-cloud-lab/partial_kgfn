import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
import matplotlib.pyplot as plt
from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor, distributions, nn
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.posteriors.posterior import Posterior
from botorch.posteriors.torch import TorchPosterior
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform

class bofn_node():
    def __init__(self, name, init_train_X, init_train_Y, func):
        self.name = name
        self.GP = SingleTaskGP(init_train_X, init_train_Y, input_transform=Normalize(init_train_X.shape[-1]))
        self.mll = ExactMarginalLogLikelihood(self.GP.likelihood, self.GP)
        fit_gpytorch_mll(self.mll)
        self.parents = []
        self.func = func
        
    # TODO: if init_train_X == None, write a sobol method such that it generates a bunch of points to test? Would need a function for the node but that makes sense
    
    def eval_posterior(self, X: Tensor):
        self.GP.eval()
        
        # Flatten q dim if present: [batch, q, d] -> [batch*q, d]
        original_shape = X.shape
        if X.dim() == 3:
            X = X.reshape(-1, X.shape[-1])

        with torch.no_grad():
            posterior = self.GP.posterior(X)
            mean = posterior.mean       # [n, 1]
            std = posterior.variance.sqrt()

        Z_k = torch.randn_like(mean)
        result = mean + std * Z_k       # [n, 1]

        # Restore batch shape if we flattened
        if len(original_shape) == 3:
            result = result.reshape(*original_shape[:-1], -1)  # [batch, q, 1]

        return result
    
    def train(self, X: Tensor):
        # ask sterling best practices on retraining bayesian op models

        # print(X.shape)
        if len(X.shape) == 1:
            X = X.unsqueeze(0)
        
        y = self.func(X)
        
        new_X = torch.cat([self.GP.train_inputs[0], X], dim=0)
        new_y = torch.cat([self.GP.train_targets, y]).unsqueeze(1)
        self.GP = SingleTaskGP(new_X, new_y, input_transform=Normalize(new_X.shape[-1]))
        self.GP.train()          # ← put in train mode before fitting
        self.mll = ExactMarginalLogLikelihood(self.GP.likelihood, self.GP)
        fit_gpytorch_mll(self.mll)
        # with torch.no_grad():
        #     posterior = self.GP.posterior(X)
        #     mean = posterior.mean
        #     lower, upper = posterior.mvn.confidence_region()

        return y.unsqueeze(1)



class BOFN(Model):
    class NetworkPosterior(Posterior):

        def __init__(self, mean: Tensor, variance: Tensor, X: Tensor, tail, input_indices, nodes) -> None:
            super().__init__()
            self._mean = mean           # [batch, q, m]
            self._variance = variance   # [batch, q, m]
            self._X = X                 # [batch, q, d] — stored for resampling
            self.tail = tail
            self.nodes = nodes
            self.input_indices = input_indices

        @property
        def mean(self) -> Tensor:
            return self._mean

        @property
        def variance(self) -> Tensor:
            return self._variance

        @property
        def device(self):
            return self._mean.device

        @property
        def dtype(self):
            return self._mean.dtype

        @property
        def base_sample_shape(self) -> torch.Size:
            # [batch, q, n_nodes] — sampler prepends sample_shape
            return torch.Size([*self._X.shape[:-1], len(self.nodes)])

        @property
        def batch_range(self):
            return (0, -2)

        def _traverse(self, node: bofn_node, X: Tensor, base_samples: dict):
            indices = self.input_indices[node.name]

            if len(node.parents) == 0:
                input = X[..., indices]
            else:
                ys = torch.cat([
                    self._traverse(self.nodes[parent], X, base_samples)
                    for parent in node.parents
                ], dim=-1)
                input = torch.cat([X[..., indices], ys], dim=-1)

            # input shape: [n_samples, batch, q, d_node]
            original_shape = input.shape
            input_2d = input.reshape(-1, input.shape[-1])

            posterior = node.GP.posterior(input_2d)
            mean = posterior.mean.reshape(*original_shape[:-1], 1)  # [..., 1]
            std  = posterior.variance.sqrt().reshape(*original_shape[:-1], 1)

            Z = base_samples[node.name]  # [n_samples, batch, q, 1]
            # Expand Z to match mean's shape in case of broadcast mismatch
            Z = Z.expand_as(mean)
            return mean + std * Z

        def rsample_from_base_samples(self, sample_shape: torch.Size, base_samples: Tensor) -> Tensor:
            # base_samples: [*sample_shape, batch, q, n_nodes]
            node_names = list(self.nodes.keys())
            base_dict = {
                name: base_samples[..., i:i+1]  # [*sample_shape, batch, q, 1]
                for i, name in enumerate(node_names)
            }
            X_expanded = self._X.expand(*sample_shape, *self._X.shape)
            return self._traverse(self.tail, X_expanded, base_dict)

        def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
            n_nodes = len(self.nodes)
            base_samples = torch.randn(*sample_shape, *self._mean.shape[:-1], n_nodes)
            node_names = list(self.nodes.keys())
            base_dict = {
                name: base_samples[..., i:i+1]
                for i, name in enumerate(node_names)
            }
            X_expanded = self._X.expand(*sample_shape, *self._X.shape)
            return self._traverse(self.tail, X_expanded, base_dict)

    def __init__(self, max):
        super(BOFN, self).__init__()
        self.nodes = {}
        self.tail = None
        self.input_indices = {}
        self.extremum = None
        self.max = max

    def posterior(self, X, output_indices=None, observation_noise=False, posterior_transform=None, **kwargs):
        dummy = torch.zeros(*X.shape[:-1], 1, dtype=X.dtype, device=X.device)
        posterior = self.NetworkPosterior(
            mean=dummy,
            variance=dummy,
            X=X,                        # ← store X for resampling
            tail=self.tail,
            input_indices=self.input_indices,
            nodes=self.nodes,
        )
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def add_node(self, name, init_train_X, init_train_Y, func, input_indices, parents=[], tail=False):
        node = bofn_node(name, init_train_X, init_train_Y, func)
        self.nodes[name] = node
        self.input_indices[name] = input_indices
        for parent in parents:
            node.parents.append(parent)
        if tail:
            self.tail = node
            if self.max:
                self.extremum = init_train_Y.max()
            else:
                self.extremum = init_train_Y.min()
    
    def query_node(self, node: bofn_node, X, train=False):
        # if node.name == "f1":
        #     print([self.query_node(self.nodes[parent], X, train) for parent in node.parents])
        indices = self.input_indices[node.name]
        if len(node.parents) == 0:
            input = X[..., indices]
        else:
            ys = torch.cat([self.query_node(self.nodes[parent], X, train) for parent in node.parents]).squeeze(0)
            
            # print("ys: ", ys.shape)
            input = torch.cat([X[..., indices], ys], dim=-1)
        # print(node.name, ' Input: ', input.shape)

        if train:
            return node.train(input)
        else:
            return node.eval_posterior(input) 
    
    def run_exp(self, X, train=False):
        soln = self.query_node(self.tail, X, train)
        if train:
            # Use the returned y value directly, before any GP transformation
            new_val = soln.squeeze()
            if self.max:
                self.extremum = torch.max(self.extremum, new_val)
            else:
                self.extremum = torch.min(self.extremum, new_val)
        return soln
    
class EIFN(MCAcquisitionFunction):
    def __init__(self, model: BOFN, best_f, n, sampler=None):
        super(MCAcquisitionFunction, self).__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([n]))
        self.sampler = sampler
        # self.best_f = best_f
        self.register_buffer(
            "best_f", torch.as_tensor(best_f, dtype=torch.double)
        )
        # self.maximize = maximize

    def forward(self, X):
        posterior = self.model.posterior(X)
        samples = self.get_posterior_samples(posterior)  # [n, b, q, 1]
        # print(samples.shape)
        EI = (samples - self.best_f).clamp(min=0.0)     # [n, b, q, 1]
        # print(EI.shape)
        # print(EI.squeeze(-1).max(dim=-1).values.mean(dim=0))
        temp = EI.squeeze(-1).max(dim=-1).values.mean(dim=0)
        # print(temp)
        # temp = temp.max(dim=-1).values
        # print(temp)
        return temp  # [b]

rosen = lambda x: torch.Tensor(-100*(x[:, 1] - x[:, 0]**2)**2 - (1-x[:, 0])**2 + x[:, 2])
rosen_init = lambda x: torch.Tensor(-100*(x[:, 1] - x[:, 0]**2)**2 - (1-x[:, 0])**2)

train_X = torch.rand(15, 6, dtype=torch.double)
train_Y1 = rosen_init(train_X[:, [0, 1]])
train_X2 = torch.cat([train_X[:, [2, 3]], train_Y1.unsqueeze(1)], dim=1)
train_Y2 = rosen(train_X2)
train_X3 = torch.cat([train_X[:, [4, 5]], train_Y2.unsqueeze(1)], dim=1)
train_Y3 = rosen(train_X3)

bofn = BOFN(True)
bofn.add_node("f1", train_X[:, [0, 1]], train_Y1.unsqueeze(1), rosen_init, [0, 1])
bofn.add_node("f2", train_X2, train_Y2.unsqueeze(1), rosen, [2, 3], parents=["f1"])
bofn.add_node("f3", train_X3, train_Y3.unsqueeze(1), rosen, [4, 5], parents=["f2"], tail=True)
# print(bofn.run_exp(torch.rand(1, 6)))


eifn = EIFN(bofn, n=64, best_f=bofn.extremum)

compo_rosen = lambda X: rosen(
    torch.cat([
        X[:, [4, 5]], 
        rosen(torch.cat([X[:, [2, 3]], rosen_init(X[:, [0, 1]]).unsqueeze(1)], dim=1)).unsqueeze(1)
    ], dim=1)
)

test = SingleTaskGP(train_X, train_Y3.unsqueeze(1))
mll = ExactMarginalLogLikelihood(test.likelihood, test)
fit_gpytorch_mll(mll)
lEI = LogExpectedImprovement(test, torch.max(train_Y3), maximize=True)

xs = train_X
ys = train_Y3.unsqueeze(1)
solns = []
test_solns = []

# candidate, val = optimize_acqf(eifn, torch.Tensor([[-2.0]*6, [2.0]*6]), q=1, num_restarts=3, raw_samples=64)
# eifn = EIFN(bofn, n=128, best_f=bofn.extremum)
# solns.append(bofn.run_exp(candidate[0], True)[0][0].item())

for i in range(30):
    # BOFN candidate
    candidate, val = optimize_acqf(eifn, torch.Tensor([[-2.0]*6, [2.0]*6]), q=1, num_restarts=2, raw_samples=64)
    print(bofn.extremum)
    eifn = EIFN(bofn, n=128, best_f=bofn.extremum)
    solns.append(bofn.run_exp(candidate[0], True)[0][0].item())

    # Baseline LEI candidate
    candidate, val = optimize_acqf(lEI, torch.Tensor([[-2.0]*6, [2.0]*6]), q=1, num_restarts=2, raw_samples=64)
    y = compo_rosen(candidate)          # [1]
    test_solns.append(y.item())

    xs = torch.cat([xs, candidate], dim=0)
    ys = torch.cat([ys, y.unsqueeze(-1)], dim=0)

    test = SingleTaskGP(xs, ys, input_transform=Normalize(6))
    mll = ExactMarginalLogLikelihood(test.likelihood, test)
    fit_gpytorch_mll(mll)
    lEI = LogExpectedImprovement(test, torch.max(ys), maximize=True)

plt.plot(np.arange(0, 30), solns)
plt.plot(np.arange(0, 30), test_solns)
plt.show()



# # Always use double precision — BoTorch will warn you if you don't
# train_X = torch.rand(20, 1, dtype=torch.double)                      # 20 points, 2D input
# train_Y = torch.sin(train_X[:, 0:1] * 3) # + torch.cos(train_X[:, 1:2] * 3)
# train_Y += 0.05 * torch.randn_like(train_Y)
# bound = torch.Tensor([[0.0], [1.0]]).to(torch.double)

# node = bofn_node("test", LogExpectedImprovement, train_X, train_Y, bound, True)
# fig, ax = plt.subplots(1, 5)
# domain = torch.linspace(0, 1, 20)
# true_y = torch.sin(3*domain).numpy()
# for i in range(5):
#     print(f"Trial {i}")
#     candidate, _ = node.get_candidate(1)
#     y = torch.sin(3*candidate)
#     node.train(candidate, y)
#     ax[i].set_title(f"Trial {i}")
#     ax[i].plot(domain.numpy(), true_y)
#     ax[i].plot(domain.numpy(), node.eval_posterior(domain)[0].numpy())
# plt.show()
# plt.subplot(121)
# domain = torch.linspace(0, 1, 20, dtype=torch.double)
# plt.plot(domain.numpy(), np.sin(3*domain.numpy()))
# plt.plot(domain.numpy(), node.eval_posterior(domain)[0].numpy())
# plt.subplot(122)
# new_train_X = torch.rand(4, 1)
# new_train_Y = torch.sin(new_train_X[:, 0:1] * 3) # + torch.cos(train_X[:, 1:2] * 3)
# new_train_Y += 0.1 * torch.randn_like(new_train_Y)
# node.train(new_train_X, new_train_Y)
# plt.plot(domain.numpy(), np.sin(3*domain.numpy()))
# plt.plot(domain.numpy(), node.eval_posterior(domain)[0].numpy())
# plt.show()