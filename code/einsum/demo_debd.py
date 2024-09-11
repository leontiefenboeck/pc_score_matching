import torch
from EinsumNetwork import Graph, EinsumNetwork
import datasets

device = 'cuda' if torch.cuda.is_available() else 'cpu'

algorithm = 'SSM'

use_em = False
if algorithm == 'EM': use_em = True

##########################################################
DEBD = ['accidents', 'ad', 'baudio', 'bbc', 'bnetflix', 'book', 'c20ng', 'cr52', 'cwebkb', 'dna', 'jester', 'kdd',
        'kosarek', 'moviereview', 'msnbc', 'msweb', 'nltcs', 'plants', 'pumsb_star', 'tmovie', 'tretail', 'voting']

dataset = 'accidents'

depth = 3
num_repetitions = 10
num_input_distributions = 20
num_sums = 20

learning_rate = 0.1
num_epochs = 10
batch_size = 100
online_em_frequency = 1
online_em_stepsize = 0.05

# ----------------------------- functions -----------------------------
def ssm_loss(einet, x, n_slices=1):
    x = x.unsqueeze(0).expand(n_slices, *x.shape).contiguous().view(-1, *x.shape[1:])
    x.requires_grad_(True)

    v = torch.randn_like(x)

    logp = einet(x)
    logp = EinsumNetwork.log_likelihoods(logp)

    # TODO: this does not work with Categorical Array 
    score = torch.autograd.grad(logp.sum(), x, create_graph=True)[0]
    loss1 = 0.5 * (torch.norm(score, dim=-1) ** 2)

    grad2 = torch.autograd.grad(torch.sum(score * v), x, create_graph=True)[0]
    loss2 = torch.sum(v * grad2, dim=-1)

    return (loss1 + loss2).mean()
# ---------------------------------------------------------------------

print(dataset)

train_x_orig, test_x_orig, valid_x_orig = datasets.load_debd(dataset, dtype='float32')

train_x = train_x_orig
test_x = test_x_orig
valid_x = valid_x_orig

# to torch
train_x = torch.from_numpy(train_x).to(torch.device(device))
valid_x = torch.from_numpy(valid_x).to(torch.device(device))
test_x = torch.from_numpy(test_x).to(torch.device(device))

train_N, num_dims = train_x.shape
valid_N = valid_x.shape[0]
test_N = test_x.shape[0]

graph = Graph.random_binary_trees(num_var=train_x.shape[1], depth=depth, num_repetitions=num_repetitions)

args = EinsumNetwork.Args(
    num_classes=1,
    num_input_distributions=num_input_distributions,
    exponential_family=EinsumNetwork.CategoricalArray,
    exponential_family_args={'K': 2},
    num_sums=num_sums,
    num_var=train_x.shape[1],
    use_em=use_em,
    online_em_frequency=1,
    online_em_stepsize=0.05)

einet = EinsumNetwork.EinsumNetwork(graph, args)
einet.initialize()
einet.to(device)
print(einet)

optimizer = torch.optim.Adam(einet.parameters(), lr=learning_rate)

for epoch_count in range(num_epochs):

    ##### evaluate
    einet.eval()
    train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x, batch_size=batch_size)
    valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x, batch_size=batch_size)
    test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, test_x, batch_size=batch_size)
    print("[{}]   train LL {}   valid LL {}   test LL {}".format(
        epoch_count,
        train_ll / train_N,
        valid_ll / valid_N,
        test_ll / test_N))
    einet.train()
    #####

    idx_batches = torch.randperm(train_N, device=device).split(batch_size)

    for idx in idx_batches:
        batch_x = train_x[idx, :]

        if algorithm == 'EM':
            logp = EinsumNetwork.log_likelihoods(einet.forward(batch_x))
            log_likelihood = logp.sum()
            log_likelihood.backward()
            einet.em_process_batch()
        else: 
            optimizer.zero_grad()

            if algorithm == 'SSM':
                loss = ssm_loss(einet, batch_x)
            if algorithm == 'SGD':
                loss = -torch.mean(EinsumNetwork.log_likelihoods(einet.forward(batch_x)))

            loss.backward()
            optimizer.step()
        
    if algorithm == 'EM': einet.em_update()

print(EinsumNetwork.eval_loglikelihood_batched(einet, train_x, batch_size=batch_size))