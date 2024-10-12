import os
import numpy as np
import torch
from EinsumNetwork import Graph, EinsumNetwork
import datasets
import utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'

algorithms = ['SSM', 'SGD', 'EM']
# algorithms = ['EM']

dataset = 'fashion-mnist'       # choose either mnist or fashion-mnist

# ------------------------ constants -------------------------------
exponential_family = EinsumNetwork.BinomialArray
exponential_family_args = {'N': 255}

classes = [2]

n_slices = 1

K = 10
learning_rate = 0.2
lr_decay_step = 5
num_epochs = 20

# 'poon-domingos'
pd_num_pieces = [4]
# pd_num_pieces = [7]
# pd_num_pieces = [7, 28]
width = 28
height = 28

batch_size = 100
online_em_frequency = 1
online_em_stepsize = 0.05

# functions 
######################################
def ssm_loss(einet, x, n_slices=1):
    x = x.unsqueeze(0).expand(n_slices, *x.shape).contiguous().view(-1, *x.shape[1:])
    x.requires_grad_(True)

    v = torch.randn_like(x)

    outputs = einet.forward(x)
    logp = EinsumNetwork.log_likelihoods(outputs)

    score = torch.autograd.grad(logp.sum(), x, create_graph=True)[0]
    loss1 = 0.5 * (torch.norm(score, dim=-1) ** 2)

    grad2 = torch.autograd.grad(torch.sum(score * v), x, create_graph=True)[0]
    loss2 = torch.sum(v * grad2, dim=-1)

    return (loss1 + loss2).mean()

def get_data(): 
    
    if dataset == 'mnist':
        train_x, train_labels, test_x, test_labels = datasets.load_mnist() 
    if dataset == 'fashion-mnist':
        train_x, train_labels, test_x, test_labels = datasets.load_fashion_mnist()

    if not exponential_family != EinsumNetwork.NormalArray:
        train_x /= 255.
        test_x /= 255.
        train_x -= .5
        test_x -= .5

    # validation split
    valid_x = train_x[-10000:, :]
    train_x = train_x[:-10000, :]
    valid_labels = train_labels[-10000:]
    train_labels = train_labels[:-10000]

    # pick the selected classes
    if classes is not None:
        train_x = train_x[np.any(np.stack([train_labels == c for c in classes], 1), 1), :]
        valid_x = valid_x[np.any(np.stack([valid_labels == c for c in classes], 1), 1), :]
        test_x = test_x[np.any(np.stack([test_labels == c for c in classes], 1), 1), :]

    train_x = torch.from_numpy(train_x).to(torch.device(device))
    valid_x = torch.from_numpy(valid_x).to(torch.device(device))
    test_x = torch.from_numpy(test_x).to(torch.device(device))

    return train_x, valid_x, test_x
#####################################

train_x, valid_x, test_x = get_data()

pd_delta = [[height / d, width / d] for d in pd_num_pieces]
graph = Graph.poon_domingos_structure(shape=(height, width), delta=pd_delta)

for a in algorithms:

    use_em = False
    if a == 'EM': use_em = True

    args = EinsumNetwork.Args(
            num_var=train_x.shape[1],
            num_dims=1,
            num_classes=1,
            num_sums=K,
            num_input_distributions=K,
            exponential_family=exponential_family,
            exponential_family_args=exponential_family_args,
            use_em=use_em,
            online_em_frequency=online_em_frequency,
            online_em_stepsize=online_em_stepsize)

    einet = EinsumNetwork.EinsumNetwork(graph, args)
    einet.initialize()
    einet.to(device)

    train_N = train_x.shape[0]
    valid_N = valid_x.shape[0]
    test_N = test_x.shape[0]

    optimizer = torch.optim.Adam(einet.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_step, 0.5)

    for epoch_count in range(num_epochs):

        #### evaluate
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
        ####

        idx_batches = torch.randperm(train_N, device=device).split(batch_size)

        for idx in idx_batches:
            batch_x = train_x[idx, :]

            if a == 'EM':
                logp = EinsumNetwork.log_likelihoods(einet.forward(batch_x))
                log_likelihood = logp.sum()
                log_likelihood.backward()
                einet.em_process_batch()
            else: 
                
                if a == 'SSM':
                    loss = ssm_loss(einet, batch_x, n_slices)
                if a == 'SGD':
                    loss = -torch.mean(EinsumNetwork.log_likelihoods(einet.forward(batch_x)))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(einet.parameters(), 1.0)
                optimizer.step()

        scheduler.step()
            
        if a == 'EM': einet.em_update()

    # draw some samples 
    #####################

    samples_dir = 'results/'
    utils.mkdir_p(samples_dir)

    samples = einet.sample(num_samples=25).cpu().numpy()
    samples = samples.reshape((-1, 28, 28))
    utils.save_image_stack(samples, 5, 5, os.path.join(samples_dir, f"{classes[0]}{dataset}_{a}.png"), margin_gray_val=0.)

    print(f'{a}-logp = {EinsumNetwork.eval_loglikelihood_batched(einet, test_x, batch_size=batch_size) / test_N}')


# ground truth
ground_truth = test_x[0:25, :].cpu().numpy()
ground_truth = ground_truth.reshape((-1, 28, 28))
utils.save_image_stack(ground_truth, 5, 5, os.path.join(samples_dir, f'{classes[0]}{dataset}_ground_truth.png'), margin_gray_val=0.)
