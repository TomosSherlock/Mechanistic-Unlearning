import torch
import copy
import os
import psutil
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from utils import evaluate, display_progress
from torch import nn

# adapted from https://github.com/Happy2Git/Unlearning_Inference_IAM/blob/13f385b62e7cbc85c8c59d0912839e756db51cde/unlearning_lib/methods/fisher_forgetting.py

# represent a VFL model as one network
class FusedVFLModel(nn.Module):

  def __init__(self, model_a, model_b, top_model):
    super(FusedVFLModel, self).__init__()
    # Store the 3 sub-models
    self.bottom_model_A = model_a
    self.bottom_model_B = model_b

    # Downstream head/fusion layer that takes concatenated outputs
    self.top_model = top_model

  def forward(self, x):
    x_a = x[:, :, :, 0:14]
    x_b = x[:, :, :, 14:28]
    # Pass each slice/modality through its respective model
    out_a = self.bottom_model_A(x_a)
    out_b = self.bottom_model_B(x_b)

    # Concatenate the feature representations along dimension 1 (batch dimension 0)
    combined = torch.cat((out_a, out_b), dim=1)

    # Final prediction
    out = self.top_model(combined)
    return out

def fisher_forgetting(active_net_A, active_net_B, passive_net, retain_loader, forget_loader, test_retain_loader, test_forget_loader, args, run, batch_size=32, REPEAT=1, alpha: float = 1e-9, Method: str = 'EWC', DEVICE: str = 'cpu', evals = 20):
    """
    Here, we implement the diagnoal fisher estimate by a more efficient way.
    Reference:  Aditya Golatkar, Alessandro Achille, Stefano Soatto:
    Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks. CVPR 2020: 9301-9309
    """

    if (passive_net == None):
        fused_model = active_net_A
    else:
        fused_model = FusedVFLModel(active_net_A, active_net_B, passive_net)

    if Method == 'EWC':
        fused_modelf = ewc_estimate_fisher(
            fused_model, retain_loader, batch_size=batch_size, REPEAT=REPEAT, DEVICE=DEVICE)
    else:
        # broken
        fused_modelf = reference_fisher_init(fused_model, retain_loader, DEVICE)

    # check the classes of retain_loader if there is a class to forget
    # if there is a class to forget, we set the mu and var of that class to 0
    num_classes = 10

    existing_class = []
    for x, y in retain_loader:
        for i in y:
            if i not in existing_class:
                existing_class.append(i)

    class_to_forget = None
    if len(existing_class) < num_classes:
        class_to_forget = [i for i in range(
            num_classes) if i not in existing_class]

    if (passive_net == None) :
        fused_bottom_model_a = fused_modelf
        fused_bottom_model_b = None
    else :
        fused_bottom_model_a = fused_modelf.bottom_model_A
        fused_bottom_model_b = fused_modelf.bottom_model_B
            

    # using forgotten sample performance, find the best place to stop unlearning early
    best_modelf = fused_modelf
    best_forgotten_samples_acc = evaluate(fused_bottom_model_a, fused_bottom_model_b, passive_net, forget_loader, args)

    for i, p in enumerate(fused_modelf.parameters()):
        mu, var = get_mean_var(p, num_classes, alpha=alpha,
                            class_to_forget=class_to_forget)
        mu, var = mu.to(DEVICE), var.to(DEVICE)
        epsilon = 1e-30
        p.data = mu + var.sqrt() * torch.full_like(p.data, epsilon).normal_()

        # currently only works for 2 active parties
        if (i % evals == 0) :
            # handle the non-federated case
            if (passive_net == None) :
                fused_bottom_model_a = fused_modelf
                fused_bottom_model_b = None
            else :
                fused_bottom_model_a = fused_modelf.bottom_model_A
                fused_bottom_model_b = fused_modelf.bottom_model_B
            
            retain_acc = evaluate(fused_bottom_model_a, fused_bottom_model_b, passive_net, test_retain_loader, args)
            biased_acc = evaluate(fused_bottom_model_a, fused_bottom_model_b, passive_net, test_forget_loader, args)
            acc_forget = evaluate(fused_bottom_model_a, fused_bottom_model_b, passive_net, forget_loader, args)

            if (acc_forget < best_forgotten_samples_acc) :
                best_modelf = fused_modelf
                best_forgotten_samples_acc = acc_forget

            print(f"Unbiased Class acc : {retain_acc}")
            print(f"Biased Class acc : {biased_acc}")
            print(f"Forgotten Samples acc : {acc_forget}")

            run.log({"Unbiased Class acc" : retain_acc})
            run.log({"Biased Class acc" : biased_acc})
            run.log({"Forgotten Samples Accuracy" : acc_forget})

    if (passive_net == None):
        return best_modelf, None
    return best_modelf.bottom_model_A, best_modelf.bottom_model_B


# ntk_fisher_forgetting: out of memory on large dataset (>10000)
def ntk_fisher_forgetting(model, retain_loader, forget_loader, alpha=1e-8, weight_decay=0.1, DSLIMIT: int = 600, DEVICE: str = 'cpu'):
    """
    Reference:  Aditya Golatkar, Alessandro Achille, Stefano Soatto:
    Forgetting Outside the Box: Scrubbing Deep Networks of Information Accessible 
    from Input-Output Observations. ECCV (29) 2020: 383-398
    """
    if not os.path.exists('NTK_data'):
        os.makedirs('NTK_data')
    net_scrubbed = copy.deepcopy(model)

    # ntk scrubbing
    # model_init = get_model_init(model, DEVICE)
    direction, scale = ntk_scrub_vector(
        model, retain_loader, forget_loader, weight_decay, DSLIMIT=DSLIMIT)
    for k, p in net_scrubbed.named_parameters():
        p.data += (direction[k]*scale).to(DEVICE)
    # fisher forgetting
    net = fisher_forgetting(net_scrubbed, retain_loader,
                            alpha=alpha, DEVICE=DEVICE)
    net.eval()
    return net


def ewc_estimate_fisher(net, retain_loader, batch_size=32, REPEAT=1, DEVICE: str = 'cpu'):
    """
    The diagonal terms of the empirical Fisher information matrix can be approximated 
        by taking the average of the squared gradients over the data distribution.

    ref: https://github.com/kuc2477/pytorch-ewc/blob/master/model.py

    self suggestions: 
        - Use a large batch size to get more stable gradient estimates. 
            limit the number of samples used to compute Fisher (say 256), which is good for most use-cases.
        - Iterate through the full dataset at least 2-3 times to get smoother values. 
            The code goes through once.
    """
    # sample loglikelihoods from the dataset.
    modelf = copy.deepcopy(net)
    for p in modelf.parameters():
        p.data0 = copy.deepcopy(p.data.clone())

    retain_loader_b1 = torch.utils.data.DataLoader(
        retain_loader.dataset, batch_size=batch_size, shuffle=True)
    num_samples = len(retain_loader_b1.dataset)
    epsilon = 1e-30
    fisher_diagonals = [torch.full_like(p, epsilon, device='cpu') for p in modelf.parameters()]
   
    for repeat in range(REPEAT):
        for enu, (x, y) in enumerate(retain_loader_b1):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            log_likelihoods = F.log_softmax(modelf(x).permute(1,0), dim=1)
            log_likelihoods = log_likelihoods.permute(1,0)

            y = y.squeeze()
            rows = torch.arange(y.shape[0], device=DEVICE)

            try:
                loglikelihoods = log_likelihoods[rows, y.long()]
            except IndexError as e:
                print("--- CRASH DETECTED ---")
                print(f"Error message: {e}")
                print(f"y shape: {y.shape}")
                print(f"y device: {y.device}")
                print(f"len(y) evaluated to: {len(y)}")
                print(f"l shape: {log_likelihoods.shape}")
                raise e

            # estimate the fisher information of the parameters.
            loglikelihoods = loglikelihoods.unbind()

            loglikelihood_grads = zip(*[torch.autograd.grad(
                l, modelf.parameters(),
                retain_graph=(i < len(loglikelihoods))
            ) for i, l in enumerate(loglikelihoods, 1)])
            loglikelihood_grads = [torch.stack(
                gs).detach() for gs in loglikelihood_grads]
            for i, g in enumerate(loglikelihood_grads):
                fisher_diagonals[i] += torch.clamp(
                    (g**2).sum(dim=0).cpu(), min=0)

            del loglikelihood_grads, loglikelihoods, x, y
            torch.cuda.empty_cache()
            display_progress("fisher estimation", enu+len(retain_loader_b1)
                             * repeat, len(retain_loader_b1)*REPEAT)

    for i in range(len(fisher_diagonals)):
        fisher_diagonals[i] /= (num_samples*REPEAT)
    # let the param.grad2_acc be the fisher information.
    for p, f in zip(modelf.parameters(), fisher_diagonals):
        p.grad2_acc = f.detach().clone()

    return modelf

def reference_fisher_init(model_input, retain_loader, DEVICE):
    """
    Reference:  Aditya Golatkar, Alessandro Achille, Stefano Soatto:
    Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks. CVPR 2020: 9301-9309
    """
    modelf = copy.deepcopy(model_input)
    for p in modelf.parameters():
        p.data0 = copy.deepcopy(p.data.clone())

    modelf.eval()
    retain_loader_b1 = torch.utils.data.DataLoader(
        retain_loader.dataset, batch_size=1, shuffle=False)
    loss_fn = torch.nn.CrossEntropyLoss()

    for p in modelf.parameters():
        p.grad2_acc = 0

    for i, (data, orig_target) in enumerate(retain_loader_b1):
        data, orig_target = data.to(DEVICE), orig_target.to(DEVICE)
        output = modelf(data)
        prob = F.softmax(output, dim=-1).data

        for y in range(output.shape[1]):
            target = torch.full_like(orig_target,y)
            loss = loss_fn(output, target)
            modelf.zero_grad()
            loss.backward(retain_graph=True)
            for p in modelf.parameters():
                if p.requires_grad:
                    p.grad2_acc += torch.clamp(prob[:, y]
                                               * p.grad.data.pow(2), min=0)

        display_progress("Fisher init", i, len(
            retain_loader_b1.dataset), enabled=True)

    for p in modelf.parameters():
        p.grad2_acc /= len(retain_loader_b1)
    return modelf


def get_mean_var(p, num_classes, alpha=1e-9, class_to_forget=None):
    mu = copy.deepcopy(p.data0.clone())
    var = copy.deepcopy(1./torch.abs(p.grad2_acc+1e-8))
    var = var.clamp(max=1e3)
    if p.size(0) == num_classes:
        var = var.clamp(max=1e2)
    var = alpha * var

    if p.ndim > 1:
        var = var.mean(dim=1, keepdim=True).expand_as(p).clone()

    # class forgetting
    if p.size(0) == num_classes and class_to_forget is not None:
        mu[class_to_forget] = 0
        var[class_to_forget] = 0.0001

    if p.size(0) == num_classes:
        # Last layer
        var *= 10
    elif p.ndim == 1:
        # BatchNorm
        var *= 10
#         var*=1
    return mu, var


def ntk_scrub_vector(model, retain_loader, forget_loader, weight_decay=0.1, DSLIMIT=None):
    """
    If len(dataset) is 600, class_number is 10, then G.shape of ResNet18 is (600*10, 11181642), 
    the size of G is ~25G in float32 format, which is too large to store in memory.
    So we split the dataset into two parts, retain and forget.
    """
    dlen = len(retain_loader.dataset)+len(forget_loader.dataset)
    plen = sum([np.prod(list(p.shape)) for p in model.parameters()])
    num_classes = 10
    if dlen * plen * num_classes > 1e10 and DSLIMIT is None:
        raise ValueError(
            "The size of G will be too large (> 40GB) to store in memory.")

    # calculate G_r
    G_r_cpt, f0_minus_y_r_cpt = delta_w_utils(copy.deepcopy(
        model), retain_loader, 'retain', DSLIMIT=int(DSLIMIT*len(retain_loader.dataset)/dlen))
    print("save the checkpoint of _cpt in {} and f0_minus_y_r in {}".format(
        G_r_cpt, f0_minus_y_r_cpt))

    G_f_cpt, f0_minus_y_f_cpt = delta_w_utils(copy.deepcopy(
        model), forget_loader, 'forget', DSLIMIT=int(DSLIMIT*len(forget_loader.dataset)/dlen))
    print("save the checkpoint of G_f in {} and f0_minus_y_f in {}".format(
        G_f_cpt, f0_minus_y_f_cpt))

    # load all checkpoints of G_r and G_f and concatenate them
    G = []
    for i in range(int(G_r_cpt.split('cpt')[-1].split('.')[0])):
        G.append(torch.load("NTK_data/G_list_retain_cpt{}.pth".format(i+1)))
    for i in range(int(G_f_cpt.split('cpt')[-1].split('.')[0])):
        G.append(torch.load("NTK_data/G_list_forget_cpt{}.pth".format(i+1)))
    G = torch.cat(G, dim=1)

    torch.save(G, "NTK_data/G.pth")
    del G
    # todo, use the checkpoint to save memory

    f0_minus_y = []
    for i in range(int(f0_minus_y_r_cpt.split('cpt')[-1].split('.')[0])):
        f0_minus_y.append(torch.load(
            "NTK_data/f0_minus_y_retain_cpt{}.pth".format(i+1)))
    for i in range(int(f0_minus_y_f_cpt.split('cpt')[-1].split('.')[0])):
        f0_minus_y.append(torch.load(
            "NTK_data/f0_minus_y_forget_cpt{}.pth".format(i+1)))
    f0_minus_y = torch.cat(f0_minus_y, dim=0)
    torch.save(f0_minus_y, "NTK_data/f0_minus_y.pth")

    G = torch.load("NTK_data/G.pth")
    print("G.shape: ", G.shape, "f0_minus_y.shape: ", f0_minus_y.shape)
    theta = G.T @ G + (len(retain_loader.dataset) +
                       len(forget_loader.dataset))*weight_decay*torch.eye(G.shape[1])
    torch.save(theta, "NTK_data/theta.pth")
    del G, f0_minus_y

    theta_inv = torch.inverse(theta)
    torch.save(theta_inv, "NTK_data/theta_inv.pth")
    del theta

    G = torch.load("NTK_data/G.pth")
    f0_minus_y = torch.load("NTK_data/f0_minus_y.pth")
    w_complete = - G @ (theta_inv @ f0_minus_y)

    torch.save(w_complete, "NTK_data/w_complete.pth")
    del G, f0_minus_y, theta_inv, w_complete

    G_r = []
    for i in range(int(G_r_cpt.split('cpt')[-1].split('.')[0])):
        G_r.append(torch.load("NTK_data/G_list_retain_cpt{}.pth".format(i+1)))
    G_r = torch.cat(G_r, dim=1)
    theta_r = G_r.T @ G_r + len(retain_loader.dataset) * \
        weight_decay*torch.eye(G_r.shape[1])
    torch.save(theta_r, "NTK_data/theta_r.pth")
    del G_r

    theta_r_inv = torch.inverse(theta_r)
    torch.save(theta_r_inv, "NTK_data/theta_r_inv.pth")
    del theta_r

    G_r = []
    for i in range(int(G_r_cpt.split('cpt')[-1].split('.')[0])):
        G_r.append(torch.load("NTK_data/G_list_retain_cpt{}.pth".format(i+1)))
    G_r = torch.cat(G_r, dim=1)

    f0_minus_y_r = []
    for i in range(int(f0_minus_y_r_cpt.split('cpt')[-1].split('.')[0])):
        f0_minus_y_r.append(torch.load(
            "NTK_data/f0_minus_y_retain_cpt{}.pth".format(i+1)))
    f0_minus_y_r = torch.cat(f0_minus_y_r, dim=0)

    w_retain = -G_r @ (theta_r_inv @ f0_minus_y_r)
    torch.save(w_retain, "NTK_data/w_retain.pth")
    del G_r, f0_minus_y_r, theta_r_inv, w_retain

    w_complete = torch.load("NTK_data/w_complete.pth")
    w_retain = torch.load("NTK_data/w_retain.pth")
    delta_w = (w_retain-w_complete).squeeze()
    scale = trapezium_trick(model, delta_w, w_retain)
    direction = get_delta_w_dict(delta_w, model)
    return direction, scale

def get_params(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

def delta_w_utils(device, model, dataloader, name='retain', lossfn='ce', dataset='cifar10', DSLIMIT=None):
    DEVICE = device
    num_classes = 10
    dataloader = torch.utils.data.DataLoader(
        dataloader.dataset, batch_size=1, shuffle=False)

    n_params = get_params(model)
    # check the memory usage and calculate DSLIMIT and decide whether to save checkpoints of G_list and f0_minus_y
    CPU_mem, GPU_mem = get_memory_usage()
    DSLIMIT = DSLIMIT if DSLIMIT is not None else len(dataloader.dataset)
    SIZE = DSLIMIT * sum([np.prod(list(p.shape))
                         for p in n_params]) * num_classes
    redundancy = 1.1
    Task_mem = redundancy*SIZE*4/(2**30)
    checkpoints_num = int(np.ceil(Task_mem/CPU_mem['free']))
    DSLIMIT = int(DSLIMIT/checkpoints_num)
    # raise ValueError("CPU_mem: {:.0f} GB, GPU_mem: {:.0f} GB, Task_mem: {:.0f} GB, checkpoints_num: {:.0f}".format(CPU_mem['free'], GPU_mem['free'], Task_mem, checkpoints_num))

    G_list = []
    f0_minus_y = []
    # (tqdm(dataloader,leave=False)):
    # use tqdm
    for idx, (input, target) in enumerate(dataloader):
        # display_progress("ntk vector estimation", idx, len(dataloader))

        if DSLIMIT is not None and idx > DSLIMIT*checkpoints_num:
            break
        # elif DSLIMIT is not None:
            # print("idx: {:.0f} with maxmium {:.0f} GB with limit {:.0f} and  size {:.0f}".format(
            #     idx, SIZE*4/(2**30), DSLIMIT*checkpoints_num, SIZE))

        input, target = input.to(DEVICE), target.to(
            DEVICE)
        if 'mnist' in dataset:
            input = input.view(input.shape[0], -1)
        target = target.cpu()
        output = model(input)
        for cls in range(num_classes):
            RETAIN_GRAPH = True if cls < num_classes-1 else False
            grads = torch.autograd.grad(
                output[0, cls], model.parameters(), retain_graph=RETAIN_GRAPH)
            G_list.append(
                torch.cat([g.view(-1).detach().cpu() for g in grads]))
        if lossfn == 'mse':
            p = output.cpu().detach().T
            # loss_hess = np.eye(len(p))
            target = 2*target-1
            f0_y_update = p-target
            f0_minus_y.append(f0_y_update)
        elif lossfn == 'ce':
            p = torch.nn.functional.softmax(output, dim=1).cpu().detach().T
            p[target] -= 1
            f0_y_update = p
            f0_minus_y.append(f0_y_update)

        if idx % DSLIMIT == 0 and idx != 0:
            # save checkpoints and clear the memory
            torch.save(torch.stack(G_list).T,
                       "NTK_data/G_list_{}_cpt{}.pth".format(name, idx//DSLIMIT))
            torch.save(torch.vstack(
                f0_minus_y), "NTK_data/f0_minus_y_{}_cpt{}.pth".format(name, idx//DSLIMIT))
            G_list = []
            f0_minus_y = []

    if len(G_list) != 0:
        print(len(G_list), len(f0_minus_y))
        torch.save(torch.stack(G_list).T,
                   "NTK_data/G_list_cpt{}.pth".format(idx//DSLIMIT+1))
        torch.save(torch.vstack(f0_minus_y),
                   "NTK_data/f0_minus_y__cpt{}.pth".format(idx//DSLIMIT+1))
        G_list = []
        f0_minus_y = []
        return 'NTK_data/G_list_cpt{}.pth'.format(idx//DSLIMIT+1), 'NTK_data/f0_minus_y__cpt{}.pth'.format(idx//DSLIMIT+1)
    else:
        return 'NTK_data/G_list_cpt{}.pth'.format(idx//DSLIMIT), 'NTK_data/f0_minus_y__cpt{}.pth'.format(idx//DSLIMIT)


def get_delta_w_dict(delta_w, model):
    # Give normalized delta_w
    delta_w_dict = OrderedDict()
    params_visited = 0
    for k, p in model.named_parameters():
        num_params = np.prod(list(p.shape))
        update_params = delta_w[params_visited:params_visited+num_params]
        delta_w_dict[k] = torch.Tensor(update_params).view_as(p)
        params_visited += num_params
    return delta_w_dict


def vectorize_params(model):
    param = []
    for p in model.parameters():
        param.append(p.data.view(-1).cpu())
    return torch.cat(param)


def trapezium_trick(model, model_init, delta_w, w_retain, VERBOSE=False):
    m_pred_error = vectorize_params(
        model)-vectorize_params(model_init)-w_retain.squeeze()
    print(
        f"Pred Error Norm: {torch.linalg.norm(delta_w)}") if VERBOSE else None

    inner = torch.inner(delta_w/torch.linalg.norm(delta_w),
                        m_pred_error/torch.linalg.norm(m_pred_error))
    print(f"Inner Product--: {inner}") if VERBOSE else None

    if inner < 0:
        angle = torch.arccos(inner)-torch.pi/2
        print(f"Angle----------:  {angle}") if VERBOSE else None

        predicted_norm = torch.linalg.norm(
            delta_w) + 2*torch.sin(angle)*torch.linalg.norm(m_pred_error)
        print(f"Pred Act Norm--:  {predicted_norm}") if VERBOSE else None
    else:
        angle = torch.arccos(inner)
        print(f"Angle----------:  {angle}") if VERBOSE else None

        predicted_norm = torch.linalg.norm(
            delta_w) + 2*torch.cos(angle)*torch.linalg.norm(m_pred_error)
        print(f"Pred Act Norm--:  {predicted_norm}") if VERBOSE else None

    predicted_scale = predicted_norm/torch.linalg.norm(delta_w)
    print(f"Predicted Scale:  {predicted_scale}") if VERBOSE else None
    return predicted_scale


def get_memory_usage():
    """Returns the memory usage of the current process in MB"""

    total = psutil.virtual_memory().total / (1024**3)
    available = psutil.virtual_memory().available / (1024**3)
    CPU_mem = {'total': total, 'free': available}

    total_mem, free_mem = torch.cuda.mem_get_info()
    GPU_mem = {'total': total_mem / (1024**3), 'free': free_mem / (1024**3)}
    return CPU_mem, GPU_mem