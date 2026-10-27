import torch
from torch import nn
import math
import numpy as np
from utils import update_logger, evaluate
import torch.nn.functional as F
from copy import deepcopy

def report(active_models, top_model, retain_loader, forget_loader, epoch, run, args, alpha=0, forgotten_accuracy=0):
    acc_remain = evaluate(active_models[0], active_models[1], top_model, retain_loader, args)
    acc_forget = evaluate(active_models[0], active_models[1], top_model, forget_loader, args)

    print(f'epoch : {epoch}')
    print(f'remain acc : {acc_remain}')
    print(f'bias acc: {acc_forget}')
    print(f'alpha: {alpha}')
    # print(f'forgot acc: {forgotten_accuracy}')

    run.log({
        "Epoch" : epoch,
        "Unbiased Class Accuracy" : acc_remain,
        "Biased Class Accuracy" : acc_forget,
        "Forgotten Sample Accuracy" : forgotten_accuracy,
        "Alpha" : alpha,
        })

def federate_data(images, labels, args):
    if args.data != 'yahoo':
        images = images.to(args.device)
        labels = labels.to(args.device)
    if args.data == 'cifar10':
        x_a = images[:, :, :, 0:16]
        x_b = images[:, :, :, 16:32]
    elif args.data == 'mnist':
        x_a = images[:, :, :, 0:14]
        x_b = images[:, :, :, 14:28]
    elif args.data == 'cifar100':
        x_a = images[:, :, :, 0:16]
        x_b = images[:, :, :, 16:32]
    elif args.data == 'mri':
        x_a = images[:, :, :, 0:112]
        x_b = images[:, :, :, 112:224]
    elif args.data == 'ct':
        x_a = images[:, :, :, 0:112]
        x_b = images[:, :, :, 112:224]
    elif args.data == 'yahoo':
        for i in range(len(images)):
            images[i] = images[i].long().to(args.device)
        labels = labels[0].long().to(args.device)
        x_b = images[1]
        x_a = images[0]
    elif args.data == 'custom':
        x_a = images[:, :, :, 0:14]
        x_b = images[:, :, :, 14:28]
    elif args.data == 'custom-1':
            x_a = torch.stack([images[:, :, 0:14]])
            x_b = torch.stack([images[:, :, 14:28]])
            x_a = torch.permute(x_a, (1,0,2,3))
            x_b = torch.permute(x_b, (1,0,2,3))
    else:
        raise ValueError(f'No dataset named {args.data}!')
    return x_a, x_b, labels

def get_bottom_embedding_exp(output_tensor_bottom_model, mixup):
    for i in range(output_tensor_bottom_model.shape[0]):
        x1_bottom_a = output_tensor_bottom_model[i]
        if i == output_tensor_bottom_model.shape[0]-2:
            mid_bottom_a = mixup[1]*x1_bottom_a + (1-mixup[1])*x2_bottom_a
            front_mid_bottom_a = mixup[0]*x1_bottom_a + (1-mixup[0])*x2_bottom_a
            back_mid_bottom_a = mixup[2]*x1_bottom_a + (1-mixup[2])*x2_bottom_a
            temperory_bottom_a = torch.stack((front_mid_bottom_a,mid_bottom_a,back_mid_bottom_a))
            bottom_a_embedding_exp = torch.cat((bottom_a_embedding_exp,temperory_bottom_a),0)

        else:
            for x in range(i+1, output_tensor_bottom_model.shape[0]):
                x2_bottom_a = output_tensor_bottom_model[x]
                mid_bottom_a = mixup[1]*x1_bottom_a + (1-mixup[1])*x2_bottom_a
                front_mid_bottom_a = mixup[0]*x1_bottom_a + (1-mixup[0])*x2_bottom_a
                back_mid_bottom_a = mixup[2]*x1_bottom_a + (1-mixup[2])*x2_bottom_a
                if x == i+1:
                    if i == 0:
                        bottom_a_embedding_exp = torch.stack((front_mid_bottom_a,mid_bottom_a,back_mid_bottom_a))
                    else:
                        temperory_bottom_a = torch.stack((front_mid_bottom_a,mid_bottom_a,back_mid_bottom_a))
                        bottom_a_embedding_exp = torch.cat((bottom_a_embedding_exp,temperory_bottom_a),0)
                else:
                    temperory_bottom_a = torch.stack((front_mid_bottom_a,mid_bottom_a,back_mid_bottom_a))
                    bottom_a_embedding_exp = torch.cat((bottom_a_embedding_exp,temperory_bottom_a),0)

    return bottom_a_embedding_exp

# One epoch of LUV Unlearning
def LUV_unlearn(active_models, top_model, active_optimizers, top_model_optimizer, criterion, LUV_trainloader, fewshot_descent_loader, alpha, args):
    for i, (images, labels) in enumerate(LUV_trainloader):
        x_a, x_b, labels = federate_data(images, labels, args)
        federated_data = [x_a, x_b]
        
        top_model_optimizer.zero_grad()

        mixup = [0.25, 0.5, 0.75]

        forward_bottoms = []
        grad_output_bottom_models = []
        output_tensors = []

        for j, model in enumerate(active_models):
            x = federated_data[j]
            output_tensor_bottom_model = model(x)
            output_tensors.append(output_tensor_bottom_model)

            bottom_a_embedding_exp = get_bottom_embedding_exp(output_tensor_bottom_model, mixup)

            input_tensor_top_model_a = torch.tensor([], requires_grad=True)
            input_tensor_top_model_a.data = bottom_a_embedding_exp.data

            forward_bottom_a = []

            for y in range(math.ceil(bottom_a_embedding_exp.shape[0]/args.batch_size)):
                if y == 0:
                    forward_bottom_a.append(input_tensor_top_model_a[0:args.batch_size])
                else:
                    forward_bottom_a.append(input_tensor_top_model_a[(y*args.batch_size):((y+1)*args.batch_size)])


            grad_output_bottom_model_a = torch.tensor([])
            grad_output_bottom_model_a = grad_output_bottom_model_a.to(args.device)

            forward_bottoms.append(forward_bottom_a)
            grad_output_bottom_models.append(grad_output_bottom_model_a)


        for z in range(len(forward_bottoms[0])):
            forward_bottoms[0][z] = forward_bottoms[0][z].detach()
            forward_bottoms[0][z].requires_grad=True
            forward_bottoms[1][z] = forward_bottoms[1][z].detach()
            forward_bottoms[1][z].requires_grad=True

            outputs = top_model(forward_bottoms[0][z], forward_bottoms[1][z])
            label = torch.full((outputs.shape[0],), args.unlearn_class, dtype=int)
            label = label.to('cuda')
            loss = -criterion(outputs, label)
            loss.backward()
            top_model_optimizer.step()

            grad_output_bottom_models[0] = torch.cat((grad_output_bottom_models[0], forward_bottoms[0][z].grad),0)
            grad_output_bottom_models[1] = torch.cat((grad_output_bottom_models[1], forward_bottoms[1][z].grad),0)

        for j, model in enumerate(active_models):
            x = federated_data[j]
            n = math.ceil(grad_output_bottom_models[j].shape[0]/output_tensors[j].shape[0])
            grad_a = torch.tensor([])
            grad_a = grad_a.to(args.device)
            for m in range(output_tensors[j].shape[0]):
                if m == 0:
                    grad_a = torch.cat((grad_a,(torch.mean(grad_output_bottom_models[j][m:n], axis=0)).unsqueeze(0)), 0)
                else:
                    grad_a = torch.cat((grad_a,(torch.mean(grad_output_bottom_models[j][(m*n):((m+1)*n)], axis=0)).unsqueeze(0)), 0)
            
            loss_bottom_A = torch.sum(grad_a * output_tensors[j])
            active_optimizers[j].zero_grad()
            loss_bottom_A.backward()
            active_optimizers[j].step()

    alphaa=1
    for i, (images, labels) in enumerate(fewshot_descent_loader):

        x_a, x_b, labels = federate_data(images, labels, args)
        
        top_model_optimizer.zero_grad()

        output_tensor_bottom_model_a = active_models[0](x_a)
        output_tensor_bottom_model_b = active_models[1](x_b)

        emb_descent_a, emb_descent_b = mixup_embeddings(output_tensor_bottom_model_a, output_tensor_bottom_model_b, labels, alpha)

        input_tensor_top_model_a_descent = torch.tensor([], requires_grad=True)
        input_tensor_top_model_a_descent.data = emb_descent_a.data
        input_tensor_top_model_b_descent = torch.tensor([], requires_grad=True)
        input_tensor_top_model_b_descent.data = emb_descent_b.data

        output_descent = top_model(input_tensor_top_model_a_descent, input_tensor_top_model_b_descent)
        loss_descent = alphaa * criterion(output_descent, labels)
        
        top_model_optimizer.zero_grad()
        loss_descent.backward()
        top_model_optimizer.step()

        grad_descent_a = input_tensor_top_model_a_descent.grad
        grad_descent_b = input_tensor_top_model_b_descent.grad

        loss_bottom_A = torch.sum(grad_descent_a * output_tensor_bottom_model_a)
        active_optimizers[0].zero_grad()
        loss_bottom_A.backward()
        active_optimizers[0].step()

        loss_bottom_B = torch.sum(grad_descent_b * output_tensor_bottom_model_b)
        active_optimizers[1].zero_grad()
        loss_bottom_B.backward()
        active_optimizers[1].step()

# LUV for the unfederated case
def LUV_unfederated(LUV_trainloader, fewshot_descent_loader, model, criterion, args, run, retain_loader, forget_loader):
    print("unlearn method : ", args.unlearn_method)
    model.train()

    optimizer = torch.optim.SGD(
                        model.parameters(),
                        args.unlearn_lr, 
                        momentum=args.momentum, 
                        weight_decay=args.weight_decay)
    epochs = args.unlearn_epochs

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(LUV_trainloader):

            if args.data == 'custom-1':
                    x = torch.stack([images])
                    x = torch.permute(x, (1,0,2,3))
            else:
                x = images.to(args.device)

            optimizer.zero_grad()

            # instead of mixing up embeddings between federated models, mixup embeddings in the middle layer of a model.
            mid_embeddings = model.layer2(model.layer1(model.layer0(x)))

            mixup = [0.25, 0.5, 0.75]
            for i in range(mid_embeddings.shape[0]):
                x1_bottom = mid_embeddings[i]
                if i == mid_embeddings.shape[0]-2:
                    mid_bottom = mixup[1]*x1_bottom + (1-mixup[1])*x2_bottom
                    front_mid_bottom = mixup[0]*x1_bottom + (1-mixup[0])*x2_bottom
                    back_mid_bottom = mixup[2]*x1_bottom + (1-mixup[2])*x2_bottom
                    temperory_bottom = torch.stack((front_mid_bottom,mid_bottom,back_mid_bottom))
                    bottom_embedding_exp = torch.cat((bottom_embedding_exp,temperory_bottom),0)

                else:
                    for x in range(i+1, mid_embeddings.shape[0]):
                        x2_bottom = mid_embeddings[x]
                        mid_bottom = mixup[1]*x1_bottom + (1-mixup[1])*x2_bottom
                        front_mid_bottom = mixup[0]*x1_bottom + (1-mixup[0])*x2_bottom
                        back_mid_bottom = mixup[2]*x1_bottom + (1-mixup[2])*x2_bottom
                        if x == i+1:
                            if i == 0:
                                bottom_embedding_exp = torch.stack((front_mid_bottom,mid_bottom,back_mid_bottom))
                            else:
                                temperory_bottom = torch.stack((front_mid_bottom,mid_bottom,back_mid_bottom))
                                bottom_embedding_exp = torch.cat((bottom_embedding_exp,temperory_bottom),0)
                        else:
                            temperory_bottom = torch.stack((front_mid_bottom,mid_bottom,back_mid_bottom))
                            bottom_embedding_exp = torch.cat((bottom_embedding_exp,temperory_bottom),0)

            input_tensor_top_model = torch.tensor([], requires_grad=True)
            input_tensor_top_model.data = bottom_embedding_exp.data

            forward_bottom = []

            for y in range(math.ceil(bottom_embedding_exp.shape[0]/args.batch_size)):
                if y == 0:
                    forward_bottom.append(input_tensor_top_model[0:args.batch_size])
                else:
                    forward_bottom.append(input_tensor_top_model[(y*args.batch_size):((y+1)*args.batch_size)])


            grad_output_bottom_model = torch.tensor([])
            grad_output_bottom_model = grad_output_bottom_model.to(args.device)


            for z in range(len(forward_bottom)):
                forward_bottom[z] = forward_bottom[z].detach()
                forward_bottom[z].requires_grad=True

                # pass mixed up embeddings through the rest of the model
                outputs = model.layer3(forward_bottom[z])
                outputs = model.layer4(outputs)
                outputs = F.avg_pool2d(outputs, outputs.size()[2:])
                outputs = outputs.view(outputs.size(0), -1)
                outputs = model.linear(outputs)

                label = torch.full((outputs.shape[0],), args.unlearn_class, dtype=int)
                label = label.to('cuda')
                loss = -criterion(outputs, label)
                loss.backward()
                optimizer.step()

                grad_output_bottom_model = torch.cat((grad_output_bottom_model, forward_bottom[z].grad),0)


            n = math.ceil(grad_output_bottom_model.shape[0]/mid_embeddings.shape[0])
            grad = torch.tensor([])
            grad = grad.to(args.device)
            for m in range(mid_embeddings.shape[0]):
                if m == 0:
                    grad = torch.cat((grad,(torch.mean(grad_output_bottom_model[m:n], axis=0)).unsqueeze(0)), 0)
                else:
                    grad = torch.cat((grad,(torch.mean(grad_output_bottom_model[(m*n):((m+1)*n)], axis=0)).unsqueeze(0)), 0)
            
            loss_bottom = torch.sum(grad * mid_embeddings)
            optimizer.zero_grad()
            loss_bottom.backward()
            optimizer.step()


        alphaa=1
        for i, (images, labels) in enumerate(fewshot_descent_loader):
            if args.data == 'custom-1':
                    x = torch.stack([images])
                    x = torch.permute(x, (1,0,2,3))
            else:
                x = images
            optimizer.zero_grad()

            mid_embeddings = model.layer2(model.layer1(model.layer0(x)))

            emb_descent = mixup_embeddings_unfederated(mid_embeddings, labels, 1)

            input_tensor_top_model_descent = torch.tensor([], requires_grad=True)
            input_tensor_top_model_descent.data = emb_descent.data

            outputs = model.layer3(input_tensor_top_model_descent[z])
            outputs = model.layer4(outputs)
            outputs = F.avg_pool2d(outputs, outputs.size()[2:])
            outputs = outputs.view(outputs.size(0), -1)
            outputs = model.linear(outputs)

            output_descent = outputs
            loss_descent = alphaa * criterion(output_descent, labels)

            optimizer.zero_grad()
            loss_descent.backward()
            optimizer.step()

            grad_descent = input_tensor_top_model_descent.grad

            loss_bottom = torch.sum(grad_descent * mid_embeddings)
            optimizer.zero_grad()
            loss_bottom.backward()
            optimizer.step()

        acc_remain = evaluate(model, None, None, retain_loader, args)
        acc_forget = evaluate(model, None, None, forget_loader, args)

        run.log({"Epoch" : epoch})
        run.log({"Unbiased Class Accuracy" : acc_remain})
        run.log({"Biased Class Accuracy" : acc_forget})

def LUV(LUV_trainloader, fewshot_descent_loader, bottom_model_A, bottom_model_B, top_model, criterion, class_to_forget, args, run, retain_loader, forget_loader, alpha):
    print("unlearn method : ", args.unlearn_method)

    if (top_model == None):
        return LUV_unfederated(LUV_trainloader, fewshot_descent_loader, bottom_model_A, criterion, args, run, retain_loader, forget_loader)

    bottom_model_A.train()
    bottom_model_B.train()
    top_model.train()

    top_model_optimizer = torch.optim.SGD(
                        top_model.parameters(),
                        args.unlearn_lr, 
                        momentum=args.momentum, 
                        weight_decay=args.weight_decay)
    
    epochs = args.unlearn_epochs

    active_models = [bottom_model_A, bottom_model_B]
    active_optimizers = []

    for model in active_models:
        active_optimizers.append(
            torch.optim.SGD(
                model.parameters(),
                args.unlearn_lr, 
                momentum=args.momentum, 
                weight_decay=args.weight_decay
            )
        )

    # if no target is specified, unlearn for a set number of epochs
    if (args.target_unlearned_performance == 999):
        for epoch in range(epochs):
            LUV_unlearn(active_models, top_model, active_optimizers, top_model_optimizer, criterion,
                        LUV_trainloader, fewshot_descent_loader, alpha, args)
            model_a, model_b, t_model = deepcopy(active_models[0]), deepcopy(active_models[1]), deepcopy(top_model)
            c_active_models = [model_a, model_b]
            report(c_active_models, t_model, retain_loader, forget_loader, epoch, run, args)

    # unlearn at an increasing rate until target is met
    else:
        epoch = 0
        model_a, model_b, t_model = deepcopy(active_models[0]), deepcopy(active_models[1]), deepcopy(top_model)
        forgotten_accuracy = evaluate(model_a, model_b, t_model, LUV_trainloader, args)
        while (forgotten_accuracy > args.target_unlearned_performance):
            LUV_unlearn(active_models, top_model, active_optimizers, top_model_optimizer, criterion,
                        LUV_trainloader, fewshot_descent_loader, alpha, args)
            model_a, model_b, t_model = deepcopy(active_models[0]), deepcopy(active_models[1]), deepcopy(top_model)
            c_active_models = [model_a, model_b]
            report(c_active_models, t_model, retain_loader, forget_loader, epoch, run, args, alpha, 10)
            epoch += 1
            # speed up unlearning
            alpha *= args.alpha_momentum
            forgotten_accuracy = evaluate(model_a, model_b, t_model, LUV_trainloader, args)





def mixup_embeddings_unfederated(embeddings_a, targets, alpha=1.0):
    """Performs manifold mixup on a batch of unfederated embeddings."""

    mixed_embeddings_a = embeddings_a.clone()

    #print("mix embedding a : ", mixed_embeddings_a)


    for class_label in targets.unique():
        # Get indices of the current class
        idx = (targets == class_label).nonzero(as_tuple=True)[0]
        if len(idx) < 2:
            continue  # Cannot mix if only one sample of a class

         # Shuffle indices within the class
        perm = idx[torch.randperm(len(idx))]

        #Sample lambda from Beta distribution
        lam = torch.distributions.Beta(alpha, alpha).sample((len(idx),)).to(embeddings_a.device)
        lam = lam.view(-1, 1)
        

        # Perform mixup
        emb1_a = embeddings_a[idx]
        emb2_a = embeddings_a[perm]


        mixed_a = lam * emb1_a + (1 - lam) * emb2_a

        
        mixed_embeddings_a[idx] = mixed_a


    return mixed_embeddings_a


def mixup_embeddings(embeddings_a, embeddings_b, targets, alpha=1.0):
    """Performs manifold mixup on a batch of embeddings."""
    # if alpha > 0:
    #     lam = torch.distributions.Beta(alpha, alpha).sample().item()
    # else:
    #     lam = 1

    mixed_embeddings_a = embeddings_a.clone()
    mixed_embeddings_b = embeddings_b.clone()

    #print("mix embedding a : ", mixed_embeddings_a)


    for class_label in targets.unique():
        # Get indices of the current class
        idx = (targets == class_label).nonzero(as_tuple=True)[0]
        if len(idx) < 2:
            continue  # Cannot mix if only one sample of a class

         # Shuffle indices within the class
        perm = idx[torch.randperm(len(idx))]

        #Sample lambda from Beta distribution
        lam = torch.distributions.Beta(alpha, alpha).sample((len(idx),)).to(embeddings_a.device)
        lam = lam.view(-1, 1)
        

        # Perform mixup
        emb1_a = embeddings_a[idx]
        emb2_a = embeddings_a[perm]
        emb1_b = embeddings_b[idx]
        emb2_b = embeddings_b[perm]


        mixed_a = lam * emb1_a + (1 - lam) * emb2_a
        mixed_b = lam * emb1_b + (1 - lam) * emb2_b

        
        mixed_embeddings_a[idx] = mixed_a
        mixed_embeddings_b[idx] = mixed_b


    return mixed_embeddings_a, mixed_embeddings_b
