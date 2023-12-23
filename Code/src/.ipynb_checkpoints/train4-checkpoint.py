# +
import torch
from transformers import get_constant_schedule_with_warmup
import numpy as np
from tqdm import tqdm
from math import ceil
from .evaluation_metrics import print_topl_statistics
from .losses import binary_crossentropy_2d
import pandas as pd

def one_hot_decode(batch_tensor):
    # Define the bases
    bases = ['A', 'C', 'G', 'T', 'N']

    # Get the index of the maximum value along the bases dimension
    _, indices = batch_tensor.max(dim=1)

    # Map indices to bases
    decoded_sequences = [[bases[i] for i in sequence] for sequence in indices.T]

    # Join the bases to form strings
    decoded_strings = [''.join(sequence) for sequence in decoded_sequences]

    return decoded_strings

def trainModel(model,fileName,criterion,train_loader,val_loader,optimizer,scheduler,warmup,BATCH_SIZE,epochs,device,tokenizer,verbose=1,CL_max=40000,lowValidationGPUMem=False,skipValidation=False,NUM_ACCUMULATION_STEPS=1,reinforce=True,continous_labels=False,no_softmax=False):
    losses = {}
    losses['train'] = []
    losses['val'] = []
    val_results = []
    dataLoaders = {}
    dataLoaders['train'] = train_loader
    dataLoaders['val'] = val_loader
    multiplier = 0.01
    eps = torch.finfo(torch.float32).eps
    acceptor_acc_avg = 0
    donor_acc_avg = 0
    
    for epoch in range(epochs):
        for phase in ['train','val']:
            if skipValidation and (phase=='val'):
                continue
            loop =tqdm(dataLoaders[phase])
            if phase=='train':
                model.train(True)
            else:
                model.train(False)
            loss = 0
            ema_loss = 0
            ema_l1 = 0
            ema_a_recall = 0
            ema_d_recall = 0
            n_steps_completed = 0
            outputs_list = []
            targets_list = []

            Y_true_acceptor,Y_true_donor,Y_pred_acceptor,Y_pred_donor=[],[],[],[]
            n_accum = 0
            for i,(batch_features, targets) in enumerate(loop):
                #batch_features = batch_features.type(torch.FloatTensor).to(device)
                batch_features = one_hot_decode(batch_features)
                batch_features = torch.LongTensor(tokenizer(batch_features)["input_ids"]).to(device).T
                print(batch_features.shape)
                targets_full = targets.to(device)
                targets = targets_full[:,:,CL_max//2:-CL_max//2]
                if (i % NUM_ACCUMULATION_STEPS == 0):
                    optimizer.zero_grad()
                outputs = model(batch_features)
                
                if type(outputs) is tuple:
                    if reinforce:
                        acceptor_actions = outputs[1]
                        donor_actions = outputs[2]
                        acceptor_log_probs = outputs[3]
                        donor_log_probs = outputs[4]
                    outputs = outputs[0]
                
                #print(l1loss)
                print(outputs.shape)
                print(targets.shape)
                train_loss = criterion(outputs,targets)/ NUM_ACCUMULATION_STEPS
                
                if no_softmax:
                    outputs = torch.nn.Softmax(dim=1)(outputs)

                if reinforce:
                    acceptor_reward = torch.gather(targets_full[:,1,:]-targets_full[:,2,:],1,acceptor_actions)
                    donor_reward = torch.gather(targets_full[:,2,:]-targets_full[:,1,:],1,donor_actions)
                    if phase == 'train':
                        acceptor_acc =  torch.nanmean(torch.sum(acceptor_reward>0,dim=1)/torch.sum(targets_full[:,1,:]>0,dim=1))
                        acceptor_acc_avg = acceptor_acc*multiplier + acceptor_acc_avg*(1-multiplier)
                        donor_acc =  torch.nanmean(torch.sum(donor_reward>0,dim=1)/torch.sum(targets_full[:,2,:]>0,dim=1))
                        donor_acc_avg = donor_acc*multiplier + donor_acc_avg*(1-multiplier)

                    acceptor_loss = -torch.mean(torch.sum(acceptor_log_probs * acceptor_reward,dim=1))/ NUM_ACCUMULATION_STEPS
                    donor_loss = -torch.mean(torch.sum(donor_log_probs * donor_reward,dim=1))/ NUM_ACCUMULATION_STEPS
                    reinforce_loss = acceptor_loss+donor_loss
                    train_loss = train_loss + 1e-6*reinforce_loss
                
                if phase == 'train':
                    train_loss.backward()
                    n_accum = n_accum+1
                    if (n_accum == NUM_ACCUMULATION_STEPS) or (i + 1 == len(loop)):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2.0)
                        optimizer.step()
                        n_accum = 0

                        if epoch<5:
                            warmup.step()

                with torch.no_grad():
                    out_argmax = torch.flatten(torch.argmax(outputs,dim=1))
                    target_argmax = torch.flatten(torch.argmax(targets,dim=1))
                    a_recall = torch.nanmean((out_argmax[target_argmax==1]==1).type(torch.float32))
                    d_recall = torch.nanmean((out_argmax[target_argmax==2]==2).type(torch.float32))

                if phase == 'val':
                    if lowValidationGPUMem:
                        outputs = outputs.detach().cpu().numpy()
                        targets_list.extend(np.expand_dims(targets.cpu().numpy(),0))
                        outputs_list.extend(np.expand_dims(outputs,0))
                    else:
                        outputs = outputs.detach()
                        outputs_list.extend(outputs.unsqueeze(0))
                
                

                loss = NUM_ACCUMULATION_STEPS*train_loss.item()+loss
                loop.set_description('Epoch ({}) {}/{}'.format(phase,epoch + 1, epochs))
                n_steps_completed += 1
                if i==0:
                    ema_loss = NUM_ACCUMULATION_STEPS*train_loss.item()
                    if ~a_recall.isnan():
                        ema_a_recall = a_recall.cpu().numpy()
                    if ~d_recall.isnan():
                        ema_d_recall = a_recall.cpu().numpy()
                else:
                    ema_loss = NUM_ACCUMULATION_STEPS*train_loss.item()*multiplier + ema_loss*(1-multiplier)

                    if ~a_recall.isnan():
                        ema_a_recall = a_recall.cpu().numpy()*multiplier + ema_a_recall*(1-multiplier)
                    if ~d_recall.isnan():
                        ema_d_recall= d_recall.cpu().numpy()*multiplier + ema_d_recall*(1-multiplier)
                if reinforce:
                    if i==0:
                        ema_reinforce_loss = NUM_ACCUMULATION_STEPS*reinforce_loss.item()
                    else:
                        ema_reinforce_loss = NUM_ACCUMULATION_STEPS*reinforce_loss.item()*multiplier + ema_reinforce_loss*(1-multiplier)
                    loop.set_postfix(loss=ema_loss, r_a=acceptor_acc_avg.item(),r_d=donor_acc_avg.item(),r_loss=ema_reinforce_loss,a_r = ema_a_recall , d_r=ema_d_recall)
                else:
                    loop.set_postfix(loss=ema_loss,a_r = ema_a_recall , d_r=ema_d_recall)
            if phase == 'val':
                if lowValidationGPUMem:
                    targets = np.swapaxes(np.vstack(targets_list),1,2)
                    outputs = np.swapaxes(np.vstack(outputs_list),1,2)
                else:
                    targets = torch.transpose(torch.vstack(targets_list),1,2).cpu().numpy()
                    outputs = torch.transpose(torch.vstack(outputs_list),1,2).cpu().numpy()

                is_expr = (targets.sum(axis=(1,2)) >= 1)
                Y_true_acceptor.extend(targets[is_expr, :, 1].flatten())
                Y_true_donor.extend(targets[is_expr, :, 2].flatten())
                Y_pred_acceptor.extend(outputs[is_expr, :, 1].flatten())
                Y_pred_donor.extend(outputs[is_expr, :, 2].flatten())

            loss = loss / (n_steps_completed)
            losses[phase].append(loss)
            
            if phase == 'val':
                Y_true_acceptor, Y_pred_acceptor,Y_true_donor, Y_pred_donor = np.array(Y_true_acceptor), np.array(Y_pred_acceptor),np.array(Y_true_donor), np.array(Y_pred_donor)
                if continous_labels:
                    Y_true_acceptor = np.round_(Y_true_acceptor,decimals=0)
                    Y_true_donor = np.round_(Y_true_donor,decimals=0)
                print("\n\033[1m{}:\033[0m".format('Acceptor'))
                acceptor_val_results = print_topl_statistics(Y_true_acceptor, Y_pred_acceptor)
                print("\n\033[1m{}:\033[0m".format('Donor'))
                donor_val_results =print_topl_statistics(Y_true_donor, Y_pred_donor)
                val_results.append([acceptor_val_results,donor_val_results])
                
            
            if verbose == 1:
                print("epoch: {}/{}, {} loss = {:.6f}".format(epoch + 1, epochs, phase, loss))
            if phase == 'val' or skipValidation:
                torch.save(model.state_dict(), fileName)
        if epoch>=5:
            scheduler.step()
        
    if skipValidation:
        return pd.DataFrame({'loss':losses['train']})
    else:
        return pd.DataFrame({'loss':losses['train'],'val_loss':losses['val']})
# -


