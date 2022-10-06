# -*- coding: utf-8 -*-
"""losses.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10XqSVHkvHpf-XabH9bEXqDZsx-UxagV-
"""

import torch 
import numpy as np
import torch.nn as nn
import pdb
from lifelines.utils import concordance_index
from survci.utilities import lindisc, mmd2_lin, auc, auc_unconditional

np.random.seed(1234)
torch.manual_seed(seed=1234)

def imb_loss(model,x,w):
  ''' Returns the IPM (Integral Probability Metric) term '''
  imb = 0.0
  fi = model.get_repr(x.cuda())
  if model.imb_func=='lin_disc':
   imb = lindisc(fi, w, model.p_ipm)

  elif model.imb_func=='mmd2_lin':
    imb = mmd2_lin(fi, w, model.p_ipm)
  return imb

######################################################################################

def mse_loss(model,x,t,w):
  '''Returns MSE Loss '''
  shape_co,scale_co, logits_co , shape_tr,scale_tr, logits_tr = model.forward(x.cuda(),w.cuda())

  t_pred_co = auc(model,t[w==0].cuda(),shape_co,scale_co,logits_co)
  t_pred_tr = auc(model,t[w==1].cuda(), shape_tr, scale_tr,logits_tr)
  #pdb.set_trace()
  true_t = torch.cat((t[w==0],t[w==1]),0)
  pred_t = torch.cat((t_pred_co,t_pred_tr),0)
  # p_w = w.sum()/len(w) #Probability of Treatement=1
  # w_tr = w/(2*p_w) 
  # w_co = (1-w)/(2*(1-p_w))
  # wts = w_tr + w_co
  #return torch.mean(torch.multiply(wts,torch.square((true_t-pred_t))))
  return torch.mean(torch.square((true_t.cuda()-pred_t.cuda())))


def factual_loss(model, x,t,e,w):
  shape_co,scale_co, logits_co , shape_tr,scale_tr, logits_tr = model.forward(x.cuda(),w.cuda())
  t_pred_co = auc(model,t[w==0],shape_co,scale_co,logits_co)
  t_pred_tr = auc(model,t[w==1], shape_tr, scale_tr,logits_tr)
  relu = torch.nn.ReLU()
  true_t = torch.cat((t[w==0],t[w==1]),0)
  pred_t = torch.cat((t_pred_co,t_pred_tr),0)
  e = torch.cat((e[w==0],e[w==1]),0)
  return  torch.mean(e.cuda()*torch.abs(pred_t.cuda()-true_t.cuda()) + (1-e.cuda())*relu(pred_t.cuda()-true_t.cuda()))

def mse_total(model,x,t,e,w):
  shape_co,scale_co, logits_co , shape_tr,scale_tr, logits_tr = model.forward(x,w)
  #pdb.set_trace()
  t_pred_co = auc(model,t[w==0],shape_co,scale_co,logits_co)
  t_pred_tr = auc(model,t[w==1], shape_tr, scale_tr,logits_tr)
  true_t = torch.cat((t[w==0],t[w==1]),0)
  pred_t = torch.cat((t_pred_co,t_pred_tr),0)
  p_w_e = w[e==1].sum()/len(w[e==1]) #Probability of Treatement=1 for Uncensored
  w_tr_e = w[e==1]/(2*p_w_e) 
  w_co_e = (1-w[e==1])/(2*(1-p_w_e))
  wts_e = w_tr_e + w_co_e
  p_w_c = w[e==0].sum()/len(w[e==0]) #Probability of Treatement=1 for Censored
  w_tr_c = w[e==0]/(2*p_w_c) 
  w_co_c = (1-w[e==0])/(2*(1-p_w_c))
  wts_c = w_tr_c + w_co_c
  true_t_e = true_t[e==1]
  pred_t_e = pred_t[e==1]
  true_t_c = true_t[e==0]
  pred_t_c = pred_t[e==0]
  loss_uncen = torch.mean(torch.multiply(wts_e,torch.square((true_t_e-pred_t_e))))
  loss_cen = torch.mean(torch.multiply(wts_c,torch.square((true_t_c-pred_t_c))))
  return loss_uncen + loss_cen

######################################################################################

def _lognormal_loss(t, e, shape, scale):
  '''Returns Unconditional Log Normal Loss (Does not depends on features x)'''

  eta_ = shape.expand(t.shape[0], -1) 
  beta_ = scale.expand(t.shape[0], -1)

  ll = 0.

  mu = eta_
  sigma = beta_

  f = - torch.log(t)- sigma - 0.5*np.log(2*np.pi)
  f = f - torch.div((torch.log(t) - mu)**2, 2.*torch.exp(2*sigma))
  s = torch.div(torch.log(t) - mu, torch.exp(sigma)*np.sqrt(2))
  s = 0.5 - 0.5*torch.erf(s)
  s = torch.log(s)

  uncens = np.where(e.cpu().data.numpy() == 1)[0]
  cens = np.where(e.cpu().data.numpy() != 1)[0]
  ll += f[uncens].sum() + s[cens].sum()

  return -ll.mean()


def _weibull_loss(t, e, shape, scale):
  '''Returns Unconditional Weibull Loss (Does not depends on features x)'''

  eta_ = shape.expand(t.shape[0], -1)
  beta_ = scale.expand(t.shape[0], -1)

  ll = 0.


  s = - (torch.pow(torch.exp(beta_)*t, torch.exp(eta_)))
  f = eta_ + beta_ + ((torch.exp(eta_)-1)*(beta_+torch.log(t)))
  f = f + s

  uncens = np.where(e.cpu().data.numpy() == 1)[0]
  cens = np.where(e.cpu().data.numpy() != 1)[0]
  ll += f[uncens].sum() + s[cens].sum()

  return -ll.mean()

def unconditional_mse(model,t,e,w):
  shape_co,scale_co,shape_tr,scale_tr = model.get_shape_scale(w)
  t_pred_co = auc_unconditional(model,t[w==0],shape_co,scale_co)
  t_pred_tr = auc_unconditional(model,t[w==1], shape_tr, scale_tr)
  true_t = torch.cat((t[w==0],t[w==1]),0)
  pred_t = torch.cat((t_pred_co,t_pred_tr),0)
  return torch.mean(torch.square((true_t-pred_t)))
  

def unconditional_loss(model, t, e, w):
  '''Returns Unconditional  Loss (Does not depends on features x)'''

  shape_co,scale_co,shape_tr,scale_tr = model.get_shape_scale(w)

  tot_unc_loss = 0.0
  if model.dist == 'Weibull':
    unco_loss_co = _weibull_loss(t[w==0], e[w==0], shape_co,scale_co)
    unco_loss_tr = _weibull_loss(t[w==1], e[w==1], shape_tr,scale_tr)

  elif model.dist == 'LogNormal':
    unco_loss_co = _lognormal_loss(t[w==0], e[w==0], shape_co,scale_co)
    unco_loss_tr = _lognormal_loss(t[w==1], e[w==1], shape_tr,scale_tr)

  else:
    raise NotImplementedError('Distribution: '+model.dist+
                              ' not implemented yet.')
  tot_unc_loss = unco_loss_co + unco_loss_tr

  return tot_unc_loss
 
######################################################################################
    

def _conditional_lognormal_loss(t, e, shape,scale, logits,alpha, elbo=True):
  '''Returns Conditional  Loss (Depends on features x)'''

  lossf = []
  losss = []
  k = shape.shape[1] #Number of Primitive Distributions 
  eta_ = shape
  beta_ = scale

  for g in range(k):

    mu = eta_[:, g]
    sigma = beta_[:, g]

    f = - torch.log(t) - sigma - 0.5*np.log(2*np.pi)
    f = f - torch.div((torch.log(t) - mu)**2, 2.*torch.exp(2*sigma))
    s = torch.div(torch.log(t) - mu, torch.exp(sigma)*np.sqrt(2))
    s = 0.5 - 0.5*torch.erf(s)
    s = torch.log(s)

    lossf.append(f)
    losss.append(s)

  losss = torch.stack(losss, dim=1)
  lossf = torch.stack(lossf, dim=1)

  if elbo:

    lossg = nn.Softmax(dim=1)(logits)
    losss = lossg*losss
    lossf = lossg*lossf

    losss = losss.sum(dim=1)
    lossf = lossf.sum(dim=1)

  else:

    lossg = nn.LogSoftmax(dim=1)(logits)
    losss = lossg + losss
    lossf = lossg + lossf

    losss = torch.logsumexp(losss, dim=1)
    lossf = torch.logsumexp(lossf, dim=1)

  uncens = np.where(e.cpu().data.numpy() == 1)[0]
  cens = np.where(e.cpu().data.numpy() != 1)[0]
  ll = lossf[uncens].sum() + alpha*losss[cens].sum()

  return -ll/float(len(uncens)+len(cens))


def _conditional_weibull_loss(t, e, shape, scale, logits, alpha, elbo=True):
  '''Returns Conditional  Loss (Depends on features x)'''

  k = shape.shape[1] #Number of Primitive Distributions 
  eta_ = shape
  beta_ = scale

  lossf = []
  losss = []

  for g in range(k):

    shape_ = eta_[:, g]
    scale_ = beta_[:, g]

    s = - (torch.pow(torch.exp(scale_)*t, torch.exp(shape_)))
    f = shape_ + scale_ + ((torch.exp(scale_)-1)*(scale_+torch.log(t)))
    f = f + s

    lossf.append(f)
    losss.append(s)

  losss = torch.stack(losss, dim=1)
  lossf = torch.stack(lossf, dim=1)

  if elbo:

    lossg = nn.Softmax(dim=1)(logits)
    losss = lossg*losss
    lossf = lossg*lossf
    losss = losss.sum(dim=1)
    lossf = lossf.sum(dim=1)

  else:

    lossg = nn.LogSoftmax(dim=1)(logits)
    losss = lossg + losss
    lossf = lossg + lossf
    losss = torch.logsumexp(losss, dim=1)
    lossf = torch.logsumexp(lossf, dim=1)

  uncens = np.where(e.cpu().data.numpy() == 1)[0]
  cens = np.where(e.cpu().data.numpy() != 1)[0]
  ll = lossf[uncens].sum() + alpha*losss[cens].sum()

  return -ll/float(len(uncens)+len(cens))


def conditional_loss(model, x, t, e, w, elbo=True):

  alpha = model.discount
  # pdb.set_trace()
  shape_co, scale_co, logits_co, shape_tr, scale_tr, logits_tr = model.forward(x.cuda(), w.cuda())
  if model.dist == 'Weibull':
    cond_loss_co = _conditional_weibull_loss(t[w==0].cuda(), e[w==0].cuda(), shape_co, scale_co, logits_co,alpha,elbo)
    cond_loss_tr = _conditional_weibull_loss(t[w==1].cuda(), e[w==1].cuda(),shape_tr,scale_tr,logits_tr, alpha,elbo)
  elif model.dist == 'LogNormal':
    cond_loss_co =  _conditional_lognormal_loss(t[w==0].cuda(), e[w==0].cuda(),shape_co,scale_co,logits_co,alpha, elbo)
    cond_loss_tr = _conditional_lognormal_loss(t[w==1].cuda(), e[w==1].cuda(),shape_tr,scale_tr,logits_tr, alpha,elbo)
  else:
    raise NotImplementedError('Distribution: '+model.dist+
                              ' not implemented yet.')
    
  tot_cond_loss = cond_loss_co + cond_loss_tr
  return tot_cond_loss



def l2_loss(model):
  layers = model.get_layers()
  #n_layers = model.get_layers_dim()
  # pdb.set_trace()
  #reg_loss = sum([torch.square(layers[i].weight).sum()/2 for i in range(len(layers))])
  reg_loss = sum([torch.square(layers[i].weight).sum()/2 for i in range(0,len(layers)) if i%2==0])
  return reg_loss

######################################################################################

# def calculate_ci(model,x,t,e,w):
    
#   treated_idx = torch.where(w>0)[0]                        
#   control_idx = torch.where(w<1)[0]
#   shape_co,scale_co,logits_co, shape_tr,scale_tr, logits_tr = model.forward(x,w) #Predicted Factual Parameters 
#   shape_co ,scale_co = softmax_out(shape_co, scale_co, logits_co)
#   shape_tr, scale_tr = softmax_out(shape_tr,scale_tr,logits_tr)
#   shape_co = shape_co.detach().numpy()
#   scale_co  = scale_co.detach().numpy()
#   shape_tr = shape_tr.detach().numpy()
#   scale_tr = scale_tr.detach().numpy()
#   if model.dist == 'LogNormal':
#     t_co_samples = sample_lognormal(mu=shape_co, sigma=np.exp(scale_co))
#     t_tr_samples = sample_lognormal(mu=shape_tr, sigma=np.exp(scale_tr))
#   elif model.dist == 'Weibull':
#     t_co_samples = sample_weibull(shape=shape_co, scale=np.exp(scale_co))
#     t_tr_samples = sample_weibull(shape=shape_tr, scale=np.exp(scale_tr))
#   else:
#     print('Sampling Distribution function not defined')
#     t_pred_co = np.median(t_co_samples,axis=1)
#     t_pred_tr = np.median(t_tr_samples,axis=1)
#     c_index_co = concordance_index(event_times=t[control_idx],predicted_scores=t_pred_co,event_observed=e[control_idx])
#     c_index_tr = concordance_index(event_times=t[treated_idx],predicted_scores=t_pred_tr,event_observed=e[treated_idx])
#   return ((c_index_co + c_index_tr) * 0.5)