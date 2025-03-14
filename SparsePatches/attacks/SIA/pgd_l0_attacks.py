#import tensorflow as tf
#import scipy.io
import numpy as np
import torch

from attacks.SIA.utils import get_predictions, get_predictions_and_gradients

def project_L0_box(y, k, lb, ub):
  ''' projection of the batch y to a batch x such that:
        - each image of the batch x has at most k pixels with non-zero channels
        - lb <= x <= ub '''
      
  x = np.copy(y)
  p1 = np.sum(x**2, axis=-1)
  p2 = np.minimum(np.minimum(ub - x, x - lb), 0)
  p2 = np.sum(p2**2, axis=-1)
  p3 = np.sort(np.reshape(p1-p2, [p2.shape[0],-1]))[:,-k]
  x = x*(np.logical_and(lb <=x, x <= ub)) + lb*(lb > x) + ub*(x > ub)
  x *= np.expand_dims((p1 - p2) >= p3.reshape([-1, 1, 1]), -1)
    
  return x
  
def project_L0_sigma(y, k, sigma, kappa, x_nat):
  ''' projection of the batch y to a batch x such that:
        - 0 <= x <= 1
        - each image of the batch x differs from the corresponding one of
          x_nat in at most k pixels
        - (1 - kappa*sigma)*x_nat <= x <= (1 + kappa*sigma)*x_nat '''
  
  x = np.copy(y)
  p1 = 1.0/np.maximum(1e-12, sigma)*(x_nat > 0).astype(float) + 1e12*(x_nat == 0).astype(float)
  p2 = 1.0/np.maximum(1e-12, sigma)*(1.0/np.maximum(1e-12, x_nat) - 1)*(x_nat > 0).astype(float) + 1e12*(x_nat == 0).astype(float) + 1e12*(sigma == 0).astype(float)
  lmbd_l = np.maximum(-kappa, np.amax(-p1, axis=-1, keepdims=True))
  lmbd_u = np.minimum(kappa, np.amin(p2, axis=-1, keepdims=True))
  
  lmbd_unconstr = np.sum((y - x_nat)*sigma*x_nat, axis=-1, keepdims=True)/np.maximum(1e-12, np.sum((sigma*x_nat)**2, axis=-1, keepdims=True))
  lmbd = np.maximum(lmbd_l, np.minimum(lmbd_unconstr, lmbd_u))
  
  p12 = np.sum((y - x_nat)**2, axis=-1, keepdims=True)
  p22 = np.sum((y - (1 + lmbd*sigma)*x_nat)**2, axis=-1, keepdims=True)
  p3 = np.sort(np.reshape(p12 - p22, [x.shape[0],-1]))[:,-k]
  
  x  = x_nat + lmbd*sigma*x_nat*((p12 - p22) >= p3.reshape([-1, 1, 1, 1]))
  
  return x
  
def perturb_L0_box(attack, x_nat, y_nat, lb, ub):
  ''' PGD attack wrt L0-norm + box constraints
  
      it returns adversarial examples (if found) adv for the images x_nat, with correct labels y_nat,
      such that:
        - each image of the batch adv differs from the corresponding one of
          x_nat in at most k pixels
        - lb <= adv - x_nat <= ub
      
      it returns also a vector of flags where 1 means no adversarial example found
      (in this case the original image is returned in adv) '''
  
  if attack.rs:
    pert = np.random.uniform(lb, ub, x_nat.shape)
    x2 = x_nat + pert
    x2 = np.clip(x2, attack.data_RGB_start, attack.data_RGB_end)
    pert = x2 - x_nat
  else:
    pert = np.zeros_like(x_nat)

  adv_not_found = np.ones(y_nat.shape)
  adv_pert = np.zeros(x_nat.shape)

  for i in range(attack.n_iter):
    if i > 0:
      #pred, grad = sess.run([attack.model.correct_prediction, attack.model.grad], feed_dict={attack.model.x_input: x2, attack.model.y_input: y_nat})
      x2 = x_nat + pert
      pred, grad = get_predictions_and_gradients(attack.model, x2, y_nat)
      adv_not_found = np.minimum(adv_not_found, pred.astype(int))
      adv_pert[np.logical_not(pred)] = np.copy(pert[np.logical_not(pred)])
      
      grad /= (1e-10 + np.sum(np.abs(grad), axis=(1,2,3), keepdims=True))
      pert = np.add(pert, (np.random.random_sample(grad.shape)-0.5)*1e-12 + attack.step_size * grad, casting='unsafe')
      
    pert = project_L0_box(pert, attack.k, lb, ub)
    
  return adv_pert, adv_not_found
  
def perturb_L0_sigma(attack, x_nat, y_nat):
  ''' PGD attack wrt L0-norm + sigma-map constraints
  
      it returns adversarial examples (if found) adv for the images x_nat, with correct labels y_nat,
      such that:
        - each image of the batch adv differs from the corresponding one of
          x_nat in at most k pixels
        - (1 - kappa*sigma)*x_nat <= adv <= (1 + kappa*sigma)*x_nat
      
      it returns also a vector of flags where 1 means no adversarial example found
      (in this case the original image is returned in adv) '''
  
  if attack.rs:
    x2 = x_nat + np.random.uniform(-attack.kappa, attack.kappa, x_nat.shape)
    x2 = np.clip(x2, attack.data_RGB_start, attack.data_RGB_end)
  else:
    x2 = np.copy(x_nat)
      
  adv_not_found = np.ones(y_nat.shape)
  adv = np.zeros(x_nat.shape)

  for i in range(attack.n_iter):
    if i > 0:
      #pred, grad = sess.run([attack.model.correct_prediction, attack.model.grad], feed_dict={attack.model.x_input: x2, attack.model.y_input: y_nat})
      pred, grad = get_predictions_and_gradients(attack.model, x2, y_nat)
      adv_not_found = np.minimum(adv_not_found, pred.astype(int))
      adv[np.logical_not(pred)] = np.copy(x2[np.logical_not(pred)])
      
      grad /= (1e-10 + np.sum(np.abs(grad), axis=(1,2,3), keepdims=True))
      x2 = np.add(x2, (np.random.random_sample(grad.shape)-0.5)*1e-12 + attack.step_size * grad, casting='unsafe')
      
    x2 = project_L0_sigma(x2, attack.k, attack.sigma, attack.kappa, x_nat)
    
  return adv, adv_not_found
  
def sigma_map(x):
  ''' creates the sigma-map for the batch x '''
  
  sh = [4]
  sh.extend(x.shape)
  t = np.zeros(sh)
  t[0,:,:-1] = x[:,1:]
  t[0,:,-1] = x[:,-1]
  t[1,:,1:] = x[:,:-1]
  t[1,:,0] = x[:,0]
  t[2,:,:,:-1] = x[:,:,1:]
  t[2,:,:,-1] = x[:,:,-1]
  t[3,:,:,1:] = x[:,:,:-1]
  t[3,:,:,0] = x[:,:,0]

  mean1 = (t[0] + x + t[1])/3
  sd1 = np.sqrt(((t[0]-mean1)**2 + (x-mean1)**2 + (t[1]-mean1)**2)/3)

  mean2 = (t[2] + x + t[3])/3
  sd2 = np.sqrt(((t[2]-mean2)**2 + (x-mean2)**2 + (t[3]-mean2)**2)/3)

  sd = np.minimum(sd1, sd2)
  sd = np.sqrt(sd)
  
  return sd

class PGDL0attack():
  def __init__(self, model, args):
    self.batch_size = args['batch_size']
    self.data_shape = [self.batch_size] + args['data_shape']
    self.data_channels = self.data_shape[1]
    self.data_w = self.data_shape[2]
    self.data_h = self.data_shape[3]
    self.n_data_pixels = self.data_w * self.data_h
    self.shape_img = [self.data_w, self.data_h, self.data_channels]
    self.data_RGB_start = 0
    self.data_RGB_end = 1
    self.data_RGB_size = 1

    self.model = model
    self.type_attack = args['type_attack'] # 'L0', 'L0+Linf', 'L0+sigma'
    self.n_iter = args['n_iter']           # number of iterations of gradient descent for each restart
    self.step_size = args['step_size']     # step size for gradient descent (\eta in the paper)
    self.n_restarts = args['n_restarts']   # number of random restarts to perform
    self.rs = True                         # random starting point
    self.eps_ratio = args['epsilon']         # for L0+Linf, the bound on the Linf-norm of the perturbation
    self.kappa = args['kappa']             # for L0+sigma (see kappa in the paper), larger kappa means easier and more visible pgd_attacks
    self.k = args['sparsity']              # maximum number of pixels that can be modified (k_max in the paper)
    self.verbose = args['verbose']         # verbose
    self.step_size *= self.data_RGB_size
    self.epsilon = self.eps_ratio * self.data_RGB_size

    self.output_l0_norms = [self.k]
    self.report_info = False
    self.name = "PGD_L0"

  def report_schematics(self):
      print("Running PGD_L0 attack based on the paper: \" Sparse and Imperceivable Adversarial Attacks \" ")
      print("Attack L0 norm limitation:")
      print(self.k)
      print("Attack L_inf norm limitation:")
      print(self.eps_ratio)
      print("Number of iterations for perturbation optimization:")
      print(self.n_iter)
      print("Number of restarts for perturbation optimization:")
      print(self.n_restarts)

  def perturb(self, x_tensor, y_tensor, targeted=False):
    x_nat = x_tensor.permute(0, 2, 3, 1).cpu().numpy()
    y_nat = y_tensor.cpu().numpy()
    adv_pert = np.zeros_like(x_nat)
    
    if self.type_attack == 'L0+sigma': self.sigma = sigma_map(x_nat)
      
    for counter in range(self.n_restarts):
      if counter == 0:
        #corr_pred = sess.run(self.model.correct_prediction, {self.model.x_input: x_nat, self.model.y_input: y_nat})
        corr_pred = get_predictions(self.model, x_nat, y_nat)
        pgd_adv_acc = np.copy(corr_pred)
        box_start = self.data_RGB_start - x_nat
        box_end = self.data_RGB_end - x_nat

      if self.type_attack == 'L0':
        x_batch_adv_pert, curr_pgd_adv_acc = perturb_L0_box(self, x_nat, y_nat, box_start, box_end)

      elif self.type_attack == 'L0+Linf':
        x_batch_adv_pert, curr_pgd_adv_acc = perturb_L0_box(self, x_nat, y_nat, np.maximum(-self.epsilon, box_start), np.minimum(self.epsilon, box_end))

      elif self.type_attack == 'L0+sigma' and x_nat.shape[3] == 3:
        x_batch_adv_pert, curr_pgd_adv_acc = perturb_L0_sigma(self, x_nat, y_nat)

      elif self.type_attack == 'L0+sigma' and x_nat.shape[3] == 1:
        x_batch_adv_pert, curr_pgd_adv_acc = perturb_L0_box(self, x_nat, y_nat, np.maximum(-self.kappa*self.sigma, box_start), np.minimum(self.kappa*self.sigma, box_end))

      pgd_adv_acc = np.minimum(pgd_adv_acc, curr_pgd_adv_acc)
      if self.verbose:
        print("Restart {} - Robust accuracy: {}".format(counter + 1, np.sum(pgd_adv_acc)/x_nat.shape[0]))
      adv_pert[np.logical_not(curr_pgd_adv_acc)] = x_batch_adv_pert[np.logical_not(curr_pgd_adv_acc)]
    

    if self.verbose:
      pixels_changed = np.sum(np.amax(np.abs(adv_pert) > 1e-10, axis=-1), axis=(1, 2))
      print('Pixels changed: ', pixels_changed)
      #corr_pred = sess.run(self.model.correct_prediction, {self.model.x_input: adv, self.model.y_input: y_nat})
      print('Maximum perturbation size: {:.5f}'.format(np.amax(np.abs(adv_pert))))
      corr_pred = get_predictions(self.model, x_nat + adv_pert, y_nat)
      print('Robust accuracy at {} pixels: {:.2f}%'.format(self.k, np.sum(corr_pred)/x_nat.shape[0]*100.0))

    l0_adv_pert = torch.tensor(adv_pert.transpose(0, 3, 1, 2)).reshape_as(x_tensor).to(x_tensor).unsqueeze(0)

    return l0_adv_pert, None, None