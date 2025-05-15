import torch


def loss_fn(est_ri, sigma_z, tgt_ri, z, sigma):
    loss = loss_fn_g(sigma_z, z, sigma) + loss_fn_p(est_ri, tgt_ri)
    return loss


def loss_fn_g(pred, target, sigma):
    # pred: sigma * z, target: z
    loss = torch.square(pred / sigma - target)
    loss = torch.mean(loss)
    return loss


def loss_fn_p(pred, target):
    # pred: (b,2,t,f)
    complex_loss = torch.mean(torch.square(pred - target))
    pred_mag = torch.sqrt(pred[:,0]**2 + pred[:,1]**2 + 1e-8)
    target_mag = torch.sqrt(target[:,0]**2 + target[:,1]**2 + 1e-8)
    mag_loss = torch.mean(torch.square(pred_mag - target_mag))
    return 0.5 * complex_loss + 0.5 * mag_loss
