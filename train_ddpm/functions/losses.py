import torch
import numpy as np
# import yaml

# config = yaml.safe_load("/teamspace/studios/this_studio/Diffusion-based-Fluid-Super-resolution/train_ddpm/configs/km_re1000_rs256_conditional.yml")


def derivative_6th_torch(f, h, axis=0):
    """
    6th-order finite-difference derivative implemented in PyTorch.
    Supports input shapes:
      - (H, W)
      - (N, H, W)  where N is batch (or B*F after reshaping)
    axis: 0 -> derivative wrt rows (y), 1 -> derivative wrt cols (x)
    Returns tensor same shape as input.
    """
    # ensure tensor
    if not torch.is_tensor(f):
        f = torch.as_tensor(f)

    # unify to (N, H, W)
    squeeze_after = False
    if f.dim() == 2:
        f = f.unsqueeze(0)
        squeeze_after = True
    elif f.dim() == 3:
        pass
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {tuple(f.shape)}")

    N, H, W = f.shape
    device = f.device
    dtype = f.dtype

    df = torch.zeros_like(f, device=device, dtype=dtype)

    # if dimensions too small for 6th-order interior, return zeros (or handle forward/backward if possible)
    if axis == 0:
        # y-derivative (rows)
        if H >= 9:
            # central interior
            df[:, 3:-3, :] = (
                -f[:, 6:, :]
                + 9.0 * f[:, 5:-1, :]
                - 45.0 * f[:, 4:-2, :]
                + 45.0 * f[:, 2:-4, :]
                - 9.0 * f[:, 1:-5, :]
                + f[:, 0:-6, :]
            ) / (60.0 * h)

            # forward (first 3 rows)
            df[:, 0, :] = (-147.0*f[:,0,:] + 360.0*f[:,1,:] - 450.0*f[:,2,:]
                           + 400.0*f[:,3,:] - 225.0*f[:,4,:] + 72.0*f[:,5,:] - 10.0*f[:,6,:]) / (60.0*h)
            df[:, 1, :] = (-147.0*f[:,1,:] + 360.0*f[:,2,:] - 450.0*f[:,3,:]
                           + 400.0*f[:,4,:] - 225.0*f[:,5,:] + 72.0*f[:,6,:] - 10.0*f[:,7,:]) / (60.0*h)
            df[:, 2, :] = (-147.0*f[:,2,:] + 360.0*f[:,3,:] - 450.0*f[:,4,:]
                           + 400.0*f[:,5,:] - 225.0*f[:,6,:] + 72.0*f[:,7,:] - 10.0*f[:,8,:]) / (60.0*h)

            # backward (last 3 rows)
            df[:, -1, :] = (147.0*f[:, -1, :] - 360.0*f[:, -2, :] + 450.0*f[:, -3, :]
                            - 400.0*f[:, -4, :] + 225.0*f[:, -5, :] - 72.0*f[:, -6, :] + 10.0*f[:, -7, :]) / (60.0*h)
            df[:, -2, :] = (147.0*f[:, -2, :] - 360.0*f[:, -3, :] + 450.0*f[:, -4, :]
                            - 400.0*f[:, -5, :] + 225.0*f[:, -6, :] - 72.0*f[:, -7, :] + 10.0*f[:, -8, :]) / (60.0*h)
            df[:, -3, :] = (147.0*f[:, -3, :] - 360.0*f[:, -4, :] + 450.0*f[:, -5, :]
                            - 400.0*f[:, -6, :] + 225.0*f[:, -7, :] - 72.0*f[:, -8, :] + 10.0*f[:, -9, :]) / (60.0*h)
        else:
            # If H small, fall back to lower-order central differences (3-point) to avoid indexing errors
            # central difference (2nd order) as fallback
            if H >= 3:
                df[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2.0 * h)
                df[:, 0, :] = (f[:, 1, :] - f[:, 0, :]) / h
                df[:, -1, :] = (f[:, -1, :] - f[:, -2, :]) / h
            else:
                # too small, leave zeros
                pass

    elif axis == 1:
        # x-derivative (cols)
        if W >= 9:
            df[:, :, 3:-3] = (
                -f[:, :, 6:]
                + 9.0 * f[:, :, 5:-1]
                - 45.0 * f[:, :, 4:-2]
                + 45.0 * f[:, :, 2:-4]
                - 9.0 * f[:, :, 1:-5]
                + f[:, :, 0:-6]
            ) / (60.0 * h)

            df[:, :, 0] = (-147.0*f[:,:,0] + 360.0*f[:,:,1] - 450.0*f[:,:,2]
                           + 400.0*f[:,:,3] - 225.0*f[:,:,4] + 72.0*f[:,:,5] - 10.0*f[:,:,6]) / (60.0*h)
            df[:, :, 1] = (-147.0*f[:,:,1] + 360.0*f[:,:,2] - 450.0*f[:,:,3]
                           + 400.0*f[:,:,4] - 225.0*f[:,:,5] + 72.0*f[:,:,6] - 10.0*f[:,:,7]) / (60.0*h)
            df[:, :, 2] = (-147.0*f[:,:,2] + 360.0*f[:,:,3] - 450.0*f[:,:,4]
                           + 400.0*f[:,:,5] - 225.0*f[:,:,6] + 72.0*f[:,:,7] - 10.0*f[:,:,8]) / (60.0*h)

            df[:, :, -1] = (147.0*f[:,:,-1] - 360.0*f[:,:,-2] + 450.0*f[:,:,-3]
                            - 400.0*f[:,:,-4] + 225.0*f[:,:,-5] - 72.0*f[:,:,-6] + 10.0*f[:,:,-7]) / (60.0*h)
            df[:, :, -2] = (147.0*f[:,:,-2] - 360.0*f[:,:,-3] + 450.0*f[:,:,-4]
                            - 400.0*f[:,:,-5] + 225.0*f[:,:,-6] - 72.0*f[:,:,-7] + 10.0*f[:,:,-8]) / (60.0*h)
            df[:, :, -3] = (147.0*f[:,:,-3] - 360.0*f[:,:,-4] + 450.0*f[:,:,-5]
                            - 400.0*f[:,:,-6] + 225.0*f[:,:,-7] - 72.0*f[:,:,-8] + 10.0*f[:,:,-9]) / (60.0*h)
        else:
            # fallback 2nd-order central
            if W >= 3:
                df[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2.0 * h)
                df[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / h
                df[:, :, -1] = (f[:, :, -1] - f[:, :, -2]) / h
            else:
                pass
    else:
        raise ValueError("axis must be 0 or 1")

    if squeeze_after:
        return df.squeeze(0)
    return df


def second_deriv6_torch(f, h, axis):
    """
    second derivative by applying 6th-order operator twice.
    Input f: (H,W) or (N,H,W)
    """
    return derivative_6th_torch(derivative_6th_torch(f, h, axis), h, axis)


def time_derivative_fd_torch(w, w_next, dt):
    """
    Finite-difference time derivative. Supports shapes:
      - (H,W), (N,H,W) or (B,F,H,W) (if B*F flattened beforehand)
    """
    if not torch.is_tensor(w):
        w = torch.as_tensor(w)
    if not torch.is_tensor(w_next):
        w_next = torch.as_tensor(w_next)
    return (w_next - w) / dt


def pde_residual_loss(omega, omega_next, u, v, dx, dy, dt, Re):
    """
    Pure-PyTorch PDE residual loss.
    Inputs can be tensors on GPU. Supported shapes:
      - omega: (B, F, H, W) or (B, C, H, W) or (B, H, W) or (H, W)
      - omega_next: same shape as omega (next snapshot)
      - u, v: same shapes as omega
    Returns: (loss_tensor, R_tensor) where R_tensor has shape (N, H, W) and loss is scalar on same device.
    """

    # convert numpy -> tensor if needed
    if not torch.is_tensor(omega):
        omega = torch.as_tensor(omega)
    if not torch.is_tensor(omega_next):
        omega_next = torch.as_tensor(omega_next)
    if not torch.is_tensor(u):
        u = torch.as_tensor(u)
    if not torch.is_tensor(v):
        v = torch.as_tensor(v)

    device = omega.device
    dtype = omega.dtype

    # squeeze channel dim if present: (B,C,H,W) -> (B,H,W) picking first channel
    def squeeze_channel_t(x):
        if x.dim() == 4 and x.size(1) == 1:
            return x[:, 0, :, :]
        if x.dim() == 4 and x.size(1) > 1:
            # choose first channel; if you want mean across channels, change here
            return x[:, 0, :, :]
        return x

    omega = squeeze_channel_t(omega)
    omega_next = squeeze_channel_t(omega_next)
    u = squeeze_channel_t(u)
    v = squeeze_channel_t(v)

    # If input is (B,F,H,W) -> reshape to (B*F, H, W)
    def flatten_time(x):
        if x.dim() == 4:
            B, F, H, W = x.shape
            return x.reshape(B * F, H, W)
        return x

    omega = flatten_time(omega).to(device=device, dtype=dtype)
    omega_next = flatten_time(omega_next).to(device=device, dtype=dtype)
    u = flatten_time(u).to(device=device, dtype=dtype)
    v = flatten_time(v).to(device=device, dtype=dtype)

    # sanity
    if omega.shape != omega_next.shape:
        raise ValueError(f"omega and omega_next must have same shape after reshape; got {omega.shape} vs {omega_next.shape}")

    # Time derivative
    dw_dt = time_derivative_fd_torch(omega, omega_next, dt)

    # Spatial derivatives
    dw_dx = derivative_6th_torch(omega, dx, axis=1)
    dw_dy = derivative_6th_torch(omega, dy, axis=0)

    d2w_dx2 = second_deriv6_torch(omega, dx, axis=1)
    d2w_dy2 = second_deriv6_torch(omega, dy, axis=0)

    # convective
    conv = u * dw_dx + v * dw_dy

    visc = (d2w_dx2 + d2w_dy2) / Re

    R = dw_dt + conv - visc

    loss = torch.mean(R ** 2)

    return loss, R

def conditional_noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor,
                          x_scale,
                          x_offset,
                          u, v,
                          dx, dy, dt, Re,
                          w_base,
                          keepdim=False, p=0.1):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()

    flag = torch.rand(1).item()
    output = model(x, t.float())

    base_loss = (e - output).square().sum(dim=(1,2,3))
    if not keepdim:
        base_loss = base_loss.mean()
    if flag >= p:
        omega = (x * x_scale + x_offset).squeeze(1)
        omega_next = omega

        u = u.squeeze(1)
        v = v.squeeze(1)

        pde_loss, _ = pde_residual_loss(
            omega=omega,
            omega_next=omega_next,
            u=u, v=v,
            dx=dx, dy=dy, dt=dt, Re=Re
        )
        
        w_pde = 1 - w_base
        total_loss = w_base * base_loss + w_pde * pde_loss

        return total_loss, base_loss, pde_loss
    else:
        return base_loss, base_loss, base_loss


loss_registry = {
    'conditional': conditional_noise_estimation_loss
}
