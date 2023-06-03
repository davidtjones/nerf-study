from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from tqdm import tqdm

from util.sampling import sample_stratified, sample_hierarchical


class Chunker:
    def __init__(self, batch_size, points_encoder, viewdirs_encoder):
        self.batch_size = batch_size
        self.points_encoder = points_encoder
        self.viewdirs_encoder = viewdirs_encoder

    def __call__(self, points, viewdirs):
    
        # There is only have 1 viewdir for each ray, but there are `sample_size`
        # points for each ray. We can expand the viewdirs s.t. the viewdirs
        # is repeated on a new axis for each point.

        viewdirs = viewdirs[:, None, ...].expand(points.shape).reshape((-1, 3))
        points = points.reshape((-1, 3))

        # Encode:
        points = self.points_encoder(points)
        viewdirs = self.viewdirs_encoder(viewdirs)

        # return a generator that returns the next batch
        for i in range(0, len(points), self.batch_size):
            pts_batch = points[i:i+self.batch_size]
            viewdirs_batch = viewdirs[i:i+self.batch_size]
            yield pts_batch, viewdirs_batch


class PositionalEncoder(nn.Module):
    """
    Sine-cosine positional encoder for input points. Specify an input size 
    `d_input` and the number of frequency ranges `n_freqs`. PositionalEncoder
    generates a list of encoding functions to be applied to the input. 

    `log_space` determines the spacing (linear or logarithmic) between
        frequencies. Setting this to true gives a stronger distinction for the
        model between at upper ranges, improving performance on high-frequency 
        features, overcoming model biases for low-frequency features.

    E.g., using log space and n_freqs=5, our frequencies will be 
    [1, 2, 4, 8, 16] vs [1, ~4, ~7, ~10, ~13, 16]. 
    
    The list of frequencies is applied to the input. For n_freq=N, the input x
    will be expanded from [x] to [x, sin(x*f_0), cos(x*f_0), ..., cos(x*f_N)]
    which is 2N+1 in size. Multiply this by the input dimension to get the final
    shape. For 3-channel input (e.g., ray origins or ray directions), expect
    6N + 3. `d_output` denotes this value.
    
    This is a lot like basis expansion, but we are intentionally using
    high-frequency encodings of the data to captgure complex spatial variations
    in the scene, while still being relatively efficeint to compute and be 
    trainable via gradient descent. This is important since we want to capture
    3D scenes with potentially intricate structures and lighting conditions.
    

    """
    def __init__(self, d_input: int, n_freqs: int, log_space: bool = False):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x: x]

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(self, x) -> torch.Tensor:
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)


class OriginalNeRF(pl.LightningModule):
    
    def __init__(self, **kwargs):
        super().__init__()

        # Misc
        self.pbkw = lambda x: {
            "leave": False, 
            "disable": not kwargs.get("progress_bars_enabled", False), 
            "position": x}

        # Chunking
        self.ray_chunk_size = kwargs.get("ray_chunk_size", 3500)
        self.pts_chunk_size = kwargs.get("pts_chunk_size", 1000)

        # Sampling
        # TODO: fix naming discrepancy: 
        self.stratified_sampling_sample_size = kwargs.get("n_stratified_sampling_size", 128)
        self.n_samples_hierarchical = kwargs.get("n_samples_hierarchical", 196)
        self.near = kwargs.get("near", 2.)
        self.far = kwargs.get("far", 6.)

        # Model Settings
        self.d_input = kwargs.get('d_input', 3)
        self.n_freqs = kwargs.get('n_freqs', 10)
        self.use_log_space = kwargs.get("use_log_space", True)
        
        self.use_viewdirs = kwargs.get("use_viewdirs", True)
        self.n_freqs_views = kwargs.get("n_freqs_views", 16)
        self.d_viewdirs = None

        self.lr = kwargs.get("lr", 1e-4)
    
        self.n_layers = 8
        self.d_filter = 128
        self.skip = (4, )

        # End settings # 
        
        self.pe = PositionalEncoder(
            self.d_input, 
            self.n_freqs, 
            self.use_log_space)
        
        self.encode = lambda x: self.pe(x)

        self.pe_viewdirs = PositionalEncoder(
            self.d_input,
            self.n_freqs_views,
            self.use_log_space)
        
        self.encode_viewdirs = lambda x: self.pe_viewdirs(x)

        self.d_viewdirs = self.pe_viewdirs.d_output
        
        self.models = nn.ModuleDict({
            "coarse": OriginalNeRF_(
                        self.pe.d_output,
                        n_layers=self.n_layers,
                        d_filter=self.d_filter,
                        skip=self.skip,
                        d_viewdirs=self.d_viewdirs),
            "fine": OriginalNeRF_(
                        self.pe.d_output,
                        n_layers=self.n_layers,
                        d_filter=self.d_filter,
                        skip=self.skip,
                        d_viewdirs=self.d_viewdirs)
        })

        self.output_keys = {
            "coarse": ["rgb_map0", "depth_map0", "acc_map0", "weights0"],
            "fine": ["rgb_map", "depth_map", "acc_map", "weights"]
        }
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            (list(self.models['coarse'].parameters()) + 
             list(self.models['fine'].parameters())),
             self.lr
        )

        return {"optimizer": optimizer}

    
    def forward(self, rays):
        rays_o, rays_d = torch.split(rays, 3, dim=1)

        C = Chunker(self.pts_chunk_size, self.pe, self.pe_viewdirs)

        outputs = defaultdict(dict)
        
        for p in ["coarse", "fine"]:
            if p is "coarse":
                pts, z_vals = sample_stratified(
                    rays_o, rays_d, self.near, self.far, 
                    self.stratified_sampling_sample_size, 
                    perturb=True, inverse_depth=False)
            else:
                pts, z_vals_combined, z_hierarch = sample_hierarchical(
                    rays_o, rays_d, z_vals, 
                    outputs['coarse']['outputs']['weights0'], 
                    self.n_samples_hierarchical, perturb=True)
                
                outputs[p]['sampling_hierarchical'] = z_hierarch
                z_vals = z_vals_combined
            
            chunks = tqdm(C(pts, rays_d), desc=f"\t{p} pass", **self.pbkw(2))
            
            preds = [self.models[p](chk[0], chk[1]) for chk in chunks]

            raw = torch.cat(preds, dim=0)
            raw = raw.reshape(list(pts.shape[:2]) + [raw.shape[-1]])

            outputs[p]['outputs'] = dict(zip(self.output_keys[p], raw2outputs(raw, z_vals, rays_d)))
            outputs[p]["sampling"] = z_vals

        return outputs

    def training_step(self, batch, batch_idx):
        rays, rgb = batch
        rays = rays.squeeze()  # handle dataloader batch dim
        rgb = rgb.squeeze()
        rgb_flat = rgb.reshape(-1, 3)

        outputs = self.forward(rays)

        rgb_pred = outputs['fine']['outputs']['rgb_map']
        loss = F.mse_loss(rgb_flat, rgb_pred)
        
        pnsr = -10 * torch.log10(loss).item()

        self.log('train/mse-loss', loss.item(), 
                 prog_bar=True, on_step=True)
        
        self.log('train/PNSR', pnsr, on_epoch=True)

        return {
            "loss": loss,
            "rgb_pred": rgb_pred.reshape(rgb.shape), 
            "rgb_gt": rgb}
            

    def validation_step(self, batch, batch_idx):
        rays, rgb = batch
        rays = rays.squeeze()
        rgb = rgb.squeeze()
        rgb_flat = rgb.reshape(-1, 3)

        outputs = self.forward(rays)

        rgb_pred = outputs['fine']['outputs']['rgb_map']
        loss = F.mse_loss(rgb_flat, rgb_pred)
        
        pnsr = -10 * torch.log10(loss).item()

        self.log('val/mse-loss', loss.item(), 
                 prog_bar=True, on_epoch=True)
        
        self.log('val/PNSR', pnsr, on_epoch=True)

        return {
            "loss": loss,
            "rgb_pred": rgb_pred.reshape(rgb.shape), 
            "rgb_gt": rgb}
    
    def test_step(self, batch, batch_idx):
        rays, rgb = batch
        rays = rays.squeeze()
        rgb = rgb.squeeze()
        rgb_flat = rgb.reshape(-1, 3)

        outputs = self.forward(rays)

        rgb_pred = outputs['fine']['outputs']['rgb_map']
        loss = F.mse_loss(rgb_flat, rgb_pred)
        
        pnsr = -10 * torch.log10(loss).item()

        self.log('test/mse-loss', loss.item(), 
                 prog_bar=False, on_epoch=True)
        
        self.log('test/PNSR', pnsr, on_epoch=True)

        return {
            "loss": loss,
            "rgb_pred": rgb_pred.reshape(rgb.shape), 
            "rgb_gt": rgb}



def raw2outputs(raw: torch.Tensor, z_vals: torch.Tensor, rays_d: torch.Tensor,
        raw_noise_std: float=0.0, white_bkgd: bool=False) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    # get the distance between z_vals, assume the last point is at inf. (~1e10)
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Add noise to model's predictions for density. Can be used to 
    # regularize network during training (prevents floater artifacts).
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point. [n_rays, n_samples]
    alpha = 1.0 - torch.exp(-nn.functional.relu(raw[..., 3] + noise) * dists)

    # compute the weight for rgb of each sample along each ray. The higher the 
    # alpha, the lower subsequent weights are driven (e.g., more transparent
    # values don't contribute as much to the final RGB value)
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

    # Compute the weighted RGB map
    rgb = torch.sigmoid(raw[..., :3])
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)

    # est. depth map is predicted distance
    depth_map = torch.sum(weights * z_vals, dim=-1)

    # disparity map is inverse depth
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map),
                              depth_map / torch.sum(weights, -1))
    
    # Sum of weights along each ray. In [0, 1] up to numerical error
    acc_map = torch.sum(weights, dim=-1)

    # To composite onto a white background, use the accumulated alpha map
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, depth_map, acc_map, weights


class OriginalNeRF_(nn.Module):
    def __init__(
        self, d_input: int=3, n_layers: int=8, d_filter: int=256,
        skip: tuple[int]=(4, ), d_viewdirs: int=None,
    ):
        super().__init__()

        self.d_input = d_input
        self.skip = skip
        self.act = nn.functional.relu
        self.d_viewdirs = d_viewdirs

        self.layers = nn.ModuleList(
            [nn.Linear(self.d_input, d_filter)] +
            [nn.Linear(d_filter + self.d_input, d_filter) if i in skip \
             else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
        )

        if self.d_viewdirs is not None:
            self.alpha_out = nn.Linear(d_filter, 1)
            self.rgb_filters = nn.Linear(d_filter, d_filter)
            self.branch = nn.Linear(d_filter + self.d_viewdirs, d_filter//2)
            self.output = nn.Linear(d_filter // 2, 3)
        else:
            self.output = nn.Linear(d_filter, 4)

    def forward(self, x:torch.Tensor, viewdirs:torch.Tensor=None) -> torch.Tensor:

        if self.d_viewdirs is None and viewdirs is not None:
            raise ValueError('Cannot input x_direction if d_viewdirs was not given')
        
        x_input = x
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
            if i in self.skip:
                x = torch.cat([x, x_input], dim=-1)

        if self.d_viewdirs is not None:
            alpha = self.alpha_out(x)

            x = self.rgb_filters(x)
            x = torch.concat([x, viewdirs], dim=-1)
            x = self.act(self.branch(x))
            x = self.output(x)

            x = torch.concat([x, alpha], dim=-1)
        else:
            x = self.output(x)
        
        return x
    
def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    cumprod = torch.cumprod(tensor, -1)
    cumprod = torch.roll(cumprod, 1, -1)
    cumprod[..., 0] = 1

    return cumprod