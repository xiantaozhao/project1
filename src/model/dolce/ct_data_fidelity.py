"""
CT Data Fidelity class using ASTRA toolbox
Replaces LEAP-based implementation from DOLCE with ASTRA
"""

import torch
import numpy as np
import astra


class CTClass_astra:
    """
    CT data fidelity class using ASTRA for forward/backward projection.
    Implements proximal operators for data consistency in DOLCE.
    
    Args:
        target_sinogram: Target sinogram data (numpy array or torch tensor)
        angles: Projection angles in radians
        det_count: Number of detector pixels
        det_pixel_mm: Detector pixel size in mm
        source_origin: Source to origin distance (DSO) in mm
        origin_det: Origin to detector distance (ODD) in mm
        img_size: Image size (assumes square)
        device: torch device
    """
    
    def __init__(
        self,
        target_sinogram,
        angles,
        det_count=1000,
        det_pixel_mm=0.7,
        source_origin=1000.0,
        origin_det=600.0,
        img_size=512,
        device='cuda'
    ):
        self.device = device
        self.det_count = det_count
        self.det_pixel_mm = det_pixel_mm
        self.source_origin = source_origin
        self.origin_det = origin_det
        self.img_size = img_size
        
        # Convert target sinogram to tensor
        if isinstance(target_sinogram, np.ndarray):
            self.target_sinogram = torch.from_numpy(target_sinogram).float().to(device)
        else:
            self.target_sinogram = target_sinogram.to(device)
            
        # Store angles
        if isinstance(angles, np.ndarray):
            self.angles = angles
        else:
            self.angles = angles.cpu().numpy() if isinstance(angles, torch.Tensor) else np.array(angles)
            
        # Create ASTRA geometries
        self._create_astra_geometries()
        
    def _create_astra_geometries(self):
        """Create ASTRA projection and volume geometries."""
        # Volume geometry (2D square)
        self.vol_geom = astra.create_vol_geom(self.img_size, self.img_size)
        
        # Projection geometry (fanflat)
        self.proj_geom = astra.create_proj_geom(
            'fanflat',
            self.det_pixel_mm,
            self.det_count,
            self.angles,
            self.source_origin,
            self.origin_det
        )
        
    def forward_project(self, image):
        """
        Forward projection: image -> sinogram
        
        Args:
            image: Input image (B, 1, H, W) torch tensor or (H, W) numpy array
            
        Returns:
            Sinogram (B, 1, num_angles, det_count) torch tensor
        """
        is_tensor = isinstance(image, torch.Tensor)
        device = image.device if is_tensor else self.device
        
        # Handle batch dimension
        if is_tensor:
            if image.dim() == 4:
                batch_size = image.shape[0]
                images = image.cpu().numpy()[:, 0, :, :]  # (B, H, W)
            elif image.dim() == 3:
                batch_size = image.shape[0]
                images = image.cpu().numpy()  # (B, H, W)
            else:
                batch_size = 1
                images = image.cpu().numpy().reshape(1, self.img_size, self.img_size)
        else:
            batch_size = 1
            images = image.reshape(1, self.img_size, self.img_size)
            
        # Project each image in batch
        sinograms = []
        for i in range(batch_size):
            # Create ASTRA data objects
            vol_id = astra.data2d.create('-vol', self.vol_geom, images[i])
            sino_id = astra.data2d.create('-sino', self.proj_geom)
            
            # Configure and run forward projection
            cfg = astra.astra_dict('FP_CUDA')
            cfg['VolumeDataId'] = vol_id
            cfg['ProjectionDataId'] = sino_id
            
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            
            # Get sinogram
            sino = astra.data2d.get(sino_id)
            sinograms.append(sino)
            
            # Cleanup
            astra.algorithm.delete(alg_id)
            astra.data2d.delete([vol_id, sino_id])
            
        sinograms = np.stack(sinograms, axis=0)  # (B, num_angles, det_count)
        
        if is_tensor:
            sinograms = torch.from_numpy(sinograms).float().to(device)
            sinograms = sinograms.unsqueeze(1)  # (B, 1, num_angles, det_count)
            
        return sinograms
    
    def backward_project(self, sinogram):
        """
        Backward projection: sinogram -> image
        
        Args:
            sinogram: Input sinogram (B, 1, num_angles, det_count) torch tensor or (num_angles, det_count) numpy
            
        Returns:
            Image (B, 1, H, W) torch tensor
        """
        is_tensor = isinstance(sinogram, torch.Tensor)
        device = sinogram.device if is_tensor else self.device
        
        # Handle batch dimension
        if is_tensor:
            if sinogram.dim() == 4:
                batch_size = sinogram.shape[0]
                sinograms = sinogram.cpu().numpy()[:, 0, :, :]  # (B, num_angles, det_count)
            elif sinogram.dim() == 3:
                batch_size = sinogram.shape[0]
                sinograms = sinogram.cpu().numpy()  # (B, num_angles, det_count)
            else:
                batch_size = 1
                sinograms = sinogram.cpu().numpy().reshape(1, len(self.angles), self.det_count)
        else:
            batch_size = 1
            sinograms = sinogram.reshape(1, len(self.angles), self.det_count)
            
        # Backproject each sinogram in batch
        images = []
        for i in range(batch_size):
            # Create ASTRA data objects
            sino_id = astra.data2d.create('-sino', self.proj_geom, sinograms[i])
            vol_id = astra.data2d.create('-vol', self.vol_geom)
            
            # Configure and run backward projection
            cfg = astra.astra_dict('BP_CUDA')
            cfg['ProjectionDataId'] = sino_id
            cfg['ReconstructionDataId'] = vol_id
            
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            
            # Get reconstructed image
            img = astra.data2d.get(vol_id)
            images.append(img)
            
            # Cleanup
            astra.algorithm.delete(alg_id)
            astra.data2d.delete([sino_id, vol_id])
            
        images = np.stack(images, axis=0)  # (B, H, W)
        
        if is_tensor:
            images = torch.from_numpy(images).float().to(device)
            images = images.unsqueeze(1)  # (B, 1, H, W)
            
        return images
    
    def eval(self, image):
        """
        Evaluate data fidelity: ||A(x) - y||^2
        
        Args:
            image: Input image (B, 1, H, W)
            
        Returns:
            Data fidelity value (scalar)
        """
        sino_pred = self.forward_project(image)
        
        # Handle target sinogram shape
        if self.target_sinogram.dim() == 2:
            target = self.target_sinogram.unsqueeze(0).unsqueeze(0)
        elif self.target_sinogram.dim() == 3:
            target = self.target_sinogram.unsqueeze(1)
        else:
            target = self.target_sinogram
            
        diff = sino_pred - target
        return torch.sum(diff ** 2).item()
    
    def gradient(self, image):
        """
        Compute gradient of data fidelity: A^T(A(x) - y)
        
        Args:
            image: Input image (B, 1, H, W)
            
        Returns:
            Gradient (B, 1, H, W)
        """
        sino_pred = self.forward_project(image)
        
        # Handle target sinogram shape
        if self.target_sinogram.dim() == 2:
            target = self.target_sinogram.unsqueeze(0).unsqueeze(0)
        elif self.target_sinogram.dim() == 3:
            target = self.target_sinogram.unsqueeze(1)
        else:
            target = self.target_sinogram
            
        diff = sino_pred - target
        grad = self.backward_project(diff)
        
        return grad
    
    def prox_apgm(self, x, rho, max_iter=10, tol=1e-6):
        """
        Proximal operator using Accelerated Proximal Gradient Method (APGM).
        Solves: argmin_z 0.5*||z - x||^2 + (rho/2)*||A(z) - y||^2
        
        Args:
            x: Input image (B, 1, H, W)
            rho: Proximal parameter
            max_iter: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            Proximal solution (B, 1, H, W)
        """
        z = x.clone()
        z_prev = x.clone()
        t = 1.0
        
        # Step size (Lipschitz constant approximation)
        L = rho * 2.0  # Conservative estimate
        step_size = 1.0 / L
        
        for k in range(max_iter):
            # Momentum update
            y = z + (t - 1) / (t + 2) * (z - z_prev)
            
            # Gradient step
            grad = self.gradient(y)
            z_new = y - step_size * (y - x + rho * grad)
            
            # Check convergence
            diff = torch.norm(z_new - z) / (torch.norm(z) + 1e-8)
            if diff < tol:
                break
                
            z_prev = z
            z = z_new
            t = t + 1
            
        return z
    
    def prox_cgrad(self, x, rho, max_iter=10, tol=1e-6):
        """
        Proximal operator using Conjugate Gradient (CG) method.
        Solves: argmin_z 0.5*||z - x||^2 + (rho/2)*||A(z) - y||^2
        
        Args:
            x: Input image (B, 1, H, W)
            rho: Proximal parameter
            max_iter: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            Proximal solution (B, 1, H, W)
        """
        # Initialize
        z = x.clone()
        
        # Compute initial residual: r = (x - z) - rho * A^T(A(z) - y)
        r = x - z - rho * self.gradient(z)
        p = r.clone()
        rs_old = torch.sum(r * r)
        
        for k in range(max_iter):
            # Compute Ap: A^T A p
            Ap = p + rho * self.gradient(p)
            
            # Compute step size
            pAp = torch.sum(p * Ap)
            alpha = rs_old / (pAp + 1e-8)
            
            # Update solution
            z = z + alpha * p
            
            # Update residual
            r = r - alpha * Ap
            rs_new = torch.sum(r * r)
            
            # Check convergence
            if torch.sqrt(rs_new) < tol:
                break
                
            # Update search direction
            beta = rs_new / (rs_old + 1e-8)
            p = r + beta * p
            rs_old = rs_new
            
        return z
    
    def __del__(self):
        """Cleanup ASTRA objects."""
        try:
            astra.clear()
        except:
            pass


def create_ct_data_fidelity(target_sinogram, angles, config, device='cuda'):
    """
    Factory function to create CTClass_astra from config.
    
    Args:
        target_sinogram: Target sinogram
        angles: Projection angles
        config: Configuration dict with geometry parameters
        device: torch device
        
    Returns:
        CTClass_astra instance
    """
    return CTClass_astra(
        target_sinogram=target_sinogram,
        angles=angles,
        det_count=config.get('det_count', 1000),
        det_pixel_mm=config.get('det_pixel_mm', 0.7),
        source_origin=config.get('source_origin', 1000.0),
        origin_det=config.get('origin_det', 600.0),
        img_size=config.get('img_size', 512),
        device=device
    )
