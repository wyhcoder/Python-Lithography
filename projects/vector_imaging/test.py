import torch
import math


class VectorAerialModel:
    def __init__(
        self,
        N,
        dx_nm,
        lambda0_nm=193.0,
        n_image=1.44,
        NA=1.35,
        device="cpu",
        dtype=torch.complex64,
        eps=1e-12,
    ):
        self.N = N
        self.dx_nm = dx_nm
        self.lambda0_nm = lambda0_nm
        self.n_image = n_image
        self.NA = NA
        self.device = device
        self.dtype = dtype
        self.real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        self.eps = eps

        self._build_grids()
        self._build_pupil_and_vector_transfer()


    def _fft2c(self, x):
        return torch.fft.fftshift(torch.fft.fft2(x))
    def _ifft2c(self, X):
        return torch.fft.ifft2(torch.fft.ifftshift(X))

    def _build_grids(self):
        N = self.N
        dx = self.dx_nm
        device = self.device
        rdtype = self.real_dtype

        # spatial grid, unit: nm
        coord = (torch.arange(N, device=device, dtype=rdtype) - N // 2) * dx
        self.X, self.Y = torch.meshgrid(coord, coord, indexing="ij")

        # frequency grid, unit: cycles / nm
        freq = torch.fft.fftfreq(N, d=dx, device=device).to(rdtype)
        freq = torch.fft.fftshift(freq)
        self.FX, self.FY = torch.meshgrid(freq, freq, indexing="ij")

        # cutoff frequency, using vacuum wavelength
        self.fmax = self.NA / self.lambda0_nm

    def _build_pupil_and_vector_transfer(self):
        FX = self.FX
        FY = self.FY
        lambda0 = self.lambda0_nm
        n = self.n_image
        eps = self.eps

        # objective pupil
        FR2 = FX**2 + FY**2
        pupil = FR2 <= self.fmax**2
        
        self.pupil = pupil.to(self.real_dtype)

        # direction cosines in image medium
        alpha = lambda0 * FX / n
        beta = lambda0 * FY / n
        ab2 = alpha**2 + beta**2

        gamma2 = 1.0 - ab2
        gamma2 = torch.clamp(gamma2, min=0.0)
        gamma = torch.sqrt(gamma2)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        rho2 = ab2
        rho2_safe = torch.clamp(rho2, min=eps)

        # vector polarization mapping M0:
        # [Ex_out, Ey_out, Ez_out]^T = M0 [Ex_in, Ey_in]^T
        Mxx = (beta**2 + alpha**2 * gamma) / rho2_safe
        Mxy = alpha * beta * (gamma - 1.0) / rho2_safe
        Myx = Mxy
        Myy = (alpha**2 + beta**2 * gamma) / rho2_safe
        Mzx = -alpha
        Mzy = -beta

        # handle optical axis rho=0 limit:
        center = rho2 < eps
        Mxx = torch.where(center, torch.ones_like(Mxx), Mxx)
        Myy = torch.where(center, torch.ones_like(Myy), Myy)
        Mxy = torch.where(center, torch.zeros_like(Mxy), Mxy)
        Myx = torch.where(center, torch.zeros_like(Myx), Myx)
        Mzx = torch.where(center, torch.zeros_like(Mzx), Mzx)
        Mzy = torch.where(center, torch.zeros_like(Mzy), Mzy)

        # shape: [3, 2, N, N]
        M0 = torch.stack(
            [
                torch.stack([Mxx, Mxy], dim=0),
                torch.stack([Myx, Myy], dim=0),
                torch.stack([Mzx, Mzy], dim=0),
            ],
            dim=0,
        )

        self.M0 = M0.to(self.dtype)

    def make_annular_source(self, Ns=29, sigma_in=0.6, sigma_out=0.9):
        """
        Source coordinates are normalized to objective pupil radius.
        sx, sy in [-sigma_out, sigma_out].
        """
        rdtype = self.real_dtype
        device = self.device

        s = torch.linspace(-sigma_out, sigma_out, Ns, device=device, dtype=rdtype)
        sx, sy = torch.meshgrid(s, s, indexing="ij")
        sr = torch.sqrt(sx**2 + sy**2)

        J = ((sr >= sigma_in) & (sr <= sigma_out)).to(rdtype)

        # normalize source intensity
        J_sum = J.sum()
        if J_sum > 0:
            J = J / J_sum

        return sx, sy, J

    def forward_single_polarization(self, mask, sx, sy, J, jones=(1.0, 0.0)):
        """
        mask: [N, N], real or complex tensor
        sx, sy, J: source grids [Ns, Ns]
        jones: incident polarization [Ex, Ey]
        """
        device = self.device
        dtype = self.dtype
        rdtype = self.real_dtype

        if not torch.is_complex(mask):
            mask_c = mask.to(device=device, dtype=rdtype).to(dtype)
        else:
            mask_c = mask.to(device=device, dtype=dtype)

        Ex_in = torch.as_tensor(jones[0], device=device, dtype=dtype)
        Ey_in = torch.as_tensor(jones[1], device=device, dtype=dtype)

        # vector transfer for this incident Jones vector:
        # T_p(f,g) = M0[p,0] Ex + M0[p,1] Ey
        # shape: [3, N, N]
        T = self.M0[:, 0] * Ex_in + self.M0[:, 1] * Ey_in

        pupil_c = self.pupil.to(dtype)
        T = T * pupil_c[None, :, :]

        aerial = torch.zeros((self.N, self.N), device=device, dtype=rdtype)

        # flatten source
        sx_flat = sx.reshape(-1)
        sy_flat = sy.reshape(-1)
        J_flat = J.reshape(-1)

        two_pi = 2.0 * math.pi

        for sx_i, sy_i, Ji in zip(sx_flat, sy_flat, J_flat):
            if Ji <= 0:
                continue

            # source coordinate normalized to objective pupil:
            # fs = sx * fmax
            fs = sx_i * self.fmax
            gs = sy_i * self.fmax

            ramp = torch.exp(
                1j * two_pi * (fs * self.X + gs * self.Y)
            ).to(dtype)

            u = mask_c * ramp
            U = torch.fft.fft2(u)

            # each vector component
            Ex = torch.fft.ifft2(U * T[0])
            Ey = torch.fft.ifft2(U * T[1])
            Ez = torch.fft.ifft2(U * T[2])

            I = (Ex.abs() ** 2 + Ey.abs() ** 2 + Ez.abs() ** 2).to(rdtype)
            aerial = aerial + Ji * I

        return aerial

    def forward_unpolarized(self, mask, sx, sy, J):
        """
        Unpolarized illumination:
        compute x-pol and y-pol separately, then intensity-average.
        """
        Ix = self.forward_single_polarization(mask, sx, sy, J, jones=(1.0, 0.0))
        Iy = self.forward_single_polarization(mask, sx, sy, J, jones=(0.0, 1.0))
        return 0.5 * (Ix + Iy)

if __name__ == "__main__":
    N = 257
    dx_nm = 4.0

    model = VectorAerialModel(
        N=N,
        dx_nm=dx_nm,
        lambda0_nm=193.0,
        n_image=1.44,
        NA=1.35,
        device="cpu",
    )

    # example mask
    mask = torch.zeros((N, N))
    mask[100:157, 120:137] = 1.0

    # annular source
    sx, sy, J = model.make_annular_source(
        Ns=29,
        sigma_in=0.6,
        sigma_out=0.9,
    )

    # x-polarized aerial image
    Ix = model.forward_single_polarization(
        mask,
        sx,
        sy,
        J,
        jones=(1.0, 0.0),
    )

    # y-polarized aerial image
    Iy = model.forward_single_polarization(
        mask,
        sx,
        sy,
        J,
        jones=(0.0, 1.0),
    )

