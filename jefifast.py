import os, math, time
import numpy as np
from numba import cuda, float32
from loguru import logger
import concurrent.futures

# ==========================================
# 1. OPTIMIZED PACKED KERNELS
# ==========================================

@cuda.jit(fastmath=True)
def jefimenko_kernel_packed(
    packed_src_data, src_x, src_y, src_z,
    obs_x, obs_y, obs_z,
    inv_dt, t, num_src, num_obs,
    Ex, Ey, Ez, Bx, By, Bz,
    src_tile_size=64
):
    # Dynamic shared memory for positions
    shared = cuda.shared.array(shape=0, dtype=float32)
    
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    block_dim = cuda.blockDim.x
    
    obs_idx = bid * block_dim + tid
    
    # Pre-fetch observation point
    xo, yo, zo = 0.0, 0.0, 0.0
    valid_obs = obs_idx < num_obs
    
    if valid_obs:
        xo = obs_x[obs_idx]
        yo = obs_y[obs_idx]
        zo = obs_z[obs_idx]
    
    # Initialize accumulators
    Exl, Eyl, Ezl = 0.0, 0.0, 0.0
    Bxl, Byl, Bzl = 0.0, 0.0, 0.0
    
    num_src_tiles = (num_src + src_tile_size - 1) // src_tile_size
    
    for tile_idx in range(num_src_tiles):
        src_start = tile_idx * src_tile_size
        src_end = min(src_start + src_tile_size, num_src)
        tile_actual_size = src_end - src_start
        
        # --- LOADING PHASE (Coalesced) ---
        for j in range(tid, tile_actual_size, block_dim):
            global_src = src_start + j
            shared[j * 3] = src_x[global_src]
            shared[j * 3 + 1] = src_y[global_src]
            shared[j * 3 + 2] = src_z[global_src]
            
        cuda.syncthreads()
        
        # --- CALCULATION PHASE ---
        if valid_obs:
            for i in range(tile_actual_size):
                base_idx = i * 3
                dx = xo - shared[base_idx]
                dy = yo - shared[base_idx + 1]
                dz = zo - shared[base_idx + 2]
                
                dr_sq = dx*dx + dy*dy + dz*dz
                
                if dr_sq <= 1e-10: continue
                
                dr = math.sqrt(dr_sq)
                tr = t - dr
                
                if tr < 0: continue
                
                ti = int(tr * inv_dt)
                
                src_idx = src_start + i
                
                # Packed Memory Access (Cache Optimized)
                # Layout: 0:rho, 1:drho, 2:Jx, 3:dJx, 4:Jy, 5:dJy, 6:Jz, 7:dJz
                rho_val  = packed_src_data[src_idx, ti, 0]
                drho_val = packed_src_data[src_idx, ti, 1]
                Jx_val   = packed_src_data[src_idx, ti, 2]
                dJx_val  = packed_src_data[src_idx, ti, 3]
                Jy_val   = packed_src_data[src_idx, ti, 4]
                dJy_val  = packed_src_data[src_idx, ti, 5]
                Jz_val   = packed_src_data[src_idx, ti, 6]
                dJz_val  = packed_src_data[src_idx, ti, 7]
                
                inv_dr = 1.0 / dr
                inv_dr2 = inv_dr * inv_dr
                inv_dr3 = inv_dr2 * inv_dr
                
                E_term = inv_dr3 * rho_val + inv_dr2 * drho_val
                
                Exl += dx * E_term - inv_dr * dJx_val
                Eyl += dy * E_term - inv_dr * dJy_val
                Ezl += dz * E_term - inv_dr * dJz_val
                
                Bxl -= ((dy * Jz_val - dz * Jy_val) * inv_dr3 + 
                        (dy * dJz_val - dz * dJy_val) * inv_dr2)
                Byl -= ((dz * Jx_val - dx * Jz_val) * inv_dr3 + 
                        (dz * dJx_val - dx * dJz_val) * inv_dr2)
                Bzl -= ((dx * Jy_val - dy * Jx_val) * inv_dr3 + 
                        (dx * dJy_val - dy * dJx_val) * inv_dr2)
        
        cuda.syncthreads()

    if valid_obs:
        Ex[obs_idx] = Exl
        Ey[obs_idx] = Eyl
        Ez[obs_idx] = Ezl
        Bx[obs_idx] = Bxl
        By[obs_idx] = Byl
        Bz[obs_idx] = Bzl

@cuda.jit
def update_src_packed(t_idx, dt, rho, Jx, Jy, Jz, packed_data, num_src):
    i = cuda.grid(1)
    if i < num_src:
        rho_c = rho[i]
        Jx_c = Jx[i]
        Jy_c = Jy[i]
        Jz_c = Jz[i]
        
        drho, dJx, dJy, dJz = 0.0, 0.0, 0.0, 0.0
        
        if t_idx > 0:
            rho_p = packed_data[i, t_idx - 1, 0]
            Jx_p  = packed_data[i, t_idx - 1, 2]
            Jy_p  = packed_data[i, t_idx - 1, 4]
            Jz_p  = packed_data[i, t_idx - 1, 6]
            
            drho = (rho_c - rho_p) / dt
            dJx  = (Jx_c - Jx_p) / dt
            dJy  = (Jy_c - Jy_p) / dt
            dJz  = (Jz_c - Jz_p) / dt
            
            packed_data[i, t_idx - 1, 1] = drho
            packed_data[i, t_idx - 1, 3] = dJx
            packed_data[i, t_idx - 1, 5] = dJy
            packed_data[i, t_idx - 1, 7] = dJz
            
        packed_data[i, t_idx, 0] = rho_c
        packed_data[i, t_idx, 2] = Jx_c
        packed_data[i, t_idx, 4] = Jy_c
        packed_data[i, t_idx, 6] = Jz_c
        
        packed_data[i, t_idx, 1] = drho
        packed_data[i, t_idx, 3] = dJx
        packed_data[i, t_idx, 5] = dJy
        packed_data[i, t_idx, 7] = dJz

# ==========================================
# 2. INTELLIGENT WORKER (Handles Batching)
# ==========================================

class SmartWorker:
    def __init__(self, gpu_id, total_steps, obs_slice_idx, obs_x_full, obs_y_full, obs_z_full, 
                 src_x, src_y, src_z, ds, dt, tpb, src_tile_size):
        
        cuda.select_device(gpu_id)
        self.gpu_id = gpu_id
        self.dt = dt
        self.inv_dt = 1.0 / dt
        self.src_tile_size = src_tile_size
        self.tpb = tpb
        self.coeff = 1 / (4 * math.pi) * ds[0] * ds[1] * ds[2]

        # 1. Store Source Data (Must fit in GPU, otherwise we need multi-node)
        self.num_src = src_x.size
        self.src_x_gpu = cuda.to_device(src_x)
        self.src_y_gpu = cuda.to_device(src_y)
        self.src_z_gpu = cuda.to_device(src_z)
        
        # 2. Allocate Packed History (Static Size)
        # Size: N_src * Steps * 8 floats * 4 bytes
        self.packed_data = cuda.device_array((self.num_src, total_steps, 8), dtype=np.float32)
        self.src_blocks = (self.num_src + tpb - 1) // tpb
        self.t_step = 0

        # 3. Observation Data (Subset for this GPU)
        self.obs_x_host = np.ascontiguousarray(obs_x_full[obs_slice_idx])
        self.obs_y_host = np.ascontiguousarray(obs_y_full[obs_slice_idx])
        self.obs_z_host = np.ascontiguousarray(obs_z_full[obs_slice_idx])
        self.num_obs = self.obs_x_host.size
        
        # 4. MEMORY MANAGEMENT & BATCHING
        self._configure_batching()

    def _configure_batching(self):
        """
        Determines if we can fit all observation points in VRAM.
        If not, sets up batching.
        """
        free_mem, total_mem = cuda.current_context().get_memory_info()
        
        # Reserve 20% buffer for system overhead / fragmentation
        available_mem = free_mem * 0.8 
        
        # Memory per observation point (Input: 3 floats, Output: 6 floats) = 9 * 4 bytes = 36 bytes
        bytes_per_obs = 36 
        
        # Check if full fit is possible
        required_mem = self.num_obs * bytes_per_obs
        
        if required_mem < available_mem:
            self.batch_mode = False
            self.batch_size = self.num_obs
            # Pre-allocate everything
            self.obs_x_gpu = cuda.to_device(self.obs_x_host)
            self.obs_y_gpu = cuda.to_device(self.obs_y_host)
            self.obs_z_gpu = cuda.to_device(self.obs_z_host)
            self.Ex_gpu = cuda.device_array(self.num_obs, dtype=np.float32)
            self.Ey_gpu = cuda.device_array(self.num_obs, dtype=np.float32)
            self.Ez_gpu = cuda.device_array(self.num_obs, dtype=np.float32)
            self.Bx_gpu = cuda.device_array(self.num_obs, dtype=np.float32)
            self.By_gpu = cuda.device_array(self.num_obs, dtype=np.float32)
            self.Bz_gpu = cuda.device_array(self.num_obs, dtype=np.float32)
        else:
            self.batch_mode = True
            # Calculate max batch size
            self.batch_size = int(available_mem // bytes_per_obs)
            # Align to 1024 for safety
            self.batch_size = (self.batch_size // 1024) * 1024
            logger.warning(f"GPU {self.gpu_id}: OOM risk. Batching {self.num_obs} obs points into batches of size {self.batch_size}")

            # Pre-allocate Reusable Buffers (Size = Batch Size)
            self.obs_x_gpu = cuda.device_array(self.batch_size, dtype=np.float32)
            self.obs_y_gpu = cuda.device_array(self.batch_size, dtype=np.float32)
            self.obs_z_gpu = cuda.device_array(self.batch_size, dtype=np.float32)
            self.Ex_gpu = cuda.device_array(self.batch_size, dtype=np.float32)
            self.Ey_gpu = cuda.device_array(self.batch_size, dtype=np.float32)
            self.Ez_gpu = cuda.device_array(self.batch_size, dtype=np.float32)
            self.Bx_gpu = cuda.device_array(self.batch_size, dtype=np.float32)
            self.By_gpu = cuda.device_array(self.batch_size, dtype=np.float32)
            self.Bz_gpu = cuda.device_array(self.batch_size, dtype=np.float32)

    def update_src(self, rho, Jx, Jy, Jz):
        cuda.select_device(self.gpu_id)
        update_src_packed[self.src_blocks, self.tpb](
            self.t_step, self.dt,
            cuda.to_device(rho), cuda.to_device(Jx), 
            cuda.to_device(Jy), cuda.to_device(Jz),
            self.packed_data,
            self.num_src
        )
        self.curr_t = self.t_step * self.dt
        self.t_step += 1

    def solve(self):
        cuda.select_device(self.gpu_id)
        shared_mem_size = self.src_tile_size * 3 * 4
        
        if not self.batch_mode:
            # === FAST PATH: Single Kernel Launch ===
            blocks = (self.num_obs + self.tpb - 1) // self.tpb
            jefimenko_kernel_packed[blocks, self.tpb, 0, shared_mem_size](
                self.packed_data, self.src_x_gpu, self.src_y_gpu, self.src_z_gpu,
                self.obs_x_gpu, self.obs_y_gpu, self.obs_z_gpu,
                self.inv_dt, self.curr_t, self.num_src, self.num_obs,
                self.Ex_gpu, self.Ey_gpu, self.Ez_gpu, self.Bx_gpu, self.By_gpu, self.Bz_gpu,
                self.src_tile_size
            )
            # Return copies
            return [self.coeff * x.copy_to_host() for x in 
                    [self.Ex_gpu, self.Ey_gpu, self.Ez_gpu, self.Bx_gpu, self.By_gpu, self.Bz_gpu]]
        
        else:
            # === SAFE PATH: Batch Processing ===
            # We must accumulate results on host to save GPU memory
            final_res = [np.empty(self.num_obs, dtype=np.float32) for _ in range(6)]
            
            num_batches = (self.num_obs + self.batch_size - 1) // self.batch_size
            
            for b in range(num_batches):
                start = b * self.batch_size
                end = min(start + self.batch_size, self.num_obs)
                curr_size = end - start
                
                # 1. Copy batch coordinates to pre-allocated GPU buffer
                self.obs_x_gpu[:curr_size].copy_to_device(self.obs_x_host[start:end])
                self.obs_y_gpu[:curr_size].copy_to_device(self.obs_y_host[start:end])
                self.obs_z_gpu[:curr_size].copy_to_device(self.obs_z_host[start:end])
                
                # 2. Reset Output buffers (optional, but good practice if reusing)
                # Actually not strictly needed as kernel overwrites, but careful with accumulation logic if changed
                
                # 3. Run Kernel
                blocks = (curr_size + self.tpb - 1) // self.tpb
                jefimenko_kernel_packed[blocks, self.tpb, 0, shared_mem_size](
                    self.packed_data, self.src_x_gpu, self.src_y_gpu, self.src_z_gpu,
                    self.obs_x_gpu, self.obs_y_gpu, self.obs_z_gpu,
                    self.inv_dt, self.curr_t, self.num_src, curr_size,
                    self.Ex_gpu, self.Ey_gpu, self.Ez_gpu, self.Bx_gpu, self.By_gpu, self.Bz_gpu,
                    self.src_tile_size
                )
                
                # 4. Copy Back
                final_res[0][start:end] = self.Ex_gpu[:curr_size].copy_to_host()
                final_res[1][start:end] = self.Ey_gpu[:curr_size].copy_to_host()
                final_res[2][start:end] = self.Ez_gpu[:curr_size].copy_to_host()
                final_res[3][start:end] = self.Bx_gpu[:curr_size].copy_to_host()
                final_res[4][start:end] = self.By_gpu[:curr_size].copy_to_host()
                final_res[5][start:end] = self.Bz_gpu[:curr_size].copy_to_host()
            
            # Apply coeff
            return [r * self.coeff for r in final_res]


# ==========================================
# 3. UNIFIED MANAGER
# ==========================================

class UnifiedEMSolver:
    def __init__(self, total_steps, obs_n, src_n, do, obs_lc, ds, src_lc, dt, 
                 tpb=256, src_tile_size=1024, force_single_gpu=False):
        
        self.dt = dt
        
        # 1. Generate Geometry
        obs_x, obs_y, obs_z = self._get_positions(obs_lc, do, obs_n)
        src_x, src_y, src_z = self._get_positions(src_lc, ds, src_n)
        
        self.arrival_time = self._get_arrival_time(do, obs_lc, ds, src_lc, obs_n, src_n)
        total_obs = obs_x.size
        
        # 2. Determine Strategy
        available_gpus = list(range(len(cuda.gpus)))
        
        # HEURISTIC: Use Single GPU if grid is small (< 50^3) to avoid thread overhead
        # UNLESS user forces single GPU or there is only 1 GPU
        is_small_grid = total_obs < (40**3) 
        
        if force_single_gpu or len(available_gpus) == 1 or is_small_grid:
            self.active_gpus = [0]
            strategy = "SINGLE GPU (Small Grid Optimized)" if is_small_grid else "SINGLE GPU"
        else:
            self.active_gpus = available_gpus
            strategy = f"MULTI-GPU ({len(self.active_gpus)} GPUs)"
            
        logger.info(f"Strategy: {strategy} | Total Obs: {total_obs}")
        
        # 3. Initialize Workers
        chunk_size = (total_obs + len(self.active_gpus) - 1) // len(self.active_gpus)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.active_gpus))
        futures = []
        
        for i, gpu_id in enumerate(self.active_gpus):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total_obs)
            slc = slice(start, end)
            
            f = self.executor.submit(
                SmartWorker,
                gpu_id=gpu_id,
                total_steps=total_steps,
                obs_slice_idx=slc,
                obs_x_full=obs_x, obs_y_full=obs_y, obs_z_full=obs_z,
                src_x=src_x, src_y=src_y, src_z=src_z,
                ds=ds, dt=dt, tpb=tpb, src_tile_size=src_tile_size
            )
            futures.append(f)
            
        self.workers = [f.result() for f in futures]

    def update_src(self, rho, Jx, Jy, Jz):
        futures = [self.executor.submit(w.update_src, rho, Jx, Jy, Jz) for w in self.workers]
        concurrent.futures.wait(futures)

    def solve(self):
        futures = [self.executor.submit(w.solve) for w in self.workers]
        results = [f.result() for f in futures]
        
        # Gather logic: [[Ex1, Ey1..], [Ex2, Ey2..]] -> [Ex_full, Ey_full..]
        transposed = list(zip(*results))
        final_fields = [np.concatenate(comp_parts) for comp_parts in transposed]
        return final_fields
    
    @staticmethod
    def _get_positions(lcs, grid_sizes, grids):
        x, y, z = [np.array([lcs[i] + grid_sizes[i] * (j + 0.5) for j in range(grids[i])], 
                                 dtype=np.float32) for i in range(3)]
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        return X.flatten(), Y.flatten(), Z.flatten()

    @staticmethod
    def _get_arrival_time(do, obs_lc, ds, src_lc, obs_n, src_n):
        obs_cnp = [obs_lc[i] + do[i] * obs_n[i] for i in range(3)]
        src_cnp = [src_lc[i] + ds[i] * src_n[i] for i in range(3)]
        d_min = [max(src_lc[i] - obs_cnp[i], obs_lc[i] - src_cnp[i], 0) for i in range(3)]
        return math.sqrt(d_min[0]**2 + d_min[1]**2 + d_min[2]**2)

# ==========================================
# 4. HELPER: Source Generation
# ==========================================
def get_src(src_type, timesteps, src_n, ds, src_lc):
    # Re-generating geometry just for source function calculation
    x, y, z = [np.array([src_lc[i] + ds[i] * (j + 0.5) for j in range(src_n[i])], 
                             dtype=np.float32) for i in range(3)]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    src_x, src_y, src_z = X.flatten(), Y.flatten(), Z.flatten()
    
    time_len = len(timesteps)
    len_src = len(src_x) 
    if src_type == "const":
        return (np.ones((time_len, len_src), dtype=np.float32) for _ in range(4))
    
    xyz = src_x + src_y + src_z
    sin_xyz = np.sin(xyz)
    cos_xyz = np.cos(xyz)
    sint = np.sin(timesteps).reshape(-1, 1)
    cost = np.cos(timesteps).reshape(-1, 1)

    val = sin_xyz * sint
    return val, val, val, 3 * (cost - 1) * cos_xyz

# ==========================================
# 5. MAIN
# ==========================================
def run_simulation(grid_size, src_type='const'):
    len_buffer = 420 
    dt = 0.05
    
    obs_n = (grid_size, grid_size, grid_size)
    src_n = (grid_size, grid_size, grid_size)
    do = (6/grid_size, 6/grid_size, 6/grid_size)
    ds = (6/grid_size, 6/grid_size, 6/grid_size)
    obs_lc = (-3, -3, -3)
    src_lc = (-3, -3, -3) if src_type == 'const' else (-3, -3, 10)
    
    logger.info(f"--- Running {src_type} Grid: {grid_size} ---")
    
    solver = UnifiedEMSolver(len_buffer, obs_n, src_n, do, obs_lc, ds, src_lc, dt)
    
    timesteps = np.arange(0, 20.5, dt, dtype=np.float32)
    Jx, Jy, Jz, rho = get_src(src_type, timesteps, src_n, ds, src_lc)
    
    Ex_list, Ey_list, Ez_list = [], [], []
    Bx_list, By_list, Bz_list = [], [], []
    
    start = time.time()
    for t_idx in range(410):
        solver.update_src(rho[t_idx], Jx[t_idx], Jy[t_idx], Jz[t_idx])
        
        # 2. Check retardation limit
        if t_idx * dt >= solver.arrival_time:
            # 3. Solve (Parallel) and Gather
            Ex, Ey, Ez, Bx, By, Bz = solver.solve()
            
            if t_idx % 1 == 0:
                shape = (grid_size, grid_size, grid_size)
                Ex_list.append(Ex.reshape(shape)); Ey_list.append(Ey.reshape(shape)); Ez_list.append(Ez.reshape(shape))
                Bx_list.append(Bx.reshape(shape)); By_list.append(By.reshape(shape)); Bz_list.append(Bz.reshape(shape))

    logger.info(f'{src_type} time ({solver.active_gpus} GPUs): {time.time() - start:.2f}s')
            
    Ex = np.array(Ex_list)
    Ey = np.array(Ey_list)
    Bx = np.array(Bx_list)
    By = np.array(By_list)
    
    try:
        data = np.load(f"{src_type}_{grid_size}_full.npz")  # Here you can use JefiGPU to generate the data for comparison

        # Assuming shape is (Time, Grid) or just (Grid)
        E_base = np.stack([data["Ex"], data["Ey"]], axis=-1)
        B_base = np.stack([data["Bx"], data["By"]], axis=-1)
        
        # Group your "Fast" results into vectors similarly
        # (Assuming Ex, Ey, Bx, By are available in local scope)
        E_fast = np.stack([Ex, Ey], axis=-1)
        B_fast = np.stack([Bx, By], axis=-1)
        
        for name, V_fast, V_base in zip(["E", "B"], [E_fast, B_fast], [E_base, B_base]):
            
            # 1. Calculate Pointwise Vector Difference (The 'delta' in your latex)
            # sqrt((Ax - Bx)^2 + (Ay - By)^2)
            diff_norm = np.linalg.norm(V_fast - V_base, axis=-1)
            
            # 2. Calculate Base Vector Norm (for relative error denominator)
            base_norm = np.linalg.norm(V_base, axis=-1)

            # 3. Compute Global Metrics
            max_err = diff_norm.max()
            mean_err = diff_norm.mean()
            rel_err = max_err / (base_norm.max() + 1e-10)

            logger.info(f"[{name}-Field] Max Abs: {max_err:.6f}, Mean Abs: {mean_err:.6f}, Rel Error: {rel_err:.6f}")
    except Exception: pass


if __name__ == "__main__":
    logger.add("experiment.log")
    for grid_size in [15, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        logger.info(f"Grid size {grid_size}")
        run_simulation(grid_size, 'const')
        run_simulation(grid_size, 'sin')
