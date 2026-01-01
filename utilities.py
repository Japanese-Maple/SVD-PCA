"""Helper File For the SVD_via_PCA Notebook"""

import numpy as np
import pandas as pd
import torch
import ctypes as C
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import plotly.express as px
import plotly.graph_objects as go

"""Main functions can be found below"""

def Eigendecomposition_via_QR(A_in: np.ndarray, 
                              tol: float = 1e-12, max_iter: int = 10000, return_eigenvectors: bool = True, 
                              library_path: str = "./libqr.so"):
    """
    Computes the Eigendecomposition of a symmetric matrix by interfacing 
    with a high-performance C shared library.

    This function serves as a Python bridge to a C implementation that uses 
    Householder Tridiagonalization followed by the QL algorithm with 
    Wilkinson shifts.

    Args:
        A_in (np.ndarray): Square symmetric input matrix of shape (n, n).
        tol (float): Convergence tolerance. Note: This is currently 
            ignored by the underlying fast C algorithm. Defaults to 1e-12.
        max_iter (int): Maximum iterations allowed. Note: This is currently 
            ignored by the underlying fast C algorithm. Defaults to 10000.
        return_eigenvectors (bool): If True, computes and returns both eigenvalues 
            and eigenvectors. If False, only eigenvalues are returned.

    Returns:
        tuple or np.ndarray: 
            - If return_eigenvectors is True: Returns (eigvals, V) where eigvals is 
              an array of shape (n,) and V is a matrix of shape (n, n).
            - If return_eigenvectors is False: Returns eigvals of shape (n,).

    Technical Details:
        1. Memory Management: Uses np.ascontiguousarray to ensure the 
           matrix memory layout is compatible with C pointers.
        2. C-Interfacing: Utilizes ctypes to pass memory addresses of NumPy 
           arrays directly to "libqr.so".
        3. Implementation: The underlying C code performs a two-phase 
           reduction: first to tridiagonal form, then to diagonal form 
           using implicit shifts for cubic convergence.
    """

    A = np.ascontiguousarray(A_in, dtype=np.float64)
    n = A.shape[0]
    
    # Prepare Output Arrays
    eigvals = np.empty(n, dtype=np.float64)
    
    if return_eigenvectors:
        V = np.empty((n, n), dtype=np.float64)
        V_ptr = V.ctypes.data_as(C.POINTER(C.c_double))
    else:
        V = None
        V_ptr = None

    lib = C.CDLL(library_path)
    lib.qr_wilkinson_symmetric.argtypes = [
        C.POINTER(C.c_double),  # A
        C.c_int,                # n
        C.c_double,             # tol (ignored by fast alg)
        C.c_int,                # max_iter (ignored by fast alg)
        C.POINTER(C.c_double),  # eigvals
        C.POINTER(C.c_double)   # V (can be NULL)
    ]
    lib.qr_wilkinson_symmetric.restype = None

    lib.qr_wilkinson_symmetric(
        A.ctypes.data_as(C.POINTER(C.c_double)),
        n,
        tol,
        max_iter,
        eigvals.ctypes.data_as(C.POINTER(C.c_double)),
        V_ptr
    )
    
    return (eigvals, V) if return_eigenvectors else eigvals

#-----------------------------------------------------------------------------------------------------------------------------------------

def svd(A: np.ndarray, device: str = 'cuda') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the Singular Value Decomposition (SVD) of a matrix using the 
    Normal Equations method and a custom C-based Eigen-solver.

    This implementation computes A = U * D * V^T by first solving the 
    eigen-problem for the symmetric matrix (A^T * A).

    Note:
        This method is highly efficient for 'tall' matrices where the number 
        of samples far exceeds the number of features, as the core decomposition 
        happens on a smaller (d x d) matrix.

    Args:
        A (np.ndarray): The input matrix of shape (m, n).
        device (str): The torch device to perform matrix operations on 
            (e.g., 'cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        U (torch.Tensor): Left singular vectors of shape (m, n).
        D (torch.Tensor): Diagonal matrix D of singular values, shape (n, n).
        V (torch.Tensor): Right singular vectors of shape (n, n).

    Process & Logic:
        1. Normal Equations: Computes the Gram matrix (A^T * A).
        2. Eigen-Decomposition: Calls a custom C implementation (Eigendecomposition_via_QR) 
           to find eigenvalues and eigenvectors (V).
        3. Singular Values: Calculates singular_value = sqrt(max(0, eigenvalue)).
        4. Left Singular Vectors: Recovers U using the relationship:
           U = A * V * inv(D)
        5. Numerical Stability: Clamps eigenvalues at 0 and uses a mask 
           (1e-12) for safe inversion of singular values to prevent 
           division by zero.
    """
    A_torch = torch.from_numpy(A).to(device=device, dtype=torch.float64)
    
    aTa = A_torch.T @ A_torch
    aTa_cpu = aTa.cpu().numpy()

    # 1. Call C Implementation
    eig_vals_np, eig_vecs_np = Eigendecomposition_via_QR(aTa_cpu, return_eigenvectors=True)
    eigenvalues  = torch.from_numpy(eig_vals_np).to(device)
    eigenvectors = torch.from_numpy(eig_vecs_np).to(device)

    # 2. Sort Eigenvalues/Vectors (Descending)
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]    
    
    # 3. SVD
    S = torch.sqrt(torch.clamp(eigenvalues, min=0))
    D = torch.diag(S)
    V = eigenvectors
    inv_S_vals = torch.zeros_like(S)
    mask = S > 1e-12
    inv_S_vals[mask] = 1.0 / S[mask]
    T = torch.diag(inv_S_vals)

    U = A_torch @ V @ T

    return U, D, V

#-----------------------------------------------------------------------------------------------------------------------------------------

def PCA_plot_2D_data(data: np.ndarray, 
                     resolution: int = 150, figure_size: tuple[int, int] = (15,10), grid_density: int = 30, main_plot_projections_height: float = 5, contours: int = 30, set_aspect_equal: bool = True):
    
    """
    An educational and diagnostic visualizer for Principal Component Analysis (PCA) 
    specifically designed for 2D datasets.

    This function provides a comprehensive breakdown of the "mechanics" of PCA by 
    using Custom Singular Value Decomposition (SVD) and Kernel Density Estimation (KDE) 
    to illustrate how data is projected and how the coordinate system is transformed.

    Note:
        This function is a demonstration tool for 2D data. It expects an input 
        matrix with exactly two features (Nx2).

    Visual Components:
        * **Main Visual (Left)**: Displays raw data scatter, 2D density contours, 
          and a warped coordinate grid showing Euclidean space re-alignment.
        * **PDF Overlays**: 1D Kernel Density Estimates "lifted" and plotted 
          directly along the PC axes.
        * **PC Vectors**: Red (PC1) and Blue (PC2) arrows downscaled for visibility.
        * **Side Panels (Right)**: Independent 1D plots showing distribution 
          and density after projection.

    Args:
        data (np.ndarray): Input dataset of shape (N, 2).
        resolution (int): The DPI (dots per inch) for the rendered figure. 
            Defaults to 150.
        figure_size (tuple[int, int]): sets the overall figure size of the output plot.
        grid_density (int): Number of lines for the background coordinate 
            transformation grid. Defaults to 50.
        main_plot_projections_height (float): Visual amplitude of the PDF curves 
            on the main plot. Defaults to 5.
        contours (int): Number of levels for the 2D density contour map. 
            Defaults to 20.
        set_aspect_equal (bool): Forces X and Y axes to the same scale. 
            Critical for visual orthogonality. Defaults to True.

    Math & Logic:
        1. **Decomposition**: Uses SVD to find right-singular vectors (V) 
           and singular values (D).
        2. **Projection**: Projects centered data onto the principal basis:
           Z = X_centered @ V[:, :k]
        3. **Transformation**: Maps the standard basis to the principal basis 
           to render the background grid and projected densities.
    """
    
    data = data - data.mean(axis=0)
    n = 300
    x = data[:,0]
    y = data[:,1]
    X,Y = np.meshgrid(np.linspace(np.min(x)*2,np.max(x)*2,n),
                    np.linspace(np.min(y)*2,np.max(y)*2,n))
    
    positions = np.vstack([X.ravel(), Y.ravel()])
    kde = gaussian_kde(data.T)
    Z = kde(positions).reshape(X.shape)

    def dist(x,y):
        return np.sqrt(x**2+y**2)

    _, D, V = svd(data)
    V, D = V.cpu().numpy(), D.cpu().numpy()

    Z_y, Z_x  = data @ V[:,0], data @ V[:,1]

    x_1d_x = np.linspace(np.min(Z_x),np.max(Z_x),n)
    kde_1d_x = gaussian_kde(Z_x.ravel())
    z_1d_x = kde_1d_x(x_1d_x).reshape(x_1d_x.shape)   

    x_1d_y = np.linspace(np.min(Z_y),np.max(Z_y),n)
    kde_1d_y = gaussian_kde(Z_y.ravel())
    z_1d_y = kde_1d_y(x_1d_y).reshape(x_1d_y.shape) 

    origin = [0, 0] 

    L1 = np.max(Z_y) 
    L2 = np.max(Z_x)
    pc1 = V[:, 0] 
    pc2 = V[:, 1]

    coord_1 = origin + (L1 * pc1) + np.outer(Z_x, pc2)
    coord_2 = origin + (L2 * pc2) + np.outer(Z_y, pc1)

    dens_x = kde_1d_x(Z_x.ravel())   
    dens_y = kde_1d_y(Z_y.ravel())  

    # Plots
    fig = plt.figure(figsize=figure_size, dpi=resolution)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1])
    plt.suptitle("PCA via SVD", color="#519226", fontsize=25)

    # Plot 1
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.set_title('General Visual', color="#0069A1")

    ax0.scatter(x, y, c=dist(x,y), cmap='viridis', s=10, alpha=0.7, zorder = 0)
    ax0.scatter(x, y, s=13, alpha=0.3, facecolors='none', edgecolors='blue', zorder = -1)
    ax0.scatter(x=coord_1[:,0], y=coord_1[:,1], c=dens_x, 
            cmap='jet', s=10, alpha=0.7, zorder = 2)

    ax0.scatter(x=coord_2[:,0], y=coord_2[:,1], c=dens_y, 
            cmap='inferno', s=10, alpha=0.7, zorder = 2)

    x_min, x_max = ax0.get_xlim()
    y_min, y_max = ax0.get_ylim()



    # --- PDF COORDINATE CALCULATION SEGMENT ---
    distr_height = main_plot_projections_height  # controls how "tall" the PDF curves appear

    # Transform PC1 PDF (Red)
    x_pc1_centered = x_1d_y 
    y_pdf_pc1_scaled = z_1d_y / np.max(z_1d_y) * distr_height
    pdf_pc1_base = origin + (L2 * pc2) + np.outer(x_pc1_centered, pc1)
    pdf_pc1_curve = pdf_pc1_base + np.outer(y_pdf_pc1_scaled, pc2)

    ax0.plot(pdf_pc1_curve[:, 0], pdf_pc1_curve[:, 1], color='r', linewidth=2, zorder=5)
    fill_x_pc1 = np.concatenate([pdf_pc1_curve[:, 0], pdf_pc1_base[::-1, 0]])
    fill_y_pc1 = np.concatenate([pdf_pc1_curve[:, 1], pdf_pc1_base[::-1, 1]])
    ax0.fill(fill_x_pc1, fill_y_pc1, alpha=0.3, hatch="//", color="#FF5816", zorder=4)

    # Transform PC2 PDF (Blue)
    x_pc2_centered = x_1d_x 
    y_pdf_pc2_scaled = z_1d_x / np.max(z_1d_x) * distr_height
    pdf_pc2_base = origin + (L1 * pc1) + np.outer(x_pc2_centered, pc2)
    pdf_pc2_curve = pdf_pc2_base + np.outer(y_pdf_pc2_scaled, pc1)

    ax0.plot(pdf_pc2_curve[:, 0], pdf_pc2_curve[:, 1], color='b', linewidth=2, zorder=5)
    fill_x_pc2 = np.concatenate([pdf_pc2_curve[:, 0], pdf_pc2_base[::-1, 0]])
    fill_y_pc2 = np.concatenate([pdf_pc2_curve[:, 1], pdf_pc2_base[::-1, 1]])
    ax0.fill(fill_x_pc2, fill_y_pc2, alpha=0.3, hatch="//", color="#4876FF", zorder=4)
    # ------------------------------------------



    p1 = origin + (L2 * pc2) - 1000*pc1
    p2 = origin + (L2 * pc2) + 1000*pc1
    ax0.plot([p1[0], p2[0]], [p1[1], p2[1]], color="b", linewidth=0.3, zorder = 0)

    p1 = origin + (L1 * pc1) - 1000*pc2
    p2 = origin + (L1 * pc1) + 1000*pc2
    ax0.plot([p1[0], p2[0]], [p1[1], p2[1]], color="r", linewidth=0.3, zorder = 0)

    #-------------------------------------------
    # Grid
    num  = grid_density
    mult = 4
    zzz  = -10
    alph = 0.1

    X_g, Y_g = np.mgrid[x_min*mult:x_max*mult:num*1j, x_min*mult:x_max*mult:num*1j]
    points = np.vstack([X_g.ravel(), Y_g.ravel()]).T 
    grid_points = points @ V.T
    grid_lines  = grid_points.reshape(num, num, 2)

    ax0.scatter(grid_points[:,0], grid_points[:,1], s=3, alpha=alph, c='k', zorder = zzz)

    for i in range(num):
        ax0.plot([grid_lines[i,0,0], grid_lines[i,-1,0]], 
                [grid_lines[i,0,1], grid_lines[i,-1,1]], color = 'k', linewidth=0.3, alpha=alph, zorder = zzz)
        
        ax0.plot([grid_lines[0,i,0], grid_lines[-1,i,0]], 
                [grid_lines[0,i,1], grid_lines[-1,i,1]], color = 'k', linewidth=0.3, alpha=alph, zorder = zzz)

    # ------------------------------------------

    ax0.quiver(*origin, *(pc1 * L1), color='r', scale=1, units='xy', label='PC1', zorder = 3)
    ax0.quiver(*origin, *(pc2 * L2), color='b', scale=1, units='xy', label='PC2', zorder = 3)
    
    # Extract singular values (magnitudes)
    s = np.diag(D) 

    # PC1 Signature (Red Arrow)
    label1 = f"σ = {s[0]:.2f}" 
    ax0.text(*(pc1 * L1 * 1.3), label1, color="#6E0000", 
                fontsize=11, fontweight='normal', ha='center', va='center', zorder = 100)

    # PC2 Signature (Blue Arrow)
    label2 = f"σ = {s[1]:.2f}" 
    ax0.text(*(pc2 * L2 * 1.3), label2, color="#00126E", 
                fontsize=11, fontweight='normal', ha='center', va='center', zorder = 100)

    ax0.contour(X, Y, Z, contours, cmap='viridis', alpha = 0.3, zorder = -2)
    ax0.scatter(x=origin[0], y=origin[1], c='k', zorder = 4)

    ax0.grid(True, alpha=0.3)
    if set_aspect_equal:
        ax0.set_aspect('equal')

    ax0.set_xlim(x_min - 3, x_max + 3)
    ax0.set_ylim(y_min - 3, y_max + 3)
    ax0.legend()

    # Plot 2
    distr_height = 23

    ax1 = fig.add_subplot(gs[1, 1])
    ax1.set_title('Principal Component 2 distribution', color="#0069A1")
    # ax1.set_aspect('equal')
    ax1.scatter(x=Z_x, y=np.zeros_like(Z_x), c=dens_x, s=5, cmap='jet', zorder=1)
    
    x_1d_x = np.linspace(np.min(Z_x),np.max(Z_x),n+4)
    z_1d_x = np.concatenate(([0, z_1d_x[0]], z_1d_x, [z_1d_x[-1], 0]))
    
    ax1.plot(x_1d_x, z_1d_x / np.max(z_1d_x) * distr_height, color='b')
    ax1.fill(x_1d_x, z_1d_x / np.max(z_1d_x) * distr_height, alpha=0.3, hatch="//", color="#4876FF")
    ax1.set_xlim(np.min(Z_x) - 3, np.max(Z_x) + 3)
    ax1.set_ylim(-1, distr_height + 1)
    ax1.grid(True, alpha=0.3)

    # Plot 3
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Principal Component 1 distribution', color="#0069A1")
    # ax2.set_aspect('equal')
    ax2.scatter(x=Z_y, y=np.zeros_like(Z_y), c=dens_y, s=5, cmap='inferno', zorder=1)  

    x_1d_y = np.linspace(np.min(Z_y),np.max(Z_y),n+4)
    z_1d_y = np.concatenate(([0, z_1d_y[0]], z_1d_y, [z_1d_y[-1], 0])) 
      
    ax2.plot(x_1d_y, z_1d_y / np.max(z_1d_y) * distr_height, color='r')
    ax2.fill(x_1d_y, z_1d_y / np.max(z_1d_y) * distr_height, alpha=0.3, hatch="//", color="#FF5816")
    ax2.set_xlim(np.min(Z_y) - 3, np.max(Z_y) + 3)
    ax2.set_ylim(-1, distr_height + 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------------

def PCA_nd(data: np.ndarray, k: int, labels: np.ndarray = None):
    """
    Performs Principal Component Analysis (PCA) with automated multi-modal visualization.

    This function reduces high-dimensional data into a k-dimensional subspace using 
    a custom C-optimized SVD backend. It automatically selects the most appropriate 
    visualization style based on the value of 'k' and the presence of 'labels'.

    Parameters:
    -----------
    data : np.ndarray
        The input matrix of shape (n_samples, n_features).
    k : int
        The target number of principal components.
        - k=1: Plots 1D density distributions (KDE).
        - k=2: Plots 2D scatter with optional density contours.
        - k=3: Generates an interactive 3D Plotly scatter plot.
        - k>3: Returns the transformed array without visualization.
    labels : np.ndarray, optional
        Categorical labels for the samples. If provided, the function performs 
        Supervised Visualization (class-wise coloring and distributions). 
        If None, it performs Unsupervised Visualization (density-based coloring).

    Returns:
    --------
    pca_nd : np.ndarray
        The transformed data projected into the k-dimensional subspace.
        Shape: (n_samples, k).

    Visual Output Modes:
    --------------------
    - 1D: Class-wise Gaussian KDE curves (Supervised) or a global KDE (Unsupervised).
    - 2D: Scatter plot with class clusters (Supervised) or KDE density contours (Unsupervised).
    - 3D: Interactive Plotly environment with rotation/zoom and density/class mapping.

    Technical Process:
    ------------------
    1. Centering: Feature-wise mean subtraction.
    2. SVD Backend: Interfaces with 'libqr.so' using Householder Tridiagonalization 
       and QL iterations with Wilkinson shifts for Eigendecomposition.
    3. Projection: Z = X @ V[:, :k], mapping data onto the top 'k' eigenvectors.
    """

    data_centered = data - data.mean(axis=0)
    _, D, V = svd(data_centered)
    V, D = V.cpu().numpy(), D.cpu().numpy()
    pca_nd = data_centered @ V[:, :k]

    if k == 1:
        print('Principal Components Used: 1')
        pca_1d = pca_nd.ravel()
        grid_size = 500
        x_grid = np.linspace(np.min(pca_1d) - 2, np.max(pca_1d) + 2, grid_size)
        distr_height = 1
        
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # CASE 1: Classes are available -> Plot multiple Gaussians
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.colormaps['viridis'].resampled(len(unique_labels))
            
            for i, label_val in enumerate(unique_labels):

                # KDE
                class_data = pca_1d[labels == label_val]
                class_kde = gaussian_kde(class_data)
                class_density = class_kde(x_grid)

                norm_density = (class_density / np.max(class_density)) * distr_height
                color = colors(i)

                # PLOT
                ax.plot(x_grid, norm_density, label=f'Class {label_val}', color=color, lw=2, zorder = 0)
                ax.fill_between(x_grid, 0, norm_density, alpha=0.2, hatch="///", color=color, zorder = 0)
                ax.scatter(class_data, np.zeros_like(class_data), color=color, s=35, alpha=0.7, zorder = 10)
                ax.scatter(class_data, np.zeros_like(class_data), s=37, alpha=0.3, facecolors='none', edgecolors='k', zorder = 11)

            ax.set_title('Class-wise PCA Distributions (k=1)', fontsize=14)
            ax.legend()

        # CASE 2: No classes -> Plot single global Gaussian
        else:
            kde_func = gaussian_kde(pca_1d)
            density = kde_func(x_grid)
            norm_density = (density / np.max(density)) * distr_height
            
            ax.plot(x_grid, norm_density, color='r', lw=2)
            ax.fill_between(x_grid, 0, norm_density, alpha=0.1, hatch="//", color="#FF5816")
            ax.scatter(pca_1d, np.zeros_like(pca_1d), c=kde_func(pca_1d), cmap='inferno', s=17, alpha=0.9)
            ax.set_title('Global Data Distribution (k=1)', fontsize=14)

        ax.set_xlabel('Principle Component 1')
        ax.set_xlim(min(pca_1d), max(pca_1d))
        ax.set_ylim(-distr_height/10, distr_height + distr_height/10)
        ax.grid(True, alpha=0.2)
        plt.show()
        return pca_nd

    elif k == 2:
        print('Principal Components Used: 2')  

        fig, ax = plt.subplots(figsize=(10, 8))
        # CASE 1: Classes are available
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.colormaps['viridis'].resampled(len(unique_labels))

            for i, label_val in enumerate(unique_labels): 
                
                mask = (labels == label_val)
                ax.scatter(pca_nd[mask, 0], pca_nd[mask, 1], 
                           color=colors(i), label=f'Class {label_val}', s=70, alpha=0.7, edgecolors='k', linewidth=0.7, zorder = 10)
            ax.set_title('Class-wise PCA Distributions (k=2)', fontsize=14)
            ax.legend()

        # CASE 2: No classes
        else: 
            
            n = 300
            x = pca_nd[:,0]
            y = pca_nd[:,1]
            X_2d, Y_2d = np.meshgrid(np.linspace(np.min(x)*2,np.max(x)*2,n),
                            np.linspace(np.min(y)*2,np.max(y)*2,n))    
            positions = np.vstack([X_2d.ravel(), Y_2d.ravel()])

            kde = gaussian_kde(pca_nd.T)
            Z_2d = kde(positions).reshape(X_2d.shape)

            ax.scatter(pca_nd[:, 0], pca_nd[:, 1], c=kde(pca_nd.T), 
                       cmap='viridis', s=50, alpha=0.7, edgecolors='k')
            
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()

            ax.contour(X_2d, Y_2d, Z_2d, 30, cmap='viridis', alpha = 0.3, zorder = -2)

            ax.set_title('PCA (k=2)', fontsize=14)
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])            
        
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.grid(True, alpha=0.3)
        plt.show()
        return pca_nd

    elif k == 3:
        print('Principal Components Used: 3')

        df_pca = pd.DataFrame(pca_nd, columns=['PC1', 'PC2', 'PC3'])
        
        # CASE 1: Classes are available
        if labels is not None:
            df_pca['Class'] = labels.astype(str)
            
            fig = px.scatter_3d(
                df_pca, x='PC1', y='PC2', z='PC3',
                color='Class',
                title='Class-wise PCA Distributions (k=3)',
                labels={'Class': 'Classes:'},
                opacity=0.7,
                color_discrete_sequence=px.colors.qualitative.Vivid
            )

        # CASE 2: No classes
        else:        
            kde = gaussian_kde(pca_nd.T)
            density = kde(pca_nd.T)
            df_pca['Density'] = density

            fig = px.scatter_3d(
                df_pca, x='PC1', y='PC2', z='PC3',
                color='Density',
                title='PCA (k=3)',
                color_continuous_scale='Viridis',
                opacity=0.8
            )

        fig.update_scenes(
            xaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="lightgrey",
                showbackground=False,
                zerolinecolor="black",
            ),
            yaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="lightgrey",
                showbackground=False,
                zerolinecolor="black",
            ),
            zaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="lightgrey",
                showbackground=False,
                zerolinecolor="black",
            )
        )
        
        fig.update_traces(marker=dict(size=5, line=dict(width=1, color='Black')))

        # Make it square and larger
        fig.update_layout(
            width=700,               # Set width in pixels
            height=600,              # Set height to match width for a square
            margin=dict(l=0, r=0, b=0, t=40),
            plot_bgcolor='rgba(0,0,0,0)',
        )
        
        fig.show()
        return pca_nd

    else:
        print(f'Principal Components Used: {k}')
        print(pca_nd)
        return pca_nd
