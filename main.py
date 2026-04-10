def main():
    import numpy as np
    import matplotlib.pyplot as plt

    # 参数
    period = 6
    length = 36
    window_size = 6
    stride = 1


    t = np.arange(length)
    signal = np.sin(2 * np.pi * t / period)

    # 提取滑动窗口
    windows = []
    idxs = []
    for start in range(0, length - window_size + 1, stride):
        w = signal[start:start + window_size].copy()
        windows.append(w)
        idxs.append(start)
    windows = np.stack(windows)  # shape (n_windows, window_size)

    # 方法1：对每个窗口按 (start % period) 旋转对齐后计算 f2（平方欧氏）距离
    def rotate_by_phase(window, phase, period):
        # phase: start index % period，窗口长度等于 period
        # 将窗口按 phase 旋转，使得相位 0 放到开头
        # rotation amount = (period - phase) % period
        r = (-phase) % period
        return np.roll(window, -r)

    n = windows.shape[0]
    f2_dist = np.zeros((n, n))
    for i in range(n):
        phase_i = idxs[i] % period
        wi = rotate_by_phase(windows[i], phase_i, period)
        for j in range(n):
            phase_j = idxs[j] % period
            wj = rotate_by_phase(windows[j], phase_j, period)
            f2_dist[i, j] = np.sum((wi - wj) ** 2)

    # 方法2：用原序列的自协函数（autocovariance）构造 Toeplitz 协方差矩阵，
    # 然后计算马氏距离（符合平稳假设时的自协方差矩阵）
    T = len(signal)
    x_mean = np.mean(signal)

    def autocov(h):
        # 无偏/有偏估计都可，这里按除以 (T-h) 的形式计算样本自协（与 README 描述一致）
        if h >= T:
            return 0.0
        return np.sum((signal[: T - h] - x_mean) * (signal[h:] - x_mean)) / (T - h)

    # 构造 Toeplitz 矩阵 Σ_{i,j} = γ(|i-j|)
    gam = [autocov(h) for h in range(window_size)]
    cov_toeplitz = np.zeros((window_size, window_size))
    for i in range(window_size):
        for j in range(window_size):
            cov_toeplitz[i, j] = gam[abs(i - j)]

    # 特征值截断/正则化以保证协方差为正定（避免数值上出现负特征值）
    eps = 1e-8
    eigvals, eigvecs = np.linalg.eigh(cov_toeplitz)
    eigvals_reg = np.where(eigvals < eps, eps, eigvals)
    cov_reg = eigvecs @ np.diag(eigvals_reg) @ eigvecs.T
    # 额外加一个极小的对角项以增强数值稳定性
    cov_reg += 1e-12 * np.eye(window_size)
    inv_cov = np.linalg.inv(cov_reg)

    maha_dist = np.zeros((n, n))
    for i in range(n):
        xi = windows[i]
        for j in range(n):
            xj = windows[j]
            diff = xi - xj
            maha_dist[i, j] = diff.T @ inv_cov @ diff

    # 绘图：信号、f2 距离矩阵、马氏距离矩阵
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(t, signal, marker='o')
    axes[0].set_title('periodic signal (period=%d)' % period)
    axes[0].set_xlabel('index')

    im1 = axes[1].imshow(f2_dist, cmap='viridis', vmin=0, vmax=1.0)
    axes[1].set_title('rotated f2 distance')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(maha_dist, cmap='viridis')
    axes[2].set_title('Mahalanobis distance (cov from windows)')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
