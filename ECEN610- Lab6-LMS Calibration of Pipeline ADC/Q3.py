import numpy as np
import matplotlib.pyplot as plt
V = 1.0
B = 13
S = 6
R = 2
Fs = 500e6
f = 200e6
T = 1e-6
N = int(Fs * T)
K = 4
alpha = 0.01
I = 500
t = np.linspace(0, T, N, endpoint=False)
x = 0.5 * np.sin(2 * np.pi * f * t)
def ref_adc(x, B=16, V=1.0):
    L = 2 ** B
    d = V / L
    y = np.round((x + V / 2) / d)
    y = np.clip(y, 0, L - 1)
    z = (y * d) - (V / 2)
    return y, z
def mdac(x, V=1.0, e=0.01, n=1e-3, c=1.0):
    th = np.array([-0.25, -0.125, 0, 0.125, 0.25])
    lv = np.array([-0.5, -0.25, 0, 0.25, 0.5, 0.5])
    idx = np.digitize(x, th, right=True)
    d = lv[idx]
    G = 2 ** R
    g = G * (1 + e) * c
    r = (x - d) * g
    r += np.random.normal(0, n, size=x.shape)
    r = np.clip(r, -V / 2, V / 2)
    return idx, r
def pipeline(x, S, R, V, c_arr):
    d_all = []
    r_all = []
    r = x.copy()
    for i in range(S):
        e = 0.01 * (1 + 0.1 * i)
        c = c_arr[i] if i < len(c_arr) else 1.0
        d, r = mdac(r, V, e, 1e-3, c)
        d_all.append(d)
        r_all.append(r.copy())
    return d_all, r_all
def snr_calc(y, Fs):
    N = len(y)
    Y = np.fft.fft(y * np.hanning(N))
    mag = np.abs(Y[:N // 2])
    p_sig = mag[np.argmax(mag)] ** 2
    mask = np.ones(len(mag), dtype=bool)
    mask[[0, np.argmax(mag)]] = False
    p_n = np.sum(mag[mask] ** 2)
    return 10 * np.log10(p_sig / p_n)
c_arr = np.ones(K)
err = []
w_hist = [[] for _ in range(K)]
snr_hist = []
snr_lim = 6.02 * B + 1.76

_, y_ref = ref_adc(x, B=16, V=V)

for i in range(I):
    d_all, r_all = pipeline(x, S, R, V, c_arr)
    y_dig = np.zeros(N, dtype=int)
    for j in range(S):
        y_dig += d_all[j] * (2 ** (R * (S - j - 1)))
    y_out = (y_dig / (2 ** B)) * V - (V / 2)
    e = y_ref - y_out
    err.append(np.mean(e ** 2))
    for j in range(K):
        r_prev = x if j == 0 else r_all[j - 1]
        c_arr[j] += alpha * np.mean(e * r_prev)
        w_hist[j].append(c_arr[j])
    snr = snr_calc(y_out, Fs)
    snr_hist.append(snr)
    if snr >= snr_lim - 1:
        iters = i + 1
        break
else:
    iters = I
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(err, color='green')
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.title("Error")
plt.grid(True)
plt.subplot(3, 1, 2)
for j in range(K):
    plt.plot(w_hist[j], color='green', label=f"S{j+1}")
plt.xlabel("Iteration")
plt.ylabel("Weights")
plt.title("Weights")
plt.legend()
plt.grid(True)
plt.subplot(3, 1, 3)
plt.plot(snr_hist, color='green')
plt.xlabel("Iteration")
plt.ylabel("SNR (dB)")
plt.title("SNR")
plt.grid(True)
plt.tight_layout()
plt.savefig("lms_convergence.png")
print(f"Iterations: {iters}")
print(f"Weights: {c_arr}")
