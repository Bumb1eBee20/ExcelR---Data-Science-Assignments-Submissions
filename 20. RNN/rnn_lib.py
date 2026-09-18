"""
Minimal, from-scratch implementations of a Vanilla RNN, an LSTM, and a GRU
for many-to-one sequence regression, trained with manual backpropagation
through time (BPTT) and the Adam optimizer.

Written from scratch in NumPy because this execution environment has no
TensorFlow / PyTorch installed and no internet access to install them.
"""
import numpy as np

np.random.seed(42)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def xavier(shape):
    fan_in = shape[1]
    limit = np.sqrt(1.0 / fan_in)
    return np.random.uniform(-limit, limit, size=shape)


class BaseRNNModel:
    """Shared training loop (Adam optimizer) for all three cell types."""

    def init_adam(self):
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0

    def adam_step(self, grads, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, clip=5.0):
        self.t += 1
        for k in self.params:
            g = np.clip(grads[k], -clip, clip)
            self.m[k] = beta1 * self.m[k] + (1 - beta1) * g
            self.v[k] = beta2 * self.v[k] + (1 - beta2) * (g ** 2)
            m_hat = self.m[k] / (1 - beta1 ** self.t)
            v_hat = self.v[k] / (1 - beta2 ** self.t)
            self.params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def predict(self, X):
        """X: (n_samples, window, n_features) -> (n_samples,) predictions"""
        preds = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            preds[i], _ = self.forward(X[i])
        return preds

    def fit(self, X, y, X_val=None, y_val=None, epochs=100, lr=0.01, verbose=False, patience=15):
        self.init_adam()
        n = X.shape[0]
        history = {"loss": [], "val_loss": []}
        best_val = np.inf
        best_params = {k: v.copy() for k, v in self.params.items()}
        wait = 0
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            epoch_loss = 0.0
            for i in idx:
                y_hat, cache = self.forward(X[i])
                loss = 0.5 * (y_hat - y[i]) ** 2
                epoch_loss += loss
                grads = self.backward(cache, y_hat, y[i])
                self.adam_step(grads, lr=lr)
            epoch_loss /= n
            history["loss"].append(epoch_loss)

            if X_val is not None:
                val_pred = self.predict(X_val)
                val_loss = np.mean(0.5 * (val_pred - y_val) ** 2)
                history["val_loss"].append(val_loss)
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_params = {k: v.copy() for k, v in self.params.items()}
                    wait = 0
                else:
                    wait += 1
                if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                    print(f"Epoch {epoch+1:3d}/{epochs} - loss: {epoch_loss:.5f} - val_loss: {val_loss:.5f}")
                if wait >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1} (best val_loss={best_val:.5f})")
                    break
            else:
                if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                    print(f"Epoch {epoch+1:3d}/{epochs} - loss: {epoch_loss:.5f}")

        if X_val is not None:
            self.params = best_params
        return history


class SimpleRNN(BaseRNNModel):
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        self.input_size = input_size
        H, D = hidden_size, input_size
        self.params = {
            "Wx": xavier((H, D)),
            "Wh": xavier((H, H)),
            "bh": np.zeros(H),
            "Wy": xavier((1, H)),
            "by": np.zeros(1),
        }

    def forward(self, x_seq):
        H = self.hidden_size
        T = x_seq.shape[0]
        h = np.zeros(H)
        hs = {0: h}
        for t in range(1, T + 1):
            x_t = x_seq[t - 1]
            a = self.params["Wx"] @ x_t + self.params["Wh"] @ hs[t - 1] + self.params["bh"]
            hs[t] = np.tanh(a)
        y_hat = (self.params["Wy"] @ hs[T] + self.params["by"])[0]
        cache = (x_seq, hs, T)
        return y_hat, cache

    def backward(self, cache, y_hat, y_true):
        x_seq, hs, T = cache
        H = self.hidden_size
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        dy = (y_hat - y_true)
        grads["Wy"] += dy * hs[T].reshape(1, -1)
        grads["by"] += np.array([dy])
        dh = self.params["Wy"].T.flatten() * dy
        for t in range(T, 0, -1):
            da = dh * (1 - hs[t] ** 2)
            grads["Wx"] += np.outer(da, x_seq[t - 1])
            grads["Wh"] += np.outer(da, hs[t - 1])
            grads["bh"] += da
            dh = self.params["Wh"].T @ da
        return grads


class LSTM(BaseRNNModel):
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        self.input_size = input_size
        H, D = hidden_size, input_size
        Z = H + D
        self.params = {
            "Wf": xavier((H, Z)), "bf": np.ones(H),   # forget bias init 1 (helps training)
            "Wi": xavier((H, Z)), "bi": np.zeros(H),
            "Wo": xavier((H, Z)), "bo": np.zeros(H),
            "Wg": xavier((H, Z)), "bg": np.zeros(H),
            "Wy": xavier((1, H)), "by": np.zeros(1),
        }

    def forward(self, x_seq):
        H = self.hidden_size
        T = x_seq.shape[0]
        h = np.zeros(H)
        c = np.zeros(H)
        cache_t = {0: (h, c)}
        gates = {}
        for t in range(1, T + 1):
            x_t = x_seq[t - 1]
            h_prev, c_prev = cache_t[t - 1]
            z = np.concatenate([h_prev, x_t])
            f = sigmoid(self.params["Wf"] @ z + self.params["bf"])
            i = sigmoid(self.params["Wi"] @ z + self.params["bi"])
            o = sigmoid(self.params["Wo"] @ z + self.params["bo"])
            g = np.tanh(self.params["Wg"] @ z + self.params["bg"])
            c_new = f * c_prev + i * g
            h_new = o * np.tanh(c_new)
            cache_t[t] = (h_new, c_new)
            gates[t] = (z, f, i, o, g, c_prev)
        y_hat = (self.params["Wy"] @ cache_t[T][0] + self.params["by"])[0]
        cache = (x_seq, cache_t, gates, T)
        return y_hat, cache

    def backward(self, cache, y_hat, y_true):
        x_seq, cache_t, gates, T = cache
        H = self.hidden_size
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        dy = (y_hat - y_true)
        h_T = cache_t[T][0]
        grads["Wy"] += dy * h_T.reshape(1, -1)
        grads["by"] += np.array([dy])
        dh_next = self.params["Wy"].T.flatten() * dy
        dc_next = np.zeros(H)
        for t in range(T, 0, -1):
            z, f, i, o, g, c_prev = gates[t]
            h_t, c_t = cache_t[t]
            dh = dh_next
            tanh_c = np.tanh(c_t)
            do = dh * tanh_c
            dc = dc_next + dh * o * (1 - tanh_c ** 2)
            df = dc * c_prev
            di = dc * g
            dg = dc * i
            dc_prev = dc * f

            da_f = df * f * (1 - f)
            da_i = di * i * (1 - i)
            da_o = do * o * (1 - o)
            da_g = dg * (1 - g ** 2)

            grads["Wf"] += np.outer(da_f, z); grads["bf"] += da_f
            grads["Wi"] += np.outer(da_i, z); grads["bi"] += da_i
            grads["Wo"] += np.outer(da_o, z); grads["bo"] += da_o
            grads["Wg"] += np.outer(da_g, z); grads["bg"] += da_g

            dz = (self.params["Wf"].T @ da_f + self.params["Wi"].T @ da_i +
                  self.params["Wo"].T @ da_o + self.params["Wg"].T @ da_g)
            dh_next = dz[:H]
            dc_next = dc_prev
        return grads


class GRU(BaseRNNModel):
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        self.input_size = input_size
        H, D = hidden_size, input_size
        Z = H + D
        self.params = {
            "Wr": xavier((H, Z)), "br": np.zeros(H),
            "Wz": xavier((H, Z)), "bz": np.zeros(H),
            "Wh": xavier((H, Z)), "bhh": np.zeros(H),
            "Wy": xavier((1, H)), "by": np.zeros(1),
        }

    def forward(self, x_seq):
        H = self.hidden_size
        T = x_seq.shape[0]
        h = np.zeros(H)
        hs = {0: h}
        gates = {}
        for t in range(1, T + 1):
            x_t = x_seq[t - 1]
            h_prev = hs[t - 1]
            z_in = np.concatenate([h_prev, x_t])
            r = sigmoid(self.params["Wr"] @ z_in + self.params["br"])
            zg = sigmoid(self.params["Wz"] @ z_in + self.params["bz"])
            z_tilde_in = np.concatenate([r * h_prev, x_t])
            h_tilde = np.tanh(self.params["Wh"] @ z_tilde_in + self.params["bhh"])
            h_new = (1 - zg) * h_prev + zg * h_tilde
            hs[t] = h_new
            gates[t] = (z_in, r, zg, z_tilde_in, h_tilde, h_prev)
        y_hat = (self.params["Wy"] @ hs[T] + self.params["by"])[0]
        cache = (x_seq, hs, gates, T)
        return y_hat, cache

    def backward(self, cache, y_hat, y_true):
        x_seq, hs, gates, T = cache
        H = self.hidden_size
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        dy = (y_hat - y_true)
        h_T = hs[T]
        grads["Wy"] += dy * h_T.reshape(1, -1)
        grads["by"] += np.array([dy])
        dh_next = self.params["Wy"].T.flatten() * dy
        for t in range(T, 0, -1):
            z_in, r, zg, z_tilde_in, h_tilde, h_prev = gates[t]
            dh = dh_next

            d_zg = dh * (h_tilde - h_prev)
            d_htilde = dh * zg
            dh_prev_direct = dh * (1 - zg)

            da_htilde = d_htilde * (1 - h_tilde ** 2)
            grads["Wh"] += np.outer(da_htilde, z_tilde_in)
            grads["bhh"] += da_htilde
            dz_tilde = self.params["Wh"].T @ da_htilde
            d_rh = dz_tilde[:H]

            dr = d_rh * h_prev
            dh_prev_from_r_term = d_rh * r

            da_r = dr * r * (1 - r)
            da_zg = d_zg * zg * (1 - zg)

            grads["Wr"] += np.outer(da_r, z_in)
            grads["br"] += da_r
            grads["Wz"] += np.outer(da_zg, z_in)
            grads["bz"] += da_zg

            dz_concat = self.params["Wr"].T @ da_r + self.params["Wz"].T @ da_zg
            dh_prev_from_gates = dz_concat[:H]

            dh_next = dh_prev_direct + dh_prev_from_r_term + dh_prev_from_gates
        return grads
