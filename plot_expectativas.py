"""Gera figuras sintéticas com o comportamento esperado das soluções."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402

# Configurações globais para figuras acadêmicas de alta resolução
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'lines.linewidth': 2.5,
    'savefig.dpi': 400,
    'savefig.bbox': 'tight'
})

FIG_DIR = Path("figuras")
FIG_DIR.mkdir(exist_ok=True)


def salvar_1d_linear() -> None:
    z = np.linspace(0.0, 6.0, 60)
    h = np.linspace(-2.0, 0.0, z.size)
    plt.figure(figsize=(4, 5))
    plt.plot(h, z, color="#1f77b4", linewidth=2)
    plt.gca().invert_yaxis()
    plt.xlabel("Carga hidráulica h (m)")
    plt.ylabel("Profundidade z (m)")
    plt.title("Perfil 1D Estacionário (esperado)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "estacionario_1d_linear.png", dpi=400)
    plt.close()


def _campo_2d(base_level: float = -1.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 10.0, 80)
    y = np.linspace(0.0, 6.0, 60)
    X, Y = np.meshgrid(x, y)
    H = base_level - 0.05 * X + 0.15 * (Y - 3.0)
    return X, Y, H


def salvar_perfil_linear() -> None:
    X, Y, H = _campo_2d()
    idx_central = X.shape[1] // 2
    plt.figure(figsize=(5, 3))
    plt.plot(H[:, idx_central], Y[:, idx_central], color="#d62728", linewidth=2)
    plt.gca().invert_yaxis()
    plt.xlabel("Carga h (m)")
    plt.ylabel("Cota y (m)")
    plt.title("Perfil vertical no plano 2D")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "estacionario_2d_perfil.png", dpi=400)
    plt.close()


def salvar_contorno() -> None:
    X, Y, H = _campo_2d()
    plt.figure(figsize=(5.5, 3.5))
    cont = plt.contourf(X, Y, H, levels=20, cmap="viridis")
    plt.colorbar(cont, label="h (m)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Curvas de nível esperadas (2D estacionário)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "estacionario_2d_contorno.png", dpi=400)
    plt.close()


def salvar_superficie() -> None:
    X, Y, H = _campo_2d()
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, H, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("h (m)")
    ax.set_title("Superfície de carga prevista (2D)")
    fig.colorbar(surf, shrink=0.6, aspect=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "estacionario_2d_superficie.png", dpi=400)
    plt.close()


def salvar_transientes() -> None:
    X, Y, H_base = _campo_2d(-1.4)
    h_t0 = H_base - 0.4 * np.exp(-(Y / 3.0))
    h_t24 = H_base + 0.2 * (1.0 - np.exp(-(Y / 3.0)))
    for campo, nome, titulo in [
        (h_t0, "transiente_t0.png", "t = 0 h"),
        (h_t24, "transiente_t24.png", "t = 24 h"),
    ]:
        plt.figure(figsize=(5.5, 3.5))
        img = plt.imshow(
            campo,
            extent=(X.min(), X.max(), Y.max(), Y.min()),
            cmap="plasma",
            aspect="auto",
        )
        plt.colorbar(img, label="h (m)")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title(f"Distribuição transiente esperada ({titulo})")
        plt.tight_layout()
        plt.savefig(FIG_DIR / nome, dpi=400)
        plt.close()


def salvar_elemento_1d() -> None:
    plt.figure(figsize=(5, 2))
    plt.plot([0, 1], [0, 0], color="#1f77b4", linewidth=3, zorder=2)
    plt.scatter([0, 1], [0, 0], s=150, color="#d62728", zorder=3, edgecolors='black')
    plt.text(0, 0.1, r"$N_1(z)$", ha="center", fontsize=14)
    plt.text(1, 0.1, r"$N_2(z)$", ha="center", fontsize=14)
    plt.text(0, -0.15, r"$z_i$", ha="center", fontsize=14)
    plt.text(1, -0.15, r"$z_j$", ha="center", fontsize=14)
    
    # Seta para L_e
    plt.annotate('', xy=(1, -0.35), xytext=(0, -0.35),
                 arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    plt.text(0.5, -0.50, r"$L_e = z_j - z_i$", ha="center", fontsize=13, color="#555555")
    
    plt.ylim(-0.7, 0.3)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "elemento_1d.png", dpi=400)
    plt.close()


def salvar_elemento_2d() -> None:
    coords = np.array([[0, 0], [1, 0], [0.3, 0.9]])
    # Triângulo preenchido para ficar mais bonito
    tri = plt.Polygon(coords, facecolor="#e0ecf4", edgecolor="#1f77b4", linewidth=3, zorder=2)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.add_patch(tri)
    
    # Nós grandes e vermelhos
    ax.scatter(coords[:, 0], coords[:, 1], s=150, color="#d62728", zorder=3, edgecolors='black')
    
    labels = ["i", "j", "k"]
    alignments = [('right', 'top'), ('left', 'top'), ('center', 'bottom')]
    offsets = [(-0.05, -0.05), (0.05, -0.05), (0.0, 0.05)]
    
    for (x, y), lbl, (ha, va), (ox, oy) in zip(coords, labels, alignments, offsets):
        ax.text(x + ox, y + oy, rf"$N_{lbl}(x,y)$", ha=ha, va=va, fontsize=14)
        
    ax.text(0.4, 0.3, r"$\Omega_e$", fontsize=16, color="#1f77b4", ha="center", va="center")
    
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "elemento_2d.png", dpi=400)
    plt.close()


def main() -> None:
    salvar_1d_linear()
    salvar_perfil_linear()
    salvar_contorno()
    salvar_superficie()
    salvar_transientes()
    salvar_elemento_1d()
    salvar_elemento_2d()


if __name__ == "__main__":
    main()
