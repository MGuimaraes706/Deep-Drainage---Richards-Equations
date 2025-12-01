"""Gera figuras sintéticas com o comportamento esperado das soluções."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402

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
    plt.savefig(FIG_DIR / "estacionario_1d_linear.png", dpi=200)
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
    plt.savefig(FIG_DIR / "estacionario_2d_perfil.png", dpi=200)
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
    plt.savefig(FIG_DIR / "estacionario_2d_contorno.png", dpi=200)
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
    plt.savefig(FIG_DIR / "estacionario_2d_superficie.png", dpi=200)
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
        plt.savefig(FIG_DIR / nome, dpi=200)
        plt.close()


def salvar_elemento_1d() -> None:
    plt.figure(figsize=(4, 1.8))
    plt.plot([0, 1], [0, 0], "k-", linewidth=2)
    plt.scatter([0, 1], [0, 0], color="black")
    plt.text(0, 0.1, "N1", ha="center", fontsize=12)
    plt.text(1, 0.1, "N2", ha="center", fontsize=12)
    plt.text(0, -0.15, "z_i", ha="center", fontsize=11)
    plt.text(1, -0.15, "z_j", ha="center", fontsize=11)
    plt.arrow(0, -0.35, 1, 0, head_width=0.03, head_length=0.04, fc="gray", ec="gray")
    plt.text(0.5, -0.45, "L_e", ha="center", fontsize=11, color="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "elemento_1d.png", dpi=200)
    plt.close()


def salvar_elemento_2d() -> None:
    coords = np.array([[0, 0], [1, 0], [0.3, 0.9]])
    tri = plt.Polygon(coords, fill=False, linewidth=2)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.add_patch(tri)
    ax.scatter(coords[:, 0], coords[:, 1], color="black")
    labels = ["i", "j", "k"]
    for (x, y), lbl in zip(coords, labels):
        ax.text(x, y + 0.05, f"N_{lbl}", ha="center", fontsize=12)
    ax.text(0.15, 0.2, "B", fontsize=12, color="gray")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "elemento_2d.png", dpi=200)
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
