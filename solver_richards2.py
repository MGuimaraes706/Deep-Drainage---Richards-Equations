"""Solver simplificado para o caso estacionário da Equação de Richards."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

import base
import config


# ---------------------------------------------------------------------------
# Utilidades de log (recicla o arquivo textual do pipeline clássico)
# ---------------------------------------------------------------------------
def print_log(*args, **kwargs) -> None:
    text = " ".join(str(a) for a in args)
    print(*args, **kwargs)
    with open("saida_terminal_pipeline.txt", "a", encoding="utf-8") as f_log:
        f_log.write(text + "\n")


# ---------------------------------------------------------------------------
# Caso estacionário 1D
# ---------------------------------------------------------------------------
def montar_sistema_1d(
    altura: float,
    n_pontos: int,
    h_topo: float,
    h_fundo: float,
    ks: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discretiza d/dz[K (dh/dz + 1)] = 0 assumindo K=Ks constante."""

    z = np.linspace(0.0, altura, n_pontos)
    dz = z[1] - z[0]

    # Usamos um operador de segunda derivada para o termo dh/dz, assumindo K constante.
    main = -2.0 * np.ones(n_pontos)
    off = np.ones(n_pontos - 1)
    lap = diags([off, main, off], offsets=[-1, 0, 1], format="csr") / dz**2

    # Termo da gravidade: d/dz(K) = 0, então sobra apenas o gradiente imposto.
    # O RHS representa a derivada da carga geodésica (1). Como K é constante, fica -K * d(1)/dz = 0.
    rhs = np.zeros(n_pontos)

    # Condições de contorno Dirichlet
    lap = lap.tolil()
    lap[0, :] = 0.0
    lap[0, 0] = 1.0
    rhs[0] = h_topo
    lap[-1, :] = 0.0
    lap[-1, -1] = 1.0
    rhs[-1] = h_fundo

    return z, lap.tocsr() * ks, rhs


def resolver_1d() -> None:
    params = base.MATERIAIS[base.TAG_SOLO_INFERIOR]
    altura = config.PROFUNDIDADE_DOMINIO if hasattr(config, "PROFUNDIDADE_DOMINIO") else 6.0
    n_pontos = 51
    h_topo = -2.0
    h_fundo = 0.0
    ks = params["ks"]

    print_log("=" * 72)
    print_log("== Caso Estacionário 1D da Equação de Richards (K constante) ==")
    print_log("=" * 72)
    print_log(f"Altura da coluna: {altura:.2f} m | Pontos: {n_pontos}")
    print_log(f"Ks utilizado: {ks:.2e} m/s")
    print_log(f"Cond. topo: {h_topo:.2f} m | Cond. fundo: {h_fundo:.2f} m")

    z, matriz, rhs = montar_sistema_1d(altura, n_pontos, h_topo, h_fundo, ks)
    h_sol = spsolve(matriz, rhs)

    gradiente = np.gradient(h_sol, z)
    fluxo = -ks * (gradiente + 1.0)  # q = -K (dh/dz + 1)

    print_log("\nPerfil obtido (primeiros 5 pontos):")
    for zi, hi, qi in zip(z[:5], h_sol[:5], fluxo[:5]):
        print_log(f"z = {zi:5.2f} m | h = {hi:6.3f} m | q = {qi: .3e} m/s")

    print_log("...")
    print_log(f"z = {z[-1]:5.2f} m | h = {h_sol[-1]:6.3f} m | q = {fluxo[-1]: .3e} m/s")

    if not os.path.exists("resultados"):
        os.makedirs("resultados")
    np.savez(
        "resultados/caso_estacionario_1d.npz",
        z=z,
        h=h_sol,
        fluxo=fluxo,
        ks=ks,
    )
    print_log("\nResultados salvos em resultados/caso_estacionario_1d.npz")
    print_log("=" * 72)


if __name__ == "__main__":
    resolver_1d()

    print_log(
        "\nObservação: este solver trata apenas o problema 1D estacionário. "
        "Extensões para 2D podem reutilizar a estrutura atual trocando o operador "
        "discreto laplaciano por uma montagem FEM com malha triangular."
    )


