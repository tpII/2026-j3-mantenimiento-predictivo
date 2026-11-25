"""
Espectro de frecuencia por clase (normal / anómalo) a partir de CSV del MPU6050.

Uso:
    python analisis_vibraciones.py carpeta_con_csv

Columnas esperadas: ax, ay, az, label, ventana, sesion   (opcional: pwm)
Etiquetas: 0 = apagado, 1 = normal, 2 = anómalo
Instalar: pip install pandas numpy scipy matplotlib
"""
import sys, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

FS = 1000                 # Hz (cambiar si tu muestreo real es otro)
NAMES = {0: "apagado", 1: "normal", 2: "anomalo"}

# ---------------------------------------------------------------- carga
carpeta = sys.argv[1] if len(sys.argv) > 1 else "."
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{carpeta}/*.csv"))],
               ignore_index=True)
df = df[df.label.isin([1, 2])]          # solo encendido: normal vs anómalo
print(df.groupby(["sesion", "label"]).size().unstack(fill_value=0))


def ventanas(g):
    """Agrupa las filas en ventanas de 256 muestras. Las ventanas NO son contiguas en el
    tiempo (hay una pausa entre capturas), por eso nunca se analiza a través de sus bordes."""
    n = len(g) // 256
    return g[["ax", "ay", "az"]].values[: n * 256].reshape(n, 256, 3)


# ---------------------------------------------------------------- 1) espectro por clase (todas las sesiones)
# Para cada clase: espectro de cada sesión (promedio de sus ventanas) y luego promedio de las
# sesiones con el MISMO peso, así una sesión larga no domina a las demás.
# Línea gruesa = promedio de sesiones | banda = mínimo-máximo entre sesiones | líneas finas = cada sesión.
COLORES = {1: "tab:blue", 2: "tab:red"}
fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for lab in (1, 2):
    por_sesion = []
    for sesion, g in df[df.label == lab].groupby("sesion"):
        X = ventanas(g)                         # (n_ventanas, 256, 3)
        X = X - X.mean(1, keepdims=True)        # quitar la gravedad (DC) en cada ventana
        f, P = welch(X, fs=FS, nperseg=X.shape[1], axis=1)
        por_sesion.append(P.mean(0))            # (frecuencias, 3 ejes)
    if not por_sesion:
        continue
    S = np.array(por_sesion)                    # (n_sesiones, frecuencias, 3 ejes)
    for k, eje in enumerate("xyz"):
        for s_ in S[:, :, k]:
            ax[k].semilogy(f, s_, color=COLORES[lab], alpha=0.25, lw=0.8)
        ax[k].fill_between(f, S[:, :, k].min(0), S[:, :, k].max(0), color=COLORES[lab], alpha=0.15)
        ax[k].semilogy(f, S[:, :, k].mean(0), color=COLORES[lab], lw=2,
                       label=f"{NAMES[lab]} ({len(S)} sesiones)")
        ax[k].set_title(f"Eje {eje}"); ax[k].set_xlabel("Hz")
ax[0].set_ylabel("PSD"); ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig("espectro_por_clase.png", dpi=130)

print("\nImagen guardada: espectro_por_clase.png")
