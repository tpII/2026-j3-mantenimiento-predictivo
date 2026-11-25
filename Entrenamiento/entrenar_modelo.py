#!/usr/bin/env python3
"""
Entrenamiento de la 1D-CNN con los datos REALES del MPU6050 (CSV generados por crear_csv.py).

Cambios respecto a la version anterior:
  * Cada "ventana" del CSV ahora es una CAPTURA continua (por ejemplo, 1024 muestras).
    Se corta en ventanas de 256 muestras con solapamiento del 50 % DENTRO de la misma captura.
  * 3 clases: 0 = apagado, 1 = normal, 2 = anomalo.
  * Evaluacion dejando una SESION afuera (leave-one-session-out): se entrena con las demas
    sesiones y se prueba con la que quedo afuera. Un archivo CSV = una sesion.

Uso:
    py -m pip install numpy pandas scikit-learn tensorflow
    py entrenar_modelo.py --datos carpeta_con_los_csv

Entrada : carpeta con los CSV (columnas ax, ay, az, label, ventana, sesion).
Salida  : modelo_int8.tflite, model.h (arreglo C para el ESP32), normalizacion.json
          y las metricas (matriz de confusion, reporte) en pantalla.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models

SEED = 42
MUESTRAS = 256          # muestras por ventana que ve la red (no tiene que ser igual a la captura)
PASO = 128              # salto entre ventanas dentro de una captura: 128 = solapamiento del 50 %
CANALES = 3             # ax, ay, az
NOMBRES = ["0: Apagado", "1: Normal", "2: Anomalo"]
N_CLASES = len(NOMBRES)
REMOVER_DC = True       # restar a cada ventana su promedio (quita la gravedad / la inclinacion)

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ---------------------------------------------------------------- 1. CARGA DE LOS CSV
def cargar_ventanas(carpeta):
    """
    Lee todos los CSV y devuelve:
      X         (N, 256, 3)  ventanas
      y         (N,)         etiqueta de cada ventana
      sesiones  (N,)         archivo (sesion) de cada ventana
      capturas  (N,)         id de la captura de la que salio cada ventana
    Las ventanas de una misma captura se solapan, asi que NUNCA deben quedar repartidas
    entre entrenamiento y prueba: por eso se guarda el id de captura.
    """
    archivos = sorted(glob.glob(os.path.join(carpeta, "*.csv")))
    if not archivos:
        raise SystemExit(f"No se encontraron CSV en '{carpeta}'.")
    X, y, sesiones, capturas = [], [], [], []
    n_cap, descartadas = 0, 0
    for ruta in archivos:
        df = pd.read_csv(ruta, comment="#")
        faltan = {"ax", "ay", "az", "label", "ventana"} - set(df.columns)
        if faltan:
            raise SystemExit(f"{ruta}: faltan las columnas {faltan}")
        sesion = os.path.basename(ruta)                 # cada archivo = una sesion
        claves = ["sesion", "ventana"] if "sesion" in df.columns else ["ventana"]
        for _, v in df.groupby(claves, sort=True):      # cada grupo = una captura continua
            etiqueta = int(v["label"].iloc[0])
            if v["label"].nunique() != 1 or len(v) < MUESTRAS or not 0 <= etiqueta < N_CLASES:
                descartadas += 1                        # captura incompleta, mezclada o con etiqueta rara
                continue
            datos = v[["ax", "ay", "az"]].to_numpy(dtype=np.float32)
            n_cap += 1
            for ini in range(0, len(datos) - MUESTRAS + 1, PASO):
                X.append(datos[ini:ini + MUESTRAS])
                y.append(etiqueta)
                sesiones.append(sesion)
                capturas.append(n_cap)
    if descartadas:
        print(f"AVISO: {descartadas} capturas descartadas (menos de {MUESTRAS} filas, etiquetas "
              f"mezcladas o etiqueta fuera de 0..{N_CLASES - 1}).")
    if not X:
        raise SystemExit("No quedo ninguna ventana valida.")
    return np.array(X), np.array(y), np.array(sesiones), np.array(capturas)


# ---------------------------------------------------------------- 2. DIVISIONES DE DATOS
def separar_validacion(idx, y, sesiones, capturas):
    """
    De los indices de entrenamiento aparta ~1/6 de las CAPTURAS (no de las ventanas) para
    validar y frenar el entrenamiento a tiempo. Capturas vecinas en el tiempo se parecen, asi
    que esta validacion es optimista: sirve solo para elegir cuando parar. La medida honesta
    es la de la sesion que queda afuera.
    """
    val_caps = set()
    for s, c in sorted(set(zip(sesiones[idx], y[idx]))):
        m = idx[(sesiones[idx] == s) & (y[idx] == c)]
        val_caps.update(np.unique(capturas[m])[5::6].tolist())
    es_val = np.array([capturas[i] in val_caps for i in idx])
    if not es_val.any():
        raise SystemExit("Hay muy pocas capturas para separar validacion. Grabar mas.")
    return idx[~es_val], idx[es_val]


def division_cronologica(y, sesiones, capturas):
    """Reparto 80/20 por capturas, en orden de grabacion, con una captura de separacion."""
    train, test = [], []
    for s, c in sorted(set(zip(sesiones, y))):
        m = np.where((sesiones == s) & (y == c))[0]
        caps = np.unique(capturas[m])
        corte = int(0.8 * len(caps))
        cap_tr = set(caps[:max(corte - 1, 0)].tolist())
        cap_te = set(caps[corte:].tolist())
        train += [i for i in m if capturas[i] in cap_tr]
        test += [i for i in m if capturas[i] in cap_te]
    return np.array(train), np.array(test)


def armar_pruebas(y, sesiones, capturas):
    """Lista de (nombre, indices_train, indices_test). Una prueba por cada sesion dejada afuera."""
    unicas = np.unique(sesiones)
    if len(unicas) >= 3:
        return [(f"sesion {s}", np.where(sesiones != s)[0], np.where(sesiones == s)[0])
                for s in unicas]
    print("AVISO: hay menos de 3 sesiones, no se puede dejar una sesion afuera. Se usa un reparto "
          "cronologico por capturas dentro de cada sesion: es una estimacion OPTIMISTA. "
          "Grabar mas sesiones (idealmente 3 o mas por clase) para una evaluacion confiable.")
    tr, te = division_cronologica(y, sesiones, capturas)
    return [("division cronologica", tr, te)]


# ---------------------------------------------------------------- 3. PREPROCESO
def preprocesar(X, escala=None):
    """Quita el promedio de cada ventana (opcional) y divide por una escala global."""
    X = X.copy()
    if REMOVER_DC:
        X -= X.mean(axis=1, keepdims=True)
    if escala is None:
        escala = float(X.std()) or 1.0                  # se calcula SOLO con train
    return X / escala, escala


# ---------------------------------------------------------------- 4. MODELO
def construir_modelo():
    return models.Sequential([
        layers.Input(shape=(MUESTRAS, CANALES)),
        layers.Conv1D(16, 3, activation="relu"),
        layers.MaxPooling1D(4),
        layers.Conv1D(32, 3, activation="relu"),
        layers.MaxPooling1D(4),
        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(N_CLASES, activation="softmax"),
    ])


def entrenar(X, y, sesiones, capturas, idx_train, epocas, verbose=0):
    """Entrena un modelo con las ventanas idx_train. Devuelve (modelo, escala, X de entrenamiento)."""
    tr, va = separar_validacion(idx_train, y, sesiones, capturas)
    X_tr, escala = preprocesar(X[tr])
    X_va, _ = preprocesar(X[va], escala)
    y_tr, y_va = y[tr], y[va]

    # Pesos por clase, por si algunas tienen menos ventanas que otras
    clases = np.unique(y_tr)
    pesos = {int(c): float(w) for c, w in
             zip(clases, compute_class_weight("balanced", classes=clases, y=y_tr))}

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    model = construir_modelo()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epocas,
              batch_size=32, class_weight=pesos, callbacks=[stop], verbose=verbose)
    return model, escala, X_tr


# ---------------------------------------------------------------- 5. CUANTIZACION int8
def convertir_int8(model, X_rep):
    """Cuantizacion entera completa: el modelo y sus entradas/salidas quedan en int8."""
    def representativo():
        for i in np.random.default_rng(SEED).permutation(len(X_rep))[:200]:
            yield [X_rep[i:i + 1].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representativo
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def evaluar_tflite(tflite_bytes, X, y):
    """Corre el modelo int8 con el interprete de TFLite y devuelve las predicciones y los parametros de entrada."""
    it = tf.lite.Interpreter(model_content=tflite_bytes)
    it.allocate_tensors()
    ent, sal = it.get_input_details()[0], it.get_output_details()[0]
    e_scale, e_zero = ent["quantization"]
    pred = []
    for x in X:
        q = np.clip(np.round(x / e_scale + e_zero), -128, 127).astype(np.int8)
        it.set_tensor(ent["index"], q[None, ...])
        it.invoke()
        pred.append(int(np.argmax(it.get_tensor(sal["index"])[0])))
    return np.array(pred), float(e_scale), int(e_zero)


def a_header(datos, ruta, nombre="model_data"):
    """Equivalente de 'xxd -i': escribe el .tflite como arreglo C para el ESP32."""
    with open(ruta, "w") as f:
        f.write(f"alignas(16) const unsigned char {nombre}[] = {{\n")
        for i in range(0, len(datos), 12):
            f.write("  " + ", ".join(f"0x{b:02x}" for b in datos[i:i + 12]) + ",\n")
        f.write(f"}};\nconst unsigned int {nombre}_len = {len(datos)};\n")


# ---------------------------------------------------------------- 6. INFORME
def informe(y, pred):
    """Matriz de confusion y reporte con todas las predicciones hechas sobre sesiones no vistas."""
    m = pred >= 0
    if not m.any():
        return
    etiquetas = sorted(set(y[m].tolist()) | set(pred[m].tolist()))
    print("Filas = clase real, columnas = clase predicha. Orden:",
          ", ".join(NOMBRES[c] for c in etiquetas))
    print(confusion_matrix(y[m], pred[m], labels=etiquetas))
    print(classification_report(y[m], pred[m], labels=etiquetas,
                                target_names=[NOMBRES[c] for c in etiquetas], zero_division=0))


# ---------------------------------------------------------------- PRINCIPAL
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default="dataset", help="carpeta con los CSV")
    ap.add_argument("--epocas", type=int, default=40)
    ap.add_argument("--salida", default=".", help="carpeta de salida")
    ap.add_argument("--sin-cv", action="store_true",
                    help="saltar la evaluacion por sesion y entrenar solo el modelo final (mas rapido)")
    args = ap.parse_args()

    X, y, sesiones, capturas = cargar_ventanas(args.datos)
    print(f"\n{len(X)} ventanas de {MUESTRAS}x{CANALES} (de {len(np.unique(capturas))} capturas) "
          f"en {len(np.unique(sesiones))} sesiones.")
    for c in range(N_CLASES):
        print(f"  {NOMBRES[c]}: {int(np.sum(y == c))} ventanas")
    faltan = [NOMBRES[c] for c in range(N_CLASES) if np.sum(y == c) == 0]
    if faltan:
        print("AVISO: no hay datos de:", ", ".join(faltan))
    if len(np.unique(y)) < 2:
        raise SystemExit("Hace falta al menos 2 clases para entrenar.")

    # ---- Evaluacion honesta: dejar una sesion afuera
    if not args.sin_cv:
        pred = np.full(len(y), -1)
        for nombre, tr_idx, te_idx in armar_pruebas(y, sesiones, capturas):
            print(f"\n--- Prueba con {nombre} (entrenando con el resto) ---")
            if len(np.unique(y[tr_idx])) < 2:
                print("   Saltada: el resto de los datos tiene una sola clase.")
                continue
            sin_ver = set(np.unique(y[te_idx]).tolist()) - set(np.unique(y[tr_idx]).tolist())
            if sin_ver:
                print("   AVISO: el entrenamiento no tiene datos de",
                      ", ".join(NOMBRES[c] for c in sorted(sin_ver)), "(esas ventanas fallaran).")
            model, escala, _ = entrenar(X, y, sesiones, capturas, tr_idx, args.epocas)
            X_te, _ = preprocesar(X[te_idx], escala)
            p = np.argmax(model.predict(X_te, verbose=0), axis=1)
            pred[te_idx] = p
            por_clase = [f"{NOMBRES[c]} {np.mean(p[y[te_idx] == c] == c):.3f}"
                         for c in range(N_CLASES) if np.any(y[te_idx] == c)]
            print(f"   Accuracy total {np.mean(p == y[te_idx]):.3f} | acierto por clase: "
                  + " | ".join(por_clase))
        print("\n=== RESULTADO DE LA EVALUACION (modelo float, datos no usados para entrenar) ===")
        informe(y, pred)

    # ---- Modelo final: todas las sesiones
    print("=== MODELO FINAL (entrenado con todas las sesiones) ===")
    todos = np.arange(len(y))
    model, escala, X_tr = entrenar(X, y, sesiones, capturas, todos, args.epocas, verbose=2)
    model.summary()

    tflite = convertir_int8(model, X_tr)
    muestra = np.random.default_rng(SEED).permutation(len(y))[:2000]
    X_m, _ = preprocesar(X[muestra], escala)
    p_float = np.argmax(model.predict(X_m, verbose=0), axis=1)
    p8, e_scale, e_zero = evaluar_tflite(tflite, X_m, y[muestra])
    print(f"\nCoincidencia entre el modelo float y el int8 en {len(muestra)} ventanas: "
          f"{np.mean(p_float == p8):.3f}")
    print("(Es solo para controlar la cuantizacion. NO es una medida de accuracy: "
          "esas ventanas ya se usaron para entrenar.)")

    os.makedirs(args.salida, exist_ok=True)
    ruta_tflite = os.path.join(args.salida, "modelo_int8.tflite")
    with open(ruta_tflite, "wb") as f:
        f.write(tflite)
    a_header(tflite, os.path.join(args.salida, "model.h"))
    with open(os.path.join(args.salida, "normalizacion.json"), "w") as f:
        json.dump({"remover_dc": REMOVER_DC, "escala": escala, "muestras": MUESTRAS,
                   "paso": PASO, "canales": CANALES, "clases": NOMBRES,
                   "entrada_scale": e_scale, "entrada_zero_point": e_zero}, f, indent=2)
    print(f"\nModelo int8: {len(tflite) / 1024:.1f} KB -> {ruta_tflite}, model.h y normalizacion.json")


if __name__ == "__main__":
    main()
