#!/usr/bin/env python3
"""
Crea el CSV del dataset a partir del firmware captura_256.ino.

Estados: 0 = apagado, 1 = normal, 2 = anomalo.

Uso:
    pip install pyserial
    python crear_csv.py --port COM5 --baud 460800

El --baud tiene que ser IGUAL al de Serial.begin() del ESP32.
Cerrar antes el Monitor Serie del Arduino IDE (el puerto no se comparte).
"""
import argparse
import csv
import re
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

import serial

MUESTRAS_POR_VENTANA = 1024
ETIQUETAS = {0: "apagado", 1: "normal", 2: "anomalo"}

# Una linea de datos valida: ax,ay,az,label  (numeros con decimales y una etiqueta entera)
NUM = r"-?\d+\.\d+"
LINEA_DATOS = re.compile(rf"^({NUM}),({NUM}),({NUM}),(\d)$")

AYUDA = """
Comandos:
  <etiqueta>              graba 1 ventana de 1024 muestras     ej.:  2
  <etiqueta> <cantidad>   graba varias ventanas seguidas      ej.:  2 20
  e                       muestra las etiquetas
  q                       salir
Etiquetas: 0 = apagado, 1 = normal, 2 = anomalo
"""


def leer_linea(ser):
    """Lee una linea del ESP32 como texto. Si hay bytes basura no rompe: los reemplaza."""
    return ser.readline().decode("utf-8", errors="replace").strip()


def esperar_prompt(ser, timeout):
    """Lee hasta ver 'ESPERANDO ENTRADA' (el ESP32 esta listo para otra etiqueta)."""
    fin = time.time() + timeout
    while time.time() < fin:
        if "ESPERANDO ENTRADA" in leer_linea(ser):
            return True
    return False


def capturar_ventana(ser, etiqueta):
    """
    Envia la etiqueta y devuelve la lista de 256 filas (ax, ay, az) si todo salio bien,
    o None si hubo algun problema (la ventana se descarta entera).
    """
    ser.reset_input_buffer()                       # tirar lo que haya quedado en el buffer
    ser.write(f"{etiqueta}\n".encode("ascii"))     # el firmware ignora el salto de linea

    filas, en_datos, error = [], False, None
    fin = time.time() + 10                         # una ventana dura ~0,3 s; 10 s es de sobra
    while time.time() < fin:
        linea = leer_linea(ser)
        if not linea:
            continue
        if linea.startswith("[ERROR"):
            error = linea
            break
        if linea.startswith("[OK]"):
            en_datos = True                        # desde aca vienen las 256 filas
            continue
        if en_datos:
            m = LINEA_DATOS.match(linea)
            if m:
                if int(m.group(4)) != etiqueta:
                    error = f"etiqueta inesperada en la linea: {linea}"
                    break
                filas.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
            elif "ESPERANDO ENTRADA" in linea or linea.startswith("---"):
                break                              # termino la transmision
            else:
                error = f"linea danada: {linea!r}"  # caracter perdido o corrupto
                break
            if len(filas) == MUESTRAS_POR_VENTANA:
                break

    if error is None and len(filas) != MUESTRAS_POR_VENTANA:
        error = f"llegaron {len(filas)} filas y se esperaban {MUESTRAS_POR_VENTANA}"
    esperar_prompt(ser, 2)                         # dejar al ESP32 listo para la siguiente
    if error:
        print(f"   Ventana descartada: {error}")
        return None
    return filas


def resumen(filas):
    """Estadisticas rapidas para ver a ojo si el dato tiene sentido."""
    partes = []
    for i, eje in enumerate(("ax", "ay", "az")):
        col = [f[i] for f in filas]
        partes.append(f"{eje} {mean(col):+7.3f} (±{pstdev(col):.3f})")
    modulo = mean((f[0] ** 2 + f[1] ** 2 + f[2] ** 2) ** 0.5 for f in filas)
    return "  ".join(partes) + f"  |a| = {modulo:.2f} m/s²"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="COM5, /dev/ttyUSB0, ...")
    ap.add_argument("--baud", type=int, default=460800, help="igual al Serial.begin() del ESP32")
    ap.add_argument("--out", default=None, help="archivo CSV de salida (por defecto, uno nuevo por sesion)")
    args = ap.parse_args()

    # Un archivo por sesion: sirve para separar entrenamiento y prueba por sesion mas adelante.
    sesion = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = Path(args.out or f"dataset_{sesion}.csv")

    ser = serial.Serial(args.port, args.baud, timeout=1)
    print(f"Conectando a {args.port} a {args.baud} baud... (se espera el reinicio del ESP32)")
    if not esperar_prompt(ser, 8):
        print("El ESP32 no respondio. Revisar puerto, baud y que tenga cargado captura_256.")
        return
    print(AYUDA)

    nuevo = not ruta.exists()
    ventana = 0
    with open(ruta, "a", newline="") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["ax", "ay", "az", "label", "ventana", "sesion"])
        try:
            while True:
                try:
                    cmd = input("> ").split()
                except (EOFError, KeyboardInterrupt):
                    break
                if not cmd:
                    continue
                if cmd[0].lower() == "q":
                    break
                if cmd[0].lower() == "e":
                    for k, v in ETIQUETAS.items():
                        print(f"  {k}: {v}")
                    continue
                if not (cmd[0].isdigit() and int(cmd[0]) in ETIQUETAS and
                        (len(cmd) == 1 or (len(cmd) == 2 and cmd[1].isdigit() and int(cmd[1]) > 0))):
                    print(AYUDA)
                    continue
                etiqueta = int(cmd[0])
                cantidad = int(cmd[1]) if len(cmd) == 2 else 1

                ok = 0
                for _ in range(cantidad):
                    filas = capturar_ventana(ser, etiqueta)
                    if filas is None:
                        continue
                    ventana += 1
                    ok += 1
                    for ax, ay, az in filas:
                        w.writerow([f"{ax:.4f}", f"{ay:.4f}", f"{az:.4f}", etiqueta, ventana, sesion])
                    f.flush()                      # guardar en disco en cada ventana
                    print(f"   Ventana {ventana} guardada. {resumen(filas)}")
                print(f"  {ok}/{cantidad} ventanas guardadas de '{ETIQUETAS[etiqueta]}'. "
                      f"Archivo: {ruta}")
        finally:
            ser.close()
    print(f"Listo. {ventana} ventanas en {ruta}")


if __name__ == "__main__":
    main()
