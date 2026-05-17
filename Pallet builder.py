import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image

st.set_page_config(page_title="Detector de Pallets", layout="wide")

st.title("Detector de Pallets y Rollos")

# Captura desde cámara del celular
camera_image = st.camera_input("Tomar foto")

# Variables globales
contours = {}
scale = 2

# Helper contour
def c(index):
    global contours
    return contours[index]

# Validar contour
def keep(contour):
    approx = cv2.approxPolyDP(
        contour,
        cv2.arcLength(contour, True) * 0.03,
        True
    )

    if (
        abs(cv2.contourArea(contour)) < 100
        or not cv2.isContourConvex(approx)
    ):
        return False

    return True

# Contar hijos
def count_children(index, h_, contour):

    if h_[0][index][2] < 0:
        return 0

    else:

        if keep(c(h_[0][index][2])):
            count = 1
        else:
            count = 0

        count += count_siblings(
            h_[0][index][2],
            h_,
            contour,
            True
        )

        return count

# Contar hermanos
def count_siblings(index, h_, contour, inc_children=False):

    count = 0

    # Adelante
    p_ = h_[0][index][0]

    while p_ > 0:

        if keep(c(p_)):
            count += 1

        p_ = h_[0][p_][0]

    # Atrás
    n = h_[0][index][1]

    while n > 0:

        if keep(c(n)):
            count += 1

        if inc_children:
            count += count_children(n, h_, contour)

        n = h_[0][n][1]

    return count

# Calcular ángulo
def angle(pt1, pt2, pt0):

    dx1 = pt1[0][0] - pt0[0][0]
    dy1 = pt1[0][1] - pt0[0][1]

    dx2 = pt2[0][0] - pt0[0][0]
    dy2 = pt2[0][1] - pt0[0][1]

    return float(
        (dx1 * dx2 + dy1 * dy2)
    ) / math.sqrt(
        float((dx1 * dx1 + dy1 * dy1))
        * (dx2 * dx2 + dy2 * dy2)
        + 1e-10
    )

# Si se tomó foto
if camera_image is not None:

    # Convertir imagen
    image = Image.open(camera_image)
    frame = np.array(image)

    # RGB → BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    palletCount = 0

    # Escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Bordes
    canny = cv2.Canny(gray, 80, 240)

    # Contornos
    contours, hierarchy = cv2.findContours(
        canny,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Guardar global
    globals()['contours'] = contours

    maxRolls = []

    for i in range(len(contours)):

        approx = cv2.approxPolyDP(
            contours[i],
            cv2.arcLength(contours[i], True) * 0.03,
            True
        )

        # Ignorar pequeños
        if (
            abs(cv2.contourArea(contours[i])) < 100
            or not cv2.isContourConvex(approx)
        ):
            continue

        # Cuadrados / rectángulos
        if len(approx) == 4:

            rect = cv2.minAreaRect(contours[i])

            box = cv2.boxPoints(rect)
            box = box.astype(int)

            cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)

            x, y, w, h = cv2.boundingRect(contours[i])

            parentNum = hierarchy[0][i][3]

            if parentNum > -1:

                chi = count_children(
                    i,
                    hierarchy,
                    contours[i]
                )

                palletCount += 1

                maxRolls.append(chi)

                cv2.putText(
                    frame,
                    f'Pallet {palletCount} ({chi} rolls)',
                    (x, y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        # Círculos
        else:

            area = cv2.contourArea(contours[i])

            x, y, w, h = cv2.boundingRect(contours[i])

            radius = w / 2

            if (
                abs(1 - (float(w) / h)) <= 0.2
                and abs(
                    1 - (
                        area / (math.pi * radius * radius)
                    )
                ) <= 0.2
            ):

                (cx, cy), radius1 = cv2.minEnclosingCircle(
                    contours[i]
                )

                center1 = (int(cx), int(cy))

                cv2.circle(
                    frame,
                    center1,
                    int(radius1),
                    (0, 255, 0),
                    2
                )

    # Mostrar resultado
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    st.subheader("Resultado")

    st.image(
        frame_rgb,
        use_container_width=True
    )

    st.subheader("Canny")

    st.image(
        canny,
        clamp=True,
        use_container_width=True
    )

    st.success(f"Pallets detectados: {palletCount}")
