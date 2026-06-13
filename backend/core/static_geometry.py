#static_geometry.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.features import (
    _split_detection_result,
    _collect_detected_hands,
    _select_hands_for_class,
)


# ============================================================
# Validación geométrica estática - Abecedario LSA V2
# ============================================================
#
# Objetivo:
#   Complementar al clasificador GRU con reglas geométricas finas
#   para letras donde importan dedos específicos, distancias,
#   ángulos, orientación de la mano, ubicación facial/corporal
#   o relación entre ambas manos.
#
# Reglas implementadas:
#   - A
#   - B
#   - C
#   - D
#   - E
#   - F
#   - G
#   - H
#   - I
#   - J
#   - K
#   - L
#   - M
#   - N
#   - Ñ
#   - O
#   - Q
#   - R
#   - S
#   - T
#   - U
#   - V
#   - W
#   - X
#   - Y
#   - Z
#
# Nota:
#   H, J y Z son letras dinámicas, pero también tienen una regla
#   geométrica estática complementaria para validar la configuración
#   de mano y/o zona corporal durante la ventana capturada.
#
# Uso previsto:
#   evaluate_live_abecedario.py:
#       validar contra target_label, porque la letra objetivo es conocida.
#
#   spell_auto_abecedario_pose.py:
#       validar contra pred_label, porque no hay letra objetivo.
#
# Para letras sin regla:
#   required=False, ok=True
# ============================================================


STATIC_GEOMETRY_LABELS = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "Z", "Y", "U", "V", "W", "X", "L", "M", "N", "Ñ", "O", "Q", "K", "R", "S", "T"}


# Reglas iniciales calibradas con debug_static_geometry_abecedario.py.
#
# C:
#   Pose estática con índice y pulgar formando una C abierta;
#   medio, anular y meñique recogidos.
#
# L:
#   Pose estática con pulgar horizontal e índice vertical. Palma/yema hacia cámara.
#
# K:
#   Pose estática con dorso/nudillos hacia cámara:
#     - índice diagonal hacia arriba-derecha;
#     - medio casi horizontal hacia la derecha;
#     - pulgar visible en el vértice entre índice y medio;
#     - anular y meñique cerrados.
#
# Nota general:
#   Algunas reglas laterales usan right/left según la cámara y la mano usada.
#   Si se cambia de mano o se espeja la cámara, puede ser necesario ajustar
#   *_right_of_base_min / *_left_of_base_max.
STATIC_GEOMETRY_RULES = {
    "A": {
        "min_valid_ratio": 0.80,

        # En A correcta, MediaPipe cuenta principalmente el pulgar
        # como dedo extendido/visible. Los dedos largos deben quedar
        # recogidos dentro del puño.
        "finger_count_min": 0.70,
        "finger_count_max": 1.35,

        "thumb_extended_score_min": 0.60,

        "index_extended_score_max": 0.30,
        "middle_extended_score_max": 0.38,
        "ring_extended_score_max": 0.53,
        "pinky_extended_score_max": 0.55,

        # Pulgar apoyado/recostado sobre el índice:
        # no debe estar escondido, no debe estar demasiado lejos
        # y no debe apuntar vertical como gesto de pulgar arriba.
        "thumb_index_tip_distance_norm_min": 0.38,
        "thumb_index_tip_distance_norm_max": 0.85,

        "thumb_mcp_tip_angle_deg_min": -75.0,
        "thumb_mcp_tip_angle_deg_max": -20.0,
        "thumb_mcp_tip_horizontality_min": 0.35,
        "thumb_mcp_tip_above_base_min": 0.80,

        # Mantener el mismo supuesto lateral que L/K para tu cámara/mano actual.
        # Si luego se usa la otra mano o la imagen espejada, puede requerir ajuste.
        "thumb_mcp_tip_right_of_base_min": 0.80,
    },

    "B": {
        "min_valid_ratio": 0.80,

        # B correcta según la variante probada:
        # mano abierta hacia arriba, palma visible pero no completamente frontal,
        # cuatro dedos largos extendidos y juntos, y pulgar doblado hacia dentro.
        # En esta configuración MediaPipe suele contar también el pulgar como
        # extendido, por eso el conteo esperado queda cerca de 5.
        "finger_count_min": 4.50,
        "finger_count_max": 5.25,

        "thumb_extended_score_min": 0.55,
        "thumb_extended_score_max": 0.85,

        "index_extended_score_min": 0.80,
        "middle_extended_score_min": 0.90,
        "ring_extended_score_min": 0.90,
        "pinky_extended_score_min": 0.85,

        # Dedos largos juntos. El máximo de ring-pinky es más flexible
        # porque en la B correcta fue el par con mayor separación natural.
        "index_middle_tip_distance_norm_max": 0.11,
        "middle_ring_tip_distance_norm_max": 0.12,
        "ring_pinky_tip_distance_norm_max": 0.24,

        # Dedos largos verticales/paralelos.
        "index_mcp_tip_angle_deg_min": -100.0,
        "index_mcp_tip_angle_deg_max": -78.0,
        "index_mcp_tip_verticality_min": 0.97,
        "index_mcp_tip_above_base_min": 0.80,

        "middle_mcp_tip_angle_deg_min": -100.0,
        "middle_mcp_tip_angle_deg_max": -78.0,
        "middle_mcp_tip_verticality_min": 0.97,
        "middle_mcp_tip_above_base_min": 0.80,

        "index_middle_axis_angle_3d_deg_max": 8.0,

        # Pulgar doblado hacia dentro, no extendido hacia arriba ni abierto
        # lateralmente. También ayuda a rechazar la B demasiado frontal.
        "thumb_mcp_tip_angle_deg_min": -118.0,
        "thumb_mcp_tip_angle_deg_max": -92.0,
        "thumb_mcp_tip_horizontality_max": 0.45,
        "thumb_mcp_tip_left_of_base_min": 0.80,
        "thumb_mcp_tip_right_of_base_max": 0.20,

        "thumb_index_axis_angle_3d_deg_max": 28.0,
        "thumb_middle_axis_angle_3d_deg_max": 28.0,

        # Rechazo de dorso a cámara. No diferencia por sí solo palma frontal
        # vs. diagonal, por eso se combina con reglas del pulgar/ejes.
        "use_palm_normal_z": True,
        "palm_normal_z_approx_max": 0.001,
    },

    "C": {
        "min_valid_ratio": 0.80,

        # C correcta según la variante probada:
        # pulgar extendido casi horizontal hacia el lado correcto,
        # índice semiextendido/doblado formando la parte superior de la C,
        # y medio/anular/meñique cerrados.
        # Ajuste 2026-05-18: rangos ampliados con 10 nuevos debugs correctos
        # de C y 10 de E; C/E se diferencian principalmente por capa espacial.
        "finger_count_min": 1.50,
        "finger_count_max": 2.20,

        "thumb_extended_score_min": 0.70,
        "thumb_extended_score_max": 1.00,

        "index_extended_score_min": 0.48,
        "index_extended_score_max": 0.85,

        "middle_extended_score_max": 0.35,
        "ring_extended_score_max": 0.35,
        "pinky_extended_score_max": 0.33, #0.35

        # Apertura principal de la C. Se rechaza tanto la pinza cerrada
        # como una apertura excesiva tipo L/deformación.
        "thumb_index_tip_distance_norm_min": 0.75,
        "thumb_index_tip_distance_norm_max": 1.30,

        # El índice debe estar separado de los dedos cerrados.
        # Esto rechaza variantes donde el medio acompaña al índice
        # o queda pegado como parte de la apertura.
        "index_middle_tip_distance_norm_min": 0.80,

        # Índice inclinado hacia arriba formando la curva superior.
        "index_mcp_tip_angle_deg_min": -60.0,
        "index_mcp_tip_angle_deg_max": -18.0,
        "index_mcp_tip_verticality_min": 0.35,
        "index_mcp_tip_verticality_max": 0.85,
        "index_mcp_tip_above_base_min": 0.80,

        # Pulgar casi horizontal formando la base de la C.
        "thumb_mcp_tip_angle_deg_min": -30.0,
        "thumb_mcp_tip_angle_deg_max": 15.0,
        "thumb_mcp_tip_horizontality_min": 0.88,
        "thumb_mcp_tip_right_of_base_min": 0.80,
        "thumb_mcp_tip_left_of_base_max": 0.20,

        # Relaciones 3D útiles para separar C correcta de:
        # - C demasiado abierta;
        # - pulgar mal orientado;
        # - medio activo/pegado.
        "thumb_index_axis_angle_3d_deg_min": 12.0,
        "thumb_index_axis_angle_3d_deg_max": 60.0,
        "thumb_middle_axis_angle_3d_deg_min": 50.0,
        "thumb_middle_axis_angle_3d_deg_max": 105.0,

        # Exclusión semántica C/E: si la misma forma de C aparece
        # rodeando el ojo del mismo lado, corresponde a E y no a C.
        # No se exige pose para aceptar C; solo se rechaza cuando hay
        # evidencia espacial clara de E.
        "reject_eye_region_for_c": True,
        "same_eye_inside_hand_bbox_ratio_for_e_min": 0.75,
        "other_eye_inside_hand_bbox_ratio_for_e_max": 0.25,
        "nose_inside_hand_bbox_ratio_for_e_max": 0.25,
    },


    "D": {
        "min_valid_ratio": 0.80,

        # D correcta según el dataset actual:
        #   - mano derecha con palma visible en orientación diagonal;
        #   - pulgar y dedo medio se tocan o quedan muy próximos;
        #   - índice, anular y meñique quedan extendidos hacia arriba;
        #   - el dedo medio no se valida como "extendido vertical":
        #     MediaPipe lo lee con score bajo porque está lateralizado
        #     hacia el pulgar;
        #   - la forma se rechaza si el contacto ocurre con índice/anular
        #     o si el medio queda arriba como un dedo largo más.
        #
        # Decisión 2026-05-24:
        # La métrica más discriminante es la combinación:
        #   contacto pulgar-medio + medio lateral/hacia abajo-derecha
        #   + índice/anular/meñique verticales hacia arriba.
        # Por eso se exige thumb_middle_tip_distance_norm bajo, pero
        # middle_extended_score máximo bajo/moderado.
        "finger_count_min": 3.50,
        "finger_count_max": 4.40,

        "thumb_extended_score_min": 0.85,
        "index_extended_score_min": 0.80,
        "middle_extended_score_max": 0.55,
        "ring_extended_score_min": 0.85,
        "pinky_extended_score_min": 0.85,

        # Contacto correcto: pulgar con medio.
        # El mínimo pulgar-índice evita aceptar variantes donde el contacto
        # se arma con índice en lugar de medio.
        "thumb_middle_tip_distance_norm_max": 0.13,
        "thumb_index_tip_distance_norm_min": 0.65,

        # Separaciones características: el medio baja/lateraliza hacia el
        # pulgar y queda lejos de índice/anular. Anular y meñique quedan
        # abiertos arriba, pero dentro de un rango natural.
        "index_middle_tip_distance_norm_min": 0.70,
        "middle_ring_tip_distance_norm_min": 0.65, #0.75,
        "ring_pinky_tip_distance_norm_min": 0.38,
        "ring_pinky_tip_distance_norm_max": 0.75,

        # Índice arriba, con tolerancia a ejecución relajada.
        "index_mcp_tip_angle_deg_min": -95.0,
        "index_mcp_tip_angle_deg_max": -50.0,
        "index_mcp_tip_verticality_min": 0.80,
        "index_mcp_tip_above_base_min": 0.80,

        # Medio lateralizado hacia el pulgar: no debe quedar vertical hacia
        # arriba como dedo extendido adicional.
        "middle_mcp_tip_angle_deg_min": 5.0, #8.0, #15.0,
        "middle_mcp_tip_angle_deg_max": 55.0,
        "middle_mcp_tip_horizontality_min": 0.60,
        "middle_mcp_tip_below_base_min": 0.75,
        "middle_mcp_tip_right_of_base_min": 0.80,

        # Anular y meñique extendidos hacia arriba.
        "ring_mcp_tip_angle_deg_min": -90.0,
        "ring_mcp_tip_angle_deg_max": -55.0,
        "ring_mcp_tip_verticality_min": 0.85,
        "ring_mcp_tip_above_base_min": 0.80,

        "pinky_mcp_tip_angle_deg_min": -112.0,
        "pinky_mcp_tip_angle_deg_max": -85.0,
        "pinky_mcp_tip_verticality_min": 0.88,
        "pinky_mcp_tip_above_base_min": 0.80,

        # Pulgar activo hacia el medio, no doblado/oculto ni invertido.
        "thumb_mcp_tip_angle_deg_min": -75.0,
        "thumb_mcp_tip_angle_deg_max": -35.0,
        "thumb_mcp_tip_horizontality_min": 0.34, #0.38,
        "thumb_mcp_tip_horizontality_max": 0.82,
        "thumb_mcp_tip_right_of_base_min": 0.80,
        "thumb_mcp_tip_left_of_base_max": 0.20,

        # Relaciones de ejes:
        #   - índice y medio no son paralelos;
        #   - pulgar y medio mantienen un ángulo compatible con contacto
        #     pulgar-medio real, no con contacto pulgar-índice/anular.
        "index_middle_axis_angle_3d_deg_min": 72.0, #80.0,
        "index_middle_axis_angle_3d_deg_max": 125.0,

        "thumb_middle_axis_angle_3d_deg_min": 60.0, #65.0,
        "thumb_middle_axis_angle_3d_deg_max": 90.0,

        # Palma visible/diagonal hacia cámara. La variante con dorso invierte
        # el signo de la normal.
        "use_palm_normal_z": True,
        "palm_normal_z_approx_max": 0.001,
    },

    "E": {
        "min_valid_ratio": 0.80,

        # E usa prácticamente la misma configuración manual que C,
        # pero la validación final exige que la C rodee el ojo del
        # mismo lado de la mano. Ajuste 2026-05-18: rangos manuales
        # alineados con C y E correctas nuevas; la diferencia semántica
        # queda en la capa espacial ojo-mano.
        "finger_count_min": 1.50,
        "finger_count_max": 2.20,

        "thumb_extended_score_min": 0.70,
        "thumb_extended_score_max": 1.00,

        "index_extended_score_min": 0.48,
        "index_extended_score_max": 0.85,

        "middle_extended_score_max": 0.35,
        "ring_extended_score_max": 0.35,
        "pinky_extended_score_max": 0.33, #0.35

        "thumb_index_tip_distance_norm_min": 0.75,
        "thumb_index_tip_distance_norm_max": 1.30,

        "index_middle_tip_distance_norm_min": 0.80,

        "index_mcp_tip_angle_deg_min": -60.0,
        "index_mcp_tip_angle_deg_max": -18.0,
        "index_mcp_tip_verticality_min": 0.35,
        "index_mcp_tip_verticality_max": 0.85,
        "index_mcp_tip_above_base_min": 0.80,

        "thumb_mcp_tip_angle_deg_min": -30.0,
        "thumb_mcp_tip_angle_deg_max": 15.0,
        "thumb_mcp_tip_horizontality_min": 0.88,
        "thumb_mcp_tip_right_of_base_min": 0.80,
        "thumb_mcp_tip_left_of_base_max": 0.20,

        "thumb_index_axis_angle_3d_deg_min": 12.0,
        "thumb_index_axis_angle_3d_deg_max": 60.0,
        "thumb_middle_axis_angle_3d_deg_min": 50.0,
        "thumb_middle_axis_angle_3d_deg_max": 105.0,

        # Capa espacial: calibrada para la mano/cámara actual, donde
        # el ojo del mismo lado en los CSV corresponde al right_eye.
        # La regla no intenta medir contacto con la cara; solo si el
        # ojo queda visualmente enmarcado por la C.
        "pose_valid_ratio_min": 0.75,
        "same_eye_inside_hand_bbox_ratio_min": 0.75,
        "other_eye_inside_hand_bbox_ratio_max": 0.25,
        "nose_inside_hand_bbox_ratio_max": 0.25,
        "same_eye_thumb_index_t_min": 0.00,
        "same_eye_thumb_index_t_max": 1.20,
        "same_eye_thumb_index_distance_norm_max": 0.35,
    },

    "F": {
        "min_valid_ratio": 0.80,

        # F correcta según el dataset actual:
        #   - mano derecha abierta;
        #   - dedos largos extendidos y compactos/juntos;
        #   - pulgar visible como borde de la mano;
        #   - mano lateral o "de canto", sin palma ni dorso completamente frontal;
        #   - eje muñeca->dedos ascendente/diagonal;
        #   - ubicación en pecho izquierdo superior, cerca del hombro izquierdo.
        #
        # Decisión 2026-05-25:
        # La forma de mano por sí sola no alcanza: una mano abierta y compacta
        # en otra zona del cuerpo no debe aceptarse como F. Por eso se combina
        # forma + orientación de canto + capa espacial relativa a hombros.
        #
        # Ajuste live 2026-05-25:
        # En evaluate_live aparecieron falsos rechazos correctos por umbrales
        # demasiado ajustados en pulgar, compactación índice-medio, paralelismo
        # índice-medio y ubicación. Se amplían solo esos rangos necesarios,
        # manteniendo las capas que bloquean palma/dorso frontal, ubicación baja,
        # dedos abiertos, dedo plegado, pulgar demasiado abierto y eje horizontal.
        "finger_count_min": 4.60,
        "finger_count_max": 5.25,

        "thumb_extended_score_min": 0.85,
        "index_extended_score_min": 0.80,
        "middle_extended_score_min": 0.88,
        "ring_extended_score_min": 0.88,
        "pinky_extended_score_min": 0.78,

        # Dedos largos juntos/compactos. Bloquea dedos abiertos y el caso
        # de un dedo largo plegado que deforma la mano compacta.
        "index_middle_tip_distance_norm_max": 0.105,
        "middle_ring_tip_distance_norm_max": 0.145,
        "ring_pinky_tip_distance_norm_max": 0.175,

        # Pulgar visible pero integrado al borde de la mano. Un pulgar oculto
        # cae por score bajo/distancia/ejes; un pulgar demasiado separado se
        # bloquea por distancia/ejes del pulgar.
        "thumb_index_tip_distance_norm_min": 0.18,
        "thumb_index_tip_distance_norm_max": 0.345,
        "thumb_mcp_tip_angle_deg_min": -88.0,
        "thumb_mcp_tip_angle_deg_max": -60.0,
        "thumb_mcp_tip_horizontality_max": 0.52,
        "thumb_mcp_tip_right_of_base_min": 0.80,
        "thumb_index_axis_angle_3d_deg_max": 15.0,
        "thumb_middle_axis_angle_3d_deg_max": 18.0,

        # Eje ascendente de los dedos. Se rechaza F horizontal y también una
        # versión excesivamente vertical si sale del rango observado.
        "index_mcp_tip_angle_deg_min": -95.0,
        "index_mcp_tip_angle_deg_max": -60.0,
        "index_mcp_tip_verticality_min": 0.88,
        "index_mcp_tip_above_base_min": 0.80,

        "middle_mcp_tip_angle_deg_min": -96.0,
        "middle_mcp_tip_angle_deg_max": -60.0,
        "middle_mcp_tip_horizontality_max": 0.43,
        "middle_mcp_tip_below_base_max": 0.20,

        # Índice y medio casi paralelos. Esto discrimina frente a dedos abiertos
        # o dedo plegado, pero se deja margen para la variación live de MediaPipe.
        "index_middle_axis_angle_3d_deg_max": 12.00,
        "middle_index_axis_signed_range_max": 0.115,

        # Orientación de canto: normal z cercana a cero y positiva leve.
        # El mínimo permite jitter pequeño; el máximo bloquea dorso frontal.
        "use_palm_normal_z": True,
        "palm_normal_z_approx_min": -0.0002,
        "palm_normal_z_approx_max": 0.0022,

        # Capa espacial: mano en pecho izquierdo superior/cerca del hombro
        # izquierdo, medida respecto al centro de hombros y normalizada por
        # ancho de hombros. Para la cámara actual, esa zona queda con dx positivo.
        "pose_valid_for_f_chest_geometry_ratio_min": 0.75,
        "hand_center_shoulder_dx_min": 0.18,
        "hand_center_shoulder_dx_max": 0.45,
        "hand_center_shoulder_dy_min": -0.05,
        "hand_center_shoulder_dy_max": 0.40,
    },

    "G": {
        "min_valid_ratio": 0.80,

        # G correcta según la variante probada:
        # mano derecha en puño suave/semicerrado junto a la oreja derecha.
        # MediaPipe suele interpretar el pulgar como extendido por el agarre
        # del lóbulo, por eso el conteo esperado queda cerca de 1, no de 0.
        "finger_count_min": 0.50,
        "finger_count_max": 1.50,

        "thumb_extended_score_min": 0.75,

        "index_extended_score_max": 0.28,
        "middle_extended_score_max": 0.30,
        "ring_extended_score_max": 0.32,
        "pinky_extended_score_max": 0.34,

        # En los positivos de G el pulgar queda visible pero cerca del índice.
        # Esto rechaza variantes con índice extendido o pulgar separado.
        "thumb_index_tip_distance_norm_min": 0.15,
        "thumb_index_tip_distance_norm_max": 0.55,

        # Capa espacial: la mano debe estar junto a la oreja derecha.
        # Los valores están normalizados por la distancia entre ambas orejas.
        "pose_valid_for_ear_geometry_ratio_min": 0.75,

        "right_ear_center_dist_earspan_max": 1.15,

        "right_ear_dx_earspan_min": -0.60,
        "right_ear_dx_earspan_max": 0.05,

        "right_ear_dy_earspan_min": 0.35,
        "right_ear_dy_earspan_max": 1.15,

        # Evita aceptar puños cerca de la oreja/lado izquierdo.
        "left_ear_center_dist_earspan_min": 1.00,
    },

    "H": {
        "min_valid_ratio": 0.80,

        # H es una letra dinámica. Esta regla NO reemplaza la
        # validación de trayectoria de dynamic_gesture.py; solo valida
        # que la configuración de mano se mantenga compatible durante
        # la ventana capturada.
        #
        # Variante calibrada con el dataset actual:
        #   - pulgar, índice y medio activos/extendidos;
        #   - anular y meñique cerrados;
        #   - índice y medio separados, sin exigir apertura extrema;
        #   - gesto realizado sobre la región de la cara.
        "finger_count_min": 2.45,
        "finger_count_max": 3.35,

        "thumb_extended_score_min": 0.85,
        "index_extended_score_min": 0.72,
        "middle_extended_score_min": 0.50,

        "ring_extended_score_max": 0.35,
        "pinky_extended_score_max": 0.38,

        # Pulgar presente pero no como condición ultra rígida de variante.
        # Se permite que se vea más por detrás o por delante, siempre que
        # sea claramente activo y conserve la orientación general observada.
        "thumb_index_tip_distance_norm_min": 0.35,
        "thumb_index_tip_distance_norm_max": 0.80,

        # Índice y medio deben estar separados, pero se acepta una variante
        # más natural donde no abren tanto como una V exagerada.
        "index_middle_tip_distance_norm_min": 0.25,
        "middle_ring_tip_distance_norm_min": 0.50,
        "ring_pinky_tip_distance_norm_max": 0.22,

        "index_middle_axis_angle_3d_deg_min": 20.0,
        "index_middle_axis_angle_3d_deg_max": 70.0,

        "thumb_mcp_tip_angle_deg_min": -100.0,
        "thumb_mcp_tip_angle_deg_max": -55.0,
        "thumb_mcp_tip_right_of_base_min": 0.72,

        # La H rota durante el movimiento; por eso estos rangos son
        # más amplios que en letras estáticas puras.
        "index_mcp_tip_angle_deg_min": -80.0,
        "index_mcp_tip_angle_deg_max": -15.0,
        "index_mcp_tip_above_base_min": 0.58,

        "middle_mcp_tip_angle_deg_min": -50.0,
        "middle_mcp_tip_angle_deg_max": 35.0,
        "middle_mcp_tip_right_of_base_min": 0.80,

        # Capa espacial: H debe ejecutarse sobre la cara, no en el pecho
        # ni completamente al costado. Las métricas se normalizan por la
        # distancia entre orejas y se agregan como promedio de la ventana.
        "pose_valid_for_h_face_geometry_ratio_min": 0.75,
        "nose_inside_hand_bbox_ratio_min": 0.15,
        "nose_dx_earspan_min": -1.30,
        "nose_dx_earspan_max": -0.20,
        "nose_dy_earspan_min": -0.20,
        "nose_dy_earspan_max": 1.25,
        "nose_center_dist_earspan_max": 1.25,
    },


    "I": {
        "min_valid_ratio": 0.80,

        # I correcta según el dataset actual:
        #   - mano derecha con dorso hacia cámara;
        #   - puño cerrado salvo el índice extendido;
        #   - índice vertical o casi vertical hacia arriba;
        #   - punta del índice apoyada/cercana al pómulo derecho,
        #     debajo del ojo derecho y por encima de la boca.
        #
        # MediaPipe puede contar entre 1 y casi 2 dedos extendidos porque
        # el pulgar queda visible según la orientación. Por eso el conteo
        # global no se fuerza a 1 exacto: se prioriza índice alto y medio,
        # anular y meñique bajos.
        "finger_count_min": 0.85,
        "finger_count_max": 2.10,

        "index_extended_score_min": 0.90,
        "middle_extended_score_max": 0.35,
        "ring_extended_score_max": 0.40,
        "pinky_extended_score_max": 0.42,

        # El índice extendido debe quedar claramente separado de los dedos
        # recogidos. Esto rechaza puño cerrado, índice+medio juntos y mano abierta.
        "index_middle_tip_distance_norm_min": 0.90,

        # Índice casi vertical hacia arriba. El negativo diagonal hacia nariz
        # cayó cerca de -65° y verticality 0.91, por eso se exige más verticalidad.
        "index_mcp_tip_angle_deg_min": -92.0,
        "index_mcp_tip_angle_deg_max": -72.0,
        "index_mcp_tip_verticality_min": 0.95,
        "index_mcp_tip_above_base_min": 0.80,

        # Pulgar no protagonista. En I correcta el pulgar puede aparecer visible,
        # pero no debe abrirse como L. Estos criterios separan i_thumb_open_l_like.
        "thumb_extended_score_max": 0.80,
        "thumb_mcp_tip_angle_deg_min": -70.0,
        "thumb_mcp_tip_angle_deg_max": -10.0,
        "thumb_mcp_tip_right_of_base_min": 0.70,

        # Capa espacial: I debe ubicarse en pómulo/mejilla derecha. La punta
        # del índice queda debajo del ojo derecho, por encima de la boca, y no
        # debe ir hacia la nariz/centro ni al lado contrario de la cara.
        # Todas estas métricas se normalizan por la distancia entre orejas.
        "pose_valid_for_i_face_geometry_ratio_min": 0.75,

        "index_tip_right_eye_dx_earspan_min": -0.30,
        "index_tip_right_eye_dx_earspan_max": -0.05,
        "index_tip_right_eye_dy_earspan_min": 0.35,
        "index_tip_right_eye_dy_earspan_max": 0.75,

        "index_tip_mouth_dy_earspan_min": -0.75,
        "index_tip_mouth_dy_earspan_max": -0.20,

        "index_tip_nose_dx_earspan_min": -0.55,
        "index_tip_nose_dx_earspan_max": -0.20,
    },

    "J": {
        "min_valid_ratio": 0.80,

        # J es una letra dinámica. Esta regla NO reemplaza la
        # validación de trayectoria de dynamic_gesture.py; solo valida
        # que la configuración de mano y la zona facial sean compatibles.
        #
        # Variante calibrada con el dataset actual:
        #   - mano derecha lateral;
        #   - dedos largos extendidos/activos y juntos;
        #   - pulgar oculto/tapado, no abierto como L;
        #   - recorrido sobre la zona de mentón/mandíbula.
        #
        # MediaPipe no ve esta mano lateral como "5 dedos extendidos":
        # los positivos correctos se agruparon cerca de 3 dedos visibles.
        "finger_count_min": 2.50,
        "finger_count_max": 4.20,

        "index_extended_score_min": 0.52,
        "middle_extended_score_min": 0.52,
        "ring_extended_score_min": 0.52,
        "pinky_extended_score_min": 0.52,

        # Dedos largos juntos y casi paralelos.
        "index_middle_tip_distance_norm_max": 0.080,
        "middle_ring_tip_distance_norm_max": 0.055,
        "ring_pinky_tip_distance_norm_max": 0.170,
        "index_middle_axis_angle_3d_deg_max": 8.0,

        # El pulgar debe permanecer compacto/oculto, no abierto hacia afuera.
        # No se exige thumb_extended_score porque MediaPipe puede estimarlo
        # como activo aunque visualmente esté tapado por los demás dedos.
        "thumb_index_tip_distance_norm_min": 0.25,
        "thumb_index_tip_distance_norm_max": 0.60,
        "thumb_mcp_tip_angle_deg_min": -75.0,
        "thumb_mcp_tip_angle_deg_max": -32.0,

        # La mano está lateral y los dedos apuntan aproximadamente en la
        # orientación observada durante el recorrido bajo la mandíbula.
        "index_mcp_tip_angle_deg_min": -30.0,
        "index_mcp_tip_angle_deg_max": 15.0,
        "middle_mcp_tip_angle_deg_min": -32.0,
        "middle_mcp_tip_angle_deg_max": 15.0,

        # Capa espacial: J debe ejecutarse sobre la zona de mentón/mandíbula,
        # no en el pecho, no al costado del cuerpo y no frente a la cara sin
        # bajar por el borde inferior de la boca/mandíbula.
        # Como MediaPipe Pose no tiene landmark exacto de mentón, se aproxima
        # con el centro de boca y la distancia entre orejas.
        "pose_valid_for_j_jaw_geometry_ratio_min": 0.75,

        "mouth_dx_earspan_min": -0.30,
        "mouth_dx_earspan_max": 0.35,

        "mouth_dy_earspan_min": 0.25,
        "mouth_dy_earspan_max": 0.75,
    },

    "Z": {
        "min_valid_ratio": 0.80,

        # Z es una letra dinámica. Esta regla NO reemplaza la
        # validación de trayectoria de dynamic_gesture.py; solo valida
        # que la configuración de mano se mantenga compatible durante
        # la ventana capturada.
        #
        # Variante calibrada con el dataset actual:
        #   - mano derecha en puño;
        #   - solo el meñique es protagonista;
        #   - índice, medio y anular permanecen recogidos;
        #   - el pulgar puede aparecer con score alto por la orientación,
        #     por lo que NO se usa thumb_extended_score como dedo abierto;
        #   - la palma queda principalmente hacia abajo y el meñique apunta
        #     hacia cámara/afuera, no hacia la cara.
        "finger_count_min": 1.20,
        "finger_count_max": 2.20,

        # Meñique activo, pero no se exige score extremo porque con esta
        # orientación MediaPipe puede estimarlo moderado aunque visualmente
        # sea el dedo extendido.
        "pinky_extended_score_min": 0.40, #0.45,

        # Los otros dedos largos deben permanecer recogidos/bajos.
        "index_extended_score_max": 0.35,
        "middle_extended_score_max": 0.35,
        "ring_extended_score_max": 0.36,

        # La separación anular-meñique confirma que el meñique es el dedo
        # proyectado y evita aceptar un puño cerrado sin meñique.
        "ring_pinky_tip_distance_norm_min": 0.30, # 0.34,

        # Pulgar compacto/no protagonista. El score del pulgar puede ser alto,
        # por eso se controla más por distancia, ángulo y horizontalidad.
        "thumb_index_tip_distance_norm_min": 0.18,
        "thumb_index_tip_distance_norm_max": 0.55,
        "thumb_mcp_tip_angle_deg_min": -100, #-96.0,
        "thumb_mcp_tip_angle_deg_max": -68.0,
        "thumb_mcp_tip_horizontality_max": 0.32,

        # Dedos recogidos apuntando hacia abajo/lateral en la orientación
        # observada. Esto ayuda a rechazar índice extendido, mano abierta,
        # pulgar abierto y meñique orientado hacia la cara.
        "index_mcp_tip_angle_deg_min": 25.0, #38.0,
        "index_mcp_tip_angle_deg_max": 72.0,
        "middle_mcp_tip_angle_deg_min": 50.0,
        "middle_mcp_tip_angle_deg_max": 78.0,
        "middle_mcp_tip_horizontality_max": 0.58,

        # Con la orientación correcta de Z, palm_normal_z quedó negativa.
        # La variante con meñique hacia la cara invirtió este signo.
        "use_palm_normal_z": True,
        "palm_normal_z_approx_max": 0.001,
    },

    "Y": {
        "min_valid_ratio": 0.80,

        # Y correcta según el dataset actual:
        #   - mano derecha tipo "teléfono";
        #   - pulgar y meñique extendidos;
        #   - índice, medio y anular recogidos;
        #   - dorso/parte posterior de la mano hacia cámara;
        #   - punta del meñique junto a la comisura derecha de la boca.
        #
        # La validación combina forma de mano + capa espacial de boca.
        # No se exige que el meñique esté perfectamente vertical: en los
        # positivos aparece extendido pero en diagonal natural.
        "finger_count_min": 1.75,
        "finger_count_max": 2.25,

        "thumb_extended_score_min": 0.90,
        "pinky_extended_score_min": 0.88,

        "index_extended_score_max": 0.32,
        "middle_extended_score_max": 0.35,
        "ring_extended_score_max": 0.32,

        # Apertura característica entre pulgar y meñique. Ayuda a rechazar
        # falta de pulgar, mano abierta y configuraciones compactas.
        "thumb_pinky_tip_distance_norm_min": 1.00,
        "ring_pinky_tip_distance_norm_min": 0.55,

        # Pulgar extendido hacia el lado observado en la cámara actual.
        # En y_no_thumb el score puede quedar medio-alto por ruido, por eso
        # se combina score + orientación + lateralidad.
        "thumb_mcp_tip_angle_deg_min": -122.0,
        "thumb_mcp_tip_angle_deg_max": -96.0,
        "thumb_mcp_tip_left_of_base_min": 0.80,

        # Meñique extendido en diagonal ascendente, no cerrado ni reemplazado
        # por índice/pulgar tipo L.
        "pinky_mcp_tip_angle_deg_min": -78.0,
        "pinky_mcp_tip_angle_deg_max": -48.0,
        "pinky_mcp_tip_verticality_min": 0.75,
        "pinky_mcp_tip_above_base_min": 0.80,

        # Capa espacial: el extremo relevante de Y es la punta del meñique.
        # Debe quedar al costado derecho de la boca/comisura para la mano
        # derecha usada en el dataset. Las métricas están normalizadas por
        # la distancia entre orejas.
        "pose_valid_for_y_mouth_geometry_ratio_min": 0.75,

        "pinky_tip_mouth_dx_earspan_min": -0.52,
        "pinky_tip_mouth_dx_earspan_max": -0.28,

        # Poca tolerancia vertical: permite estar apenas arriba/abajo de la
        # comisura, pero rechaza mentón, pecho, pómulo alto u ojo.
        "pinky_tip_mouth_dy_earspan_min": -0.08,
        "pinky_tip_mouth_dy_earspan_max": 0.22,

        "pinky_tip_mouth_dist_earspan_max": 0.52,
    },

    "U": {
        "min_valid_ratio": 0.80,

        # U correcta según el dataset actual:
        #   - mano derecha con dorso hacia cámara;
        #   - índice y meñique extendidos hacia arriba;
        #   - medio y anular recogidos;
        #   - pulgar escondido/no protagonista.
        #
        # Nota importante: aunque visualmente el pulgar no es protagonista,
        # MediaPipe lo estima parcialmente extendido en la U correcta
        # (aprox. 0.69..0.72), por eso NO se exige conteo exacto de 2 dedos.
        # La regla prioriza índice + meñique altos, medio/anular bajos,
        # verticalidad de índice/meñique y pulgar no abierto como Y/L.
        "finger_count_min": 1.80,
        "finger_count_max": 3.35,

        "index_extended_score_min": 0.90,
        "pinky_extended_score_min": 0.88,

        "middle_extended_score_max": 0.35,
        "ring_extended_score_max": 0.35,

        # Pulgar tolerado como visible por MediaPipe, pero no protagonista.
        # Rechaza pulgar claramente abierto o orientación tipo Y/mano abierta.
        "thumb_extended_score_max": 0.88,

        # Índice y meñique deben apuntar hacia arriba. Esto bloquea la U
        # rotada/inclinada donde los dedos quedan casi horizontales.
        "index_mcp_tip_angle_deg_min": -105.0,
        "index_mcp_tip_angle_deg_max": -72.0,
        "index_mcp_tip_verticality_min": 0.90,
        "index_mcp_tip_above_base_min": 0.80,

        "pinky_mcp_tip_angle_deg_min": -95.0,
        "pinky_mcp_tip_angle_deg_max": -60.0,
        "pinky_mcp_tip_verticality_min": 0.85,
        "pinky_mcp_tip_above_base_min": 0.80,

        # Separaciones características de índice/meñique extendidos con
        # medio/anular cerrados. Ayudan a rechazar índice+medio, mano abierta
        # y medio extra abierto.
        "index_middle_tip_distance_norm_min": 0.85,

        "ring_pinky_tip_distance_norm_min": 0.55,
        "ring_pinky_tip_distance_norm_max": 0.90,

        "index_pinky_tip_distance_norm_min": 0.60,
        "index_pinky_tip_distance_norm_max": 0.85,

        "thumb_pinky_tip_distance_norm_min": 0.38,
        "thumb_pinky_tip_distance_norm_max": 0.75,

        # Pulgar escondido hacia el lado esperado en la cámara actual.
        # Si se usa otra mano o cámara espejada, esta lateralidad puede
        # requerir ajuste.
        "thumb_mcp_tip_angle_deg_min": -75.0,
        "thumb_mcp_tip_angle_deg_max": -35.0,
        "thumb_mcp_tip_right_of_base_min": 0.70,
        "thumb_mcp_tip_left_of_base_max": 0.20,
    },

    "V": {
        "min_valid_ratio": 0.80,

        # V correcta según el dataset actual:
        #   - mano derecha con dorso hacia cámara;
        #   - índice y medio extendidos apuntando hacia arriba;
        #   - índice y medio separados, formando una V visible;
        #   - anular y meñique recogidos;
        #   - el pulgar puede verse, pero no debe volverse protagonista.
        #
        # Decisión 2026-05-24:
        # La comparación relevante para V es N: ambas usan índice + medio,
        # pero V apunta hacia arriba y N hacia abajo. Por eso la regla se
        # apoya fuerte en los ángulos negativos de índice/medio y en que
        # ambos queden por encima de su base.
        #
        # La separación índice-medio bloquea la variante con dedos juntos.
        # El pulgar no es la señal principal, pero se limita de forma suave
        # para rechazar los casos donde MediaPipe lo lee como demasiado
        # abierto/protagonista.
        "finger_count_min": 1.80,
        "finger_count_max": 3.20,

        "index_extended_score_min": 0.90,
        "middle_extended_score_min": 0.90,

        "ring_extended_score_max": 0.28, #0.25,
        "pinky_extended_score_max": 0.25,

        # Pulgar tolerado como visible, pero no protagonista.
        "thumb_extended_score_max": 0.82,

        # Índice y medio separados: evita aceptar una forma de dos dedos
        # juntos/paralelos. El máximo evita aceptar casos donde uno de los
        # dos dedos falta y MediaPipe mide una distancia artificial grande.
        "index_middle_tip_distance_norm_min": 0.15,
        "index_middle_tip_distance_norm_max": 0.35,

        # Anular y meñique compactos/recogidos.
        "ring_pinky_tip_distance_norm_max": 0.18,

        # Índice y medio apuntando hacia arriba. En coordenadas de imagen,
        # ángulos negativos cercanos a -90° indican que la punta queda por
        # encima de la base. Esto bloquea directamente la variante tipo N
        # hacia abajo.
        "index_mcp_tip_angle_deg_min": -100.0,
        "index_mcp_tip_angle_deg_max": -70.0,
        "index_mcp_tip_verticality_min": 0.96,
        "index_mcp_tip_above_base_min": 0.80,
        "index_mcp_tip_below_base_max": 0.20,

        "middle_mcp_tip_angle_deg_min": -85.0,
        "middle_mcp_tip_angle_deg_max": -55.0,
        "middle_mcp_tip_verticality_min": 0.82, #0.85,
        "middle_mcp_tip_above_base_min": 0.80,
        "middle_mcp_tip_below_base_max": 0.20,
        "middle_mcp_tip_horizontality_max": 0.55,

        # Apertura angular entre índice y medio compatible con V:
        # suficientemente separados, pero no como una configuración deformada.
        "index_middle_axis_angle_3d_deg_min": 8.0,
        "index_middle_axis_angle_3d_deg_max": 28.0,

        # Dorso hacia cámara. En positivos de V quedó con normal z positiva;
        # la orientación con palma hacia cámara invirtió el signo.
        "use_palm_normal_z": True,
        "palm_normal_z_approx_min": 0.001,
    },

    "W": {
        "min_valid_ratio": 0.80,

        # W correcta según el dataset actual:
        #   - letra estática de dos manos;
        #   - cada mano tiene una forma equivalente a U:
        #       índice y meñique extendidos, medio/anular recogidos,
        #       pulgar no protagonista;
        #   - los meñiques se conectan en el centro.
        #
        # Decisión 2026-05-21:
        # No se exige cruce geométrico perfecto de los meñiques porque
        # MediaPipe puede distorsionar/acortar los landmarks del dedo que
        # queda atrás. Se acepta tanto cruce visible como contacto fuerte
        # punta con punta/borde con borde, siempre que haya conexión central.
        "expected_hands_min": 1.0,

        # Mano primaria: equivalente a U derecha.
        "finger_count_min": 1.80,
        "finger_count_max": 3.40,

        "index_extended_score_min": 0.90,
        "pinky_extended_score_min": 0.80,

        "middle_extended_score_max": 0.42,
        "ring_extended_score_max": 0.42,

        "thumb_extended_score_max": 0.90,

        "index_mcp_tip_angle_deg_min": -105.0,
        "index_mcp_tip_angle_deg_max": -65.0,
        "index_mcp_tip_verticality_min": 0.88,
        "index_mcp_tip_above_base_min": 0.80,

        "pinky_mcp_tip_angle_deg_min": -90.0,
        "pinky_mcp_tip_angle_deg_max": -50.0,
        "pinky_mcp_tip_verticality_min": 0.75,
        "pinky_mcp_tip_above_base_min": 0.70,

        "thumb_mcp_tip_right_of_base_min": 0.70,
        "thumb_mcp_tip_left_of_base_max": 0.25,

        # Mano secundaria: misma configuración, pero espejada.
        "secondary_valid_ratio_min": 0.80,

        "secondary_finger_count_min": 1.80,
        "secondary_finger_count_max": 3.40,

        "secondary_index_extended_score_min": 0.90,
        "secondary_pinky_extended_score_min": 0.75,

        "secondary_middle_extended_score_max": 0.42,
        "secondary_ring_extended_score_max": 0.42,

        "secondary_thumb_extended_score_max": 0.90,

        "secondary_index_mcp_tip_angle_deg_min": -112.0,
        "secondary_index_mcp_tip_angle_deg_max": -75.0,
        "secondary_index_mcp_tip_verticality_min": 0.88,
        "secondary_index_mcp_tip_above_base_min": 0.80,

        "secondary_pinky_mcp_tip_angle_deg_min": -130.0,
        "secondary_pinky_mcp_tip_angle_deg_max": -85.0,
        "secondary_pinky_mcp_tip_verticality_min": 0.75,
        "secondary_pinky_mcp_tip_above_base_min": 0.70,

        "secondary_thumb_mcp_tip_left_of_base_min": 0.70,
        "secondary_thumb_mcp_tip_right_of_base_max": 0.25,

        # Relación entre manos:
        #   - los meñiques deben quedar conectados en el centro;
        #   - los índices deben quedar más separados hacia afuera;
        #   - se acepta contacto fuerte o cruce con oclusión/distorsión.
        "w_pinky_tip_distance_norm_max": 0.35,
        "w_pinky_segment_min_distance_norm_max": 0.18,

        "w_index_tip_distance_norm_min": 0.85,
        "w_pinky_index_tip_distance_ratio_max": 0.32,

        "w_index_tip_x_distance_norm_min": 0.80,
        "w_pinky_tip_x_distance_norm_max": 0.18,
    },


    "X": {
        "min_valid_ratio": 0.80,

        # X correcta según el dataset actual:
        #   - letra estática de dos manos;
        #   - ambas manos en puño/casi puño;
        #   - solo los índices son protagonistas;
        #   - medio, anular y meñique deben quedar recogidos;
        #   - los pulgares pueden verse por la orientación, pero no son
        #     protagonistas;
        #   - los índices se cruzan en el centro formando una X.
        #
        # Decisión 2026-05-22:
        # A diferencia de W, en X sí se exige cruce/intersección de los
        # segmentos de los índices, no solo contacto de puntas. En las
        # pruebas positivas los landmarks de índice fueron suficientemente
        # estables y no presentaron la distorsión fuerte observada en los
        # meñiques de W.
        "expected_hands_min": 1.0,

        # Mano primaria.
        "finger_count_min": 0.85, #0.95,
        "finger_count_max": 2.35,

        "index_extended_score_min": 0.72,
        "middle_extended_score_max": 0.32,
        "ring_extended_score_max": 0.34,
        "pinky_extended_score_max": 0.42, #0.35,
        "thumb_extended_score_max": 0.93,

        # El índice queda diagonal hacia el centro, no vertical como I/U
        # ni casi horizontal por orientación incorrecta.
        "index_mcp_tip_angle_deg_min": -65.0,
        "index_mcp_tip_angle_deg_max": -25.0,
        "index_mcp_tip_verticality_min": 0.50,
        "index_mcp_tip_above_base_min": 0.80,

        # En X correcta el pulgar queda relativamente separado del índice.
        # Esto ayuda a rechazar variantes donde MediaPipe reconstruye una
        # pose ruidosa como si fuera índice/medio o una orientación frontal.
        "thumb_index_tip_distance_norm_min": 0.85, #1.05,

        # Orientación esperada de la mano primaria en la cámara actual.
        # En los positivos quedó con normal z positiva; una orientación
        # invertida/rotada tiende a cambiar este signo.
        "palm_normal_z_approx_min": 0.0,

        # Mano secundaria. No se fuerza ángulo 2D específico porque la
        # mano espejada puede aparecer con valores angulares muy distintos
        # según cuál índice queda por delante, pero sí se valida forma:
        # índice presente y dedos no protagonistas recogidos.
        "secondary_valid_ratio_min": 0.80,

        "secondary_finger_count_min": 0.85, #1.20,
        "secondary_finger_count_max": 2.35,

        "secondary_index_extended_score_min": 0.62,
        "secondary_middle_extended_score_max": 0.32,
        "secondary_ring_extended_score_max": 0.34,
        "secondary_pinky_extended_score_max": 0.42,#0.35,
        "secondary_thumb_extended_score_max": 0.93,

        # Orientación esperada de la mano secundaria en la cámara actual.
        # En los positivos quedó con normal z negativa.
        "secondary_palm_normal_z_approx_max": 0.0,

        # Relación entre manos:
        #   - los segmentos MCP->TIP de ambos índices deben cruzarse;
        #   - la distancia mínima entre segmentos debe ser baja;
        #   - las puntas no deben estar simplemente juntas punta con punta;
        #   - debe haber separación horizontal suficiente entre puntas,
        #     compatible con una X y no con contacto simple.
        "x_index_segment_strict_intersection_min": 0.60,
        "x_index_segment_min_distance_norm_max": 0.30,

        "x_index_tip_distance_norm_min": 0.25,
        "x_index_tip_distance_norm_max": 0.80,

        "x_index_tip_x_distance_norm_min": 0.30, #0.35,
    },


    "M": {
        "min_valid_ratio": 0.80,

        # M correcta según el dataset actual:
        #   - mano derecha con dorso hacia cámara;
        #   - muñeca flexionada, con los dedos largos apuntando hacia abajo;
        #   - índice, medio, anular y meñique extendidos y con separación moderada;
        #   - el pulgar puede aparecer como extendido para MediaPipe, pero no debe
        #     separarse ni volverse protagonista como en una variante abierta.
        #
        # Decisión 2026-05-22:
        # No se exige thumb_extended_score bajo porque en los positivos correctos
        # el pulgar fue estimado alto (~0.91..0.99). Se controla mediante distancia,
        # horizontalidad y relación con el índice.
        "finger_count_min": 4.70,
        "finger_count_max": 5.25,

        "index_extended_score_min": 0.90,
        "middle_extended_score_min": 0.90,
        "ring_extended_score_min": 0.90,
        "pinky_extended_score_min": 0.90,

        # Dedos largos descendentes. En coordenadas de imagen, ángulos positivos
        # cercanos a 90° indican que la punta queda por debajo de la base.
        "index_mcp_tip_angle_deg_min": 65.0,
        "index_mcp_tip_angle_deg_max": 105.0,
        "index_mcp_tip_verticality_min": 0.90,
        "index_mcp_tip_below_base_min": 0.80,
        "index_mcp_tip_above_base_max": 0.20,

        "middle_mcp_tip_angle_deg_min": 70.0,
        "middle_mcp_tip_angle_deg_max": 105.0,
        "middle_mcp_tip_below_base_min": 0.80,

        "ring_mcp_tip_angle_deg_min": 70.0,
        "ring_mcp_tip_angle_deg_max": 110.0,
        "ring_mcp_tip_below_base_min": 0.80,

        "pinky_mcp_tip_angle_deg_min": 70.0,
        "pinky_mcp_tip_angle_deg_max": 115.0,
        "pinky_mcp_tip_below_base_min": 0.80,

        # Separación moderada entre los cuatro dedos largos. Esto bloquea
        # la variante con los dedos completamente pegados.
        "index_middle_tip_distance_norm_min": 0.12,
        "middle_ring_tip_distance_norm_min": 0.075,
        "ring_pinky_tip_distance_norm_min": 0.15, #0.14,

        # Pulgar visible pero no separado/protagonista.
        # Ajuste 2026-05-22:
        #   - m_thumb_separated queda bloqueado por distancia/horizontalidad;
        #   - m_thumb_straight_attached queda bloqueado por ángulo del pulgar
        #     demasiado bajo (~73.5°), mientras los positivos correctos quedaron
        #     aprox. entre 89° y 105°.
        "thumb_index_tip_distance_norm_max": 0.45,
        "thumb_mcp_tip_horizontality_max": 0.40,
        "thumb_mcp_tip_angle_deg_min": 86.0, #82.0,
        "thumb_mcp_tip_angle_deg_max": 115.0,

        # Dorso/orientación compatible. La variante con palma/orientación
        # invertida cambió el signo de esta métrica.
        "use_palm_normal_z": True,
        "palm_normal_z_approx_min": 0.001,
    },


    "N": {
        "min_valid_ratio": 0.80,

        # N correcta según el dataset actual:
        #   - mano derecha con dorso hacia cámara;
        #   - muñeca flexionada, con índice y medio apuntando hacia abajo;
        #   - índice y medio extendidos y con separación moderada;
        #   - anular y meñique recogidos/compactos;
        #   - el pulgar puede aparecer como extendido para MediaPipe, pero no debe
        #     sobresalir ni volverse protagonista.
        #
        # Decisión 2026-05-22:
        # Al igual que en M, no se exige thumb_extended_score bajo porque
        # MediaPipe estima el pulgar alto en la N correcta. Se controla por
        # distancia, horizontalidad y ángulo del pulgar. El caso "pulgar apenas
        # sobresalido" queda bloqueado por thumb_angle/horizontality, mientras
        # el pulgar recto muy pegado se acepta como limitación/tolerancia natural.
        "finger_count_min": 2.70,
        "finger_count_max": 3.30,

        "index_extended_score_min": 0.90,
        "middle_extended_score_min": 0.90,

        "ring_extended_score_max": 0.25,
        "pinky_extended_score_max": 0.35,

        # Índice y medio descendentes. En coordenadas de imagen, ángulos positivos
        # cercanos a 90° indican que la punta queda por debajo de la base.
        "index_mcp_tip_angle_deg_min": 70.0,
        "index_mcp_tip_angle_deg_max": 108.0,
        "index_mcp_tip_verticality_min": 0.88,
        "index_mcp_tip_below_base_min": 0.80,
        "index_mcp_tip_above_base_max": 0.20,

        "middle_mcp_tip_angle_deg_min": 82.0,
        "middle_mcp_tip_angle_deg_max": 110.0,
        "middle_mcp_tip_below_base_min": 0.80,
        "middle_mcp_tip_horizontality_max": 0.35,

        # Separación característica:
        #   - índice y medio no deben estar pegados;
        #   - medio y anular deben quedar bien separados porque el anular está recogido;
        #   - anular y meñique deben quedar compactos.
        "index_middle_tip_distance_norm_min": 0.13,
        "index_middle_tip_distance_norm_max": 0.30,
        "middle_ring_tip_distance_norm_min": 0.55,
        "ring_pinky_tip_distance_norm_max": 0.13,

        # Pulgar visible pero no sobresalido/protagonista.
        "thumb_index_tip_distance_norm_max": 0.55,
        "thumb_mcp_tip_horizontality_max": 0.32,
        "thumb_mcp_tip_angle_deg_min": 86.0, #82.0,
        "thumb_mcp_tip_angle_deg_max": 115.0,

        # Dorso/orientación compatible. En positivos de N quedó con normal z positiva.
        "use_palm_normal_z": True,
        "palm_normal_z_approx_min": 0.001,
    },


    "Ñ": {
        "min_valid_ratio": 0.80,

        # Ñ correcta según el dataset actual:
        #   - letra estática de dos manos;
        #   - la mano primaria mantiene la base de N:
        #       índice y medio extendidos hacia abajo,
        #       anular y meñique recogidos,
        #       pulgar no protagonista, igual que en N aislada;
        #   - la mano secundaria forma la tilde con solo el índice extendido;
        #   - el índice secundario queda en contacto/proximidad fuerte con la
        #     línea de nudillos/base superior de índice y medio de la mano primaria.
        #
        # Decisión 2026-05-23:
        # Aunque en algunos debugs positivos el pulgar de la mano primaria quedó
        # más visible por error de ejecución, la definición correcta de Ñ hereda
        # el criterio de N/M: el pulgar no debe sobresalir ni volverse protagonista.
        # Si esta restricción rompe positivos reales en evaluate/spell, se revisará
        # con una nueva tanda específica.
        "expected_hands_min": 1.0,

        # Mano primaria: base N.
        "finger_count_min": 2.70,
        "finger_count_max": 3.30,

        "index_extended_score_min": 0.90,
        "middle_extended_score_min": 0.90,

        "ring_extended_score_max": 0.275, #0.25,
        "pinky_extended_score_max": 0.35,

        "index_mcp_tip_angle_deg_min": 70.0,
        "index_mcp_tip_angle_deg_max": 108.0,
        "index_mcp_tip_verticality_min": 0.88,
        "index_mcp_tip_below_base_min": 0.80,
        "index_mcp_tip_above_base_max": 0.20,

        "middle_mcp_tip_angle_deg_min": 82.0,
        "middle_mcp_tip_angle_deg_max": 110.0,
        "middle_mcp_tip_below_base_min": 0.80,
        "middle_mcp_tip_horizontality_max": 0.35,

        "index_middle_tip_distance_norm_min": 0.13,
        "index_middle_tip_distance_norm_max": 0.34,
        "middle_ring_tip_distance_norm_min": 0.54,
        "ring_pinky_tip_distance_norm_max": 0.13,

        # Pulgar de la mano primaria: mismo criterio conceptual que N.
        "thumb_index_tip_distance_norm_max": 0.55,
        "thumb_mcp_tip_horizontality_max": 0.42, #0.32,
        "thumb_mcp_tip_angle_deg_min": 86.0,
        "thumb_mcp_tip_angle_deg_max": 115.0,

        "use_palm_normal_z": True,
        "palm_normal_z_approx_min": 0.001,

        # Mano secundaria: tilde con solo índice.
        "secondary_valid_ratio_min": 0.80,

        "secondary_finger_count_min": 0.75,
        "secondary_finger_count_max": 1.35,

        "secondary_index_extended_score_min": 0.70,
        "secondary_middle_extended_score_max": 0.42, #0.30,
        "secondary_ring_extended_score_max": 0.30,
        "secondary_pinky_extended_score_max": 0.32,
        "secondary_thumb_extended_score_max": 0.65,

        "secondary_index_mcp_tip_angle_deg_min": 145.0,
        "secondary_index_mcp_tip_angle_deg_max": 175.0, #170.0,
        "secondary_index_mcp_tip_horizontality_min": 0.82,

        # En los positivos la mano de la tilde muestra dorso/uña hacia cámara
        # y queda con normal z negativa. La orientación con palma/lateral tiende
        # a invertir o alterar esta métrica.
        "secondary_palm_normal_z_approx_max": 0.0,

        # Contacto/proximidad 2D de la tilde:
        # distancia mínima entre el segmento MCP->TIP del índice secundario
        # y la línea de nudillos formada por los MCP de índice y medio primarios.
        # No prueba contacto 3D real, pero rechaza tilde alta, baja o separada.
        "enie_tilde_index_to_knuckle_line_distance_norm_max": 0.14, #0.09,

        # Cobertura 2D de la tilde sobre la línea de nudillos/base entre
        # MCP de índice y medio primarios. Complementa la distancia:
        # evita aceptar una tilde que toca solo un extremo o queda al costado.
        # No mide contacto 3D real; mide solapamiento proyectado en imagen.
        "enie_tilde_knuckle_line_coverage_min": 0.45,
    },


    "O": {
        "min_valid_ratio": 0.80,

        # O correcta según el dataset actual:
        #   - mano derecha con orientación hacia arriba, algo diagonal;
        #   - palma parcialmente visible, no totalmente frontal;
        #   - pulgar e índice se tocan formando un círculo/O;
        #   - medio, anular y meñique extendidos o semi-extendidos;
        #   - los tres dedos largos deben verse abiertos/naturales,
        #     no cerrados ni totalmente pegados como bloque tipo B.
        #
        # Decisión 2026-05-23:
        # La distancia pulgar-índice tiene mínimo y máximo: el máximo
        # rechaza O abierta/C-like; el mínimo evita aceptar una pinza
        # aplastada sin hueco circular visible. La separación de dedos largos
        # ayuda a rechazar variantes tipo B con medio/anular/meñique rígidos
        # y demasiado juntos.
        "finger_count_min": 3.60,
        "finger_count_max": 4.35,

        "thumb_extended_score_min": 0.85,
        "index_extended_score_max": 0.35,
        "middle_extended_score_min": 0.85,
        "ring_extended_score_min": 0.85,
        "pinky_extended_score_min": 0.85,

        # Cierre circular pulgar-índice.
        "thumb_index_tip_distance_norm_min": 0.045,
        "thumb_index_tip_distance_norm_max": 0.120,

        # Índice curvado hacia el pulgar y separado de los dedos largos.
        "index_middle_tip_distance_norm_min": 0.45,
        "index_middle_tip_distance_norm_max": 0.68, #0.66,

        # Medio, anular y meñique abiertos/naturales.
        "middle_ring_tip_distance_norm_min": 0.075, #0.080,
        "ring_pinky_tip_distance_norm_min": 0.165, # 0.190,
        "ring_pinky_tip_distance_norm_max": 0.290,

        # Índice curvado formando la parte superior de la O, no extendido
        # hacia arriba ni aplastado horizontalmente como pinza.
        "index_mcp_tip_angle_deg_min": 0.0,
        "index_mcp_tip_angle_deg_max": 35.0, #30.0,
        "index_mcp_tip_verticality_max": 0.60, #0.50,
        "index_mcp_tip_above_base_max": 0.20,

        # Dedo medio extendido hacia arriba en la orientación observada.
        "middle_mcp_tip_angle_deg_min": -88.0,
        "middle_mcp_tip_angle_deg_max": -65.0,
        "middle_mcp_tip_horizontality_max": 0.38,

        # Pulgar diagonal hacia el índice, no completamente horizontal
        # como apertura C/pinza ni oculto por orientación incorrecta.
        "thumb_mcp_tip_angle_deg_min": -70.0,
        "thumb_mcp_tip_angle_deg_max": -40.0,
        "thumb_mcp_tip_horizontality_min": 0.35, # 0.40,
        "thumb_mcp_tip_horizontality_max": 0.75,
        "thumb_mcp_tip_right_of_base_min": 0.80,

        # Relaciones de ejes útiles para separar O circular de pinza,
        # palma frontal, dorso y dedos largos tipo B.
        "index_middle_axis_angle_3d_deg_min": 65.0,
        "index_middle_axis_angle_3d_deg_max": 102.0, #97.0,

        "thumb_index_axis_angle_3d_deg_min": 55.0,
        "thumb_index_axis_angle_3d_deg_max": 72.0,

        "thumb_middle_axis_angle_3d_deg_min": 8.0,
        "thumb_middle_axis_angle_3d_deg_max": 36.0,

        # En la O correcta actual la orientación queda con normal z negativa;
        # la variante con más dorso visible invierte el signo.
        "use_palm_normal_z": True,
        "palm_normal_z_approx_max": 0.001,
    },

    "Q": {
        "min_valid_ratio": 0.80,

        # Q correcta según el dataset actual:
        #   - letra estática de dos manos;
        #   - mano primaria equivalente a O: pulgar e índice se tocan
        #     formando el círculo, con medio/anular/meñique extendidos
        #     o semi-extendidos;
        #   - mano secundaria en puño, con solo el índice extendido;
        #   - la punta del índice secundario toca el punto de unión de
        #     pulgar + índice de la mano primaria, formando la cola de Q;
        #   - el índice secundario debe quedar diagonal, no vertical ni
        #     horizontal.
        #
        # Decisión 2026-05-23:
        # La mano primaria de Q se trata como una O real, no como una forma
        # distinta. Se conservan los criterios principales de O:
        # cierre pulgar-índice, índice no extendido, medio/anular/meñique
        # activos y palma compatible.
        #
        # El contacto de la segunda mano sobre la unión pulgar-índice puede
        # modificar la lectura de MediaPipe en la mano primaria. Por eso se
        # amplían algunos rangos de índice, pulgar y ejes respecto de O
        # aislada, pero sin permitir O abierta ni pinza aplastada.
        "expected_hands_min": 1.0,

        # Mano primaria: O-like, alineada conceptualmente con O.
        "finger_count_min": 3.60,
        "finger_count_max": 4.35,

        "thumb_extended_score_min": 0.85,
        "index_extended_score_max": 0.35,
        "middle_extended_score_min": 0.85,
        "ring_extended_score_min": 0.85,
        "pinky_extended_score_min": 0.85,

        # Cierre circular pulgar-índice en mano primaria.
        # El máximo rechaza O abierta; el mínimo ayuda a rechazar pinza
        # completamente aplastada, igual que en O.
        "thumb_index_tip_distance_norm_min": 0.045,
        "thumb_index_tip_distance_norm_max": 0.130,

        # Índice curvado separado de los dedos largos.
        # En Q puede aumentar por el contacto con la segunda mano.
        "index_middle_tip_distance_norm_min": 0.45,
        "index_middle_tip_distance_norm_max": 0.82,

        # Medio, anular y meñique abiertos/naturales.
        # Se deja un poco más de margen que en O por variación live.
        "middle_ring_tip_distance_norm_min": 0.075,
        "ring_pinky_tip_distance_norm_min": 0.160,
        "ring_pinky_tip_distance_norm_max": 0.320,

        # Índice de la O primaria. En Q puede quedar más inclinado/vertical
        # que en O aislada por el apoyo del índice secundario.
        "index_mcp_tip_angle_deg_min": 0.0,
        "index_mcp_tip_angle_deg_max": 62.0,
        "index_mcp_tip_verticality_max": 0.90,
        "index_mcp_tip_above_base_max": 0.20,

        # Dedo medio extendido hacia arriba, con más tolerancia que O aislada
        # porque la mano puede rotar levemente al recibir la cola.
        "middle_mcp_tip_angle_deg_min": -88.0,
        "middle_mcp_tip_angle_deg_max": -55.0,
        "middle_mcp_tip_horizontality_max": 0.58,

        # Pulgar de la O primaria. Se permite más horizontalidad que en O,
        # pero se mantiene límite para bloquear pinza aplastada y variantes
        # donde el pulgar queda demasiado plano.
        "thumb_mcp_tip_angle_deg_min": -70.0,
        "thumb_mcp_tip_angle_deg_max": -18.0,
        "thumb_mcp_tip_horizontality_min": 0.35,
        "thumb_mcp_tip_horizontality_max": 0.96,
        "thumb_mcp_tip_right_of_base_min": 0.80,

        # Relaciones de ejes. Se amplían respecto de O porque en Q la segunda
        # mano altera la estimación, pero se mantienen límites para rechazar
        # O abierta, pinza aplastada y orientaciones incorrectas.
        "index_middle_axis_angle_3d_deg_min": 65.0,
        "index_middle_axis_angle_3d_deg_max": 110.0,

        "thumb_index_axis_angle_3d_deg_min": 58.0,
        "thumb_index_axis_angle_3d_deg_max": 82.0,

        "thumb_middle_axis_angle_3d_deg_min": 8.0,
        "thumb_middle_axis_angle_3d_deg_max": 47.0,

        # Misma orientación general que O: normal z negativa en la cámara actual.
        "use_palm_normal_z": True,
        "palm_normal_z_approx_max": 0.001,

        # Mano secundaria: puño con índice extendido.
        # No se exige finger_count=1 exacto porque MediaPipe puede leer parte
        # del pulgar visible como dedo activo, pero sí se exige que medio,
        # anular y meñique no sean protagonistas.
        "secondary_valid_ratio_min": 0.80,

        "secondary_finger_count_min": 0.85,
        "secondary_finger_count_max": 2.35,

        "secondary_index_extended_score_min": 0.70,
        "secondary_middle_extended_score_max": 0.42,
        "secondary_ring_extended_score_max": 0.38,
        "secondary_pinky_extended_score_max": 0.38,
        "secondary_thumb_extended_score_max": 0.88,

        # Índice secundario diagonal. Se bloquean variantes verticales y
        # horizontales aunque la punta se acerque al círculo.
        "secondary_index_mcp_tip_angle_deg_min": -152.0,
        "secondary_index_mcp_tip_angle_deg_max": -118.0,
        "secondary_index_mcp_tip_verticality_min": 0.45,
        "secondary_index_mcp_tip_verticality_max": 0.88,
        "secondary_index_mcp_tip_horizontality_min": 0.45,
        "secondary_index_mcp_tip_horizontality_max": 0.88,

        # Relación entre manos: la punta del índice secundario debe tocar
        # o quedar muy próxima al punto de unión pulgar-índice de la O primaria.
        # Esto separa Q de O + índice tocando cualquier otra parte del círculo.
        "q_secondary_index_tip_to_o_join_distance_norm_max": 0.11,
    },


    "S": {
        "min_valid_ratio": 0.80,

        # S correcta según el dataset actual:
        #   - mano derecha en forma de pistola;
        #   - pulgar e índice extendidos/activos;
        #   - medio, anular y meñique recogidos;
        #   - dorso de la mano hacia cámara;
        #   - punta del índice ubicada en el centro de la barbilla.
        #
        # Decisión 2026-05-24:
        # La S no se valida solo por forma de mano. La configuración
        # "pistola" también aparece fuera del contexto correcto, por lo
        # que se agrega una capa espacial facial. Como MediaPipe Pose no
        # expone un landmark directo de mentón, se aproxima la barbilla
        # con un proxy proyectado desde nariz hacia el centro de boca:
        #   chin_proxy = mouth_center + (mouth_center - nose)
        # La punta del índice debe quedar próxima a ese proxy y centrada
        # horizontalmente, evitando aceptar pistola en boca, mejilla o cuello.
        "finger_count_min": 1.70,
        "finger_count_max": 2.35,

        "thumb_extended_score_min": 0.88,
        "index_extended_score_min": 0.60,

        "middle_extended_score_max": 0.38,
        "ring_extended_score_max": 0.40,
        "pinky_extended_score_max": 0.40,

        # Apertura de pistola: pulgar e índice deben estar claramente
        # separados. No se usa un mínimo extremo para tolerar pulgar algo
        # relajado, pero sí se bloquea pulgar recogido/oculto.
        "thumb_index_tip_distance_norm_min": 0.75,
        "thumb_index_tip_distance_norm_max": 1.55,

        # El índice extendido debe separarse de los dedos recogidos.
        # Esto ayuda a rechazar índice+medio extendidos, mano abierta o
        # variantes donde los dedos largos no quedan compactos.
        "index_middle_tip_distance_norm_min": 0.68,

        # Índice casi horizontal hacia la barbilla, con tolerancia a una
        # leve diagonal. No se exige above_base porque según el contacto
        # con la barbilla puede variar.
        "index_mcp_tip_angle_deg_min": -40.0,
        "index_mcp_tip_angle_deg_max": 10.0,
        "index_mcp_tip_verticality_max":0.67, #0.62,
        "index_mcp_tip_horizontality_min": 0.74, #0.78,

        # Pulgar activo pero no doblado contra la mano ni escondido.
        "thumb_mcp_tip_angle_deg_min": -95.0,
        "thumb_mcp_tip_angle_deg_max": -62.0,
        "thumb_mcp_tip_horizontality_max": 0.42,
        "thumb_mcp_tip_right_of_base_min": 0.70,

        # Con dorso hacia cámara la normal z quedó positiva en los positivos.
        # La variante con palma hacia cámara invierte el signo.
        "use_palm_normal_z": True,
        "palm_normal_z_approx_min": 0.001,

        # Relación índice-medio compatible con medio recogido.
        "index_middle_axis_angle_3d_deg_min": 85.0,

        # Capa espacial facial: punta del índice en zona de barbilla.
        # Las métricas están normalizadas por la distancia entre orejas.
        "pose_valid_for_s_chin_geometry_ratio_min": 0.75,

        "index_tip_chin_dx_earspan_min": -0.18,
        "index_tip_chin_dx_earspan_max": 0.12,

        "index_tip_chin_dy_earspan_min": -0.10,
        "index_tip_chin_dy_earspan_max": 0.22,

        "index_tip_chin_dist_earspan_max": 0.22,
    },


    "T": {
        "min_valid_ratio": 0.80,

        # T correcta según el dataset actual:
        #   - mano derecha en puño con dorso hacia cámara;
        #   - solo el índice queda extendido;
        #   - el índice queda recto/casi vertical;
        #   - la punta del índice se apoya en el punto medio de la barbilla;
        #   - el pulgar puede verse parcialmente por la orientación, pero no
        #     debe volverse protagonista.
        #
        # Decisión 2026-05-25:
        # La T no se valida solo como "puño con índice vertical" porque una
        # configuración parecida puede aparecer lejos de la cara. Por eso se
        # combina forma de mano + orientación dorsal + capa espacial de barbilla.
        # Como MediaPipe Pose no expone un landmark exacto de mentón, se reutiliza
        # el proxy ya usado en S:
        #   chin_proxy = mouth_center + (mouth_center - nose)
        # La punta del índice debe quedar centrada y próxima a esa zona.
        "finger_count_min": 0.85,
        "finger_count_max": 2.25,

        # El índice debe ser el único dedo largo protagonista. El pulgar se
        # permite visible porque MediaPipe lo estima entre ~0.49 y ~0.77 en
        # positivos correctos, según la oclusión del puño.
        "thumb_extended_score_max": 0.85,
        "index_extended_score_min": 0.90,
        "middle_extended_score_max": 0.20,
        "ring_extended_score_max": 0.20,
        "pinky_extended_score_max": 0.22,

        # El índice extendido debe quedar claramente separado de los dedos
        # recogidos. Esto bloquea puño cerrado e índice+medio extendidos.
        "index_middle_tip_distance_norm_min": 0.85,

        # Índice vertical/recto sobre la barbilla.
        "index_mcp_tip_angle_deg_min": -92.0,
        "index_mcp_tip_angle_deg_max": -82.0,
        "index_mcp_tip_verticality_min": 0.99,
        "index_mcp_tip_above_base_min": 0.80,

        # Orientación de dorso hacia cámara. Palma frontal invierte el signo;
        # mano de canto/lateral baja demasiado la normal z.
        "use_palm_normal_z": True,
        "palm_normal_z_approx_min": 0.0035,
        "palm_normal_z_approx_max": 0.0065,

        # Señal complementaria para rechazar palma/lateral cuando el signo de
        # normal z no alcanza por sí solo o hay jitter.
        "middle_index_axis_signed_range_max": 0.18,

        # Capa espacial facial: punta del índice en zona de barbilla.
        # Las métricas están normalizadas por la distancia entre orejas.
        "pose_valid_for_t_chin_geometry_ratio_min": 0.75,

        "index_tip_chin_dx_earspan_min": -0.12,
        "index_tip_chin_dx_earspan_max": 0.08,

        "index_tip_chin_dy_earspan_min": -0.26,
        "index_tip_chin_dy_earspan_max": 0.08,

        "index_tip_chin_dist_earspan_max": 0.25,
    },


    "L": {
        "min_valid_ratio": 0.80,

        "finger_count_min": 1.80,
        "finger_count_max": 2.20,

        "thumb_extended_score_min": 0.80,
        "index_extended_score_min": 0.90,

        "middle_extended_score_max": 0.55,
        "ring_extended_score_max": 0.55,
        "pinky_extended_score_max": 0.55,

        "thumb_index_tip_distance_norm_min": 1.40,
        "thumb_index_angle_deg_min": 50.0,
        "thumb_index_angle_deg_max": 85.0,

        "index_mcp_tip_angle_deg_min": -110.0,
        "index_mcp_tip_angle_deg_max": -70.0,
        "index_mcp_tip_verticality_min": 0.95,
        "index_mcp_tip_above_base_min": 0.80,

        "thumb_mcp_tip_angle_deg_min": -25.0,
        "thumb_mcp_tip_angle_deg_max": 25.0,
        "thumb_mcp_tip_horizontality_min": 0.90,
        "thumb_mcp_tip_right_of_base_min": 0.80,
        "thumb_mcp_tip_left_of_base_max": 0.20,

        # En las pruebas actuales, L correcta quedó con normal negativa
        # y L con dorso a cámara quedó con normal positiva.
        # Si este criterio se vuelve inestable con otro sujeto/cámara,
        # conviene desactivarlo o parametrizarlo.
        "use_palm_normal_z": True,
        "palm_normal_z_approx_max": 0.0,
    },

    "K": {
        "min_valid_ratio": 0.80,

        # En la K actual el conteo global sigue dando 2,
        # aun cuando el medio esté visualmente extendido en otro plano.
        "finger_count_min": 1.80,
        "finger_count_max": 2.20,

        "thumb_extended_score_min": 0.84,
        "index_extended_score_min": 0.70,
        "middle_extended_score_min": 0.30,

        "ring_extended_score_max": 0.30,
        "pinky_extended_score_max": 0.30,

        # Pulgar cerca del vértice índice/medio; no pegado ni demasiado lejos.
        "thumb_index_tip_distance_norm_min": 0.34,
        "thumb_index_tip_distance_norm_max": 0.62,

        # Apertura índice-medio de la K real.
        "index_middle_tip_distance_norm_min": 0.62,
        "index_middle_tip_distance_norm_max": 0.95,

        # Anular y meñique cerrados/compactos.
        "middle_ring_tip_distance_norm_min": 0.45,
        "ring_pinky_tip_distance_norm_max": 0.20,

        # Índice diagonal hacia arriba.
        "index_mcp_tip_angle_deg_min": -82.0,
        "index_mcp_tip_angle_deg_max": -50.0,
        "index_mcp_tip_verticality_min": 0.80,
        "index_mcp_tip_above_base_min": 0.80,

        # Medio casi horizontal hacia la derecha.
        "middle_mcp_tip_angle_deg_min": -15.0,
        "middle_mcp_tip_angle_deg_max": 15.0,
        "middle_mcp_tip_horizontality_min": 0.90,
        "middle_mcp_tip_right_of_base_min": 0.80,

        # Pulgar visible en el vértice, en la orientación observada.
        "thumb_mcp_tip_angle_deg_min": -100.0,
        "thumb_mcp_tip_angle_deg_max": -65.0,
        "thumb_mcp_tip_right_of_base_min": 0.70,

        # Relaciones de ejes entre dedos. Estas son claves para separar:
        # - K correcta;
        # - apertura demasiado cerrada;
        # - orientación tipo L/V;
        # - pulgar mal ubicado.
        "index_middle_axis_angle_3d_deg_min": 50.0,
        "index_middle_axis_angle_3d_deg_max": 72.0,
        "thumb_index_axis_angle_3d_deg_max": 24.0,
        "thumb_middle_axis_angle_3d_deg_min": 65.0,
        "thumb_middle_axis_angle_3d_deg_max": 95.0,
    },

    "R": {
        "min_valid_ratio": 0.80,

        # En R, MediaPipe suele contar 3 dedos extendidos:
        # índice + medio + pulgar visible/doblado. No se interpreta
        # como pulgar abierto tipo L/K, sino como pulgar presente.
        "finger_count_min": 2.70,
        "finger_count_max": 3.30,

        "index_extended_score_min": 0.82,
        "middle_extended_score_min": 0.88,

        "ring_extended_score_max": 0.30,
        "pinky_extended_score_max": 0.32,

        # Índice y medio cruzados/pegados: en R correcta la separación
        # es muy baja. Esto separa R de V/U y de variantes sin cruce.
        "index_middle_tip_distance_norm_max": 0.09,

        # El medio debe separarse claramente de anular, mientras anular
        # y meñique permanecen recogidos/compactos.
        "middle_ring_tip_distance_norm_min": 0.58,
        "ring_pinky_tip_distance_norm_max": 0.13,

        # Índice y medio apuntan hacia arriba, casi paralelos.
        "index_mcp_tip_angle_deg_min": -100.0,
        "index_mcp_tip_angle_deg_max": -70.0,
        "index_mcp_tip_verticality_min": 0.95,
        "index_mcp_tip_above_base_min": 0.80,

        "middle_mcp_tip_angle_deg_min": -92.0,
        "middle_mcp_tip_angle_deg_max": -68.0,
        "middle_mcp_tip_above_base_min": 0.80,

        "index_middle_axis_angle_3d_deg_max": 15.0,

        # El medio debe envolver/cruzar al índice, no quedar solo pegado
        # en paralelo. Se mide el rango lateral firmado de MCP/PIP/DIP/TIP
        # del dedo medio respecto del eje MCP->TIP del índice.
        "middle_index_axis_signed_range_min": 0.060,

        # En esta R, el pulgar está doblado y apoyado sobre el anular.
        # La posición lateral y el ángulo respecto al medio son más
        # estables que thumb_extended_score.
        "thumb_mcp_tip_left_of_base_min": 0.80,
        "thumb_mcp_tip_right_of_base_max": 0.20,
        "thumb_mcp_tip_angle_deg_min": -115.0,
        "thumb_mcp_tip_angle_deg_max": -95.0,
        "thumb_index_axis_angle_3d_deg_min": 10.0,
        "thumb_index_axis_angle_3d_deg_max": 25.0,
        "thumb_middle_axis_angle_3d_deg_min": 20.0,
        "thumb_middle_axis_angle_3d_deg_max": 32.0,
    },
}

FINGERS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

FINGER_TIPS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}


@dataclass
class StaticGeometryValidationResult:
    label: str
    required: bool
    ok: bool
    message: str

    metrics: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    expected_hands: int = 1
    frames_used: int = 0
    valid_frames: int = 0
    valid_ratio: float = 0.0

    def to_dict(self) -> dict:
        data = {
            "label": self.label,
            "required": self.required,
            "ok": self.ok,
            "message": self.message,
            "expected_hands": self.expected_hands,
            "frames_used": self.frames_used,
            "valid_frames": self.valid_frames,
            "valid_ratio": self.valid_ratio,
            "reasons": self.reasons,
        }

        for key, value in self.metrics.items():
            data[key] = value

        return data

    def get(self, key: str, default=None):
        if hasattr(self, key):
            return getattr(self, key)

        if key in self.metrics:
            return self.metrics.get(key, default)

        return self.to_dict().get(key, default)


def is_static_geometry_label(label: str) -> bool:
    return str(label).upper() in STATIC_GEOMETRY_LABELS


def _landmark_value(landmark: Any, name: str, default: float = 0.0) -> float:
    value = default

    if hasattr(landmark, name):
        value = getattr(landmark, name)
    elif isinstance(landmark, dict) and name in landmark:
        value = landmark[name]

    if value is None:
        return float(default)

    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def landmarks_to_array(hand_landmarks: Any) -> np.ndarray:
    arr = np.full((21, 3), np.nan, dtype=np.float32)

    if hand_landmarks is None:
        return arr

    for idx, landmark in enumerate(hand_landmarks[:21]):
        arr[idx, 0] = _landmark_value(landmark, "x", np.nan)
        arr[idx, 1] = _landmark_value(landmark, "y", np.nan)
        arr[idx, 2] = _landmark_value(landmark, "z", np.nan)

    return arr


def _get_pose_landmarks_first_person(pose_result: Any):
    if pose_result is None:
        return None

    pose_landmarks = getattr(pose_result, "pose_landmarks", None)

    if pose_landmarks is None and isinstance(pose_result, dict):
        pose_landmarks = pose_result.get("pose_landmarks")

    if pose_landmarks is None:
        return None

    if hasattr(pose_landmarks, "landmark"):
        return list(pose_landmarks.landmark)

    if isinstance(pose_landmarks, (list, tuple)):
        if not pose_landmarks:
            return None

        first = pose_landmarks[0]

        if isinstance(first, (list, tuple)):
            return first

        return pose_landmarks

    return None


def pose_landmarks_to_array(pose_result: Any) -> np.ndarray:
    arr = np.full((33, 4), np.nan, dtype=np.float32)
    landmarks = _get_pose_landmarks_first_person(pose_result)

    if landmarks is None:
        return arr

    for idx, landmark in enumerate(landmarks[:33]):
        arr[idx, 0] = _landmark_value(landmark, "x", np.nan)
        arr[idx, 1] = _landmark_value(landmark, "y", np.nan)
        arr[idx, 2] = _landmark_value(landmark, "z", np.nan)
        arr[idx, 3] = _landmark_value(landmark, "visibility", 1.0)

    return arr


def safe_div(num: float, den: float) -> float:
    if den is None or not np.isfinite(den) or abs(float(den)) <= 1e-8:
        return np.nan

    return float(num / den)


def clip01(value: float) -> float:
    if not np.isfinite(value):
        return np.nan

    return float(np.clip(value, 0.0, 1.0))


def distance_2d(points: np.ndarray, a: int, b: int) -> float:
    pa = points[a, :2]
    pb = points[b, :2]

    if not np.all(np.isfinite(pa)) or not np.all(np.isfinite(pb)):
        return np.nan

    return float(np.linalg.norm(pb - pa))


def angle_between_vectors_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    if not np.all(np.isfinite(v1)) or not np.all(np.isfinite(v2)):
        return np.nan

    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))

    if n1 <= 1e-8 or n2 <= 1e-8:
        return np.nan

    cosine = float(np.dot(v1, v2) / (n1 * n2))
    cosine = float(np.clip(cosine, -1.0, 1.0))

    return float(np.degrees(np.arccos(cosine)))


def joint_angle_deg(points: np.ndarray, a: int, b: int, c: int) -> float:
    pa = points[a, :3]
    pb = points[b, :3]
    pc = points[c, :3]

    if not np.all(np.isfinite(pa)) or not np.all(np.isfinite(pb)) or not np.all(np.isfinite(pc)):
        return np.nan

    v1 = pa - pb
    v2 = pc - pb

    return angle_between_vectors_deg(v1, v2)


def estimate_hand_scale(points: np.ndarray) -> dict[str, float]:
    wrist_to_middle_mcp = distance_2d(points, 0, 9)
    index_mcp_to_pinky_mcp = distance_2d(points, 5, 17)
    wrist_to_middle_tip = distance_2d(points, 0, 12)

    values = [
        value for value in [
            wrist_to_middle_mcp,
            index_mcp_to_pinky_mcp,
            wrist_to_middle_tip,
        ]
        if np.isfinite(value) and value > 1e-8
    ]

    hand_scale = float(max(values)) if values else np.nan

    xs = points[:, 0]
    ys = points[:, 1]

    finite_x = xs[np.isfinite(xs)]
    finite_y = ys[np.isfinite(ys)]

    if len(finite_x) > 0 and len(finite_y) > 0:
        bbox_width = float(np.max(finite_x) - np.min(finite_x))
        bbox_height = float(np.max(finite_y) - np.min(finite_y))
        bbox_area = bbox_width * bbox_height
    else:
        bbox_width = np.nan
        bbox_height = np.nan
        bbox_area = np.nan

    return {
        "hand_scale": hand_scale,
        "palm_width": index_mcp_to_pinky_mcp,
        "palm_height": wrist_to_middle_mcp,
        "hand_height": wrist_to_middle_tip,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area": bbox_area,
        "bbox_aspect_ratio": safe_div(bbox_width, bbox_height),
        "bbox_area_norm": safe_div(bbox_area, hand_scale * hand_scale) if np.isfinite(hand_scale) else np.nan,
    }


def finger_geometry(points: np.ndarray, finger: str, hand_scale: float) -> dict[str, float]:
    ids = FINGERS[finger]

    if finger == "thumb":
        base, mcp, ip, tip = ids
        bone_pairs = [(base, mcp), (mcp, ip), (ip, tip)]
        base_idx = base
        mid_a, mid_b, mid_c = base, mcp, ip
        distal_a, distal_b, distal_c = mcp, ip, tip
        wrist_reference = mcp
    else:
        mcp, pip, dip, tip = ids
        bone_pairs = [(mcp, pip), (pip, dip), (dip, tip)]
        base_idx = mcp
        mid_a, mid_b, mid_c = mcp, pip, dip
        distal_a, distal_b, distal_c = pip, dip, tip
        wrist_reference = mcp

    bone_lengths = []
    for a, b in bone_pairs:
        d = distance_2d(points, a, b)
        if np.isfinite(d):
            bone_lengths.append(d)

    bone_length_sum = float(np.sum(bone_lengths)) if bone_lengths else np.nan

    base_to_tip = distance_2d(points, base_idx, tip)
    wrist_to_tip = distance_2d(points, 0, tip)
    wrist_to_reference = distance_2d(points, 0, wrist_reference)

    straightness = safe_div(base_to_tip, bone_length_sum)
    curl_score = 1.0 - straightness if np.isfinite(straightness) else np.nan

    extension_ratio = safe_div(wrist_to_tip, wrist_to_reference)

    if finger == "thumb":
        raw_extension_score = safe_div(extension_ratio - 1.00, 0.75)
    else:
        raw_extension_score = safe_div(extension_ratio - 1.10, 0.80)

    extension_score = clip01(raw_extension_score)

    if np.isfinite(extension_score) and np.isfinite(straightness):
        extended_score = clip01(0.65 * extension_score + 0.35 * straightness)
    else:
        extended_score = np.nan

    extended_est = bool(extended_score >= 0.55) if np.isfinite(extended_score) else False

    pip_angle = joint_angle_deg(points, mid_a, mid_b, mid_c)
    dip_angle = joint_angle_deg(points, distal_a, distal_b, distal_c)

    return {
        f"{finger}_base_to_tip_norm": safe_div(base_to_tip, hand_scale),
        f"{finger}_wrist_to_tip_norm": safe_div(wrist_to_tip, hand_scale),
        f"{finger}_bone_length_sum_norm": safe_div(bone_length_sum, hand_scale),
        f"{finger}_straightness": straightness,
        f"{finger}_curl": curl_score,
        f"{finger}_extension_ratio": extension_ratio,
        f"{finger}_extended_score": extended_score,
        f"{finger}_extended_est": int(extended_est),
        f"{finger}_pip_angle_deg": pip_angle,
        f"{finger}_dip_angle_deg": dip_angle,
    }


def hand_orientation_metrics(points: np.ndarray) -> dict[str, float]:
    wrist = points[0, :3]
    index_mcp = points[5, :3]
    middle_mcp = points[9, :3]
    pinky_mcp = points[17, :3]

    if not (
        np.all(np.isfinite(wrist))
        and np.all(np.isfinite(index_mcp))
        and np.all(np.isfinite(middle_mcp))
        and np.all(np.isfinite(pinky_mcp))
    ):
        return {
            "palm_axis_angle_deg": np.nan,
            "palm_width_angle_deg": np.nan,
            "palm_normal_z_approx": np.nan,
            "palm_area_2d": np.nan,
        }

    palm_axis = middle_mcp[:2] - wrist[:2]
    palm_width_vec = pinky_mcp[:2] - index_mcp[:2]

    palm_axis_angle = float(np.degrees(np.arctan2(palm_axis[1], palm_axis[0])))
    palm_width_angle = float(np.degrees(np.arctan2(palm_width_vec[1], palm_width_vec[0])))

    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist
    normal = np.cross(v1, v2)

    palm_area_2d = abs(float(v1[0] * v2[1] - v1[1] * v2[0])) / 2.0

    return {
        "palm_axis_angle_deg": palm_axis_angle,
        "palm_width_angle_deg": palm_width_angle,
        "palm_normal_z_approx": float(normal[2]),
        "palm_area_2d": palm_area_2d,
    }


def pairwise_finger_metrics(points: np.ndarray, hand_scale: float) -> dict[str, float]:
    out = {}

    pairs = [
        ("thumb", "index"),
        ("index", "middle"),
        ("middle", "ring"),
        ("ring", "pinky"),
        ("index", "ring"),
        ("index", "pinky"),
        ("thumb", "middle"),
        ("thumb", "pinky"),
    ]

    wrist = points[0, :3]

    for a_name, b_name in pairs:
        a_tip = FINGER_TIPS[a_name]
        b_tip = FINGER_TIPS[b_name]

        d = distance_2d(points, a_tip, b_tip)
        out[f"{a_name}_{b_name}_tip_distance_norm"] = safe_div(d, hand_scale)

        va = points[a_tip, :3] - wrist
        vb = points[b_tip, :3] - wrist
        out[f"{a_name}_{b_name}_angle_deg"] = angle_between_vectors_deg(va, vb)

    return out


def finger_direction_metrics(points: np.ndarray, hand_scale: float) -> dict[str, float]:
    """
    Dirección 2D y 3D de dedos clave.

    Coordenadas de imagen:
    - x aumenta hacia la derecha.
    - y aumenta hacia abajo.
    - z es profundidad relativa de MediaPipe.

    En general, en MediaPipe Hands valores z menores suelen indicar puntos
    más cercanos a la cámara. Para las reglas iniciales se usan principalmente
    ángulos/distancias 2D y 3D, no z absoluto.
    """

    def vector_metrics(prefix: str, base_idx: int, tip_idx: int) -> dict[str, float]:
        base = points[base_idx, :3]
        tip = points[tip_idx, :3]

        if not np.all(np.isfinite(base)) or not np.all(np.isfinite(tip)):
            return {
                f"{prefix}_dx_norm": np.nan,
                f"{prefix}_dy_norm": np.nan,
                f"{prefix}_dz_norm": np.nan,
                f"{prefix}_length_norm": np.nan,
                f"{prefix}_length_3d_norm": np.nan,
                f"{prefix}_angle_deg": np.nan,
                f"{prefix}_verticality": np.nan,
                f"{prefix}_horizontality": np.nan,
                f"{prefix}_depth_fraction": np.nan,
                f"{prefix}_tip_closer_than_base": np.nan,
                f"{prefix}_tip_farther_than_base": np.nan,
                f"{prefix}_above_base": np.nan,
                f"{prefix}_below_base": np.nan,
                f"{prefix}_left_of_base": np.nan,
                f"{prefix}_right_of_base": np.nan,
            }

        dx = float(tip[0] - base[0])
        dy = float(tip[1] - base[1])
        dz = float(tip[2] - base[2])

        length_2d = float(np.sqrt(dx * dx + dy * dy))
        length_3d = float(np.sqrt(dx * dx + dy * dy + dz * dz))

        if length_2d <= 1e-8:
            angle_deg = np.nan
            verticality = np.nan
            horizontality = np.nan
        else:
            angle_deg = float(np.degrees(np.arctan2(dy, dx)))
            verticality = float(abs(dy) / length_2d)
            horizontality = float(abs(dx) / length_2d)

        if length_3d <= 1e-8:
            depth_fraction = np.nan
        else:
            depth_fraction = float(abs(dz) / length_3d)

        return {
            f"{prefix}_dx_norm": safe_div(dx, hand_scale),
            f"{prefix}_dy_norm": safe_div(dy, hand_scale),
            f"{prefix}_dz_norm": safe_div(dz, hand_scale),
            f"{prefix}_length_norm": safe_div(length_2d, hand_scale),
            f"{prefix}_length_3d_norm": safe_div(length_3d, hand_scale),
            f"{prefix}_angle_deg": angle_deg,
            f"{prefix}_verticality": verticality,
            f"{prefix}_horizontality": horizontality,
            f"{prefix}_depth_fraction": depth_fraction,
            f"{prefix}_tip_closer_than_base": int(dz < 0),
            f"{prefix}_tip_farther_than_base": int(dz > 0),
            f"{prefix}_above_base": int(dy < 0),
            f"{prefix}_below_base": int(dy > 0),
            f"{prefix}_left_of_base": int(dx < 0),
            f"{prefix}_right_of_base": int(dx > 0),
        }

    def axis_vector_2d(base_idx: int, tip_idx: int) -> np.ndarray:
        return points[tip_idx, :2] - points[base_idx, :2]

    def axis_vector_3d(base_idx: int, tip_idx: int) -> np.ndarray:
        return points[tip_idx, :3] - points[base_idx, :3]

    def signed_angle_2d_deg(v1: np.ndarray, v2: np.ndarray) -> float:
        if not np.all(np.isfinite(v1)) or not np.all(np.isfinite(v2)):
            return np.nan

        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))

        if n1 <= 1e-8 or n2 <= 1e-8:
            return np.nan

        cross = float(v1[0] * v2[1] - v1[1] * v2[0])
        dot = float(np.dot(v1, v2))

        return float(np.degrees(np.arctan2(cross, dot)))

    def point_to_axis_signed_distance_norm(
        point_idx: int,
        axis_base_idx: int,
        axis_tip_idx: int,
    ) -> float:
        axis_base = points[axis_base_idx, :2]
        axis_tip = points[axis_tip_idx, :2]
        point = points[point_idx, :2]

        if (
            not np.all(np.isfinite(axis_base))
            or not np.all(np.isfinite(axis_tip))
            or not np.all(np.isfinite(point))
        ):
            return np.nan

        axis = axis_tip - axis_base
        axis_len = float(np.linalg.norm(axis))

        if axis_len <= 1e-8:
            return np.nan

        rel = point - axis_base
        signed_distance = float(axis[0] * rel[1] - axis[1] * rel[0]) / axis_len

        return safe_div(signed_distance, hand_scale)

    def sign_changes(values: list[float], eps: float = 0.005) -> float:
        signs: list[int] = []

        for value in values:
            if not np.isfinite(value):
                continue

            if value > eps:
                signs.append(1)
            elif value < -eps:
                signs.append(-1)
            else:
                signs.append(0)

        non_zero = [sign for sign in signs if sign != 0]

        if len(non_zero) <= 1:
            return 0.0

        return float(
            sum(
                1
                for prev, curr in zip(non_zero[:-1], non_zero[1:])
                if prev != curr
            )
        )

    out = {}

    out.update(vector_metrics("index_mcp_tip", 5, 8))
    out.update(vector_metrics("middle_mcp_tip", 9, 12))
    out.update(vector_metrics("ring_mcp_tip", 13, 16))
    out.update(vector_metrics("pinky_mcp_tip", 17, 20))
    out.update(vector_metrics("thumb_mcp_tip", 2, 4))
    out.update(vector_metrics("thumb_cmc_tip", 1, 4))

    index_axis_2d = axis_vector_2d(5, 8)
    middle_axis_2d = axis_vector_2d(9, 12)
    thumb_axis_2d = axis_vector_2d(2, 4)

    index_axis_3d = axis_vector_3d(5, 8)
    middle_axis_3d = axis_vector_3d(9, 12)
    thumb_axis_3d = axis_vector_3d(2, 4)

    out["index_middle_axis_angle_2d_deg"] = angle_between_vectors_deg(
        index_axis_2d,
        middle_axis_2d,
    )
    out["index_middle_axis_angle_3d_deg"] = angle_between_vectors_deg(
        index_axis_3d,
        middle_axis_3d,
    )
    out["index_middle_axis_signed_angle_2d_deg"] = signed_angle_2d_deg(
        index_axis_2d,
        middle_axis_2d,
    )

    out["thumb_index_axis_angle_2d_deg"] = angle_between_vectors_deg(
        thumb_axis_2d,
        index_axis_2d,
    )
    out["thumb_index_axis_angle_3d_deg"] = angle_between_vectors_deg(
        thumb_axis_3d,
        index_axis_3d,
    )
    out["thumb_middle_axis_angle_2d_deg"] = angle_between_vectors_deg(
        thumb_axis_2d,
        middle_axis_2d,
    )
    out["thumb_middle_axis_angle_3d_deg"] = angle_between_vectors_deg(
        thumb_axis_3d,
        middle_axis_3d,
    )

    # Métricas específicas para R:
    # el dedo medio debe cruzar/envolver el eje del índice. Para capturar
    # esa diferencia se mide la distancia lateral firmada de los puntos del
    # medio respecto al eje índice MCP->TIP. Una R falsa con medio solo
    # paralelo/pegado suele tener un rango firmado mucho menor.
    middle_axis_signed_values = {
        "middle_mcp_to_index_axis_signed_norm": point_to_axis_signed_distance_norm(9, 5, 8),
        "middle_pip_to_index_axis_signed_norm": point_to_axis_signed_distance_norm(10, 5, 8),
        "middle_dip_to_index_axis_signed_norm": point_to_axis_signed_distance_norm(11, 5, 8),
        "middle_tip_to_index_axis_signed_norm": point_to_axis_signed_distance_norm(12, 5, 8),
    }
    out.update(middle_axis_signed_values)

    signed_values = [
        value
        for value in middle_axis_signed_values.values()
        if np.isfinite(value)
    ]

    if signed_values:
        signed_min = float(np.min(signed_values))
        signed_max = float(np.max(signed_values))
        signed_range = float(signed_max - signed_min)
    else:
        signed_min = np.nan
        signed_max = np.nan
        signed_range = np.nan

    out["middle_index_axis_signed_min"] = signed_min
    out["middle_index_axis_signed_max"] = signed_max
    out["middle_index_axis_signed_range"] = signed_range
    out["middle_index_axis_sign_changes"] = sign_changes(signed_values)

    return out


def empty_eye_hand_spatial_metrics() -> dict[str, float]:
    return {
        "pose_valid_for_eye_geometry": 0.0,
        "pose_valid_for_ear_geometry": 0.0,
        "pose_valid_for_h_face_geometry": 0.0,
        "pose_valid_for_j_jaw_geometry": 0.0,
        "pose_valid_for_j_jaw_geometry_ratio": np.nan,
        "pose_valid_for_i_face_geometry": 0.0,
        "pose_valid_for_i_face_geometry_ratio": np.nan,
        "pose_valid_for_y_mouth_geometry": 0.0,
        "pose_valid_for_y_mouth_geometry_ratio": np.nan,
        "pose_valid_for_s_chin_geometry": 0.0,
        "pose_valid_for_s_chin_geometry_ratio": np.nan,
        "pose_valid_for_t_chin_geometry": 0.0,
        "pose_valid_for_t_chin_geometry_ratio": np.nan,
        "pose_valid_for_f_chest_geometry": 0.0,
        "pose_valid_for_f_chest_geometry_ratio": np.nan,

        "shoulder_center_x": np.nan,
        "shoulder_center_y": np.nan,
        "shoulder_width": np.nan,
        "hand_center_shoulder_dx": np.nan,
        "hand_center_shoulder_dy": np.nan,
        "hand_center_shoulder_dist": np.nan,

        "mouth_center_x": np.nan,
        "mouth_center_y": np.nan,
        "mouth_dx_earspan": np.nan,
        "mouth_dy_earspan": np.nan,
        "mouth_center_dist_earspan": np.nan,

        "index_tip_x": np.nan,
        "index_tip_y": np.nan,
        "index_mcp_x": np.nan,
        "index_mcp_y": np.nan,

        "thumb_tip_x": np.nan,
        "thumb_tip_y": np.nan,
        "pinky_tip_x": np.nan,
        "pinky_tip_y": np.nan,
        "pinky_tip_mouth_dx_earspan": np.nan,
        "pinky_tip_mouth_dy_earspan": np.nan,
        "pinky_tip_mouth_dist_earspan": np.nan,
        "thumb_tip_mouth_dx_earspan": np.nan,
        "thumb_tip_mouth_dy_earspan": np.nan,
        "thumb_tip_mouth_dist_earspan": np.nan,

        "index_tip_right_eye_dx_earspan": np.nan,
        "index_tip_right_eye_dy_earspan": np.nan,
        "index_tip_right_eye_dist_earspan": np.nan,
        "index_tip_left_eye_dx_earspan": np.nan,
        "index_tip_left_eye_dy_earspan": np.nan,
        "index_tip_left_eye_dist_earspan": np.nan,
        "index_tip_nose_dx_earspan": np.nan,
        "index_tip_nose_dy_earspan": np.nan,
        "index_tip_nose_dist_earspan": np.nan,
        "index_tip_mouth_dx_earspan": np.nan,
        "index_tip_mouth_dy_earspan": np.nan,
        "index_tip_mouth_dist_earspan": np.nan,
        "chin_proxy_x": np.nan,
        "chin_proxy_y": np.nan,
        "index_tip_chin_dx_earspan": np.nan,
        "index_tip_chin_dy_earspan": np.nan,
        "index_tip_chin_dist_earspan": np.nan,
        "index_mcp_right_eye_dx_earspan": np.nan,
        "index_mcp_right_eye_dy_earspan": np.nan,
        "index_mcp_right_eye_dist_earspan": np.nan,
        "hand_center_right_eye_dx_earspan": np.nan,
        "hand_center_right_eye_dy_earspan": np.nan,
        "hand_center_right_eye_dist_earspan": np.nan,

        "right_eye_inside_hand_bbox": np.nan,
        "left_eye_inside_hand_bbox": np.nan,
        "nose_inside_hand_bbox": np.nan,
        "same_eye_inside_hand_bbox": np.nan,
        "other_eye_inside_hand_bbox": np.nan,
        "same_eye_thumb_index_t": np.nan,
        "same_eye_thumb_index_distance_norm": np.nan,
        "right_eye_thumb_index_t": np.nan,
        "right_eye_thumb_index_distance_norm": np.nan,
        "left_eye_thumb_index_t": np.nan,
        "left_eye_thumb_index_distance_norm": np.nan,
        "same_eye_inside_hand_bbox_ratio": np.nan,
        "other_eye_inside_hand_bbox_ratio": np.nan,
        "right_eye_inside_hand_bbox_ratio": np.nan,
        "left_eye_inside_hand_bbox_ratio": np.nan,
        "nose_inside_hand_bbox_ratio": np.nan,
        "pose_valid_for_eye_geometry_ratio": np.nan,
        "pose_valid_for_h_face_geometry_ratio": np.nan,

        "nose_dx_earspan": np.nan,
        "nose_dy_earspan": np.nan,
        "nose_center_dist_earspan": np.nan,

        "right_ear_x": np.nan,
        "right_ear_y": np.nan,
        "left_ear_x": np.nan,
        "left_ear_y": np.nan,
        "ear_span": np.nan,
        "hand_center_x": np.nan,
        "hand_center_y": np.nan,
        "right_ear_dx_earspan": np.nan,
        "right_ear_dy_earspan": np.nan,
        "right_ear_center_dist_earspan": np.nan,
        "left_ear_dx_earspan": np.nan,
        "left_ear_dy_earspan": np.nan,
        "left_ear_center_dist_earspan": np.nan,
        "pose_valid_for_ear_geometry_ratio": np.nan,
    }


def _pose_point_center(
    pose_points: np.ndarray,
    indices: list[int],
    min_visibility: float = 0.30,
) -> np.ndarray:
    values = []

    for idx in indices:
        if idx < 0 or idx >= pose_points.shape[0]:
            continue

        point = pose_points[idx]

        if (
            np.all(np.isfinite(point[:2]))
            and (not np.isfinite(point[3]) or float(point[3]) >= min_visibility)
        ):
            values.append(point[:2].astype(np.float32))

    if not values:
        return np.array([np.nan, np.nan], dtype=np.float32)

    return np.mean(np.vstack(values), axis=0)


def compute_eye_hand_spatial_metrics(
    hand_landmarks: Any,
    pose_result: Any,
) -> dict[str, float]:
    metrics = empty_eye_hand_spatial_metrics()

    hand_points = landmarks_to_array(hand_landmarks)
    pose_points = pose_landmarks_to_array(pose_result)

    if (
        hand_points.shape != (21, 3)
        or pose_points.shape != (33, 4)
        or not np.any(np.isfinite(hand_points[:, :2]))
        or not np.any(np.isfinite(pose_points[:, :2]))
    ):
        return metrics

    finite_hand = hand_points[:, :2][np.all(np.isfinite(hand_points[:, :2]), axis=1)]

    if finite_hand.size == 0:
        return metrics

    scale_metrics = estimate_hand_scale(hand_points)
    hand_scale = float(scale_metrics.get("hand_scale", np.nan))

    if not np.isfinite(hand_scale) or hand_scale <= 1e-8:
        return metrics

    raw_min_xy = np.min(finite_hand, axis=0)
    raw_max_xy = np.max(finite_hand, axis=0)
    hand_center = (raw_min_xy + raw_max_xy) / 2.0

    min_xy = raw_min_xy
    max_xy = raw_max_xy

    # Margen pequeño para evitar rechazos por jitter de MediaPipe, sin volver
    # demasiado permisiva la condición de ojo dentro de la mano.
    margin = 0.04 * hand_scale
    min_xy = min_xy - margin
    max_xy = max_xy + margin

    right_eye = _pose_point_center(pose_points, [4, 5, 6])
    left_eye = _pose_point_center(pose_points, [1, 2, 3])
    nose = _pose_point_center(pose_points, [0])
    mouth_center = _pose_point_center(pose_points, [9, 10])
    left_shoulder = _pose_point_center(pose_points, [11])
    right_shoulder = _pose_point_center(pose_points, [12])

    shoulder_center = np.array([np.nan, np.nan], dtype=np.float32)
    shoulder_width = np.nan
    if np.all(np.isfinite(left_shoulder)) and np.all(np.isfinite(right_shoulder)):
        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))

    chin_proxy = np.array([np.nan, np.nan], dtype=np.float32)
    if np.all(np.isfinite(mouth_center)) and np.all(np.isfinite(nose)):
        # Aproximación simple de mentón/barbilla: desde nariz hacia boca
        # se proyecta una distancia equivalente hacia abajo.
        chin_proxy = mouth_center + (mouth_center - nose)

    # MediaPipe Pose: 7 = left_ear, 8 = right_ear.
    # En la cámara/configuración actual de este proyecto, G se realiza
    # junto a right_ear.
    left_ear = _pose_point_center(pose_points, [7])
    right_ear = _pose_point_center(pose_points, [8])

    def inside_bbox(point: np.ndarray) -> float:
        if not np.all(np.isfinite(point)):
            return np.nan

        return float(
            min_xy[0] <= point[0] <= max_xy[0]
            and min_xy[1] <= point[1] <= max_xy[1]
        )

    def point_to_thumb_index_segment(point: np.ndarray) -> tuple[float, float]:
        if not np.all(np.isfinite(point)):
            return np.nan, np.nan

        thumb_tip = hand_points[4, :2]
        index_tip = hand_points[8, :2]

        if not np.all(np.isfinite(thumb_tip)) or not np.all(np.isfinite(index_tip)):
            return np.nan, np.nan

        segment = index_tip - thumb_tip
        segment_len_sq = float(np.dot(segment, segment))

        if segment_len_sq <= 1e-10:
            return np.nan, np.nan

        rel = point - thumb_tip
        t = float(np.dot(rel, segment) / segment_len_sq)
        closest_t = float(np.clip(t, 0.0, 1.0))
        closest = thumb_tip + closest_t * segment
        distance_norm = safe_div(float(np.linalg.norm(point - closest)), hand_scale)

        return t, distance_norm

    right_inside = inside_bbox(right_eye)
    left_inside = inside_bbox(left_eye)
    nose_inside = inside_bbox(nose)

    right_t, right_dist = point_to_thumb_index_segment(right_eye)
    left_t, left_dist = point_to_thumb_index_segment(left_eye)

    ear_span = np.nan
    right_ear_dx = np.nan
    right_ear_dy = np.nan
    right_ear_dist = np.nan
    left_ear_dx = np.nan
    left_ear_dy = np.nan
    left_ear_dist = np.nan
    nose_dx = np.nan
    nose_dy = np.nan
    nose_dist = np.nan
    mouth_dx = np.nan
    mouth_dy = np.nan
    mouth_dist = np.nan

    index_tip = hand_points[8, :2]
    index_mcp = hand_points[5, :2]
    thumb_tip = hand_points[4, :2]
    pinky_tip = hand_points[20, :2]

    index_tip_right_eye_dx = np.nan
    index_tip_right_eye_dy = np.nan
    index_tip_right_eye_dist = np.nan
    index_tip_left_eye_dx = np.nan
    index_tip_left_eye_dy = np.nan
    index_tip_left_eye_dist = np.nan
    index_tip_nose_dx = np.nan
    index_tip_nose_dy = np.nan
    index_tip_nose_dist = np.nan
    index_tip_mouth_dx = np.nan
    index_tip_mouth_dy = np.nan
    index_tip_mouth_dist = np.nan
    index_tip_chin_dx = np.nan
    index_tip_chin_dy = np.nan
    index_tip_chin_dist = np.nan
    index_mcp_right_eye_dx = np.nan
    index_mcp_right_eye_dy = np.nan
    index_mcp_right_eye_dist = np.nan
    hand_center_right_eye_dx = np.nan
    hand_center_right_eye_dy = np.nan
    hand_center_right_eye_dist = np.nan
    hand_center_shoulder_dx = np.nan
    hand_center_shoulder_dy = np.nan
    hand_center_shoulder_dist = np.nan

    pinky_tip_mouth_dx = np.nan
    pinky_tip_mouth_dy = np.nan
    pinky_tip_mouth_dist = np.nan
    thumb_tip_mouth_dx = np.nan
    thumb_tip_mouth_dy = np.nan
    thumb_tip_mouth_dist = np.nan

    def point_relative_to_anchor_earspan(
        point: np.ndarray,
        anchor: np.ndarray,
        scale: float,
    ) -> tuple[float, float, float]:
        if (
            not np.all(np.isfinite(point))
            or not np.all(np.isfinite(anchor))
            or not np.isfinite(scale)
            or scale <= 1e-8
        ):
            return np.nan, np.nan, np.nan

        rel = point - anchor
        dx = safe_div(float(rel[0]), scale)
        dy = safe_div(float(rel[1]), scale)
        dist = safe_div(float(np.linalg.norm(rel)), scale)
        return dx, dy, dist

    if np.all(np.isfinite(hand_center)) and np.all(np.isfinite(shoulder_center)) and np.isfinite(shoulder_width) and shoulder_width > 1e-8:
        shoulder_rel = hand_center - shoulder_center
        hand_center_shoulder_dx = safe_div(float(shoulder_rel[0]), shoulder_width)
        hand_center_shoulder_dy = safe_div(float(shoulder_rel[1]), shoulder_width)
        hand_center_shoulder_dist = safe_div(float(np.linalg.norm(shoulder_rel)), shoulder_width)

    if np.all(np.isfinite(right_ear)) and np.all(np.isfinite(left_ear)):
        ear_span = float(np.linalg.norm(left_ear - right_ear))

        if ear_span > 1e-8 and np.all(np.isfinite(hand_center)):
            right_rel = hand_center - right_ear
            left_rel = hand_center - left_ear

            right_ear_dx = safe_div(float(right_rel[0]), ear_span)
            right_ear_dy = safe_div(float(right_rel[1]), ear_span)
            right_ear_dist = safe_div(float(np.linalg.norm(right_rel)), ear_span)

            left_ear_dx = safe_div(float(left_rel[0]), ear_span)
            left_ear_dy = safe_div(float(left_rel[1]), ear_span)
            left_ear_dist = safe_div(float(np.linalg.norm(left_rel)), ear_span)

            if np.all(np.isfinite(nose)):
                nose_rel = hand_center - nose
                nose_dx = safe_div(float(nose_rel[0]), ear_span)
                nose_dy = safe_div(float(nose_rel[1]), ear_span)
                nose_dist = safe_div(float(np.linalg.norm(nose_rel)), ear_span)

            if np.all(np.isfinite(mouth_center)):
                mouth_rel = hand_center - mouth_center
                mouth_dx = safe_div(float(mouth_rel[0]), ear_span)
                mouth_dy = safe_div(float(mouth_rel[1]), ear_span)
                mouth_dist = safe_div(float(np.linalg.norm(mouth_rel)), ear_span)

            index_tip_right_eye_dx, index_tip_right_eye_dy, index_tip_right_eye_dist = (
                point_relative_to_anchor_earspan(index_tip, right_eye, ear_span)
            )
            index_tip_left_eye_dx, index_tip_left_eye_dy, index_tip_left_eye_dist = (
                point_relative_to_anchor_earspan(index_tip, left_eye, ear_span)
            )
            index_tip_nose_dx, index_tip_nose_dy, index_tip_nose_dist = (
                point_relative_to_anchor_earspan(index_tip, nose, ear_span)
            )
            index_tip_mouth_dx, index_tip_mouth_dy, index_tip_mouth_dist = (
                point_relative_to_anchor_earspan(index_tip, mouth_center, ear_span)
            )
            index_tip_chin_dx, index_tip_chin_dy, index_tip_chin_dist = (
                point_relative_to_anchor_earspan(index_tip, chin_proxy, ear_span)
            )
            index_mcp_right_eye_dx, index_mcp_right_eye_dy, index_mcp_right_eye_dist = (
                point_relative_to_anchor_earspan(index_mcp, right_eye, ear_span)
            )
            hand_center_right_eye_dx, hand_center_right_eye_dy, hand_center_right_eye_dist = (
                point_relative_to_anchor_earspan(hand_center, right_eye, ear_span)
            )

            pinky_tip_mouth_dx, pinky_tip_mouth_dy, pinky_tip_mouth_dist = (
                point_relative_to_anchor_earspan(pinky_tip, mouth_center, ear_span)
            )
            thumb_tip_mouth_dx, thumb_tip_mouth_dy, thumb_tip_mouth_dist = (
                point_relative_to_anchor_earspan(thumb_tip, mouth_center, ear_span)
            )

    # Calibración actual: el ojo del mismo lado aparece como right_eye.
    # Si se cambia de mano/cámara, revisar este supuesto con CSV.
    same_eye_inside = right_inside
    other_eye_inside = left_inside
    same_eye_t = right_t
    same_eye_dist = right_dist

    pose_valid = float(
        np.all(np.isfinite(right_eye))
        and np.all(np.isfinite(left_eye))
        and np.all(np.isfinite(nose))
    )

    pose_valid_ear = float(
        np.all(np.isfinite(right_ear))
        and np.all(np.isfinite(left_ear))
        and np.isfinite(ear_span)
        and ear_span > 1e-8
    )

    pose_valid_h_face = float(
        bool(pose_valid_ear)
        and np.all(np.isfinite(nose))
        and np.all(np.isfinite(hand_center))
    )

    pose_valid_j_jaw = float(
        bool(pose_valid_ear)
        and np.all(np.isfinite(mouth_center))
        and np.all(np.isfinite(hand_center))
    )

    pose_valid_i_face = float(
        bool(pose_valid_ear)
        and np.all(np.isfinite(right_eye))
        and np.all(np.isfinite(left_eye))
        and np.all(np.isfinite(nose))
        and np.all(np.isfinite(mouth_center))
        and np.all(np.isfinite(index_tip))
        and np.all(np.isfinite(index_mcp))
    )

    pose_valid_y_mouth = float(
        bool(pose_valid_ear)
        and np.all(np.isfinite(mouth_center))
        and np.all(np.isfinite(pinky_tip))
        and np.all(np.isfinite(thumb_tip))
    )

    pose_valid_s_chin = float(
        bool(pose_valid_ear)
        and np.all(np.isfinite(nose))
        and np.all(np.isfinite(mouth_center))
        and np.all(np.isfinite(chin_proxy))
        and np.all(np.isfinite(index_tip))
    )

    pose_valid_f_chest = float(
        np.all(np.isfinite(hand_center))
        and np.all(np.isfinite(shoulder_center))
        and np.isfinite(shoulder_width)
        and shoulder_width > 1e-8
    )

    # T usa la misma aproximación geométrica de barbilla que S,
    # pero se expone con nombre propio para mantener reglas legibles.
    pose_valid_t_chin = pose_valid_s_chin

    metrics.update(
        {
            "pose_valid_for_eye_geometry": pose_valid,
            "pose_valid_for_ear_geometry": pose_valid_ear,
            "pose_valid_for_h_face_geometry": pose_valid_h_face,
            "pose_valid_for_j_jaw_geometry": pose_valid_j_jaw,
            "pose_valid_for_i_face_geometry": pose_valid_i_face,
            "pose_valid_for_y_mouth_geometry": pose_valid_y_mouth,
            "pose_valid_for_s_chin_geometry": pose_valid_s_chin,
            "pose_valid_for_t_chin_geometry": pose_valid_t_chin,
            "pose_valid_for_f_chest_geometry": pose_valid_f_chest,
            "right_eye_inside_hand_bbox": right_inside,
            "left_eye_inside_hand_bbox": left_inside,
            "nose_inside_hand_bbox": nose_inside,
            "same_eye_inside_hand_bbox": same_eye_inside,
            "other_eye_inside_hand_bbox": other_eye_inside,
            "same_eye_thumb_index_t": same_eye_t,
            "same_eye_thumb_index_distance_norm": same_eye_dist,
            "right_eye_thumb_index_t": right_t,
            "right_eye_thumb_index_distance_norm": right_dist,
            "left_eye_thumb_index_t": left_t,
            "left_eye_thumb_index_distance_norm": left_dist,

            "shoulder_center_x": float(shoulder_center[0]) if np.all(np.isfinite(shoulder_center)) else np.nan,
            "shoulder_center_y": float(shoulder_center[1]) if np.all(np.isfinite(shoulder_center)) else np.nan,
            "shoulder_width": shoulder_width,
            "hand_center_shoulder_dx": hand_center_shoulder_dx,
            "hand_center_shoulder_dy": hand_center_shoulder_dy,
            "hand_center_shoulder_dist": hand_center_shoulder_dist,

            "mouth_center_x": float(mouth_center[0]) if np.all(np.isfinite(mouth_center)) else np.nan,
            "mouth_center_y": float(mouth_center[1]) if np.all(np.isfinite(mouth_center)) else np.nan,
            "chin_proxy_x": float(chin_proxy[0]) if np.all(np.isfinite(chin_proxy)) else np.nan,
            "chin_proxy_y": float(chin_proxy[1]) if np.all(np.isfinite(chin_proxy)) else np.nan,
            "mouth_dx_earspan": mouth_dx,
            "mouth_dy_earspan": mouth_dy,
            "mouth_center_dist_earspan": mouth_dist,

            "index_tip_x": float(index_tip[0]) if np.all(np.isfinite(index_tip)) else np.nan,
            "index_tip_y": float(index_tip[1]) if np.all(np.isfinite(index_tip)) else np.nan,
            "index_mcp_x": float(index_mcp[0]) if np.all(np.isfinite(index_mcp)) else np.nan,
            "index_mcp_y": float(index_mcp[1]) if np.all(np.isfinite(index_mcp)) else np.nan,

            "thumb_tip_x": float(thumb_tip[0]) if np.all(np.isfinite(thumb_tip)) else np.nan,
            "thumb_tip_y": float(thumb_tip[1]) if np.all(np.isfinite(thumb_tip)) else np.nan,
            "pinky_tip_x": float(pinky_tip[0]) if np.all(np.isfinite(pinky_tip)) else np.nan,
            "pinky_tip_y": float(pinky_tip[1]) if np.all(np.isfinite(pinky_tip)) else np.nan,
            "pinky_tip_mouth_dx_earspan": pinky_tip_mouth_dx,
            "pinky_tip_mouth_dy_earspan": pinky_tip_mouth_dy,
            "pinky_tip_mouth_dist_earspan": pinky_tip_mouth_dist,
            "thumb_tip_mouth_dx_earspan": thumb_tip_mouth_dx,
            "thumb_tip_mouth_dy_earspan": thumb_tip_mouth_dy,
            "thumb_tip_mouth_dist_earspan": thumb_tip_mouth_dist,

            "index_tip_right_eye_dx_earspan": index_tip_right_eye_dx,
            "index_tip_right_eye_dy_earspan": index_tip_right_eye_dy,
            "index_tip_right_eye_dist_earspan": index_tip_right_eye_dist,
            "index_tip_left_eye_dx_earspan": index_tip_left_eye_dx,
            "index_tip_left_eye_dy_earspan": index_tip_left_eye_dy,
            "index_tip_left_eye_dist_earspan": index_tip_left_eye_dist,
            "index_tip_nose_dx_earspan": index_tip_nose_dx,
            "index_tip_nose_dy_earspan": index_tip_nose_dy,
            "index_tip_nose_dist_earspan": index_tip_nose_dist,
            "index_tip_mouth_dx_earspan": index_tip_mouth_dx,
            "index_tip_mouth_dy_earspan": index_tip_mouth_dy,
            "index_tip_mouth_dist_earspan": index_tip_mouth_dist,
            "index_tip_chin_dx_earspan": index_tip_chin_dx,
            "index_tip_chin_dy_earspan": index_tip_chin_dy,
            "index_tip_chin_dist_earspan": index_tip_chin_dist,
            "index_mcp_right_eye_dx_earspan": index_mcp_right_eye_dx,
            "index_mcp_right_eye_dy_earspan": index_mcp_right_eye_dy,
            "index_mcp_right_eye_dist_earspan": index_mcp_right_eye_dist,
            "hand_center_right_eye_dx_earspan": hand_center_right_eye_dx,
            "hand_center_right_eye_dy_earspan": hand_center_right_eye_dy,
            "hand_center_right_eye_dist_earspan": hand_center_right_eye_dist,

            "right_ear_x": float(right_ear[0]) if np.all(np.isfinite(right_ear)) else np.nan,
            "right_ear_y": float(right_ear[1]) if np.all(np.isfinite(right_ear)) else np.nan,
            "left_ear_x": float(left_ear[0]) if np.all(np.isfinite(left_ear)) else np.nan,
            "left_ear_y": float(left_ear[1]) if np.all(np.isfinite(left_ear)) else np.nan,
            "ear_span": ear_span,
            "hand_center_x": float(hand_center[0]) if np.all(np.isfinite(hand_center)) else np.nan,
            "hand_center_y": float(hand_center[1]) if np.all(np.isfinite(hand_center)) else np.nan,
            "nose_dx_earspan": nose_dx,
            "nose_dy_earspan": nose_dy,
            "nose_center_dist_earspan": nose_dist,
            "right_ear_dx_earspan": right_ear_dx,
            "right_ear_dy_earspan": right_ear_dy,
            "right_ear_center_dist_earspan": right_ear_dist,
            "left_ear_dx_earspan": left_ear_dx,
            "left_ear_dy_earspan": left_ear_dy,
            "left_ear_center_dist_earspan": left_ear_dist,
        }
    )

    return metrics


def empty_hand_geometry() -> dict[str, float]:
    base = {
        "hand_valid": 0,
        "hand_scale": np.nan,
        "palm_width": np.nan,
        "palm_height": np.nan,
        "hand_height": np.nan,
        "bbox_width": np.nan,
        "bbox_height": np.nan,
        "bbox_area": np.nan,
        "bbox_aspect_ratio": np.nan,
        "bbox_area_norm": np.nan,
        "palm_axis_angle_deg": np.nan,
        "palm_width_angle_deg": np.nan,
        "palm_normal_z_approx": np.nan,
        "palm_area_2d": np.nan,
        "finger_count_extended": np.nan,
    }

    for finger in ["thumb", "index", "middle", "ring", "pinky"]:
        base.update(
            {
                f"{finger}_base_to_tip_norm": np.nan,
                f"{finger}_wrist_to_tip_norm": np.nan,
                f"{finger}_bone_length_sum_norm": np.nan,
                f"{finger}_straightness": np.nan,
                f"{finger}_curl": np.nan,
                f"{finger}_extension_ratio": np.nan,
                f"{finger}_extended_score": np.nan,
                f"{finger}_extended_est": np.nan,
                f"{finger}_pip_angle_deg": np.nan,
                f"{finger}_dip_angle_deg": np.nan,
            }
        )

    for key in [
        "thumb_index",
        "index_middle",
        "middle_ring",
        "ring_pinky",
        "index_ring",
        "index_pinky",
        "thumb_middle",
        "thumb_pinky",
    ]:
        base[f"{key}_tip_distance_norm"] = np.nan
        base[f"{key}_angle_deg"] = np.nan

    for prefix in [
        "index_mcp_tip",
        "middle_mcp_tip",
        "ring_mcp_tip",
        "pinky_mcp_tip",
        "thumb_mcp_tip",
        "thumb_cmc_tip",
    ]:
        base.update(
            {
                f"{prefix}_dx_norm": np.nan,
                f"{prefix}_dy_norm": np.nan,
                f"{prefix}_dz_norm": np.nan,
                f"{prefix}_length_norm": np.nan,
                f"{prefix}_length_3d_norm": np.nan,
                f"{prefix}_angle_deg": np.nan,
                f"{prefix}_verticality": np.nan,
                f"{prefix}_horizontality": np.nan,
                f"{prefix}_depth_fraction": np.nan,
                f"{prefix}_tip_closer_than_base": np.nan,
                f"{prefix}_tip_farther_than_base": np.nan,
                f"{prefix}_above_base": np.nan,
                f"{prefix}_below_base": np.nan,
                f"{prefix}_left_of_base": np.nan,
                f"{prefix}_right_of_base": np.nan,
            }
        )

    for key in [
        "index_middle_axis_angle_2d_deg",
        "index_middle_axis_angle_3d_deg",
        "index_middle_axis_signed_angle_2d_deg",
        "thumb_index_axis_angle_2d_deg",
        "thumb_index_axis_angle_3d_deg",
        "thumb_middle_axis_angle_2d_deg",
        "thumb_middle_axis_angle_3d_deg",
        "middle_mcp_to_index_axis_signed_norm",
        "middle_pip_to_index_axis_signed_norm",
        "middle_dip_to_index_axis_signed_norm",
        "middle_tip_to_index_axis_signed_norm",
        "middle_index_axis_signed_min",
        "middle_index_axis_signed_max",
        "middle_index_axis_signed_range",
        "middle_index_axis_sign_changes",
    ]:
        base[key] = np.nan

    base.update(empty_eye_hand_spatial_metrics())

    return base


def compute_hand_geometry_from_landmarks(hand_landmarks: Any) -> dict[str, float]:
    points = landmarks_to_array(hand_landmarks)

    if points.shape != (21, 3) or not np.all(np.isfinite(points[:, :2]).any(axis=1)):
        return empty_hand_geometry()

    scale_metrics = estimate_hand_scale(points)
    hand_scale = float(scale_metrics["hand_scale"])

    if not np.isfinite(hand_scale) or hand_scale <= 1e-8:
        return empty_hand_geometry()

    out = {
        "hand_valid": 1,
        **scale_metrics,
        **hand_orientation_metrics(points),
        **pairwise_finger_metrics(points, hand_scale),
        **finger_direction_metrics(points, hand_scale),
    }

    extended_count = 0

    for finger in ["thumb", "index", "middle", "ring", "pinky"]:
        fm = finger_geometry(points, finger, hand_scale)
        out.update(fm)
        extended_count += int(fm.get(f"{finger}_extended_est", 0))

    out["finger_count_extended"] = int(extended_count)

    return out



def _prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _point_to_segment_distance_2d(
    point: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
) -> tuple[float, float]:
    if (
        not np.all(np.isfinite(point))
        or not np.all(np.isfinite(seg_a))
        or not np.all(np.isfinite(seg_b))
    ):
        return np.nan, np.nan

    segment = seg_b - seg_a
    segment_len_sq = float(np.dot(segment, segment))

    if segment_len_sq <= 1e-10:
        return np.nan, np.nan

    rel = point - seg_a
    t = float(np.dot(rel, segment) / segment_len_sq)
    t_clamped = float(np.clip(t, 0.0, 1.0))
    closest = seg_a + t_clamped * segment

    return float(np.linalg.norm(point - closest)), t


def empty_interhand_geometry_metrics() -> dict[str, float]:
    return {
        "two_hands_valid": 0.0,
        "w_avg_hand_scale": np.nan,
        "w_hand_center_distance_norm": np.nan,
        "w_pinky_tip_distance_norm": np.nan,
        "w_index_tip_distance_norm": np.nan,
        "w_pinky_index_tip_distance_ratio": np.nan,
        "w_pinky_tip_x_distance_norm": np.nan,
        "w_index_tip_x_distance_norm": np.nan,
        "w_pinky_tip_y_distance_norm": np.nan,
        "w_index_tip_y_distance_norm": np.nan,
        "w_pinky_segment_min_distance_norm": np.nan,
        "w_primary_pinky_tip_to_secondary_pinky_segment_norm": np.nan,
        "w_secondary_pinky_tip_to_primary_pinky_segment_norm": np.nan,
        "w_primary_pinky_tip_to_secondary_pinky_segment_t": np.nan,
        "w_secondary_pinky_tip_to_primary_pinky_segment_t": np.nan,

        "enie_tilde_index_to_knuckle_line_distance_norm": np.nan,
        "enie_tilde_knuckle_line_coverage": np.nan,
        "enie_secondary_index_tip_to_primary_knuckle_line_norm": np.nan,
        "enie_secondary_index_mcp_to_primary_knuckle_line_norm": np.nan,
        "enie_primary_index_mcp_to_secondary_index_segment_norm": np.nan,
        "enie_primary_middle_mcp_to_secondary_index_segment_norm": np.nan,
        "enie_secondary_index_tip_to_primary_knuckle_line_t": np.nan,
        "enie_secondary_index_mcp_to_primary_knuckle_line_t": np.nan,
        "enie_primary_index_mcp_to_secondary_index_segment_t": np.nan,
        "enie_primary_middle_mcp_to_secondary_index_segment_t": np.nan,

        "q_secondary_index_tip_to_o_join_distance_norm": np.nan,
        "q_secondary_index_mcp_to_o_join_distance_norm": np.nan,
        "q_secondary_middle_tip_to_o_join_distance_norm": np.nan,
        "q_o_join_to_secondary_index_segment_distance_norm": np.nan,
        "q_o_join_to_secondary_index_segment_t": np.nan,
        "q_primary_thumb_tip_to_secondary_index_tip_distance_norm": np.nan,
        "q_primary_index_tip_to_secondary_index_tip_distance_norm": np.nan,

        "x_index_tip_distance_norm": np.nan,
        "x_index_mcp_distance_norm": np.nan,
        "x_index_tip_x_distance_norm": np.nan,
        "x_index_tip_y_distance_norm": np.nan,
        "x_index_segment_min_distance_norm": np.nan,
        "x_primary_index_tip_to_secondary_index_segment_norm": np.nan,
        "x_secondary_index_tip_to_primary_index_segment_norm": np.nan,
        "x_primary_index_mcp_to_secondary_index_segment_norm": np.nan,
        "x_secondary_index_mcp_to_primary_index_segment_norm": np.nan,
        "x_primary_index_tip_to_secondary_index_segment_t": np.nan,
        "x_secondary_index_tip_to_primary_index_segment_t": np.nan,
        "x_primary_index_mcp_to_secondary_index_segment_t": np.nan,
        "x_secondary_index_mcp_to_primary_index_segment_t": np.nan,
        "x_index_segment_strict_intersection": np.nan,
    }


def compute_interhand_geometry_metrics(
    primary_landmarks: Any,
    secondary_landmarks: Any,
) -> dict[str, float]:
    metrics = empty_interhand_geometry_metrics()

    primary_points = landmarks_to_array(primary_landmarks)
    secondary_points = landmarks_to_array(secondary_landmarks)

    if (
        primary_points.shape != (21, 3)
        or secondary_points.shape != (21, 3)
        or not np.any(np.isfinite(primary_points[:, :2]))
        or not np.any(np.isfinite(secondary_points[:, :2]))
    ):
        return metrics

    primary_scale = estimate_hand_scale(primary_points).get("hand_scale", np.nan)
    secondary_scale = estimate_hand_scale(secondary_points).get("hand_scale", np.nan)

    valid_scales = [
        float(value)
        for value in [primary_scale, secondary_scale]
        if np.isfinite(value) and float(value) > 1e-8
    ]

    if not valid_scales:
        return metrics

    avg_hand_scale = float(np.mean(valid_scales))

    def norm_dist(a: np.ndarray, b: np.ndarray) -> float:
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            return np.nan

        return safe_div(float(np.linalg.norm(a - b)), avg_hand_scale)

    primary_finite = primary_points[:, :2][
        np.all(np.isfinite(primary_points[:, :2]), axis=1)
    ]
    secondary_finite = secondary_points[:, :2][
        np.all(np.isfinite(secondary_points[:, :2]), axis=1)
    ]

    if primary_finite.size > 0 and secondary_finite.size > 0:
        primary_center = np.mean(primary_finite, axis=0)
        secondary_center = np.mean(secondary_finite, axis=0)
    else:
        primary_center = np.array([np.nan, np.nan], dtype=np.float32)
        secondary_center = np.array([np.nan, np.nan], dtype=np.float32)

    primary_thumb_tip = primary_points[4, :2]
    primary_index_tip = primary_points[8, :2]
    secondary_index_tip = secondary_points[8, :2]
    secondary_middle_tip = secondary_points[12, :2]
    primary_pinky_tip = primary_points[20, :2]
    secondary_pinky_tip = secondary_points[20, :2]

    primary_pinky_mcp = primary_points[17, :2]
    secondary_pinky_mcp = secondary_points[17, :2]

    primary_index_mcp = primary_points[5, :2]
    primary_middle_mcp = primary_points[9, :2]
    secondary_index_mcp = secondary_points[5, :2]

    pinky_tip_distance = norm_dist(primary_pinky_tip, secondary_pinky_tip)
    index_tip_distance = norm_dist(primary_index_tip, secondary_index_tip)

    p_to_s_dist, p_to_s_t = _point_to_segment_distance_2d(
        primary_pinky_tip,
        secondary_pinky_mcp,
        secondary_pinky_tip,
    )
    s_to_p_dist, s_to_p_t = _point_to_segment_distance_2d(
        secondary_pinky_tip,
        primary_pinky_mcp,
        primary_pinky_tip,
    )

    segment_dist_values = [
        safe_div(value, avg_hand_scale)
        for value in [p_to_s_dist, s_to_p_dist]
        if np.isfinite(value)
    ]

    if segment_dist_values:
        pinky_segment_min_distance = float(np.min(segment_dist_values))
    else:
        pinky_segment_min_distance = np.nan

    enie_s_tip_to_knuckle_dist, enie_s_tip_to_knuckle_t = _point_to_segment_distance_2d(
        secondary_index_tip,
        primary_index_mcp,
        primary_middle_mcp,
    )
    enie_s_mcp_to_knuckle_dist, enie_s_mcp_to_knuckle_t = _point_to_segment_distance_2d(
        secondary_index_mcp,
        primary_index_mcp,
        primary_middle_mcp,
    )
    enie_p_index_mcp_to_s_index_dist, enie_p_index_mcp_to_s_index_t = _point_to_segment_distance_2d(
        primary_index_mcp,
        secondary_index_mcp,
        secondary_index_tip,
    )
    enie_p_middle_mcp_to_s_index_dist, enie_p_middle_mcp_to_s_index_t = _point_to_segment_distance_2d(
        primary_middle_mcp,
        secondary_index_mcp,
        secondary_index_tip,
    )

    enie_tilde_distance_values = [
        safe_div(value, avg_hand_scale)
        for value in [
            enie_s_tip_to_knuckle_dist,
            enie_s_mcp_to_knuckle_dist,
            enie_p_index_mcp_to_s_index_dist,
            enie_p_middle_mcp_to_s_index_dist,
        ]
        if np.isfinite(value)
    ]

    if enie_tilde_distance_values:
        enie_tilde_index_to_knuckle_line_distance = float(np.min(enie_tilde_distance_values))
    else:
        enie_tilde_index_to_knuckle_line_distance = np.nan

    # Cobertura proyectada de la tilde sobre la línea de nudillos de Ñ.
    # Usamos los parámetros t del MCP y TIP del índice secundario proyectados
    # sobre el segmento primary_index_mcp -> primary_middle_mcp:
    #   t = 0 corresponde al MCP del índice primario;
    #   t = 1 corresponde al MCP del medio primario.
    # La cobertura es el solapamiento del intervalo [t_mcp, t_tip] con [0, 1].
    if np.isfinite(enie_s_mcp_to_knuckle_t) and np.isfinite(enie_s_tip_to_knuckle_t):
        enie_tilde_projection_min_t = min(
            float(enie_s_mcp_to_knuckle_t),
            float(enie_s_tip_to_knuckle_t),
        )
        enie_tilde_projection_max_t = max(
            float(enie_s_mcp_to_knuckle_t),
            float(enie_s_tip_to_knuckle_t),
        )
        enie_tilde_knuckle_line_coverage = max(
            0.0,
            min(enie_tilde_projection_max_t, 1.0)
            - max(enie_tilde_projection_min_t, 0.0),
        )
    else:
        enie_tilde_knuckle_line_coverage = np.nan

    # Métricas específicas para Q:
    # el punto semántico de contacto de la cola se aproxima como el punto
    # medio entre thumb_tip e index_tip de la mano primaria, porque en la O
    # correcta ambas puntas están juntas. La punta del índice secundario
    # debe quedar muy cerca de ese punto.
    if np.all(np.isfinite(primary_thumb_tip)) and np.all(np.isfinite(primary_index_tip)):
        q_o_join = (primary_thumb_tip + primary_index_tip) / 2.0
    else:
        q_o_join = np.array([np.nan, np.nan], dtype=np.float32)

    q_join_to_secondary_index_segment_dist, q_join_to_secondary_index_segment_t = _point_to_segment_distance_2d(
        q_o_join,
        secondary_index_mcp,
        secondary_index_tip,
    )

    x_p_to_s_dist, x_p_to_s_t = _point_to_segment_distance_2d(
        primary_index_tip,
        secondary_index_mcp,
        secondary_index_tip,
    )
    x_s_to_p_dist, x_s_to_p_t = _point_to_segment_distance_2d(
        secondary_index_tip,
        primary_index_mcp,
        primary_index_tip,
    )
    x_p_mcp_to_s_dist, x_p_mcp_to_s_t = _point_to_segment_distance_2d(
        primary_index_mcp,
        secondary_index_mcp,
        secondary_index_tip,
    )
    x_s_mcp_to_p_dist, x_s_mcp_to_p_t = _point_to_segment_distance_2d(
        secondary_index_mcp,
        primary_index_mcp,
        primary_index_tip,
    )

    x_segment_dist_values = [
        safe_div(value, avg_hand_scale)
        for value in [
            x_p_to_s_dist,
            x_s_to_p_dist,
            x_p_mcp_to_s_dist,
            x_s_mcp_to_p_dist,
        ]
        if np.isfinite(value)
    ]

    if x_segment_dist_values:
        x_index_segment_min_distance = float(np.min(x_segment_dist_values))
    else:
        x_index_segment_min_distance = np.nan

    def orient_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        if (
            not np.all(np.isfinite(a))
            or not np.all(np.isfinite(b))
            or not np.all(np.isfinite(c))
        ):
            return np.nan

        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    def strict_segment_intersection_2d(
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
        eps: float = 1e-9,
    ) -> float:
        o1 = orient_2d(a, b, c)
        o2 = orient_2d(a, b, d)
        o3 = orient_2d(c, d, a)
        o4 = orient_2d(c, d, b)

        if not all(np.isfinite(value) for value in [o1, o2, o3, o4]):
            return np.nan

        # Cruce estricto: no cuenta el simple contacto de puntas/extremos.
        return float((o1 * o2 < -eps) and (o3 * o4 < -eps))

    x_index_segment_strict_intersection = strict_segment_intersection_2d(
        primary_index_mcp,
        primary_index_tip,
        secondary_index_mcp,
        secondary_index_tip,
    )

    metrics.update(
        {
            "two_hands_valid": 1.0,
            "w_avg_hand_scale": avg_hand_scale,
            "w_hand_center_distance_norm": norm_dist(primary_center, secondary_center),
            "w_pinky_tip_distance_norm": pinky_tip_distance,
            "w_index_tip_distance_norm": index_tip_distance,
            "w_pinky_index_tip_distance_ratio": safe_div(
                pinky_tip_distance,
                index_tip_distance,
            ),
            "w_pinky_tip_x_distance_norm": safe_div(
                abs(float(primary_pinky_tip[0] - secondary_pinky_tip[0])),
                avg_hand_scale,
            )
            if np.all(np.isfinite(primary_pinky_tip)) and np.all(np.isfinite(secondary_pinky_tip))
            else np.nan,
            "w_index_tip_x_distance_norm": safe_div(
                abs(float(primary_index_tip[0] - secondary_index_tip[0])),
                avg_hand_scale,
            )
            if np.all(np.isfinite(primary_index_tip)) and np.all(np.isfinite(secondary_index_tip))
            else np.nan,
            "w_pinky_tip_y_distance_norm": safe_div(
                abs(float(primary_pinky_tip[1] - secondary_pinky_tip[1])),
                avg_hand_scale,
            )
            if np.all(np.isfinite(primary_pinky_tip)) and np.all(np.isfinite(secondary_pinky_tip))
            else np.nan,
            "w_index_tip_y_distance_norm": safe_div(
                abs(float(primary_index_tip[1] - secondary_index_tip[1])),
                avg_hand_scale,
            )
            if np.all(np.isfinite(primary_index_tip)) and np.all(np.isfinite(secondary_index_tip))
            else np.nan,
            "w_pinky_segment_min_distance_norm": pinky_segment_min_distance,
            "w_primary_pinky_tip_to_secondary_pinky_segment_norm": safe_div(
                p_to_s_dist,
                avg_hand_scale,
            ),
            "w_secondary_pinky_tip_to_primary_pinky_segment_norm": safe_div(
                s_to_p_dist,
                avg_hand_scale,
            ),
            "w_primary_pinky_tip_to_secondary_pinky_segment_t": p_to_s_t,
            "w_secondary_pinky_tip_to_primary_pinky_segment_t": s_to_p_t,

            "enie_tilde_index_to_knuckle_line_distance_norm": enie_tilde_index_to_knuckle_line_distance,
            "enie_tilde_knuckle_line_coverage": enie_tilde_knuckle_line_coverage,
            "enie_secondary_index_tip_to_primary_knuckle_line_norm": safe_div(
                enie_s_tip_to_knuckle_dist,
                avg_hand_scale,
            ),
            "enie_secondary_index_mcp_to_primary_knuckle_line_norm": safe_div(
                enie_s_mcp_to_knuckle_dist,
                avg_hand_scale,
            ),
            "enie_primary_index_mcp_to_secondary_index_segment_norm": safe_div(
                enie_p_index_mcp_to_s_index_dist,
                avg_hand_scale,
            ),
            "enie_primary_middle_mcp_to_secondary_index_segment_norm": safe_div(
                enie_p_middle_mcp_to_s_index_dist,
                avg_hand_scale,
            ),
            "enie_secondary_index_tip_to_primary_knuckle_line_t": enie_s_tip_to_knuckle_t,
            "enie_secondary_index_mcp_to_primary_knuckle_line_t": enie_s_mcp_to_knuckle_t,
            "enie_primary_index_mcp_to_secondary_index_segment_t": enie_p_index_mcp_to_s_index_t,
            "enie_primary_middle_mcp_to_secondary_index_segment_t": enie_p_middle_mcp_to_s_index_t,

            "q_secondary_index_tip_to_o_join_distance_norm": norm_dist(
                secondary_index_tip,
                q_o_join,
            ),
            "q_secondary_index_mcp_to_o_join_distance_norm": norm_dist(
                secondary_index_mcp,
                q_o_join,
            ),
            "q_secondary_middle_tip_to_o_join_distance_norm": norm_dist(
                secondary_middle_tip,
                q_o_join,
            ),
            "q_o_join_to_secondary_index_segment_distance_norm": safe_div(
                q_join_to_secondary_index_segment_dist,
                avg_hand_scale,
            ),
            "q_o_join_to_secondary_index_segment_t": q_join_to_secondary_index_segment_t,
            "q_primary_thumb_tip_to_secondary_index_tip_distance_norm": norm_dist(
                primary_thumb_tip,
                secondary_index_tip,
            ),
            "q_primary_index_tip_to_secondary_index_tip_distance_norm": norm_dist(
                primary_index_tip,
                secondary_index_tip,
            ),

            "x_index_tip_distance_norm": index_tip_distance,
            "x_index_mcp_distance_norm": norm_dist(primary_index_mcp, secondary_index_mcp),
            "x_index_tip_x_distance_norm": safe_div(
                abs(float(primary_index_tip[0] - secondary_index_tip[0])),
                avg_hand_scale,
            )
            if np.all(np.isfinite(primary_index_tip)) and np.all(np.isfinite(secondary_index_tip))
            else np.nan,
            "x_index_tip_y_distance_norm": safe_div(
                abs(float(primary_index_tip[1] - secondary_index_tip[1])),
                avg_hand_scale,
            )
            if np.all(np.isfinite(primary_index_tip)) and np.all(np.isfinite(secondary_index_tip))
            else np.nan,
            "x_index_segment_min_distance_norm": x_index_segment_min_distance,
            "x_primary_index_tip_to_secondary_index_segment_norm": safe_div(
                x_p_to_s_dist,
                avg_hand_scale,
            ),
            "x_secondary_index_tip_to_primary_index_segment_norm": safe_div(
                x_s_to_p_dist,
                avg_hand_scale,
            ),
            "x_primary_index_mcp_to_secondary_index_segment_norm": safe_div(
                x_p_mcp_to_s_dist,
                avg_hand_scale,
            ),
            "x_secondary_index_mcp_to_primary_index_segment_norm": safe_div(
                x_s_mcp_to_p_dist,
                avg_hand_scale,
            ),
            "x_primary_index_tip_to_secondary_index_segment_t": x_p_to_s_t,
            "x_secondary_index_tip_to_primary_index_segment_t": x_s_to_p_t,
            "x_primary_index_mcp_to_secondary_index_segment_t": x_p_mcp_to_s_t,
            "x_secondary_index_mcp_to_primary_index_segment_t": x_s_mcp_to_p_t,
            "x_index_segment_strict_intersection": x_index_segment_strict_intersection,
        }
    )

    return metrics


def _select_primary_secondary_hands(
    result: Any,
    expected_hands: int,
) -> tuple[dict | None, dict | None]:
    hand_result, _pose_result = _split_detection_result(result)
    hands = _collect_detected_hands(hand_result)

    if not hands:
        return None, None

    primary, secondary = _select_hands_for_class(
        hands=hands,
        expected_hands=expected_hands,
    )

    return primary, secondary


def _select_primary_hand(result: Any, expected_hands: int) -> dict | None:
    hand_result, _pose_result = _split_detection_result(result)
    hands = _collect_detected_hands(hand_result)

    if not hands:
        return None

    primary, _secondary = _select_hands_for_class(
        hands=hands,
        expected_hands=expected_hands,
    )

    return primary


def _extract_primary_metrics_per_frame(
    captured_items: list[dict],
    expected_hands: int,
) -> list[dict[str, float]]:
    rows = []

    for item in captured_items:
        result = item.get("result")
        primary, secondary = _select_primary_secondary_hands(
            result=result,
            expected_hands=expected_hands,
        )

        if primary is None:
            metrics = empty_hand_geometry()
            metrics.update(_prefix_metrics(empty_hand_geometry(), "secondary"))
            metrics.update(empty_interhand_geometry_metrics())
            rows.append(metrics)
            continue

        metrics = compute_hand_geometry_from_landmarks(
            primary.get("landmarks")
        )

        _hand_result, pose_result = _split_detection_result(result)
        metrics.update(
            compute_eye_hand_spatial_metrics(
                primary.get("landmarks"),
                pose_result,
            )
        )

        if secondary is not None:
            secondary_metrics = compute_hand_geometry_from_landmarks(
                secondary.get("landmarks")
            )
            metrics.update(_prefix_metrics(secondary_metrics, "secondary"))
            metrics.update(
                compute_interhand_geometry_metrics(
                    primary.get("landmarks"),
                    secondary.get("landmarks"),
                )
            )
        else:
            metrics.update(_prefix_metrics(empty_hand_geometry(), "secondary"))
            metrics.update(empty_interhand_geometry_metrics())

        rows.append(metrics)

    return rows

def _nanmean(values: list[float], default: float = np.nan) -> float:
    arr = np.asarray(values, dtype=np.float32)

    if arr.size == 0:
        return float(default)

    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        return float(default)

    return float(np.mean(finite))


def aggregate_primary_metrics(frame_metrics: list[dict[str, float]]) -> dict[str, float]:
    if not frame_metrics:
        return {
            "valid_ratio": 0.0,
            "frames_used": 0,
            "valid_frames": 0,
        }

    keys = sorted(
        {
            key
            for row in frame_metrics
            for key in row.keys()
        }
    )

    metrics = {}

    for key in keys:
        metrics[key] = _nanmean([row.get(key, np.nan) for row in frame_metrics])

    hand_valid_values = [
        row.get("hand_valid", 0.0)
        for row in frame_metrics
    ]

    valid_ratio = _nanmean(hand_valid_values, default=0.0)

    metrics["valid_ratio"] = float(valid_ratio)
    metrics["frames_used"] = int(len(frame_metrics))
    metrics["valid_frames"] = int(
        sum(1 for value in hand_valid_values if np.isfinite(value) and value >= 0.5)
    )

    # Alias explícitos para leer estas señales como ratios de frames.
    for base_key in [
        "pose_valid_for_eye_geometry",
        "pose_valid_for_ear_geometry",
        "pose_valid_for_h_face_geometry",
        "pose_valid_for_j_jaw_geometry",
        "pose_valid_for_i_face_geometry",
        "pose_valid_for_y_mouth_geometry",
        "pose_valid_for_s_chin_geometry",
        "pose_valid_for_t_chin_geometry",
        "pose_valid_for_f_chest_geometry",
        "same_eye_inside_hand_bbox",
        "other_eye_inside_hand_bbox",
        "right_eye_inside_hand_bbox",
        "left_eye_inside_hand_bbox",
        "nose_inside_hand_bbox",
    ]:
        metrics[f"{base_key}_ratio"] = metrics.get(base_key, np.nan)

    # Alias para reglas bimanuales:
    # compute_hand_geometry_from_landmarks() expone "hand_valid".
    # Al prefijar la mano secundaria queda "secondary_hand_valid".
    # Como aquí se promedian frames, este valor funciona como ratio
    # de visibilidad de la segunda mano durante la ventana.
    metrics["secondary_valid_ratio"] = metrics.get("secondary_hand_valid", np.nan)

    return metrics


def _metric(metrics: dict[str, float], key: str) -> float:
    value = metrics.get(key, np.nan)

    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan

    return value


def _add_missing_or_min(
    reasons: list[str],
    metrics: dict[str, float],
    key: str,
    minimum: float,
    description: str,
):
    value = _metric(metrics, key)

    if not np.isfinite(value):
        reasons.append(f"{description}: métrica ausente ({key})")
    elif value < minimum:
        reasons.append(f"{description} insuficiente ({value:.4f} < {minimum:.4f})")


def _add_missing_or_max(
    reasons: list[str],
    metrics: dict[str, float],
    key: str,
    maximum: float,
    description: str,
):
    value = _metric(metrics, key)

    if not np.isfinite(value):
        reasons.append(f"{description}: métrica ausente ({key})")
    elif value > maximum:
        reasons.append(f"{description} demasiado alto ({value:.4f} > {maximum:.4f})")


def _add_missing_or_range(
    reasons: list[str],
    metrics: dict[str, float],
    key: str,
    minimum: float,
    maximum: float,
    description: str,
):
    value = _metric(metrics, key)

    if not np.isfinite(value):
        reasons.append(f"{description}: métrica ausente ({key})")
    elif value < minimum or value > maximum:
        reasons.append(
            f"{description} fuera de rango ({value:.4f}; esperado {minimum:.4f}..{maximum:.4f})"
        )


def _validate_a_geometry(metrics: dict[str, float], rule: dict) -> list[str]:
    reasons: list[str] = []

    _add_missing_or_min(
        reasons,
        metrics,
        "valid_ratio",
        float(rule["min_valid_ratio"]),
        "mano primaria visible para A",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "finger_count_extended",
        float(rule["finger_count_min"]),
        float(rule["finger_count_max"]),
        "cantidad de dedos extendidos/visibles para A",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_extended_score",
        float(rule["thumb_extended_score_min"]),
        "pulgar visible/apoyado para A",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "index_extended_score",
        float(rule["index_extended_score_max"]),
        "dedo índice debe estar recogido en A",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "middle_extended_score",
        float(rule["middle_extended_score_max"]),
        "dedo medio debe estar recogido en A",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "ring_extended_score",
        float(rule["ring_extended_score_max"]),
        "dedo anular debe estar recogido en A",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "pinky_extended_score",
        float(rule["pinky_extended_score_max"]),
        "dedo meñique debe estar recogido en A",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_index_tip_distance_norm",
        float(rule["thumb_index_tip_distance_norm_min"]),
        float(rule["thumb_index_tip_distance_norm_max"]),
        "distancia pulgar-índice para A",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_mcp_tip_angle_deg",
        float(rule["thumb_mcp_tip_angle_deg_min"]),
        float(rule["thumb_mcp_tip_angle_deg_max"]),
        "dirección del pulgar para A",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_mcp_tip_horizontality",
        float(rule["thumb_mcp_tip_horizontality_min"]),
        "horizontalidad/recostado del pulgar para A",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_mcp_tip_above_base",
        float(rule["thumb_mcp_tip_above_base_min"]),
        "pulgar debe quedar por encima de su base en A",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_mcp_tip_right_of_base",
        float(rule["thumb_mcp_tip_right_of_base_min"]),
        "pulgar debe apoyarse hacia el lado correcto en A",
    )

    return reasons


def _validate_b_geometry(metrics: dict[str, float], rule: dict) -> list[str]:
    reasons: list[str] = []

    _add_missing_or_min(
        reasons,
        metrics,
        "valid_ratio",
        float(rule["min_valid_ratio"]),
        "mano primaria visible para B",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "finger_count_extended",
        float(rule["finger_count_min"]),
        float(rule["finger_count_max"]),
        "cantidad de dedos extendidos/visibles para B",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_extended_score",
        float(rule["thumb_extended_score_min"]),
        float(rule["thumb_extended_score_max"]),
        "pulgar doblado/visible para B",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_extended_score",
        float(rule["index_extended_score_min"]),
        "índice extendido para B",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_extended_score",
        float(rule["middle_extended_score_min"]),
        "dedo medio extendido para B",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "ring_extended_score",
        float(rule["ring_extended_score_min"]),
        "dedo anular extendido para B",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "pinky_extended_score",
        float(rule["pinky_extended_score_min"]),
        "dedo meñique extendido para B",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "index_middle_tip_distance_norm",
        float(rule["index_middle_tip_distance_norm_max"]),
        "separación índice-medio para B",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "middle_ring_tip_distance_norm",
        float(rule["middle_ring_tip_distance_norm_max"]),
        "separación medio-anular para B",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "ring_pinky_tip_distance_norm",
        float(rule["ring_pinky_tip_distance_norm_max"]),
        "separación anular-meñique para B",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_mcp_tip_angle_deg",
        float(rule["index_mcp_tip_angle_deg_min"]),
        float(rule["index_mcp_tip_angle_deg_max"]),
        "dirección del índice para B",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_verticality",
        float(rule["index_mcp_tip_verticality_min"]),
        "verticalidad del índice para B",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_above_base",
        float(rule["index_mcp_tip_above_base_min"]),
        "índice debe apuntar hacia arriba en B",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "middle_mcp_tip_angle_deg",
        float(rule["middle_mcp_tip_angle_deg_min"]),
        float(rule["middle_mcp_tip_angle_deg_max"]),
        "dirección del dedo medio para B",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_mcp_tip_verticality",
        float(rule["middle_mcp_tip_verticality_min"]),
        "verticalidad del dedo medio para B",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_mcp_tip_above_base",
        float(rule["middle_mcp_tip_above_base_min"]),
        "dedo medio debe apuntar hacia arriba en B",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "index_middle_axis_angle_3d_deg",
        float(rule["index_middle_axis_angle_3d_deg_max"]),
        "paralelismo 3D índice-medio para B",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_mcp_tip_angle_deg",
        float(rule["thumb_mcp_tip_angle_deg_min"]),
        float(rule["thumb_mcp_tip_angle_deg_max"]),
        "dirección del pulgar para B",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_mcp_tip_horizontality",
        float(rule["thumb_mcp_tip_horizontality_max"]),
        "horizontalidad/apertura del pulgar para B",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_mcp_tip_left_of_base",
        float(rule["thumb_mcp_tip_left_of_base_min"]),
        "pulgar debe doblarse hacia el lado correcto en B",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_mcp_tip_right_of_base",
        float(rule["thumb_mcp_tip_right_of_base_max"]),
        "pulgar extendido hacia el lado contrario en B",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_index_axis_angle_3d_deg",
        float(rule["thumb_index_axis_angle_3d_deg_max"]),
        "ángulo 3D pulgar-índice para B",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_middle_axis_angle_3d_deg",
        float(rule["thumb_middle_axis_angle_3d_deg_max"]),
        "ángulo 3D pulgar-medio para B",
    )

    if bool(rule.get("use_palm_normal_z", False)):
        _add_missing_or_max(
            reasons,
            metrics,
            "palm_normal_z_approx",
            float(rule["palm_normal_z_approx_max"]),
            "orientación palma/dorso para B",
        )

    return reasons



def _validate_c_geometry(metrics: dict[str, float], rule: dict) -> list[str]:
    reasons: list[str] = []

    _add_missing_or_min(
        reasons,
        metrics,
        "valid_ratio",
        float(rule["min_valid_ratio"]),
        "mano primaria visible para C",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "finger_count_extended",
        float(rule["finger_count_min"]),
        float(rule["finger_count_max"]),
        "cantidad de dedos extendidos/visibles para C",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_extended_score",
        float(rule["thumb_extended_score_min"]),
        float(rule["thumb_extended_score_max"]),
        "pulgar extendido/visible para C",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_extended_score",
        float(rule["index_extended_score_min"]),
        float(rule["index_extended_score_max"]),
        "índice semiextendido/doblado para C",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "middle_extended_score",
        float(rule["middle_extended_score_max"]),
        "dedo medio debe estar recogido en C",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "ring_extended_score",
        float(rule["ring_extended_score_max"]),
        "dedo anular debe estar recogido en C",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "pinky_extended_score",
        float(rule["pinky_extended_score_max"]),
        "dedo meñique debe estar recogido en C",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_index_tip_distance_norm",
        float(rule["thumb_index_tip_distance_norm_min"]),
        float(rule["thumb_index_tip_distance_norm_max"]),
        "apertura pulgar-índice para C",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_middle_tip_distance_norm",
        float(rule["index_middle_tip_distance_norm_min"]),
        "separación índice-medio para C",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_mcp_tip_angle_deg",
        float(rule["index_mcp_tip_angle_deg_min"]),
        float(rule["index_mcp_tip_angle_deg_max"]),
        "dirección del índice para C",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_mcp_tip_verticality",
        float(rule["index_mcp_tip_verticality_min"]),
        float(rule["index_mcp_tip_verticality_max"]),
        "verticalidad/curvatura aparente del índice para C",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_above_base",
        float(rule["index_mcp_tip_above_base_min"]),
        "índice debe apuntar hacia arriba en C",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_mcp_tip_angle_deg",
        float(rule["thumb_mcp_tip_angle_deg_min"]),
        float(rule["thumb_mcp_tip_angle_deg_max"]),
        "dirección del pulgar para C",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_mcp_tip_horizontality",
        float(rule["thumb_mcp_tip_horizontality_min"]),
        "horizontalidad del pulgar para C",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_mcp_tip_right_of_base",
        float(rule["thumb_mcp_tip_right_of_base_min"]),
        "pulgar debe apuntar al lado correcto en C",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_mcp_tip_left_of_base",
        float(rule["thumb_mcp_tip_left_of_base_max"]),
        "pulgar apunta al lado contrario en C",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_index_axis_angle_3d_deg",
        float(rule["thumb_index_axis_angle_3d_deg_min"]),
        float(rule["thumb_index_axis_angle_3d_deg_max"]),
        "ángulo 3D pulgar-índice para C",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_middle_axis_angle_3d_deg",
        float(rule["thumb_middle_axis_angle_3d_deg_min"]),
        float(rule["thumb_middle_axis_angle_3d_deg_max"]),
        "ángulo 3D pulgar-medio para C",
    )

    if bool(rule.get("reject_eye_region_for_c", False)):
        same_eye = _metric(metrics, "same_eye_inside_hand_bbox_ratio")
        other_eye = _metric(metrics, "other_eye_inside_hand_bbox_ratio")
        nose = _metric(metrics, "nose_inside_hand_bbox_ratio")

        if (
            np.isfinite(same_eye)
            and np.isfinite(other_eye)
            and np.isfinite(nose)
            and same_eye >= float(rule["same_eye_inside_hand_bbox_ratio_for_e_min"])
            and other_eye <= float(rule["other_eye_inside_hand_bbox_ratio_for_e_max"])
            and nose <= float(rule["nose_inside_hand_bbox_ratio_for_e_max"])
        ):
            reasons.append(
                "ubicación alrededor del ojo corresponde a E, no a C "
                f"(same_eye={same_eye:.3f}, other_eye={other_eye:.3f}, nose={nose:.3f})"
            )

    return reasons




def _validate_d_geometry(metrics: dict[str, float], rule: dict) -> list[str]:
    reasons: list[str] = []

    _add_missing_or_min(
        reasons,
        metrics,
        "valid_ratio",
        float(rule["min_valid_ratio"]),
        "mano primaria visible para D",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "finger_count_extended",
        float(rule["finger_count_min"]),
        float(rule["finger_count_max"]),
        "cantidad de dedos extendidos/visibles para D",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_extended_score",
        float(rule["thumb_extended_score_min"]),
        "pulgar activo para D",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_extended_score",
        float(rule["index_extended_score_min"]),
        "índice extendido para D",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "middle_extended_score",
        float(rule["middle_extended_score_max"]),
        "dedo medio no debe quedar vertical/protagonista en D",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "ring_extended_score",
        float(rule["ring_extended_score_min"]),
        "dedo anular extendido para D",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "pinky_extended_score",
        float(rule["pinky_extended_score_min"]),
        "dedo meñique extendido para D",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_middle_tip_distance_norm",
        float(rule["thumb_middle_tip_distance_norm_max"]),
        "contacto/proximidad pulgar-medio para D",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_index_tip_distance_norm",
        float(rule["thumb_index_tip_distance_norm_min"]),
        "pulgar no debe contactar con índice en D",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_middle_tip_distance_norm",
        float(rule["index_middle_tip_distance_norm_min"]),
        "separación índice-medio para D",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_ring_tip_distance_norm",
        float(rule["middle_ring_tip_distance_norm_min"]),
        "separación medio-anular para D",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "ring_pinky_tip_distance_norm",
        float(rule["ring_pinky_tip_distance_norm_min"]),
        float(rule["ring_pinky_tip_distance_norm_max"]),
        "separación anular-meñique para D",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_mcp_tip_angle_deg",
        float(rule["index_mcp_tip_angle_deg_min"]),
        float(rule["index_mcp_tip_angle_deg_max"]),
        "dirección ascendente del índice para D",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_verticality",
        float(rule["index_mcp_tip_verticality_min"]),
        "verticalidad del índice para D",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_above_base",
        float(rule["index_mcp_tip_above_base_min"]),
        "índice debe apuntar hacia arriba en D",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "middle_mcp_tip_angle_deg",
        float(rule["middle_mcp_tip_angle_deg_min"]),
        float(rule["middle_mcp_tip_angle_deg_max"]),
        "dirección lateral del dedo medio para D",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_mcp_tip_horizontality",
        float(rule["middle_mcp_tip_horizontality_min"]),
        "horizontalidad del dedo medio para D",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_mcp_tip_below_base",
        float(rule["middle_mcp_tip_below_base_min"]),
        "dedo medio debe bajar hacia el pulgar en D",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_mcp_tip_right_of_base",
        float(rule["middle_mcp_tip_right_of_base_min"]),
        "dedo medio debe lateralizar hacia el pulgar en D",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "ring_mcp_tip_angle_deg",
        float(rule["ring_mcp_tip_angle_deg_min"]),
        float(rule["ring_mcp_tip_angle_deg_max"]),
        "dirección ascendente del anular para D",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "ring_mcp_tip_verticality",
        float(rule["ring_mcp_tip_verticality_min"]),
        "verticalidad del anular para D",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "ring_mcp_tip_above_base",
        float(rule["ring_mcp_tip_above_base_min"]),
        "anular debe apuntar hacia arriba en D",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "pinky_mcp_tip_angle_deg",
        float(rule["pinky_mcp_tip_angle_deg_min"]),
        float(rule["pinky_mcp_tip_angle_deg_max"]),
        "dirección ascendente del meñique para D",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "pinky_mcp_tip_verticality",
        float(rule["pinky_mcp_tip_verticality_min"]),
        "verticalidad del meñique para D",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "pinky_mcp_tip_above_base",
        float(rule["pinky_mcp_tip_above_base_min"]),
        "meñique debe apuntar hacia arriba en D",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_mcp_tip_angle_deg",
        float(rule["thumb_mcp_tip_angle_deg_min"]),
        float(rule["thumb_mcp_tip_angle_deg_max"]),
        "dirección del pulgar para D",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_mcp_tip_horizontality",
        float(rule["thumb_mcp_tip_horizontality_min"]),
        float(rule["thumb_mcp_tip_horizontality_max"]),
        "horizontalidad del pulgar para D",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_mcp_tip_right_of_base",
        float(rule["thumb_mcp_tip_right_of_base_min"]),
        "pulgar debe ir hacia el medio en D",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_mcp_tip_left_of_base",
        float(rule["thumb_mcp_tip_left_of_base_max"]),
        "pulgar no debe invertirse en D",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_middle_axis_angle_3d_deg",
        float(rule["index_middle_axis_angle_3d_deg_min"]),
        float(rule["index_middle_axis_angle_3d_deg_max"]),
        "relación angular índice-medio para D",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_middle_axis_angle_3d_deg",
        float(rule["thumb_middle_axis_angle_3d_deg_min"]),
        float(rule["thumb_middle_axis_angle_3d_deg_max"]),
        "relación angular pulgar-medio para D",
    )

    if bool(rule.get("use_palm_normal_z", False)):
        _add_missing_or_max(
            reasons,
            metrics,
            "palm_normal_z_approx",
            float(rule["palm_normal_z_approx_max"]),
            "orientación palma visible para D",
        )

    return reasons


def _validate_e_geometry(metrics: dict[str, float], rule: dict) -> list[str]:
    reasons: list[str] = []

    _add_missing_or_min(
        reasons,
        metrics,
        "valid_ratio",
        float(rule["min_valid_ratio"]),
        "mano primaria visible para E",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "finger_count_extended",
        float(rule["finger_count_min"]),
        float(rule["finger_count_max"]),
        "cantidad de dedos extendidos/visibles para E",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_extended_score",
        float(rule["thumb_extended_score_min"]),
        float(rule["thumb_extended_score_max"]),
        "pulgar extendido/visible para E",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_extended_score",
        float(rule["index_extended_score_min"]),
        float(rule["index_extended_score_max"]),
        "índice semiextendido/doblado para E",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "middle_extended_score",
        float(rule["middle_extended_score_max"]),
        "dedo medio debe estar recogido en E",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "ring_extended_score",
        float(rule["ring_extended_score_max"]),
        "dedo anular debe estar recogido en E",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "pinky_extended_score",
        float(rule["pinky_extended_score_max"]),
        "dedo meñique debe estar recogido en E",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_index_tip_distance_norm",
        float(rule["thumb_index_tip_distance_norm_min"]),
        float(rule["thumb_index_tip_distance_norm_max"]),
        "apertura pulgar-índice para E",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_middle_tip_distance_norm",
        float(rule["index_middle_tip_distance_norm_min"]),
        "separación índice-medio para E",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_mcp_tip_angle_deg",
        float(rule["index_mcp_tip_angle_deg_min"]),
        float(rule["index_mcp_tip_angle_deg_max"]),
        "dirección del índice para E",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_mcp_tip_verticality",
        float(rule["index_mcp_tip_verticality_min"]),
        float(rule["index_mcp_tip_verticality_max"]),
        "verticalidad/curvatura aparente del índice para E",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_above_base",
        float(rule["index_mcp_tip_above_base_min"]),
        "índice debe apuntar hacia arriba en E",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_mcp_tip_angle_deg",
        float(rule["thumb_mcp_tip_angle_deg_min"]),
        float(rule["thumb_mcp_tip_angle_deg_max"]),
        "dirección del pulgar para E",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_mcp_tip_horizontality",
        float(rule["thumb_mcp_tip_horizontality_min"]),
        "horizontalidad del pulgar para E",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_mcp_tip_right_of_base",
        float(rule["thumb_mcp_tip_right_of_base_min"]),
        "pulgar debe apuntar al lado correcto en E",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_mcp_tip_left_of_base",
        float(rule["thumb_mcp_tip_left_of_base_max"]),
        "pulgar apunta al lado contrario en E",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_index_axis_angle_3d_deg",
        float(rule["thumb_index_axis_angle_3d_deg_min"]),
        float(rule["thumb_index_axis_angle_3d_deg_max"]),
        "ángulo 3D pulgar-índice para E",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_middle_axis_angle_3d_deg",
        float(rule["thumb_middle_axis_angle_3d_deg_min"]),
        float(rule["thumb_middle_axis_angle_3d_deg_max"]),
        "ángulo 3D pulgar-medio para E",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "pose_valid_for_eye_geometry_ratio",
        float(rule["pose_valid_ratio_min"]),
        "landmarks de cara/ojos visibles para E",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "same_eye_inside_hand_bbox_ratio",
        float(rule["same_eye_inside_hand_bbox_ratio_min"]),
        "ojo del mismo lado dentro del marco de la mano para E",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "other_eye_inside_hand_bbox_ratio",
        float(rule["other_eye_inside_hand_bbox_ratio_max"]),
        "ojo contrario dentro del marco de la mano para E",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "nose_inside_hand_bbox_ratio",
        float(rule["nose_inside_hand_bbox_ratio_max"]),
        "nariz dentro del marco de la mano para E",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "same_eye_thumb_index_t",
        float(rule["same_eye_thumb_index_t_min"]),
        float(rule["same_eye_thumb_index_t_max"]),
        "proyección del ojo sobre el segmento pulgar-índice para E",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "same_eye_thumb_index_distance_norm",
        float(rule["same_eye_thumb_index_distance_norm_max"]),
        "distancia del ojo al segmento pulgar-índice para E",
    )

    return reasons



def _validate_f_geometry(metrics: dict[str, float], rule: dict) -> list[str]:
    reasons: list[str] = []

    _add_missing_or_min(
        reasons,
        metrics,
        "valid_ratio",
        float(rule["min_valid_ratio"]),
        "mano primaria visible para F",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "finger_count_extended",
        float(rule["finger_count_min"]),
        float(rule["finger_count_max"]),
        "cantidad de dedos extendidos/visibles para F",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_extended_score",
        float(rule["thumb_extended_score_min"]),
        "pulgar visible/activo para F",
    )
    _add_missing_or_min(
        reasons,
        metrics,
        "index_extended_score",
        float(rule["index_extended_score_min"]),
        "índice extendido para F",
    )
    _add_missing_or_min(
        reasons,
        metrics,
        "middle_extended_score",
        float(rule["middle_extended_score_min"]),
        "dedo medio extendido para F",
    )
    _add_missing_or_min(
        reasons,
        metrics,
        "ring_extended_score",
        float(rule["ring_extended_score_min"]),
        "dedo anular extendido para F",
    )
    _add_missing_or_min(
        reasons,
        metrics,
        "pinky_extended_score",
        float(rule["pinky_extended_score_min"]),
        "meñique extendido para F",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "index_middle_tip_distance_norm",
        float(rule["index_middle_tip_distance_norm_max"]),
        "dedos índice y medio juntos para F",
    )
    _add_missing_or_max(
        reasons,
        metrics,
        "middle_ring_tip_distance_norm",
        float(rule["middle_ring_tip_distance_norm_max"]),
        "dedos medio y anular juntos para F",
    )
    _add_missing_or_max(
        reasons,
        metrics,
        "ring_pinky_tip_distance_norm",
        float(rule["ring_pinky_tip_distance_norm_max"]),
        "dedos anular y meñique juntos para F",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_index_tip_distance_norm",
        float(rule["thumb_index_tip_distance_norm_min"]),
        float(rule["thumb_index_tip_distance_norm_max"]),
        "distancia pulgar-índice para F",
    )
    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_mcp_tip_angle_deg",
        float(rule["thumb_mcp_tip_angle_deg_min"]),
        float(rule["thumb_mcp_tip_angle_deg_max"]),
        "dirección del pulgar para F",
    )
    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_mcp_tip_horizontality",
        float(rule["thumb_mcp_tip_horizontality_max"]),
        "horizontalidad del pulgar para F",
    )
    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_mcp_tip_right_of_base",
        float(rule["thumb_mcp_tip_right_of_base_min"]),
        "pulgar ubicado hacia el borde visible de F",
    )
    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_index_axis_angle_3d_deg",
        float(rule["thumb_index_axis_angle_3d_deg_max"]),
        "relación angular pulgar-índice para F",
    )
    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_middle_axis_angle_3d_deg",
        float(rule["thumb_middle_axis_angle_3d_deg_max"]),
        "relación angular pulgar-medio para F",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_mcp_tip_angle_deg",
        float(rule["index_mcp_tip_angle_deg_min"]),
        float(rule["index_mcp_tip_angle_deg_max"]),
        "dirección ascendente del índice para F",
    )
    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_verticality",
        float(rule["index_mcp_tip_verticality_min"]),
        "verticalidad del índice para F",
    )
    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_above_base",
        float(rule["index_mcp_tip_above_base_min"]),
        "índice debe apuntar hacia arriba en F",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "middle_mcp_tip_angle_deg",
        float(rule["middle_mcp_tip_angle_deg_min"]),
        float(rule["middle_mcp_tip_angle_deg_max"]),
        "dirección ascendente del dedo medio para F",
    )
    _add_missing_or_max(
        reasons,
        metrics,
        "middle_mcp_tip_horizontality",
        float(rule["middle_mcp_tip_horizontality_max"]),
        "horizontalidad del dedo medio para F",
    )
    _add_missing_or_max(
        reasons,
        metrics,
        "middle_mcp_tip_below_base",
        float(rule["middle_mcp_tip_below_base_max"]),
        "dedo medio no debe apuntar hacia abajo en F",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "index_middle_axis_angle_3d_deg",
        float(rule["index_middle_axis_angle_3d_deg_max"]),
        "paralelismo índice-medio para F",
    )
    _add_missing_or_max(
        reasons,
        metrics,
        "middle_index_axis_signed_range",
        float(rule["middle_index_axis_signed_range_max"]),
        "bloque compacto índice-medio para F",
    )

    if bool(rule.get("use_palm_normal_z", False)):
        _add_missing_or_range(
            reasons,
            metrics,
            "palm_normal_z_approx",
            float(rule["palm_normal_z_approx_min"]),
            float(rule["palm_normal_z_approx_max"]),
            "orientación lateral/de canto para F",
        )

    _add_missing_or_min(
        reasons,
        metrics,
        "pose_valid_for_f_chest_geometry_ratio",
        float(rule["pose_valid_for_f_chest_geometry_ratio_min"]),
        "anclas de hombros visibles para F",
    )
    _add_missing_or_range(
        reasons,
        metrics,
        "hand_center_shoulder_dx",
        float(rule["hand_center_shoulder_dx_min"]),
        float(rule["hand_center_shoulder_dx_max"]),
        "ubicación horizontal pecho/hombro izquierdo para F",
    )
    _add_missing_or_range(
        reasons,
        metrics,
        "hand_center_shoulder_dy",
        float(rule["hand_center_shoulder_dy_min"]),
        float(rule["hand_center_shoulder_dy_max"]),
        "ubicación vertical pecho superior para F",
    )

    return reasons


def _validate_g_geometry(metrics: dict[str, float], rule: dict) -> list[str]:
    reasons: list[str] = []

    _add_missing_or_min(
        reasons,
        metrics,
        "valid_ratio",
        float(rule["min_valid_ratio"]),
        "mano primaria visible para G",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "finger_count_extended",
        float(rule["finger_count_min"]),
        float(rule["finger_count_max"]),
        "cantidad de dedos extendidos/visibles para G",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_extended_score",
        float(rule["thumb_extended_score_min"]),
        "pulgar visible/activo para G",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "index_extended_score",
        float(rule["index_extended_score_max"]),
        "dedo índice debe estar recogido en G",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "middle_extended_score",
        float(rule["middle_extended_score_max"]),
        "dedo medio debe estar recogido en G",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "ring_extended_score",
        float(rule["ring_extended_score_max"]),
        "dedo anular debe estar recogido en G",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "pinky_extended_score",
        float(rule["pinky_extended_score_max"]),
        "dedo meñique debe estar recogido en G",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_index_tip_distance_norm",
        float(rule["thumb_index_tip_distance_norm_min"]),
        float(rule["thumb_index_tip_distance_norm_max"]),
        "distancia pulgar-índice para G",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "pose_valid_for_ear_geometry_ratio",
        float(rule["pose_valid_for_ear_geometry_ratio_min"]),
        "pose/capa de orejas visible para G",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "right_ear_center_dist_earspan",
        float(rule["right_ear_center_dist_earspan_max"]),
        "distancia mano-oreja derecha para G",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "right_ear_dx_earspan",
        float(rule["right_ear_dx_earspan_min"]),
        float(rule["right_ear_dx_earspan_max"]),
        "posición horizontal respecto de oreja derecha para G",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "right_ear_dy_earspan",
        float(rule["right_ear_dy_earspan_min"]),
        float(rule["right_ear_dy_earspan_max"]),
        "posición vertical respecto de oreja derecha para G",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "left_ear_center_dist_earspan",
        float(rule["left_ear_center_dist_earspan_min"]),
        "mano demasiado cercana a la oreja izquierda para G",
    )

    return reasons



def _validate_h_geometry(metrics: dict[str, float], rule: dict) -> list[str]:
    reasons: list[str] = []

    _add_missing_or_min(
        reasons,
        metrics,
        "valid_ratio",
        float(rule["min_valid_ratio"]),
        "mano primaria visible para H",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "finger_count_extended",
        float(rule["finger_count_min"]),
        float(rule["finger_count_max"]),
        "cantidad de dedos extendidos/visibles para H",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_extended_score",
        float(rule["thumb_extended_score_min"]),
        "pulgar extendido/activo para H",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_extended_score",
        float(rule["index_extended_score_min"]),
        "índice extendido para H",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_extended_score",
        float(rule["middle_extended_score_min"]),
        "dedo medio extendido para H",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "ring_extended_score",
        float(rule["ring_extended_score_max"]),
        "dedo anular debe estar recogido en H",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "pinky_extended_score",
        float(rule["pinky_extended_score_max"]),
        "dedo meñique debe estar recogido en H",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_index_tip_distance_norm",
        float(rule["thumb_index_tip_distance_norm_min"]),
        float(rule["thumb_index_tip_distance_norm_max"]),
        "distancia pulgar-índice para H",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_middle_tip_distance_norm",
        float(rule["index_middle_tip_distance_norm_min"]),
        "separación índice-medio para H",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_ring_tip_distance_norm",
        float(rule["middle_ring_tip_distance_norm_min"]),
        "separación medio-anular para H",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "ring_pinky_tip_distance_norm",
        float(rule["ring_pinky_tip_distance_norm_max"]),
        "anular y meñique deben permanecer compactos en H",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_middle_axis_angle_3d_deg",
        float(rule["index_middle_axis_angle_3d_deg_min"]),
        float(rule["index_middle_axis_angle_3d_deg_max"]),
        "ángulo 3D índice-medio para H",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_mcp_tip_angle_deg",
        float(rule["thumb_mcp_tip_angle_deg_min"]),
        float(rule["thumb_mcp_tip_angle_deg_max"]),
        "dirección del pulgar para H",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_mcp_tip_right_of_base",
        float(rule["thumb_mcp_tip_right_of_base_min"]),
        "pulgar debe quedar del lado compatible para H",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_mcp_tip_angle_deg",
        float(rule["index_mcp_tip_angle_deg_min"]),
        float(rule["index_mcp_tip_angle_deg_max"]),
        "dirección del índice para H",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_above_base",
        float(rule["index_mcp_tip_above_base_min"]),
        "índice debe mantenerse orientado hacia arriba/lateral compatible en H",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "middle_mcp_tip_angle_deg",
        float(rule["middle_mcp_tip_angle_deg_min"]),
        float(rule["middle_mcp_tip_angle_deg_max"]),
        "dirección del dedo medio para H",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_mcp_tip_right_of_base",
        float(rule["middle_mcp_tip_right_of_base_min"]),
        "dedo medio debe apuntar hacia el lado compatible en H",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "pose_valid_for_h_face_geometry_ratio",
        float(rule["pose_valid_for_h_face_geometry_ratio_min"]),
        "landmarks de cara/orejas visibles para ubicar H",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "nose_inside_hand_bbox_ratio",
        float(rule["nose_inside_hand_bbox_ratio_min"]),
        "trayectoria de H debe pasar sobre la región de la cara",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "nose_dx_earspan",
        float(rule["nose_dx_earspan_min"]),
        float(rule["nose_dx_earspan_max"]),
        "posición horizontal de H respecto de la nariz",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "nose_dy_earspan",
        float(rule["nose_dy_earspan_min"]),
        float(rule["nose_dy_earspan_max"]),
        "posición vertical de H respecto de la nariz",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "nose_center_dist_earspan",
        float(rule["nose_center_dist_earspan_max"]),
        "distancia media mano-nariz para H",
    )

    return reasons



def _validate_i_geometry(metrics: dict[str, float], rule: dict) -> list[str]:
    reasons: list[str] = []

    _add_missing_or_min(
        reasons,
        metrics,
        "valid_ratio",
        float(rule["min_valid_ratio"]),
        "mano primaria visible para I",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "finger_count_extended",
        float(rule["finger_count_min"]),
        float(rule["finger_count_max"]),
        "cantidad de dedos extendidos/visibles para I",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_extended_score",
        float(rule["index_extended_score_min"]),
        "índice extendido para I",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "middle_extended_score",
        float(rule["middle_extended_score_max"]),
        "dedo medio debe estar recogido en I",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "ring_extended_score",
        float(rule["ring_extended_score_max"]),
        "dedo anular debe estar recogido en I",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "pinky_extended_score",
        float(rule["pinky_extended_score_max"]),
        "dedo meñique debe estar recogido en I",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_middle_tip_distance_norm",
        float(rule["index_middle_tip_distance_norm_min"]),
        "índice debe estar separado de los dedos recogidos en I",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_mcp_tip_angle_deg",
        float(rule["index_mcp_tip_angle_deg_min"]),
        float(rule["index_mcp_tip_angle_deg_max"]),
        "dirección vertical del índice para I",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_verticality",
        float(rule["index_mcp_tip_verticality_min"]),
        "verticalidad del índice para I",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_above_base",
        float(rule["index_mcp_tip_above_base_min"]),
        "índice debe apuntar hacia arriba en I",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_extended_score",
        float(rule["thumb_extended_score_max"]),
        "pulgar no debe estar abierto/protagonista en I",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_mcp_tip_angle_deg",
        float(rule["thumb_mcp_tip_angle_deg_min"]),
        float(rule["thumb_mcp_tip_angle_deg_max"]),
        "dirección del pulgar compatible con I",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_mcp_tip_right_of_base",
        float(rule["thumb_mcp_tip_right_of_base_min"]),
        "pulgar no debe abrirse como L en I",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "pose_valid_for_i_face_geometry_ratio",
        float(rule["pose_valid_for_i_face_geometry_ratio_min"]),
        "pose/cara visible para validar pómulo en I",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_tip_right_eye_dx_earspan",
        float(rule["index_tip_right_eye_dx_earspan_min"]),
        float(rule["index_tip_right_eye_dx_earspan_max"]),
        "posición horizontal de la punta del índice respecto al ojo derecho para I",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_tip_right_eye_dy_earspan",
        float(rule["index_tip_right_eye_dy_earspan_min"]),
        float(rule["index_tip_right_eye_dy_earspan_max"]),
        "posición vertical de la punta del índice respecto al ojo derecho para I",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_tip_mouth_dy_earspan",
        float(rule["index_tip_mouth_dy_earspan_min"]),
        float(rule["index_tip_mouth_dy_earspan_max"]),
        "punta del índice debe quedar por encima de la boca en I",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_tip_nose_dx_earspan",
        float(rule["index_tip_nose_dx_earspan_min"]),
        float(rule["index_tip_nose_dx_earspan_max"]),
        "punta del índice no debe ir hacia nariz/centro de cara en I",
    )

    return reasons


def _validate_j_geometry(metrics: dict[str, float], rule: dict) -> list[str]:
    reasons: list[str] = []

    _add_missing_or_min(
        reasons,
        metrics,
        "valid_ratio",
        float(rule["min_valid_ratio"]),
        "mano primaria visible para J",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "finger_count_extended",
        float(rule["finger_count_min"]),
        float(rule["finger_count_max"]),
        "cantidad de dedos extendidos/visibles para J",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_extended_score",
        float(rule["index_extended_score_min"]),
        "índice extendido/activo para J",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_extended_score",
        float(rule["middle_extended_score_min"]),
        "dedo medio extendido/activo para J",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "ring_extended_score",
        float(rule["ring_extended_score_min"]),
        "dedo anular extendido/activo para J",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "pinky_extended_score",
        float(rule["pinky_extended_score_min"]),
        "dedo meñique extendido/activo para J",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "index_middle_tip_distance_norm",
        float(rule["index_middle_tip_distance_norm_max"]),
        "índice y medio deben estar juntos en J",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "middle_ring_tip_distance_norm",
        float(rule["middle_ring_tip_distance_norm_max"]),
        "medio y anular deben estar juntos en J",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "ring_pinky_tip_distance_norm",
        float(rule["ring_pinky_tip_distance_norm_max"]),
        "anular y meñique deben estar juntos en J",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "index_middle_axis_angle_3d_deg",
        float(rule["index_middle_axis_angle_3d_deg_max"]),
        "ejes índice-medio deben ser casi paralelos en J",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_index_tip_distance_norm",
        float(rule["thumb_index_tip_distance_norm_min"]),
        float(rule["thumb_index_tip_distance_norm_max"]),
        "pulgar debe permanecer compacto/oculto en J",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_mcp_tip_angle_deg",
        float(rule["thumb_mcp_tip_angle_deg_min"]),
        float(rule["thumb_mcp_tip_angle_deg_max"]),
        "dirección del pulgar compatible con J",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_mcp_tip_angle_deg",
        float(rule["index_mcp_tip_angle_deg_min"]),
        float(rule["index_mcp_tip_angle_deg_max"]),
        "dirección del índice compatible con J",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "middle_mcp_tip_angle_deg",
        float(rule["middle_mcp_tip_angle_deg_min"]),
        float(rule["middle_mcp_tip_angle_deg_max"]),
        "dirección del dedo medio compatible con J",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "pose_valid_for_j_jaw_geometry_ratio",
        float(rule["pose_valid_for_j_jaw_geometry_ratio_min"]),
        "pose/cara visible para validar zona de mandíbula en J",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "mouth_dx_earspan",
        float(rule["mouth_dx_earspan_min"]),
        float(rule["mouth_dx_earspan_max"]),
        "posición horizontal de la mano respecto de boca/mandíbula para J",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "mouth_dy_earspan",
        float(rule["mouth_dy_earspan_min"]),
        float(rule["mouth_dy_earspan_max"]),
        "posición vertical de la mano respecto de boca/mandíbula para J",
    )

    return reasons


def _validate_z_geometry(metrics: dict[str, float], rule: dict) -> list[str]:
    reasons: list[str] = []

    _add_missing_or_min(
        reasons,
        metrics,
        "valid_ratio",
        float(rule["min_valid_ratio"]),
        "mano primaria visible para Z",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "finger_count_extended",
        float(rule["finger_count_min"]),
        float(rule["finger_count_max"]),
        "cantidad de dedos extendidos/visibles para Z",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "pinky_extended_score",
        float(rule["pinky_extended_score_min"]),
        "meñique extendido/activo para Z",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "index_extended_score",
        float(rule["index_extended_score_max"]),
        "índice debe estar recogido en Z",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "middle_extended_score",
        float(rule["middle_extended_score_max"]),
        "dedo medio debe estar recogido en Z",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "ring_extended_score",
        float(rule["ring_extended_score_max"]),
        "dedo anular debe estar recogido en Z",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "ring_pinky_tip_distance_norm",
        float(rule["ring_pinky_tip_distance_norm_min"]),
        "separación anular-meñique para Z",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_index_tip_distance_norm",
        float(rule["thumb_index_tip_distance_norm_min"]),
        float(rule["thumb_index_tip_distance_norm_max"]),
        "pulgar debe permanecer compacto/no protagonista en Z",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_mcp_tip_angle_deg",
        float(rule["thumb_mcp_tip_angle_deg_min"]),
        float(rule["thumb_mcp_tip_angle_deg_max"]),
        "dirección del pulgar compatible con Z",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_mcp_tip_horizontality",
        float(rule["thumb_mcp_tip_horizontality_max"]),
        "pulgar no debe abrirse horizontalmente en Z",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_mcp_tip_angle_deg",
        float(rule["index_mcp_tip_angle_deg_min"]),
        float(rule["index_mcp_tip_angle_deg_max"]),
        "dirección del índice recogido compatible con Z",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "middle_mcp_tip_angle_deg",
        float(rule["middle_mcp_tip_angle_deg_min"]),
        float(rule["middle_mcp_tip_angle_deg_max"]),
        "dirección del dedo medio recogido compatible con Z",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "middle_mcp_tip_horizontality",
        float(rule["middle_mcp_tip_horizontality_max"]),
        "dedo medio no debe abrirse horizontalmente en Z",
    )

    if bool(rule.get("use_palm_normal_z", False)):
        _add_missing_or_max(
            reasons,
            metrics,
            "palm_normal_z_approx",
            float(rule["palm_normal_z_approx_max"]),
            "orientación palma/meñique para Z",
        )

    return reasons


def _validate_y_geometry(metrics: dict[str, float], rule: dict) -> list[str]:
    reasons: list[str] = []

    _add_missing_or_min(
        reasons,
        metrics,
        "valid_ratio",
        float(rule["min_valid_ratio"]),
        "mano primaria visible para Y",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "finger_count_extended",
        float(rule["finger_count_min"]),
        float(rule["finger_count_max"]),
        "cantidad de dedos extendidos/visibles para Y",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_extended_score",
        float(rule["thumb_extended_score_min"]),
        "pulgar extendido para Y",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "pinky_extended_score",
        float(rule["pinky_extended_score_min"]),
        "meñique extendido para Y",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "index_extended_score",
        float(rule["index_extended_score_max"]),
        "índice debe estar recogido en Y",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "middle_extended_score",
        float(rule["middle_extended_score_max"]),
        "dedo medio debe estar recogido en Y",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "ring_extended_score",
        float(rule["ring_extended_score_max"]),
        "dedo anular debe estar recogido en Y",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_pinky_tip_distance_norm",
        float(rule["thumb_pinky_tip_distance_norm_min"]),
        "apertura pulgar-meñique para Y",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "ring_pinky_tip_distance_norm",
        float(rule["ring_pinky_tip_distance_norm_min"]),
        "separación anular-meñique para Y",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_mcp_tip_angle_deg",
        float(rule["thumb_mcp_tip_angle_deg_min"]),
        float(rule["thumb_mcp_tip_angle_deg_max"]),
        "dirección del pulgar compatible con Y",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_mcp_tip_left_of_base",
        float(rule["thumb_mcp_tip_left_of_base_min"]),
        "pulgar debe abrirse hacia el lado esperado en Y",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "pinky_mcp_tip_angle_deg",
        float(rule["pinky_mcp_tip_angle_deg_min"]),
        float(rule["pinky_mcp_tip_angle_deg_max"]),
        "dirección diagonal del meñique para Y",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "pinky_mcp_tip_verticality",
        float(rule["pinky_mcp_tip_verticality_min"]),
        "verticalidad diagonal del meñique para Y",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "pinky_mcp_tip_above_base",
        float(rule["pinky_mcp_tip_above_base_min"]),
        "meñique debe apuntar hacia arriba en Y",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "pose_valid_for_y_mouth_geometry_ratio",
        float(rule["pose_valid_for_y_mouth_geometry_ratio_min"]),
        "pose/boca visible para validar Y",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "pinky_tip_mouth_dx_earspan",
        float(rule["pinky_tip_mouth_dx_earspan_min"]),
        float(rule["pinky_tip_mouth_dx_earspan_max"]),
        "posición horizontal del meñique respecto de la boca para Y",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "pinky_tip_mouth_dy_earspan",
        float(rule["pinky_tip_mouth_dy_earspan_min"]),
        float(rule["pinky_tip_mouth_dy_earspan_max"]),
        "posición vertical del meñique respecto de la boca para Y",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "pinky_tip_mouth_dist_earspan",
        float(rule["pinky_tip_mouth_dist_earspan_max"]),
        "distancia meñique-boca para Y",
    )

    return reasons




def _validate_u_geometry(metrics: dict[str, float], rule: dict) -> list[str]:
    reasons: list[str] = []

    _add_missing_or_min(
        reasons,
        metrics,
        "valid_ratio",
        float(rule["min_valid_ratio"]),
        "mano primaria visible para U",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "finger_count_extended",
        float(rule["finger_count_min"]),
        float(rule["finger_count_max"]),
        "cantidad de dedos extendidos/visibles para U",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_extended_score",
        float(rule["index_extended_score_min"]),
        "índice extendido para U",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "pinky_extended_score",
        float(rule["pinky_extended_score_min"]),
        "meñique extendido para U",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "middle_extended_score",
        float(rule["middle_extended_score_max"]),
        "dedo medio debe estar recogido en U",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "ring_extended_score",
        float(rule["ring_extended_score_max"]),
        "dedo anular debe estar recogido en U",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_extended_score",
        float(rule["thumb_extended_score_max"]),
        "pulgar no debe ser protagonista en U",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_mcp_tip_angle_deg",
        float(rule["index_mcp_tip_angle_deg_min"]),
        float(rule["index_mcp_tip_angle_deg_max"]),
        "dirección vertical del índice para U",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_verticality",
        float(rule["index_mcp_tip_verticality_min"]),
        "verticalidad del índice para U",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_above_base",
        float(rule["index_mcp_tip_above_base_min"]),
        "índice debe apuntar hacia arriba en U",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "pinky_mcp_tip_angle_deg",
        float(rule["pinky_mcp_tip_angle_deg_min"]),
        float(rule["pinky_mcp_tip_angle_deg_max"]),
        "dirección vertical del meñique para U",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "pinky_mcp_tip_verticality",
        float(rule["pinky_mcp_tip_verticality_min"]),
        "verticalidad del meñique para U",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "pinky_mcp_tip_above_base",
        float(rule["pinky_mcp_tip_above_base_min"]),
        "meñique debe apuntar hacia arriba en U",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_middle_tip_distance_norm",
        float(rule["index_middle_tip_distance_norm_min"]),
        "separación índice-medio para U",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "ring_pinky_tip_distance_norm",
        float(rule["ring_pinky_tip_distance_norm_min"]),
        float(rule["ring_pinky_tip_distance_norm_max"]),
        "separación anular-meñique para U",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_pinky_tip_distance_norm",
        float(rule["index_pinky_tip_distance_norm_min"]),
        float(rule["index_pinky_tip_distance_norm_max"]),
        "apertura índice-meñique para U",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_pinky_tip_distance_norm",
        float(rule["thumb_pinky_tip_distance_norm_min"]),
        float(rule["thumb_pinky_tip_distance_norm_max"]),
        "distancia pulgar-meñique compatible con U",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_mcp_tip_angle_deg",
        float(rule["thumb_mcp_tip_angle_deg_min"]),
        float(rule["thumb_mcp_tip_angle_deg_max"]),
        "dirección del pulgar compatible con U",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_mcp_tip_right_of_base",
        float(rule["thumb_mcp_tip_right_of_base_min"]),
        "pulgar debe quedar hacia el lado esperado en U",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_mcp_tip_left_of_base",
        float(rule["thumb_mcp_tip_left_of_base_max"]),
        "pulgar no debe abrirse hacia el lado contrario en U",
    )

    return reasons




def _validate_m_geometry(metrics: dict[str, float], rule: dict) -> list[str]:
    reasons: list[str] = []

    _add_missing_or_min(
        reasons,
        metrics,
        "valid_ratio",
        float(rule["min_valid_ratio"]),
        "mano primaria visible para M",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "finger_count_extended",
        float(rule["finger_count_min"]),
        float(rule["finger_count_max"]),
        "cantidad de dedos extendidos/visibles para M",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_extended_score",
        float(rule["index_extended_score_min"]),
        "índice extendido para M",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_extended_score",
        float(rule["middle_extended_score_min"]),
        "dedo medio extendido para M",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "ring_extended_score",
        float(rule["ring_extended_score_min"]),
        "dedo anular extendido para M",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "pinky_extended_score",
        float(rule["pinky_extended_score_min"]),
        "meñique extendido para M",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_mcp_tip_angle_deg",
        float(rule["index_mcp_tip_angle_deg_min"]),
        float(rule["index_mcp_tip_angle_deg_max"]),
        "dirección descendente del índice para M",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_verticality",
        float(rule["index_mcp_tip_verticality_min"]),
        "verticalidad del índice para M",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_below_base",
        float(rule["index_mcp_tip_below_base_min"]),
        "índice apuntando hacia abajo para M",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "index_mcp_tip_above_base",
        float(rule["index_mcp_tip_above_base_max"]),
        "índice no debe apuntar hacia arriba en M",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "middle_mcp_tip_angle_deg",
        float(rule["middle_mcp_tip_angle_deg_min"]),
        float(rule["middle_mcp_tip_angle_deg_max"]),
        "dirección descendente del dedo medio para M",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_mcp_tip_below_base",
        float(rule["middle_mcp_tip_below_base_min"]),
        "dedo medio apuntando hacia abajo para M",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "ring_mcp_tip_angle_deg",
        float(rule["ring_mcp_tip_angle_deg_min"]),
        float(rule["ring_mcp_tip_angle_deg_max"]),
        "dirección descendente del anular para M",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "ring_mcp_tip_below_base",
        float(rule["ring_mcp_tip_below_base_min"]),
        "anular apuntando hacia abajo para M",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "pinky_mcp_tip_angle_deg",
        float(rule["pinky_mcp_tip_angle_deg_min"]),
        float(rule["pinky_mcp_tip_angle_deg_max"]),
        "dirección descendente del meñique para M",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "pinky_mcp_tip_below_base",
        float(rule["pinky_mcp_tip_below_base_min"]),
        "meñique apuntando hacia abajo para M",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_middle_tip_distance_norm",
        float(rule["index_middle_tip_distance_norm_min"]),
        "separación índice-medio para M",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_ring_tip_distance_norm",
        float(rule["middle_ring_tip_distance_norm_min"]),
        "separación medio-anular para M",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "ring_pinky_tip_distance_norm",
        float(rule["ring_pinky_tip_distance_norm_min"]),
        "separación anular-meñique para M",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_index_tip_distance_norm",
        float(rule["thumb_index_tip_distance_norm_max"]),
        "pulgar demasiado separado del índice para M",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_mcp_tip_horizontality",
        float(rule["thumb_mcp_tip_horizontality_max"]),
        "pulgar demasiado horizontal/protagonista para M",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_mcp_tip_angle_deg",
        float(rule["thumb_mcp_tip_angle_deg_min"]),
        float(rule["thumb_mcp_tip_angle_deg_max"]),
        "dirección del pulgar compatible con M",
    )

    if bool(rule.get("use_palm_normal_z", False)):
        _add_missing_or_min(
            reasons,
            metrics,
            "palm_normal_z_approx",
            float(rule["palm_normal_z_approx_min"]),
            "orientación de dorso/palma para M",
        )

    return reasons


def _validate_n_geometry(metrics: dict[str, float], rule: dict) -> list[str]:
    reasons: list[str] = []

    _add_missing_or_min(
        reasons,
        metrics,
        "valid_ratio",
        float(rule["min_valid_ratio"]),
        "mano primaria visible para N",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "finger_count_extended",
        float(rule["finger_count_min"]),
        float(rule["finger_count_max"]),
        "cantidad de dedos extendidos/visibles para N",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_extended_score",
        float(rule["index_extended_score_min"]),
        "índice extendido para N",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_extended_score",
        float(rule["middle_extended_score_min"]),
        "dedo medio extendido para N",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "ring_extended_score",
        float(rule["ring_extended_score_max"]),
        "dedo anular recogido para N",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "pinky_extended_score",
        float(rule["pinky_extended_score_max"]),
        "meñique recogido para N",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_mcp_tip_angle_deg",
        float(rule["index_mcp_tip_angle_deg_min"]),
        float(rule["index_mcp_tip_angle_deg_max"]),
        "dirección descendente del índice para N",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_verticality",
        float(rule["index_mcp_tip_verticality_min"]),
        "verticalidad del índice para N",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_below_base",
        float(rule["index_mcp_tip_below_base_min"]),
        "índice apuntando hacia abajo para N",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "index_mcp_tip_above_base",
        float(rule["index_mcp_tip_above_base_max"]),
        "índice no debe apuntar hacia arriba en N",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "middle_mcp_tip_angle_deg",
        float(rule["middle_mcp_tip_angle_deg_min"]),
        float(rule["middle_mcp_tip_angle_deg_max"]),
        "dirección descendente del dedo medio para N",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_mcp_tip_below_base",
        float(rule["middle_mcp_tip_below_base_min"]),
        "dedo medio apuntando hacia abajo para N",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "middle_mcp_tip_horizontality",
        float(rule["middle_mcp_tip_horizontality_max"]),
        "dedo medio demasiado horizontal para N",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_middle_tip_distance_norm",
        float(rule["index_middle_tip_distance_norm_min"]),
        "separación índice-medio para N insuficiente",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "index_middle_tip_distance_norm",
        float(rule["index_middle_tip_distance_norm_max"]),
        "separación índice-medio para N excesiva",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_ring_tip_distance_norm",
        float(rule["middle_ring_tip_distance_norm_min"]),
        "separación medio-anular para N insuficiente",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "ring_pinky_tip_distance_norm",
        float(rule["ring_pinky_tip_distance_norm_max"]),
        "anular y meñique demasiado separados para N",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_index_tip_distance_norm",
        float(rule["thumb_index_tip_distance_norm_max"]),
        "pulgar demasiado separado del índice para N",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_mcp_tip_horizontality",
        float(rule["thumb_mcp_tip_horizontality_max"]),
        "pulgar demasiado horizontal/protagonista para N",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_mcp_tip_angle_deg",
        float(rule["thumb_mcp_tip_angle_deg_min"]),
        float(rule["thumb_mcp_tip_angle_deg_max"]),
        "dirección del pulgar compatible con N",
    )

    if bool(rule.get("use_palm_normal_z", False)):
        _add_missing_or_min(
            reasons,
            metrics,
            "palm_normal_z_approx",
            float(rule["palm_normal_z_approx_min"]),
            "orientación de dorso/palma para N",
        )

    return reasons



def _validate_enie_geometry(metrics: dict[str, float], rule: dict) -> list[str]:
    reasons: list[str] = []

    # Compatibilidad: si el agregador no creó el alias explícito,
    # usar el promedio de hand_valid de la mano secundaria.
    if "secondary_valid_ratio" not in metrics and "secondary_hand_valid" in metrics:
        metrics = dict(metrics)
        metrics["secondary_valid_ratio"] = metrics.get("secondary_hand_valid", np.nan)

    _add_missing_or_min(
        reasons,
        metrics,
        "valid_ratio",
        float(rule["min_valid_ratio"]),
        "mano primaria visible para Ñ",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "secondary_valid_ratio",
        float(rule["secondary_valid_ratio_min"]),
        "mano secundaria visible para Ñ",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "two_hands_valid",
        float(rule["expected_hands_min"]),
        "dos manos detectadas para Ñ",
    )

    # Mano primaria: base N.
    _add_missing_or_range(
        reasons,
        metrics,
        "finger_count_extended",
        float(rule["finger_count_min"]),
        float(rule["finger_count_max"]),
        "cantidad de dedos extendidos/visibles en mano primaria para Ñ",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_extended_score",
        float(rule["index_extended_score_min"]),
        "índice extendido en mano primaria para Ñ",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_extended_score",
        float(rule["middle_extended_score_min"]),
        "dedo medio extendido en mano primaria para Ñ",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "ring_extended_score",
        float(rule["ring_extended_score_max"]),
        "dedo anular recogido en mano primaria para Ñ",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "pinky_extended_score",
        float(rule["pinky_extended_score_max"]),
        "meñique recogido en mano primaria para Ñ",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_mcp_tip_angle_deg",
        float(rule["index_mcp_tip_angle_deg_min"]),
        float(rule["index_mcp_tip_angle_deg_max"]),
        "dirección descendente del índice en mano primaria para Ñ",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_verticality",
        float(rule["index_mcp_tip_verticality_min"]),
        "verticalidad del índice en mano primaria para Ñ",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_below_base",
        float(rule["index_mcp_tip_below_base_min"]),
        "índice apuntando hacia abajo en mano primaria para Ñ",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "index_mcp_tip_above_base",
        float(rule["index_mcp_tip_above_base_max"]),
        "índice no debe apuntar hacia arriba en mano primaria para Ñ",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "middle_mcp_tip_angle_deg",
        float(rule["middle_mcp_tip_angle_deg_min"]),
        float(rule["middle_mcp_tip_angle_deg_max"]),
        "dirección descendente del dedo medio en mano primaria para Ñ",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_mcp_tip_below_base",
        float(rule["middle_mcp_tip_below_base_min"]),
        "dedo medio apuntando hacia abajo en mano primaria para Ñ",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "middle_mcp_tip_horizontality",
        float(rule["middle_mcp_tip_horizontality_max"]),
        "dedo medio demasiado horizontal en mano primaria para Ñ",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_middle_tip_distance_norm",
        float(rule["index_middle_tip_distance_norm_min"]),
        "separación índice-medio en mano primaria para Ñ insuficiente",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "index_middle_tip_distance_norm",
        float(rule["index_middle_tip_distance_norm_max"]),
        "separación índice-medio en mano primaria para Ñ excesiva",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_ring_tip_distance_norm",
        float(rule["middle_ring_tip_distance_norm_min"]),
        "separación medio-anular en mano primaria para Ñ insuficiente",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "ring_pinky_tip_distance_norm",
        float(rule["ring_pinky_tip_distance_norm_max"]),
        "anular y meñique demasiado separados en mano primaria para Ñ",
    )

    # Pulgar de la base N: se usa el mismo criterio que en N aislada.
    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_index_tip_distance_norm",
        float(rule["thumb_index_tip_distance_norm_max"]),
        "pulgar demasiado separado del índice en mano primaria para Ñ",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_mcp_tip_horizontality",
        float(rule["thumb_mcp_tip_horizontality_max"]),
        "pulgar demasiado horizontal/protagonista en mano primaria para Ñ",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_mcp_tip_angle_deg",
        float(rule["thumb_mcp_tip_angle_deg_min"]),
        float(rule["thumb_mcp_tip_angle_deg_max"]),
        "dirección del pulgar compatible con base N de Ñ",
    )

    if bool(rule.get("use_palm_normal_z", False)):
        _add_missing_or_min(
            reasons,
            metrics,
            "palm_normal_z_approx",
            float(rule["palm_normal_z_approx_min"]),
            "orientación de dorso/palma en mano primaria para Ñ",
        )

    # Mano secundaria: tilde con índice.
    _add_missing_or_range(
        reasons,
        metrics,
        "secondary_finger_count_extended",
        float(rule["secondary_finger_count_min"]),
        float(rule["secondary_finger_count_max"]),
        "cantidad de dedos extendidos/visibles en mano secundaria para Ñ",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "secondary_index_extended_score",
        float(rule["secondary_index_extended_score_min"]),
        "índice extendido en mano secundaria para Ñ",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "secondary_middle_extended_score",
        float(rule["secondary_middle_extended_score_max"]),
        "dedo medio recogido en mano secundaria para Ñ",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "secondary_ring_extended_score",
        float(rule["secondary_ring_extended_score_max"]),
        "dedo anular recogido en mano secundaria para Ñ",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "secondary_pinky_extended_score",
        float(rule["secondary_pinky_extended_score_max"]),
        "meñique recogido en mano secundaria para Ñ",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "secondary_thumb_extended_score",
        float(rule["secondary_thumb_extended_score_max"]),
        "pulgar no protagonista en mano secundaria para Ñ",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "secondary_index_mcp_tip_angle_deg",
        float(rule["secondary_index_mcp_tip_angle_deg_min"]),
        float(rule["secondary_index_mcp_tip_angle_deg_max"]),
        "dirección horizontal/diagonal del índice secundario para Ñ",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "secondary_index_mcp_tip_horizontality",
        float(rule["secondary_index_mcp_tip_horizontality_min"]),
        "horizontalidad del índice secundario para Ñ",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "secondary_palm_normal_z_approx",
        float(rule["secondary_palm_normal_z_approx_max"]),
        "orientación dorso/uña de la mano secundaria para Ñ",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "enie_tilde_index_to_knuckle_line_distance_norm",
        float(rule["enie_tilde_index_to_knuckle_line_distance_norm_max"]),
        "contacto/proximidad de tilde con línea de nudillos de Ñ",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "enie_tilde_knuckle_line_coverage",
        float(rule["enie_tilde_knuckle_line_coverage_min"]),
        "cobertura de tilde sobre línea índice-medio de Ñ",
    )

    return reasons



def _validate_v_geometry(metrics: dict[str, float], rule: dict) -> list[str]:
    reasons: list[str] = []

    _add_missing_or_min(
        reasons,
        metrics,
        "valid_ratio",
        float(rule["min_valid_ratio"]),
        "mano primaria visible para V",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "finger_count_extended",
        float(rule["finger_count_min"]),
        float(rule["finger_count_max"]),
        "cantidad de dedos extendidos/visibles para V",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_extended_score",
        float(rule["index_extended_score_min"]),
        "índice extendido para V",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_extended_score",
        float(rule["middle_extended_score_min"]),
        "dedo medio extendido para V",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "ring_extended_score",
        float(rule["ring_extended_score_max"]),
        "dedo anular recogido para V",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "pinky_extended_score",
        float(rule["pinky_extended_score_max"]),
        "meñique recogido para V",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_extended_score",
        float(rule["thumb_extended_score_max"]),
        "pulgar no debe ser protagonista en V",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_middle_tip_distance_norm",
        float(rule["index_middle_tip_distance_norm_min"]),
        float(rule["index_middle_tip_distance_norm_max"]),
        "separación índice-medio para V",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "ring_pinky_tip_distance_norm",
        float(rule["ring_pinky_tip_distance_norm_max"]),
        "anular y meñique deben quedar compactos en V",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_mcp_tip_angle_deg",
        float(rule["index_mcp_tip_angle_deg_min"]),
        float(rule["index_mcp_tip_angle_deg_max"]),
        "dirección ascendente del índice para V",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_verticality",
        float(rule["index_mcp_tip_verticality_min"]),
        "verticalidad del índice para V",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_above_base",
        float(rule["index_mcp_tip_above_base_min"]),
        "índice debe apuntar hacia arriba en V",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "index_mcp_tip_below_base",
        float(rule["index_mcp_tip_below_base_max"]),
        "índice no debe apuntar hacia abajo en V",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "middle_mcp_tip_angle_deg",
        float(rule["middle_mcp_tip_angle_deg_min"]),
        float(rule["middle_mcp_tip_angle_deg_max"]),
        "dirección ascendente del dedo medio para V",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_mcp_tip_verticality",
        float(rule["middle_mcp_tip_verticality_min"]),
        "verticalidad del dedo medio para V",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_mcp_tip_above_base",
        float(rule["middle_mcp_tip_above_base_min"]),
        "dedo medio debe apuntar hacia arriba en V",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "middle_mcp_tip_below_base",
        float(rule["middle_mcp_tip_below_base_max"]),
        "dedo medio no debe apuntar hacia abajo en V",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "middle_mcp_tip_horizontality",
        float(rule["middle_mcp_tip_horizontality_max"]),
        "dedo medio demasiado horizontal para V",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_middle_axis_angle_3d_deg",
        float(rule["index_middle_axis_angle_3d_deg_min"]),
        float(rule["index_middle_axis_angle_3d_deg_max"]),
        "apertura angular índice-medio para V",
    )

    if bool(rule.get("use_palm_normal_z", False)):
        _add_missing_or_min(
            reasons,
            metrics,
            "palm_normal_z_approx",
            float(rule["palm_normal_z_approx_min"]),
            "orientación dorso hacia cámara para V",
        )

    return reasons



def _validate_w_geometry(metrics: dict[str, float], rule: dict) -> list[str]:
    reasons: list[str] = []

    # Compatibilidad: si el agregador no creó el alias explícito,
    # usar el promedio de hand_valid de la mano secundaria.
    if "secondary_valid_ratio" not in metrics and "secondary_hand_valid" in metrics:
        metrics = dict(metrics)
        metrics["secondary_valid_ratio"] = metrics.get("secondary_hand_valid", np.nan)


    _add_missing_or_min(
        reasons,
        metrics,
        "valid_ratio",
        float(rule["min_valid_ratio"]),
        "mano primaria visible para W",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "secondary_valid_ratio",
        float(rule["secondary_valid_ratio_min"]),
        "mano secundaria visible para W",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "two_hands_valid",
        float(rule["expected_hands_min"]),
        "dos manos detectadas para W",
    )

    # Mano primaria: forma tipo U.
    _add_missing_or_range(
        reasons,
        metrics,
        "finger_count_extended",
        float(rule["finger_count_min"]),
        float(rule["finger_count_max"]),
        "cantidad de dedos extendidos/visibles en mano primaria para W",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_extended_score",
        float(rule["index_extended_score_min"]),
        "índice extendido en mano primaria para W",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "pinky_extended_score",
        float(rule["pinky_extended_score_min"]),
        "meñique extendido en mano primaria para W",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "middle_extended_score",
        float(rule["middle_extended_score_max"]),
        "dedo medio recogido en mano primaria para W",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "ring_extended_score",
        float(rule["ring_extended_score_max"]),
        "dedo anular recogido en mano primaria para W",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_extended_score",
        float(rule["thumb_extended_score_max"]),
        "pulgar no protagonista en mano primaria para W",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_mcp_tip_angle_deg",
        float(rule["index_mcp_tip_angle_deg_min"]),
        float(rule["index_mcp_tip_angle_deg_max"]),
        "dirección vertical del índice en mano primaria para W",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_verticality",
        float(rule["index_mcp_tip_verticality_min"]),
        "verticalidad del índice en mano primaria para W",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_above_base",
        float(rule["index_mcp_tip_above_base_min"]),
        "índice hacia arriba en mano primaria para W",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "pinky_mcp_tip_angle_deg",
        float(rule["pinky_mcp_tip_angle_deg_min"]),
        float(rule["pinky_mcp_tip_angle_deg_max"]),
        "dirección vertical del meñique en mano primaria para W",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "pinky_mcp_tip_verticality",
        float(rule["pinky_mcp_tip_verticality_min"]),
        "verticalidad del meñique en mano primaria para W",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "pinky_mcp_tip_above_base",
        float(rule["pinky_mcp_tip_above_base_min"]),
        "meñique hacia arriba en mano primaria para W",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_mcp_tip_right_of_base",
        float(rule["thumb_mcp_tip_right_of_base_min"]),
        "pulgar al lado esperado en mano primaria para W",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_mcp_tip_left_of_base",
        float(rule["thumb_mcp_tip_left_of_base_max"]),
        "pulgar no debe abrirse al lado contrario en mano primaria para W",
    )

    # Mano secundaria: forma tipo U espejada.
    _add_missing_or_range(
        reasons,
        metrics,
        "secondary_finger_count_extended",
        float(rule["secondary_finger_count_min"]),
        float(rule["secondary_finger_count_max"]),
        "cantidad de dedos extendidos/visibles en mano secundaria para W",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "secondary_index_extended_score",
        float(rule["secondary_index_extended_score_min"]),
        "índice extendido en mano secundaria para W",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "secondary_pinky_extended_score",
        float(rule["secondary_pinky_extended_score_min"]),
        "meñique extendido en mano secundaria para W",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "secondary_middle_extended_score",
        float(rule["secondary_middle_extended_score_max"]),
        "dedo medio recogido en mano secundaria para W",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "secondary_ring_extended_score",
        float(rule["secondary_ring_extended_score_max"]),
        "dedo anular recogido en mano secundaria para W",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "secondary_thumb_extended_score",
        float(rule["secondary_thumb_extended_score_max"]),
        "pulgar no protagonista en mano secundaria para W",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "secondary_index_mcp_tip_angle_deg",
        float(rule["secondary_index_mcp_tip_angle_deg_min"]),
        float(rule["secondary_index_mcp_tip_angle_deg_max"]),
        "dirección vertical del índice en mano secundaria para W",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "secondary_index_mcp_tip_verticality",
        float(rule["secondary_index_mcp_tip_verticality_min"]),
        "verticalidad del índice en mano secundaria para W",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "secondary_index_mcp_tip_above_base",
        float(rule["secondary_index_mcp_tip_above_base_min"]),
        "índice hacia arriba en mano secundaria para W",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "secondary_pinky_mcp_tip_angle_deg",
        float(rule["secondary_pinky_mcp_tip_angle_deg_min"]),
        float(rule["secondary_pinky_mcp_tip_angle_deg_max"]),
        "dirección vertical del meñique en mano secundaria para W",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "secondary_pinky_mcp_tip_verticality",
        float(rule["secondary_pinky_mcp_tip_verticality_min"]),
        "verticalidad del meñique en mano secundaria para W",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "secondary_pinky_mcp_tip_above_base",
        float(rule["secondary_pinky_mcp_tip_above_base_min"]),
        "meñique hacia arriba en mano secundaria para W",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "secondary_thumb_mcp_tip_left_of_base",
        float(rule["secondary_thumb_mcp_tip_left_of_base_min"]),
        "pulgar al lado esperado en mano secundaria para W",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "secondary_thumb_mcp_tip_right_of_base",
        float(rule["secondary_thumb_mcp_tip_right_of_base_max"]),
        "pulgar no debe abrirse al lado contrario en mano secundaria para W",
    )

    # Conexión central por meñiques.
    # Se acepta cruce o contacto fuerte. No se exige intersección perfecta
    # porque MediaPipe puede deformar los landmarks por oclusión.
    pinky_tip_dist = _metric(metrics, "w_pinky_tip_distance_norm")
    pinky_segment_dist = _metric(metrics, "w_pinky_segment_min_distance_norm")
    tip_ok = np.isfinite(pinky_tip_dist) and pinky_tip_dist <= float(
        rule["w_pinky_tip_distance_norm_max"]
    )
    segment_ok = np.isfinite(pinky_segment_dist) and pinky_segment_dist <= float(
        rule["w_pinky_segment_min_distance_norm_max"]
    )

    if not (tip_ok or segment_ok):
        reasons.append(
            "conexión central por meñiques para W insuficiente "
            f"(tip_dist={pinky_tip_dist:.4f}, segment_dist={pinky_segment_dist:.4f}; "
            f"esperado tip <= {float(rule['w_pinky_tip_distance_norm_max']):.4f} "
            f"o segment <= {float(rule['w_pinky_segment_min_distance_norm_max']):.4f})"
        )

    _add_missing_or_min(
        reasons,
        metrics,
        "w_index_tip_distance_norm",
        float(rule["w_index_tip_distance_norm_min"]),
        "separación externa entre índices para W",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "w_pinky_index_tip_distance_ratio",
        float(rule["w_pinky_index_tip_distance_ratio_max"]),
        "relación distancia meñiques/distancia índices para W",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "w_index_tip_x_distance_norm",
        float(rule["w_index_tip_x_distance_norm_min"]),
        "separación horizontal entre índices para W",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "w_pinky_tip_x_distance_norm",
        float(rule["w_pinky_tip_x_distance_norm_max"]),
        "separación horizontal entre meñiques para W",
    )

    return reasons




def _validate_x_geometry(metrics: dict[str, float], rule: dict) -> list[str]:
    reasons: list[str] = []

    # Compatibilidad: si el agregador no creó el alias explícito,
    # usar el promedio de hand_valid de la mano secundaria.
    if "secondary_valid_ratio" not in metrics and "secondary_hand_valid" in metrics:
        metrics = dict(metrics)
        metrics["secondary_valid_ratio"] = metrics.get("secondary_hand_valid", np.nan)

    _add_missing_or_min(
        reasons,
        metrics,
        "valid_ratio",
        float(rule["min_valid_ratio"]),
        "mano primaria visible para X",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "secondary_valid_ratio",
        float(rule["secondary_valid_ratio_min"]),
        "mano secundaria visible para X",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "two_hands_valid",
        float(rule["expected_hands_min"]),
        "dos manos detectadas para X",
    )

    # Mano primaria: puño con índice protagonista.
    _add_missing_or_range(
        reasons,
        metrics,
        "finger_count_extended",
        float(rule["finger_count_min"]),
        float(rule["finger_count_max"]),
        "cantidad de dedos extendidos/visibles en mano primaria para X",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_extended_score",
        float(rule["index_extended_score_min"]),
        "índice extendido en mano primaria para X",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "middle_extended_score",
        float(rule["middle_extended_score_max"]),
        "dedo medio recogido en mano primaria para X",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "ring_extended_score",
        float(rule["ring_extended_score_max"]),
        "dedo anular recogido en mano primaria para X",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "pinky_extended_score",
        float(rule["pinky_extended_score_max"]),
        "meñique recogido en mano primaria para X",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_extended_score",
        float(rule["thumb_extended_score_max"]),
        "pulgar no protagonista en mano primaria para X",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_mcp_tip_angle_deg",
        float(rule["index_mcp_tip_angle_deg_min"]),
        float(rule["index_mcp_tip_angle_deg_max"]),
        "dirección diagonal del índice en mano primaria para X",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_verticality",
        float(rule["index_mcp_tip_verticality_min"]),
        "componente vertical del índice en mano primaria para X",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_above_base",
        float(rule["index_mcp_tip_above_base_min"]),
        "índice hacia arriba en mano primaria para X",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_index_tip_distance_norm",
        float(rule["thumb_index_tip_distance_norm_min"]),
        "separación pulgar-índice en mano primaria para X",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "palm_normal_z_approx",
        float(rule["palm_normal_z_approx_min"]),
        "orientación de nudillos/dorso en mano primaria para X",
    )

    # Mano secundaria: puño con índice protagonista, sin exigir ángulo
    # específico porque la mano espejada produce ángulos 2D menos estables.
    _add_missing_or_range(
        reasons,
        metrics,
        "secondary_finger_count_extended",
        float(rule["secondary_finger_count_min"]),
        float(rule["secondary_finger_count_max"]),
        "cantidad de dedos extendidos/visibles en mano secundaria para X",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "secondary_index_extended_score",
        float(rule["secondary_index_extended_score_min"]),
        "índice extendido en mano secundaria para X",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "secondary_middle_extended_score",
        float(rule["secondary_middle_extended_score_max"]),
        "dedo medio recogido en mano secundaria para X",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "secondary_ring_extended_score",
        float(rule["secondary_ring_extended_score_max"]),
        "dedo anular recogido en mano secundaria para X",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "secondary_pinky_extended_score",
        float(rule["secondary_pinky_extended_score_max"]),
        "meñique recogido en mano secundaria para X",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "secondary_thumb_extended_score",
        float(rule["secondary_thumb_extended_score_max"]),
        "pulgar no protagonista en mano secundaria para X",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "secondary_palm_normal_z_approx",
        float(rule["secondary_palm_normal_z_approx_max"]),
        "orientación de nudillos/dorso en mano secundaria para X",
    )

    # Relación entre manos: cruce real de índices.
    _add_missing_or_min(
        reasons,
        metrics,
        "x_index_segment_strict_intersection",
        float(rule["x_index_segment_strict_intersection_min"]),
        "cruce/intersección estricta de segmentos de índices para X",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "x_index_segment_min_distance_norm",
        float(rule["x_index_segment_min_distance_norm_max"]),
        "distancia mínima entre segmentos de índices para X",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "x_index_tip_distance_norm",
        float(rule["x_index_tip_distance_norm_min"]),
        float(rule["x_index_tip_distance_norm_max"]),
        "distancia entre puntas de índices para X",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "x_index_tip_x_distance_norm",
        float(rule["x_index_tip_x_distance_norm_min"]),
        "separación horizontal entre puntas de índices para X",
    )

    return reasons



def _validate_o_geometry(metrics: dict[str, float], rule: dict) -> list[str]:
    reasons: list[str] = []

    _add_missing_or_min(
        reasons,
        metrics,
        "valid_ratio",
        float(rule["min_valid_ratio"]),
        "mano primaria visible para O",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "finger_count_extended",
        float(rule["finger_count_min"]),
        float(rule["finger_count_max"]),
        "cantidad de dedos extendidos/visibles para O",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_extended_score",
        float(rule["thumb_extended_score_min"]),
        "pulgar activo para O",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "index_extended_score",
        float(rule["index_extended_score_max"]),
        "índice debe estar curvado en O",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_extended_score",
        float(rule["middle_extended_score_min"]),
        "dedo medio extendido para O",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "ring_extended_score",
        float(rule["ring_extended_score_min"]),
        "dedo anular extendido para O",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "pinky_extended_score",
        float(rule["pinky_extended_score_min"]),
        "meñique extendido para O",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_index_tip_distance_norm",
        float(rule["thumb_index_tip_distance_norm_min"]),
        float(rule["thumb_index_tip_distance_norm_max"]),
        "distancia pulgar-índice/cierre circular para O",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_middle_tip_distance_norm",
        float(rule["index_middle_tip_distance_norm_min"]),
        float(rule["index_middle_tip_distance_norm_max"]),
        "separación índice-medio para O",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_ring_tip_distance_norm",
        float(rule["middle_ring_tip_distance_norm_min"]),
        "separación medio-anular para O",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "ring_pinky_tip_distance_norm",
        float(rule["ring_pinky_tip_distance_norm_min"]),
        float(rule["ring_pinky_tip_distance_norm_max"]),
        "separación anular-meñique para O",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_mcp_tip_angle_deg",
        float(rule["index_mcp_tip_angle_deg_min"]),
        float(rule["index_mcp_tip_angle_deg_max"]),
        "dirección del índice curvado para O",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "index_mcp_tip_verticality",
        float(rule["index_mcp_tip_verticality_max"]),
        "verticalidad del índice debe ser baja en O",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "index_mcp_tip_above_base",
        float(rule["index_mcp_tip_above_base_max"]),
        "índice no debe apuntar hacia arriba en O",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "middle_mcp_tip_angle_deg",
        float(rule["middle_mcp_tip_angle_deg_min"]),
        float(rule["middle_mcp_tip_angle_deg_max"]),
        "dirección del dedo medio para O",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "middle_mcp_tip_horizontality",
        float(rule["middle_mcp_tip_horizontality_max"]),
        "dedo medio no debe quedar horizontal en O",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_mcp_tip_angle_deg",
        float(rule["thumb_mcp_tip_angle_deg_min"]),
        float(rule["thumb_mcp_tip_angle_deg_max"]),
        "dirección del pulgar para O",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_mcp_tip_horizontality",
        float(rule["thumb_mcp_tip_horizontality_min"]),
        "horizontalidad mínima del pulgar para O",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_mcp_tip_horizontality",
        float(rule["thumb_mcp_tip_horizontality_max"]),
        "pulgar demasiado horizontal/protagonista en O",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_mcp_tip_right_of_base",
        float(rule["thumb_mcp_tip_right_of_base_min"]),
        "pulgar debe apuntar hacia el lado esperado en O",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_middle_axis_angle_3d_deg",
        float(rule["index_middle_axis_angle_3d_deg_min"]),
        float(rule["index_middle_axis_angle_3d_deg_max"]),
        "relación eje índice-medio para O",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_index_axis_angle_3d_deg",
        float(rule["thumb_index_axis_angle_3d_deg_min"]),
        float(rule["thumb_index_axis_angle_3d_deg_max"]),
        "relación eje pulgar-índice para O",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_middle_axis_angle_3d_deg",
        float(rule["thumb_middle_axis_angle_3d_deg_min"]),
        float(rule["thumb_middle_axis_angle_3d_deg_max"]),
        "relación eje pulgar-medio para O",
    )

    if bool(rule.get("use_palm_normal_z", False)):
        _add_missing_or_max(
            reasons,
            metrics,
            "palm_normal_z_approx",
            float(rule["palm_normal_z_approx_max"]),
            "orientación palma/dorso para O",
        )

    return reasons


def _validate_q_geometry(metrics: dict[str, float], rule: dict) -> list[str]:
    reasons: list[str] = []

    # Compatibilidad: si el agregador no creó el alias explícito,
    # usar el promedio de hand_valid de la mano secundaria.
    if "secondary_valid_ratio" not in metrics and "secondary_hand_valid" in metrics:
        metrics = dict(metrics)
        metrics["secondary_valid_ratio"] = metrics.get("secondary_hand_valid", np.nan)

    _add_missing_or_min(
        reasons,
        metrics,
        "valid_ratio",
        float(rule["min_valid_ratio"]),
        "mano primaria visible para Q",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "secondary_valid_ratio",
        float(rule["secondary_valid_ratio_min"]),
        "mano secundaria visible para Q",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "two_hands_valid",
        float(rule["expected_hands_min"]),
        "dos manos detectadas para Q",
    )

    # Mano primaria: O-like.
    _add_missing_or_range(
        reasons,
        metrics,
        "finger_count_extended",
        float(rule["finger_count_min"]),
        float(rule["finger_count_max"]),
        "cantidad de dedos extendidos/visibles en mano primaria para Q",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_extended_score",
        float(rule["thumb_extended_score_min"]),
        "pulgar activo en mano primaria para Q",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "index_extended_score",
        float(rule["index_extended_score_max"]),
        "índice debe estar curvado en mano primaria para Q",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_extended_score",
        float(rule["middle_extended_score_min"]),
        "dedo medio extendido en mano primaria para Q",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "ring_extended_score",
        float(rule["ring_extended_score_min"]),
        "dedo anular extendido en mano primaria para Q",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "pinky_extended_score",
        float(rule["pinky_extended_score_min"]),
        "meñique extendido en mano primaria para Q",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_index_tip_distance_norm",
        float(rule["thumb_index_tip_distance_norm_min"]),
        float(rule["thumb_index_tip_distance_norm_max"]),
        "cierre pulgar-índice de la O primaria para Q",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_middle_tip_distance_norm",
        float(rule["index_middle_tip_distance_norm_min"]),
        float(rule["index_middle_tip_distance_norm_max"]),
        "separación índice-medio en mano primaria para Q",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_ring_tip_distance_norm",
        float(rule["middle_ring_tip_distance_norm_min"]),
        "separación medio-anular en mano primaria para Q",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "ring_pinky_tip_distance_norm",
        float(rule["ring_pinky_tip_distance_norm_min"]),
        float(rule["ring_pinky_tip_distance_norm_max"]),
        "separación anular-meñique en mano primaria para Q",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_mcp_tip_angle_deg",
        float(rule["index_mcp_tip_angle_deg_min"]),
        float(rule["index_mcp_tip_angle_deg_max"]),
        "dirección del índice curvado en mano primaria para Q",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "index_mcp_tip_verticality",
        float(rule["index_mcp_tip_verticality_max"]),
        "verticalidad del índice en mano primaria para Q",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "index_mcp_tip_above_base",
        float(rule["index_mcp_tip_above_base_max"]),
        "índice primario no debe apuntar hacia arriba en Q",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "middle_mcp_tip_angle_deg",
        float(rule["middle_mcp_tip_angle_deg_min"]),
        float(rule["middle_mcp_tip_angle_deg_max"]),
        "dirección del dedo medio en mano primaria para Q",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "middle_mcp_tip_horizontality",
        float(rule["middle_mcp_tip_horizontality_max"]),
        "dedo medio demasiado horizontal en mano primaria para Q",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_mcp_tip_angle_deg",
        float(rule["thumb_mcp_tip_angle_deg_min"]),
        float(rule["thumb_mcp_tip_angle_deg_max"]),
        "dirección del pulgar en mano primaria para Q",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_mcp_tip_horizontality",
        float(rule["thumb_mcp_tip_horizontality_min"]),
        "horizontalidad mínima del pulgar en mano primaria para Q",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_mcp_tip_horizontality",
        float(rule["thumb_mcp_tip_horizontality_max"]),
        "pulgar demasiado horizontal/protagonista en mano primaria para Q",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_mcp_tip_right_of_base",
        float(rule["thumb_mcp_tip_right_of_base_min"]),
        "pulgar al lado esperado en mano primaria para Q",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_middle_axis_angle_3d_deg",
        float(rule["index_middle_axis_angle_3d_deg_min"]),
        float(rule["index_middle_axis_angle_3d_deg_max"]),
        "relación eje índice-medio en mano primaria para Q",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_index_axis_angle_3d_deg",
        float(rule["thumb_index_axis_angle_3d_deg_min"]),
        float(rule["thumb_index_axis_angle_3d_deg_max"]),
        "relación eje pulgar-índice en mano primaria para Q",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_middle_axis_angle_3d_deg",
        float(rule["thumb_middle_axis_angle_3d_deg_min"]),
        float(rule["thumb_middle_axis_angle_3d_deg_max"]),
        "relación eje pulgar-medio en mano primaria para Q",
    )

    if bool(rule.get("use_palm_normal_z", False)):
        _add_missing_or_max(
            reasons,
            metrics,
            "palm_normal_z_approx",
            float(rule["palm_normal_z_approx_max"]),
            "orientación palma/dorso en mano primaria para Q",
        )

    # Mano secundaria: índice como cola.
    _add_missing_or_range(
        reasons,
        metrics,
        "secondary_finger_count_extended",
        float(rule["secondary_finger_count_min"]),
        float(rule["secondary_finger_count_max"]),
        "cantidad de dedos extendidos/visibles en mano secundaria para Q",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "secondary_index_extended_score",
        float(rule["secondary_index_extended_score_min"]),
        "índice extendido en mano secundaria para Q",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "secondary_middle_extended_score",
        float(rule["secondary_middle_extended_score_max"]),
        "dedo medio recogido en mano secundaria para Q",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "secondary_ring_extended_score",
        float(rule["secondary_ring_extended_score_max"]),
        "dedo anular recogido en mano secundaria para Q",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "secondary_pinky_extended_score",
        float(rule["secondary_pinky_extended_score_max"]),
        "meñique recogido en mano secundaria para Q",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "secondary_thumb_extended_score",
        float(rule["secondary_thumb_extended_score_max"]),
        "pulgar demasiado protagonista en mano secundaria para Q",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "secondary_index_mcp_tip_angle_deg",
        float(rule["secondary_index_mcp_tip_angle_deg_min"]),
        float(rule["secondary_index_mcp_tip_angle_deg_max"]),
        "dirección diagonal del índice secundario para Q",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "secondary_index_mcp_tip_verticality",
        float(rule["secondary_index_mcp_tip_verticality_min"]),
        float(rule["secondary_index_mcp_tip_verticality_max"]),
        "componente vertical del índice secundario para Q",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "secondary_index_mcp_tip_horizontality",
        float(rule["secondary_index_mcp_tip_horizontality_min"]),
        float(rule["secondary_index_mcp_tip_horizontality_max"]),
        "componente horizontal del índice secundario para Q",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "q_secondary_index_tip_to_o_join_distance_norm",
        float(rule["q_secondary_index_tip_to_o_join_distance_norm_max"]),
        "contacto de punta del índice secundario con unión pulgar-índice para Q",
    )

    return reasons



def _validate_l_geometry(metrics: dict[str, float], rule: dict) -> list[str]:
    reasons: list[str] = []

    _add_missing_or_min(
        reasons,
        metrics,
        "valid_ratio",
        float(rule["min_valid_ratio"]),
        "mano primaria visible para L",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "finger_count_extended",
        float(rule["finger_count_min"]),
        float(rule["finger_count_max"]),
        "cantidad de dedos extendidos para L",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_extended_score",
        float(rule["thumb_extended_score_min"]),
        "pulgar extendido para L",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_extended_score",
        float(rule["index_extended_score_min"]),
        "índice extendido para L",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "middle_extended_score",
        float(rule["middle_extended_score_max"]),
        "dedo medio debe estar cerrado en L",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "ring_extended_score",
        float(rule["ring_extended_score_max"]),
        "dedo anular debe estar cerrado en L",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "pinky_extended_score",
        float(rule["pinky_extended_score_max"]),
        "dedo meñique debe estar cerrado en L",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_index_tip_distance_norm",
        float(rule["thumb_index_tip_distance_norm_min"]),
        "apertura pulgar-índice para L",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_index_angle_deg",
        float(rule["thumb_index_angle_deg_min"]),
        float(rule["thumb_index_angle_deg_max"]),
        "ángulo pulgar-índice para L",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_mcp_tip_angle_deg",
        float(rule["index_mcp_tip_angle_deg_min"]),
        float(rule["index_mcp_tip_angle_deg_max"]),
        "dirección del índice para L",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_verticality",
        float(rule["index_mcp_tip_verticality_min"]),
        "verticalidad del índice para L",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_above_base",
        float(rule["index_mcp_tip_above_base_min"]),
        "índice debe apuntar hacia arriba en L",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_mcp_tip_angle_deg",
        float(rule["thumb_mcp_tip_angle_deg_min"]),
        float(rule["thumb_mcp_tip_angle_deg_max"]),
        "dirección del pulgar para L",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_mcp_tip_horizontality",
        float(rule["thumb_mcp_tip_horizontality_min"]),
        "horizontalidad del pulgar para L",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_mcp_tip_right_of_base",
        float(rule["thumb_mcp_tip_right_of_base_min"]),
        "pulgar debe apuntar al lado correcto para L",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_mcp_tip_left_of_base",
        float(rule["thumb_mcp_tip_left_of_base_max"]),
        "pulgar apunta al lado contrario en L",
    )

    if bool(rule.get("use_palm_normal_z", False)):
        _add_missing_or_max(
            reasons,
            metrics,
            "palm_normal_z_approx",
            float(rule["palm_normal_z_approx_max"]),
            "orientación palma/dorso para L",
        )

    return reasons



def _validate_k_geometry(metrics: dict[str, float], rule: dict) -> list[str]:
    reasons: list[str] = []

    _add_missing_or_min(
        reasons,
        metrics,
        "valid_ratio",
        float(rule["min_valid_ratio"]),
        "mano primaria visible para K",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "finger_count_extended",
        float(rule["finger_count_min"]),
        float(rule["finger_count_max"]),
        "cantidad de dedos extendidos para K",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_extended_score",
        float(rule["thumb_extended_score_min"]),
        "pulgar visible/extendido para K",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_extended_score",
        float(rule["index_extended_score_min"]),
        "índice extendido para K",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_extended_score",
        float(rule["middle_extended_score_min"]),
        "dedo medio activo para K",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "ring_extended_score",
        float(rule["ring_extended_score_max"]),
        "dedo anular debe estar cerrado en K",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "pinky_extended_score",
        float(rule["pinky_extended_score_max"]),
        "dedo meñique debe estar cerrado en K",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_index_tip_distance_norm",
        float(rule["thumb_index_tip_distance_norm_min"]),
        float(rule["thumb_index_tip_distance_norm_max"]),
        "distancia pulgar-índice para K",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_middle_tip_distance_norm",
        float(rule["index_middle_tip_distance_norm_min"]),
        float(rule["index_middle_tip_distance_norm_max"]),
        "apertura índice-medio para K",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_ring_tip_distance_norm",
        float(rule["middle_ring_tip_distance_norm_min"]),
        "separación medio-anular para K",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "ring_pinky_tip_distance_norm",
        float(rule["ring_pinky_tip_distance_norm_max"]),
        "compactación anular-meñique para K",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_mcp_tip_angle_deg",
        float(rule["index_mcp_tip_angle_deg_min"]),
        float(rule["index_mcp_tip_angle_deg_max"]),
        "dirección del índice para K",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_verticality",
        float(rule["index_mcp_tip_verticality_min"]),
        "verticalidad/diagonal ascendente del índice para K",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_above_base",
        float(rule["index_mcp_tip_above_base_min"]),
        "índice debe apuntar hacia arriba en K",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "middle_mcp_tip_angle_deg",
        float(rule["middle_mcp_tip_angle_deg_min"]),
        float(rule["middle_mcp_tip_angle_deg_max"]),
        "dirección del dedo medio para K",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_mcp_tip_horizontality",
        float(rule["middle_mcp_tip_horizontality_min"]),
        "horizontalidad del dedo medio para K",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_mcp_tip_right_of_base",
        float(rule["middle_mcp_tip_right_of_base_min"]),
        "dedo medio debe apuntar al lado correcto en K",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_mcp_tip_angle_deg",
        float(rule["thumb_mcp_tip_angle_deg_min"]),
        float(rule["thumb_mcp_tip_angle_deg_max"]),
        "dirección del pulgar para K",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_mcp_tip_right_of_base",
        float(rule["thumb_mcp_tip_right_of_base_min"]),
        "pulgar debe ubicarse al lado correcto en K",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_middle_axis_angle_3d_deg",
        float(rule["index_middle_axis_angle_3d_deg_min"]),
        float(rule["index_middle_axis_angle_3d_deg_max"]),
        "ángulo 3D índice-medio para K",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_index_axis_angle_3d_deg",
        float(rule["thumb_index_axis_angle_3d_deg_max"]),
        "ángulo 3D pulgar-índice para K",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_middle_axis_angle_3d_deg",
        float(rule["thumb_middle_axis_angle_3d_deg_min"]),
        float(rule["thumb_middle_axis_angle_3d_deg_max"]),
        "ángulo 3D pulgar-medio para K",
    )

    return reasons



def _validate_r_geometry(metrics: dict[str, float], rule: dict) -> list[str]:
    reasons: list[str] = []

    _add_missing_or_min(
        reasons,
        metrics,
        "valid_ratio",
        float(rule["min_valid_ratio"]),
        "mano primaria visible para R",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "finger_count_extended",
        float(rule["finger_count_min"]),
        float(rule["finger_count_max"]),
        "cantidad de dedos extendidos para R",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_extended_score",
        float(rule["index_extended_score_min"]),
        "índice extendido para R",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_extended_score",
        float(rule["middle_extended_score_min"]),
        "dedo medio extendido para R",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "ring_extended_score",
        float(rule["ring_extended_score_max"]),
        "dedo anular debe estar cerrado en R",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "pinky_extended_score",
        float(rule["pinky_extended_score_max"]),
        "dedo meñique debe estar cerrado en R",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "index_middle_tip_distance_norm",
        float(rule["index_middle_tip_distance_norm_max"]),
        "índice y medio deben estar pegados/cruzados en R",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_ring_tip_distance_norm",
        float(rule["middle_ring_tip_distance_norm_min"]),
        "separación medio-anular para R",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "ring_pinky_tip_distance_norm",
        float(rule["ring_pinky_tip_distance_norm_max"]),
        "compactación anular-meñique para R",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_mcp_tip_angle_deg",
        float(rule["index_mcp_tip_angle_deg_min"]),
        float(rule["index_mcp_tip_angle_deg_max"]),
        "dirección del índice para R",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_verticality",
        float(rule["index_mcp_tip_verticality_min"]),
        "verticalidad del índice para R",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_above_base",
        float(rule["index_mcp_tip_above_base_min"]),
        "índice debe apuntar hacia arriba en R",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "middle_mcp_tip_angle_deg",
        float(rule["middle_mcp_tip_angle_deg_min"]),
        float(rule["middle_mcp_tip_angle_deg_max"]),
        "dirección del dedo medio para R",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_mcp_tip_above_base",
        float(rule["middle_mcp_tip_above_base_min"]),
        "dedo medio debe apuntar hacia arriba en R",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "index_middle_axis_angle_3d_deg",
        float(rule["index_middle_axis_angle_3d_deg_max"]),
        "paralelismo 3D índice-medio para R",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "middle_index_axis_signed_range",
        float(rule["middle_index_axis_signed_range_min"]),
        "envolvimiento/cruce del dedo medio sobre el índice para R",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_mcp_tip_left_of_base",
        float(rule["thumb_mcp_tip_left_of_base_min"]),
        "pulgar debe ubicarse del lado correcto en R",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_mcp_tip_right_of_base",
        float(rule["thumb_mcp_tip_right_of_base_max"]),
        "pulgar apunta al lado contrario en R",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_mcp_tip_angle_deg",
        float(rule["thumb_mcp_tip_angle_deg_min"]),
        float(rule["thumb_mcp_tip_angle_deg_max"]),
        "dirección del pulgar para R",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_index_axis_angle_3d_deg",
        float(rule["thumb_index_axis_angle_3d_deg_min"]),
        float(rule["thumb_index_axis_angle_3d_deg_max"]),
        "ángulo 3D pulgar-índice para R",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_middle_axis_angle_3d_deg",
        float(rule["thumb_middle_axis_angle_3d_deg_min"]),
        float(rule["thumb_middle_axis_angle_3d_deg_max"]),
        "ángulo 3D pulgar-medio para R",
    )

    return reasons



def _validate_s_geometry(metrics: dict[str, float], rule: dict) -> list[str]:
    reasons: list[str] = []

    _add_missing_or_min(
        reasons,
        metrics,
        "valid_ratio",
        float(rule["min_valid_ratio"]),
        "mano primaria visible para S",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "finger_count_extended",
        float(rule["finger_count_min"]),
        float(rule["finger_count_max"]),
        "cantidad de dedos extendidos/visibles para S",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_extended_score",
        float(rule["thumb_extended_score_min"]),
        "pulgar extendido/activo para S",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_extended_score",
        float(rule["index_extended_score_min"]),
        "índice extendido para S",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "middle_extended_score",
        float(rule["middle_extended_score_max"]),
        "dedo medio recogido para S",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "ring_extended_score",
        float(rule["ring_extended_score_max"]),
        "dedo anular recogido para S",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "pinky_extended_score",
        float(rule["pinky_extended_score_max"]),
        "meñique recogido para S",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_index_tip_distance_norm",
        float(rule["thumb_index_tip_distance_norm_min"]),
        float(rule["thumb_index_tip_distance_norm_max"]),
        "apertura pulgar-índice tipo pistola para S",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_middle_tip_distance_norm",
        float(rule["index_middle_tip_distance_norm_min"]),
        "separación índice-medio para S",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_mcp_tip_angle_deg",
        float(rule["index_mcp_tip_angle_deg_min"]),
        float(rule["index_mcp_tip_angle_deg_max"]),
        "dirección del índice hacia barbilla para S",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "index_mcp_tip_verticality",
        float(rule["index_mcp_tip_verticality_max"]),
        "verticalidad del índice para S",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_horizontality",
        float(rule["index_mcp_tip_horizontality_min"]),
        "horizontalidad del índice para S",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "thumb_mcp_tip_angle_deg",
        float(rule["thumb_mcp_tip_angle_deg_min"]),
        float(rule["thumb_mcp_tip_angle_deg_max"]),
        "dirección del pulgar para S",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_mcp_tip_horizontality",
        float(rule["thumb_mcp_tip_horizontality_max"]),
        "pulgar demasiado horizontal/recogido para S",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "thumb_mcp_tip_right_of_base",
        float(rule["thumb_mcp_tip_right_of_base_min"]),
        "pulgar orientado al lado esperado para S",
    )

    if bool(rule.get("use_palm_normal_z", False)):
        _add_missing_or_min(
            reasons,
            metrics,
            "palm_normal_z_approx",
            float(rule["palm_normal_z_approx_min"]),
            "orientación de dorso hacia cámara para S",
        )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_middle_axis_angle_3d_deg",
        float(rule["index_middle_axis_angle_3d_deg_min"]),
        "relación eje índice-medio para S",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "pose_valid_for_s_chin_geometry_ratio",
        float(rule["pose_valid_for_s_chin_geometry_ratio_min"]),
        "pose facial válida para S",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_tip_chin_dx_earspan",
        float(rule["index_tip_chin_dx_earspan_min"]),
        float(rule["index_tip_chin_dx_earspan_max"]),
        "posición horizontal del índice respecto de barbilla para S",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_tip_chin_dy_earspan",
        float(rule["index_tip_chin_dy_earspan_min"]),
        float(rule["index_tip_chin_dy_earspan_max"]),
        "posición vertical del índice respecto de barbilla para S",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "index_tip_chin_dist_earspan",
        float(rule["index_tip_chin_dist_earspan_max"]),
        "distancia índice-barbilla para S",
    )

    return reasons



def _validate_t_geometry(metrics: dict[str, float], rule: dict) -> list[str]:
    reasons: list[str] = []

    _add_missing_or_min(
        reasons,
        metrics,
        "valid_ratio",
        float(rule["min_valid_ratio"]),
        "mano primaria visible para T",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "finger_count_extended",
        float(rule["finger_count_min"]),
        float(rule["finger_count_max"]),
        "cantidad de dedos extendidos/visibles para T",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "thumb_extended_score",
        float(rule["thumb_extended_score_max"]),
        "pulgar demasiado protagonista para T",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_extended_score",
        float(rule["index_extended_score_min"]),
        "índice extendido para T",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "middle_extended_score",
        float(rule["middle_extended_score_max"]),
        "dedo medio recogido para T",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "ring_extended_score",
        float(rule["ring_extended_score_max"]),
        "dedo anular recogido para T",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "pinky_extended_score",
        float(rule["pinky_extended_score_max"]),
        "meñique recogido para T",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_middle_tip_distance_norm",
        float(rule["index_middle_tip_distance_norm_min"]),
        "separación índice-medio para T",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_mcp_tip_angle_deg",
        float(rule["index_mcp_tip_angle_deg_min"]),
        float(rule["index_mcp_tip_angle_deg_max"]),
        "dirección vertical del índice para T",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_verticality",
        float(rule["index_mcp_tip_verticality_min"]),
        "verticalidad del índice para T",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "index_mcp_tip_above_base",
        float(rule["index_mcp_tip_above_base_min"]),
        "punta del índice por encima de su base para T",
    )

    if bool(rule.get("use_palm_normal_z", False)):
        _add_missing_or_range(
            reasons,
            metrics,
            "palm_normal_z_approx",
            float(rule["palm_normal_z_approx_min"]),
            float(rule["palm_normal_z_approx_max"]),
            "orientación de dorso hacia cámara para T",
        )

    _add_missing_or_max(
        reasons,
        metrics,
        "middle_index_axis_signed_range",
        float(rule["middle_index_axis_signed_range_max"]),
        "mano demasiado lateral/frontal para T",
    )

    _add_missing_or_min(
        reasons,
        metrics,
        "pose_valid_for_t_chin_geometry_ratio",
        float(rule["pose_valid_for_t_chin_geometry_ratio_min"]),
        "pose facial válida para T",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_tip_chin_dx_earspan",
        float(rule["index_tip_chin_dx_earspan_min"]),
        float(rule["index_tip_chin_dx_earspan_max"]),
        "posición horizontal del índice respecto de barbilla para T",
    )

    _add_missing_or_range(
        reasons,
        metrics,
        "index_tip_chin_dy_earspan",
        float(rule["index_tip_chin_dy_earspan_min"]),
        float(rule["index_tip_chin_dy_earspan_max"]),
        "posición vertical del índice respecto de barbilla para T",
    )

    _add_missing_or_max(
        reasons,
        metrics,
        "index_tip_chin_dist_earspan",
        float(rule["index_tip_chin_dist_earspan_max"]),
        "distancia índice-barbilla para T",
    )

    return reasons



def _build_result(
    *,
    label: str,
    required: bool,
    ok: bool,
    metrics: dict[str, float] | None = None,
    reasons: list[str] | None = None,
    expected_hands: int = 1,
    message: str | None = None,
) -> StaticGeometryValidationResult:
    normalized_label = str(label).upper()
    metrics = metrics or {}
    reasons = reasons or []

    frames_used = int(metrics.get("frames_used", 0) or 0)
    valid_frames = int(metrics.get("valid_frames", 0) or 0)
    valid_ratio = float(metrics.get("valid_ratio", 0.0) or 0.0)

    if message is None:
        if not required:
            message = "Validación geométrica estática no requerida."
        elif ok:
            if normalized_label == "A":
                message = (
                    f"Validación geométrica estática OK para {normalized_label}: "
                    f"valid_ratio={valid_ratio:.3f}, "
                    f"finger_count={metrics.get('finger_count_extended', np.nan):.2f}, "
                    f"thumb={metrics.get('thumb_extended_score', np.nan):.3f}, "
                    f"index={metrics.get('index_extended_score', np.nan):.3f}, "
                    f"middle={metrics.get('middle_extended_score', np.nan):.3f}, "
                    f"ring={metrics.get('ring_extended_score', np.nan):.3f}, "
                    f"pinky={metrics.get('pinky_extended_score', np.nan):.3f}, "
                    f"thumb_idx_dist={metrics.get('thumb_index_tip_distance_norm', np.nan):.3f}, "
                    f"thumb_angle={metrics.get('thumb_mcp_tip_angle_deg', np.nan):.1f}"
                )
            elif normalized_label == "B":
                message = (
                    f"Validación geométrica estática OK para {normalized_label}: "
                    f"valid_ratio={valid_ratio:.3f}, "
                    f"finger_count={metrics.get('finger_count_extended', np.nan):.2f}, "
                    f"thumb={metrics.get('thumb_extended_score', np.nan):.3f}, "
                    f"index={metrics.get('index_extended_score', np.nan):.3f}, "
                    f"middle={metrics.get('middle_extended_score', np.nan):.3f}, "
                    f"ring={metrics.get('ring_extended_score', np.nan):.3f}, "
                    f"pinky={metrics.get('pinky_extended_score', np.nan):.3f}, "
                    f"idx_mid_dist={metrics.get('index_middle_tip_distance_norm', np.nan):.3f}, "
                    f"mid_ring_dist={metrics.get('middle_ring_tip_distance_norm', np.nan):.3f}, "
                    f"ring_pinky_dist={metrics.get('ring_pinky_tip_distance_norm', np.nan):.3f}, "
                    f"thumb_angle={metrics.get('thumb_mcp_tip_angle_deg', np.nan):.1f}, "
                    f"thumb_idx_axis_3d={metrics.get('thumb_index_axis_angle_3d_deg', np.nan):.1f}"
                )
            elif normalized_label == "C":
                message = (
                    f"Validación geométrica estática OK para {normalized_label}: "
                    f"valid_ratio={valid_ratio:.3f}, "
                    f"finger_count={metrics.get('finger_count_extended', np.nan):.2f}, "
                    f"thumb={metrics.get('thumb_extended_score', np.nan):.3f}, "
                    f"index={metrics.get('index_extended_score', np.nan):.3f}, "
                    f"middle={metrics.get('middle_extended_score', np.nan):.3f}, "
                    f"ring={metrics.get('ring_extended_score', np.nan):.3f}, "
                    f"pinky={metrics.get('pinky_extended_score', np.nan):.3f}, "
                    f"thumb_idx_dist={metrics.get('thumb_index_tip_distance_norm', np.nan):.3f}, "
                    f"idx_mid_dist={metrics.get('index_middle_tip_distance_norm', np.nan):.3f}, "
                    f"index_angle={metrics.get('index_mcp_tip_angle_deg', np.nan):.1f}, "
                    f"thumb_angle={metrics.get('thumb_mcp_tip_angle_deg', np.nan):.1f}, "
                    f"thumb_idx_axis_3d={metrics.get('thumb_index_axis_angle_3d_deg', np.nan):.1f}"
                )
            elif normalized_label == "D":
                message = (
                    f"Validación geométrica estática OK para {normalized_label}: "
                    f"valid_ratio={valid_ratio:.3f}, "
                    f"finger_count={metrics.get('finger_count_extended', np.nan):.2f}, "
                    f"thumb={metrics.get('thumb_extended_score', np.nan):.3f}, "
                    f"index={metrics.get('index_extended_score', np.nan):.3f}, "
                    f"middle={metrics.get('middle_extended_score', np.nan):.3f}, "
                    f"ring={metrics.get('ring_extended_score', np.nan):.3f}, "
                    f"pinky={metrics.get('pinky_extended_score', np.nan):.3f}, "
                    f"thumb_mid_dist={metrics.get('thumb_middle_tip_distance_norm', np.nan):.3f}, "
                    f"middle_angle={metrics.get('middle_mcp_tip_angle_deg', np.nan):.1f}, "
                    f"palm_z={metrics.get('palm_normal_z_approx', np.nan):.4f}"
                )
            elif normalized_label == "E":
                message = (
                    f"Validación geométrica estática OK para {normalized_label}: "
                    f"valid_ratio={valid_ratio:.3f}, "
                    f"finger_count={metrics.get('finger_count_extended', np.nan):.2f}, "
                    f"thumb={metrics.get('thumb_extended_score', np.nan):.3f}, "
                    f"index={metrics.get('index_extended_score', np.nan):.3f}, "
                    f"thumb_idx_dist={metrics.get('thumb_index_tip_distance_norm', np.nan):.3f}, "
                    f"same_eye_in={metrics.get('same_eye_inside_hand_bbox_ratio', np.nan):.3f}, "
                    f"other_eye_in={metrics.get('other_eye_inside_hand_bbox_ratio', np.nan):.3f}, "
                    f"nose_in={metrics.get('nose_inside_hand_bbox_ratio', np.nan):.3f}, "
                    f"eye_seg_dist={metrics.get('same_eye_thumb_index_distance_norm', np.nan):.3f}"
                )
            elif normalized_label == "F":
                message = (
                    f"Validación geométrica estática OK para {normalized_label}: "
                    f"valid_ratio={valid_ratio:.3f}, "
                    f"finger_count={metrics.get('finger_count_extended', np.nan):.2f}, "
                    f"thumb={metrics.get('thumb_extended_score', np.nan):.3f}, "
                    f"index={metrics.get('index_extended_score', np.nan):.3f}, "
                    f"middle={metrics.get('middle_extended_score', np.nan):.3f}, "
                    f"ring={metrics.get('ring_extended_score', np.nan):.3f}, "
                    f"pinky={metrics.get('pinky_extended_score', np.nan):.3f}, "
                    f"idx_mid_dist={metrics.get('index_middle_tip_distance_norm', np.nan):.3f}, "
                    f"idx_mid_axis_3d={metrics.get('index_middle_axis_angle_3d_deg', np.nan):.1f}, "
                    f"palm_z={metrics.get('palm_normal_z_approx', np.nan):.4f}, "
                    f"shoulder_dx={metrics.get('hand_center_shoulder_dx', np.nan):.3f}, "
                    f"shoulder_dy={metrics.get('hand_center_shoulder_dy', np.nan):.3f}"
                )
            elif normalized_label == "G":
                message = (
                    f"Validación geométrica estática OK para {normalized_label}: "
                    f"valid_ratio={valid_ratio:.3f}, "
                    f"finger_count={metrics.get('finger_count_extended', np.nan):.2f}, "
                    f"thumb={metrics.get('thumb_extended_score', np.nan):.3f}, "
                    f"index={metrics.get('index_extended_score', np.nan):.3f}, "
                    f"middle={metrics.get('middle_extended_score', np.nan):.3f}, "
                    f"ring={metrics.get('ring_extended_score', np.nan):.3f}, "
                    f"pinky={metrics.get('pinky_extended_score', np.nan):.3f}, "
                    f"thumb_idx_dist={metrics.get('thumb_index_tip_distance_norm', np.nan):.3f}, "
                    f"right_ear_dx={metrics.get('right_ear_dx_earspan', np.nan):.3f}, "
                    f"right_ear_dy={metrics.get('right_ear_dy_earspan', np.nan):.3f}, "
                    f"right_ear_dist={metrics.get('right_ear_center_dist_earspan', np.nan):.3f}"
                )
            elif normalized_label == "H":
                message = (
                    f"Validación geométrica estática OK para {normalized_label}: "
                    f"valid_ratio={valid_ratio:.3f}, "
                    f"finger_count={metrics.get('finger_count_extended', np.nan):.2f}, "
                    f"thumb={metrics.get('thumb_extended_score', np.nan):.3f}, "
                    f"index={metrics.get('index_extended_score', np.nan):.3f}, "
                    f"middle={metrics.get('middle_extended_score', np.nan):.3f}, "
                    f"ring={metrics.get('ring_extended_score', np.nan):.3f}, "
                    f"pinky={metrics.get('pinky_extended_score', np.nan):.3f}, "
                    f"idx_mid_dist={metrics.get('index_middle_tip_distance_norm', np.nan):.3f}, "
                    f"mid_ring_dist={metrics.get('middle_ring_tip_distance_norm', np.nan):.3f}, "
                    f"idx_mid_axis_3d={metrics.get('index_middle_axis_angle_3d_deg', np.nan):.1f}, "
                    f"nose_in={metrics.get('nose_inside_hand_bbox_ratio', np.nan):.3f}, "
                    f"nose_dx={metrics.get('nose_dx_earspan', np.nan):.3f}, "
                    f"nose_dy={metrics.get('nose_dy_earspan', np.nan):.3f}"
                )
            elif normalized_label == "I":
                message = (
                    f"Validación geométrica estática OK para {normalized_label}: "
                    f"valid_ratio={valid_ratio:.3f}, "
                    f"finger_count={metrics.get('finger_count_extended', np.nan):.2f}, "
                    f"index={metrics.get('index_extended_score', np.nan):.3f}, "
                    f"middle={metrics.get('middle_extended_score', np.nan):.3f}, "
                    f"ring={metrics.get('ring_extended_score', np.nan):.3f}, "
                    f"pinky={metrics.get('pinky_extended_score', np.nan):.3f}, "
                    f"index_angle={metrics.get('index_mcp_tip_angle_deg', np.nan):.1f}, "
                    f"index_vert={metrics.get('index_mcp_tip_verticality', np.nan):.3f}, "
                    f"idx_eye_dx={metrics.get('index_tip_right_eye_dx_earspan', np.nan):.3f}, "
                    f"idx_eye_dy={metrics.get('index_tip_right_eye_dy_earspan', np.nan):.3f}, "
                    f"idx_nose_dx={metrics.get('index_tip_nose_dx_earspan', np.nan):.3f}"
                )
            elif normalized_label == "J":
                message = (
                    f"Validación geométrica estática OK para {normalized_label}: "
                    f"valid_ratio={valid_ratio:.3f}, "
                    f"finger_count={metrics.get('finger_count_extended', np.nan):.2f}, "
                    f"index={metrics.get('index_extended_score', np.nan):.3f}, "
                    f"middle={metrics.get('middle_extended_score', np.nan):.3f}, "
                    f"ring={metrics.get('ring_extended_score', np.nan):.3f}, "
                    f"pinky={metrics.get('pinky_extended_score', np.nan):.3f}, "
                    f"idx_mid_dist={metrics.get('index_middle_tip_distance_norm', np.nan):.3f}, "
                    f"mid_ring_dist={metrics.get('middle_ring_tip_distance_norm', np.nan):.3f}, "
                    f"idx_mid_axis_3d={metrics.get('index_middle_axis_angle_3d_deg', np.nan):.1f}, "
                    f"mouth_dx={metrics.get('mouth_dx_earspan', np.nan):.3f}, "
                    f"mouth_dy={metrics.get('mouth_dy_earspan', np.nan):.3f}"
                )
            elif normalized_label == "Y":
                message = (
                    f"Validación geométrica estática OK para {normalized_label}: "
                    f"valid_ratio={valid_ratio:.3f}, "
                    f"finger_count={metrics.get('finger_count_extended', np.nan):.2f}, "
                    f"thumb={metrics.get('thumb_extended_score', np.nan):.3f}, "
                    f"pinky={metrics.get('pinky_extended_score', np.nan):.3f}, "
                    f"index={metrics.get('index_extended_score', np.nan):.3f}, "
                    f"middle={metrics.get('middle_extended_score', np.nan):.3f}, "
                    f"ring={metrics.get('ring_extended_score', np.nan):.3f}, "
                    f"thumb_pinky_dist={metrics.get('thumb_pinky_tip_distance_norm', np.nan):.3f}, "
                    f"pinky_mouth_dx={metrics.get('pinky_tip_mouth_dx_earspan', np.nan):.3f}, "
                    f"pinky_mouth_dy={metrics.get('pinky_tip_mouth_dy_earspan', np.nan):.3f}"
                )
            elif normalized_label == "M":
                message = (
                    f"Validación geométrica estática OK para {normalized_label}: "
                    f"valid_ratio={valid_ratio:.3f}, "
                    f"finger_count={metrics.get('finger_count_extended', np.nan):.2f}, "
                    f"thumb={metrics.get('thumb_extended_score', np.nan):.3f}, "
                    f"index={metrics.get('index_extended_score', np.nan):.3f}, "
                    f"middle={metrics.get('middle_extended_score', np.nan):.3f}, "
                    f"ring={metrics.get('ring_extended_score', np.nan):.3f}, "
                    f"pinky={metrics.get('pinky_extended_score', np.nan):.3f}, "
                    f"index_angle={metrics.get('index_mcp_tip_angle_deg', np.nan):.1f}, "
                    f"middle_angle={metrics.get('middle_mcp_tip_angle_deg', np.nan):.1f}, "
                    f"idx_mid_dist={metrics.get('index_middle_tip_distance_norm', np.nan):.3f}, "
                    f"mid_ring_dist={metrics.get('middle_ring_tip_distance_norm', np.nan):.3f}, "
                    f"ring_pinky_dist={metrics.get('ring_pinky_tip_distance_norm', np.nan):.3f}"
                )
            elif normalized_label == "N":
                message = (
                    f"Validación geométrica estática OK para {normalized_label}: "
                    f"valid_ratio={valid_ratio:.3f}, "
                    f"finger_count={metrics.get('finger_count_extended', np.nan):.2f}, "
                    f"thumb={metrics.get('thumb_extended_score', np.nan):.3f}, "
                    f"index={metrics.get('index_extended_score', np.nan):.3f}, "
                    f"middle={metrics.get('middle_extended_score', np.nan):.3f}, "
                    f"ring={metrics.get('ring_extended_score', np.nan):.3f}, "
                    f"pinky={metrics.get('pinky_extended_score', np.nan):.3f}, "
                    f"index_angle={metrics.get('index_mcp_tip_angle_deg', np.nan):.1f}, "
                    f"middle_angle={metrics.get('middle_mcp_tip_angle_deg', np.nan):.1f}, "
                    f"idx_mid_dist={metrics.get('index_middle_tip_distance_norm', np.nan):.3f}, "
                    f"mid_ring_dist={metrics.get('middle_ring_tip_distance_norm', np.nan):.3f}, "
                    f"ring_pinky_dist={metrics.get('ring_pinky_tip_distance_norm', np.nan):.3f}, "
                    f"thumb_angle={metrics.get('thumb_mcp_tip_angle_deg', np.nan):.1f}, "
                    f"thumb_horiz={metrics.get('thumb_mcp_tip_horizontality', np.nan):.3f}"
                )
            elif normalized_label == "Ñ":
                message = (
                    f"Validación geométrica estática OK para {normalized_label}: "
                    f"valid_ratio={valid_ratio:.3f}, "
                    f"secondary_valid={metrics.get('secondary_valid_ratio', np.nan):.3f}, "
                    f"finger_count={metrics.get('finger_count_extended', np.nan):.2f}, "
                    f"index={metrics.get('index_extended_score', np.nan):.3f}, "
                    f"middle={metrics.get('middle_extended_score', np.nan):.3f}, "
                    f"ring={metrics.get('ring_extended_score', np.nan):.3f}, "
                    f"pinky={metrics.get('pinky_extended_score', np.nan):.3f}, "
                    f"thumb_angle={metrics.get('thumb_mcp_tip_angle_deg', np.nan):.1f}, "
                    f"thumb_horiz={metrics.get('thumb_mcp_tip_horizontality', np.nan):.3f}, "
                    f"sec_count={metrics.get('secondary_finger_count_extended', np.nan):.2f}, "
                    f"sec_index={metrics.get('secondary_index_extended_score', np.nan):.3f}, "
                    f"sec_middle={metrics.get('secondary_middle_extended_score', np.nan):.3f}, "
                    f"sec_angle={metrics.get('secondary_index_mcp_tip_angle_deg', np.nan):.1f}, "
                    f"tilde_dist={metrics.get('enie_tilde_index_to_knuckle_line_distance_norm', np.nan):.3f}, "
                    f"tilde_cover={metrics.get('enie_tilde_knuckle_line_coverage', np.nan):.3f}"
                )
            elif normalized_label == "O":
                message = (
                    f"Validación geométrica estática OK para {normalized_label}: "
                    f"valid_ratio={valid_ratio:.3f}, "
                    f"finger_count={metrics.get('finger_count_extended', np.nan):.2f}, "
                    f"thumb={metrics.get('thumb_extended_score', np.nan):.3f}, "
                    f"index={metrics.get('index_extended_score', np.nan):.3f}, "
                    f"middle={metrics.get('middle_extended_score', np.nan):.3f}, "
                    f"ring={metrics.get('ring_extended_score', np.nan):.3f}, "
                    f"pinky={metrics.get('pinky_extended_score', np.nan):.3f}, "
                    f"thumb_idx_dist={metrics.get('thumb_index_tip_distance_norm', np.nan):.3f}, "
                    f"idx_mid_dist={metrics.get('index_middle_tip_distance_norm', np.nan):.3f}, "
                    f"mid_ring_dist={metrics.get('middle_ring_tip_distance_norm', np.nan):.3f}, "
                    f"ring_pinky_dist={metrics.get('ring_pinky_tip_distance_norm', np.nan):.3f}, "
                    f"index_angle={metrics.get('index_mcp_tip_angle_deg', np.nan):.1f}, "
                    f"thumb_angle={metrics.get('thumb_mcp_tip_angle_deg', np.nan):.1f}, "
                    f"idx_mid_axis_3d={metrics.get('index_middle_axis_angle_3d_deg', np.nan):.1f}"
                )
            elif normalized_label == "Q":
                message = (
                    f"Validación geométrica estática OK para {normalized_label}: "
                    f"valid_ratio={valid_ratio:.3f}, "
                    f"secondary_valid={metrics.get('secondary_valid_ratio', np.nan):.3f}, "
                    f"finger_count={metrics.get('finger_count_extended', np.nan):.2f}, "
                    f"thumb={metrics.get('thumb_extended_score', np.nan):.3f}, "
                    f"index={metrics.get('index_extended_score', np.nan):.3f}, "
                    f"middle={metrics.get('middle_extended_score', np.nan):.3f}, "
                    f"thumb_idx_dist={metrics.get('thumb_index_tip_distance_norm', np.nan):.3f}, "
                    f"sec_count={metrics.get('secondary_finger_count_extended', np.nan):.2f}, "
                    f"sec_index={metrics.get('secondary_index_extended_score', np.nan):.3f}, "
                    f"sec_middle={metrics.get('secondary_middle_extended_score', np.nan):.3f}, "
                    f"sec_angle={metrics.get('secondary_index_mcp_tip_angle_deg', np.nan):.1f}, "
                    f"q_join_dist={metrics.get('q_secondary_index_tip_to_o_join_distance_norm', np.nan):.3f}, "
                    f"thumb_angle={metrics.get('thumb_mcp_tip_angle_deg', np.nan):.1f}, "
                    f"thumb_idx_axis_3d={metrics.get('thumb_index_axis_angle_3d_deg', np.nan):.1f}"
                )
            elif normalized_label == "V":
                message = (
                    f"Validación geométrica estática OK para {normalized_label}: "
                    f"valid_ratio={valid_ratio:.3f}, "
                    f"finger_count={metrics.get('finger_count_extended', np.nan):.2f}, "
                    f"thumb={metrics.get('thumb_extended_score', np.nan):.3f}, "
                    f"index={metrics.get('index_extended_score', np.nan):.3f}, "
                    f"middle={metrics.get('middle_extended_score', np.nan):.3f}, "
                    f"ring={metrics.get('ring_extended_score', np.nan):.3f}, "
                    f"pinky={metrics.get('pinky_extended_score', np.nan):.3f}, "
                    f"idx_mid_dist={metrics.get('index_middle_tip_distance_norm', np.nan):.3f}, "
                    f"index_angle={metrics.get('index_mcp_tip_angle_deg', np.nan):.1f}, "
                    f"middle_angle={metrics.get('middle_mcp_tip_angle_deg', np.nan):.1f}, "
                    f"idx_mid_axis_3d={metrics.get('index_middle_axis_angle_3d_deg', np.nan):.1f}, "
                    f"palm_z={metrics.get('palm_normal_z_approx', np.nan):.4f}"
                )
            elif normalized_label == "S":
                message = (
                    f"Validación geométrica estática OK para {normalized_label}: "
                    f"valid_ratio={valid_ratio:.3f}, "
                    f"finger_count={metrics.get('finger_count_extended', np.nan):.2f}, "
                    f"thumb={metrics.get('thumb_extended_score', np.nan):.3f}, "
                    f"index={metrics.get('index_extended_score', np.nan):.3f}, "
                    f"middle={metrics.get('middle_extended_score', np.nan):.3f}, "
                    f"ring={metrics.get('ring_extended_score', np.nan):.3f}, "
                    f"pinky={metrics.get('pinky_extended_score', np.nan):.3f}, "
                    f"thumb_idx_dist={metrics.get('thumb_index_tip_distance_norm', np.nan):.3f}, "
                    f"index_angle={metrics.get('index_mcp_tip_angle_deg', np.nan):.1f}, "
                    f"thumb_angle={metrics.get('thumb_mcp_tip_angle_deg', np.nan):.1f}, "
                    f"palm_z={metrics.get('palm_normal_z_approx', np.nan):.4f}, "
                    f"idx_chin_dx={metrics.get('index_tip_chin_dx_earspan', np.nan):.3f}, "
                    f"idx_chin_dy={metrics.get('index_tip_chin_dy_earspan', np.nan):.3f}, "
                    f"idx_chin_dist={metrics.get('index_tip_chin_dist_earspan', np.nan):.3f}"
                )
            elif normalized_label == "T":
                message = (
                    f"Validación geométrica estática OK para {normalized_label}: "
                    f"valid_ratio={valid_ratio:.3f}, "
                    f"finger_count={metrics.get('finger_count_extended', np.nan):.2f}, "
                    f"thumb={metrics.get('thumb_extended_score', np.nan):.3f}, "
                    f"index={metrics.get('index_extended_score', np.nan):.3f}, "
                    f"middle={metrics.get('middle_extended_score', np.nan):.3f}, "
                    f"ring={metrics.get('ring_extended_score', np.nan):.3f}, "
                    f"pinky={metrics.get('pinky_extended_score', np.nan):.3f}, "
                    f"idx_mid_dist={metrics.get('index_middle_tip_distance_norm', np.nan):.3f}, "
                    f"index_angle={metrics.get('index_mcp_tip_angle_deg', np.nan):.1f}, "
                    f"index_vert={metrics.get('index_mcp_tip_verticality', np.nan):.3f}, "
                    f"palm_z={metrics.get('palm_normal_z_approx', np.nan):.4f}, "
                    f"idx_chin_dx={metrics.get('index_tip_chin_dx_earspan', np.nan):.3f}, "
                    f"idx_chin_dy={metrics.get('index_tip_chin_dy_earspan', np.nan):.3f}, "
                    f"idx_chin_dist={metrics.get('index_tip_chin_dist_earspan', np.nan):.3f}"
                )
            elif normalized_label == "K":
                message = (
                    f"Validación geométrica estática OK para {normalized_label}: "
                    f"valid_ratio={valid_ratio:.3f}, "
                    f"finger_count={metrics.get('finger_count_extended', np.nan):.2f}, "
                    f"thumb={metrics.get('thumb_extended_score', np.nan):.3f}, "
                    f"index={metrics.get('index_extended_score', np.nan):.3f}, "
                    f"middle={metrics.get('middle_extended_score', np.nan):.3f}, "
                    f"idx_mid_axis_3d={metrics.get('index_middle_axis_angle_3d_deg', np.nan):.1f}"
                )
            elif normalized_label == "R":
                message = (
                    f"Validación geométrica estática OK para {normalized_label}: "
                    f"valid_ratio={valid_ratio:.3f}, "
                    f"finger_count={metrics.get('finger_count_extended', np.nan):.2f}, "
                    f"index={metrics.get('index_extended_score', np.nan):.3f}, "
                    f"middle={metrics.get('middle_extended_score', np.nan):.3f}, "
                    f"idx_mid_dist={metrics.get('index_middle_tip_distance_norm', np.nan):.3f}, "
                    f"idx_mid_axis_3d={metrics.get('index_middle_axis_angle_3d_deg', np.nan):.1f}, "
                    f"mid_axis_range={metrics.get('middle_index_axis_signed_range', np.nan):.3f}, "
                    f"thumb_mid_axis_3d={metrics.get('thumb_middle_axis_angle_3d_deg', np.nan):.1f}"
                )
            else:
                message = (
                    f"Validación geométrica estática OK para {normalized_label}: "
                    f"valid_ratio={valid_ratio:.3f}, "
                    f"finger_count={metrics.get('finger_count_extended', np.nan):.2f}, "
                    f"thumb={metrics.get('thumb_extended_score', np.nan):.3f}, "
                    f"index={metrics.get('index_extended_score', np.nan):.3f}"
                )
        else:
            message = (
                f"Validación geométrica estática NO OK para {normalized_label}: "
                + " | ".join(reasons)
            )

    return StaticGeometryValidationResult(
        label=normalized_label,
        required=required,
        ok=ok,
        message=message,
        metrics=metrics,
        reasons=reasons,
        expected_hands=int(expected_hands),
        frames_used=frames_used,
        valid_frames=valid_frames,
        valid_ratio=valid_ratio,
    )


def validate_static_geometry_from_metrics(
    *,
    label: str,
    metrics: dict[str, float],
    expected_hands: int = 1,
) -> StaticGeometryValidationResult:
    normalized_label = str(label).upper()

    if normalized_label not in STATIC_GEOMETRY_RULES:
        return _build_result(
            label=normalized_label,
            required=False,
            ok=True,
            metrics=metrics,
            expected_hands=expected_hands,
        )

    rule = STATIC_GEOMETRY_RULES[normalized_label]

    if normalized_label == "A":
        reasons = _validate_a_geometry(metrics, rule)
    elif normalized_label == "B":
        reasons = _validate_b_geometry(metrics, rule)
    elif normalized_label == "C":
        reasons = _validate_c_geometry(metrics, rule)
    elif normalized_label == "D":
        reasons = _validate_d_geometry(metrics, rule)
    elif normalized_label == "E":
        reasons = _validate_e_geometry(metrics, rule)
    elif normalized_label == "F":
        reasons = _validate_f_geometry(metrics, rule)
    elif normalized_label == "G":
        reasons = _validate_g_geometry(metrics, rule)
    elif normalized_label == "H":
        reasons = _validate_h_geometry(metrics, rule)
    elif normalized_label == "I":
        reasons = _validate_i_geometry(metrics, rule)
    elif normalized_label == "J":
        reasons = _validate_j_geometry(metrics, rule)
    elif normalized_label == "Z":
        reasons = _validate_z_geometry(metrics, rule)
    elif normalized_label == "Y":
        reasons = _validate_y_geometry(metrics, rule)
    elif normalized_label == "U":
        reasons = _validate_u_geometry(metrics, rule)
    elif normalized_label == "V":
        reasons = _validate_v_geometry(metrics, rule)
    elif normalized_label == "W":
        reasons = _validate_w_geometry(metrics, rule)
    elif normalized_label == "X":
        reasons = _validate_x_geometry(metrics, rule)
    elif normalized_label == "M":
        reasons = _validate_m_geometry(metrics, rule)
    elif normalized_label == "N":
        reasons = _validate_n_geometry(metrics, rule)
    elif normalized_label == "Ñ":
        reasons = _validate_enie_geometry(metrics, rule)
    elif normalized_label == "O":
        reasons = _validate_o_geometry(metrics, rule)
    elif normalized_label == "Q":
        reasons = _validate_q_geometry(metrics, rule)
    elif normalized_label == "L":
        reasons = _validate_l_geometry(metrics, rule)
    elif normalized_label == "K":
        reasons = _validate_k_geometry(metrics, rule)
    elif normalized_label == "R":
        reasons = _validate_r_geometry(metrics, rule)
    elif normalized_label == "S":
        reasons = _validate_s_geometry(metrics, rule)
    elif normalized_label == "T":
        reasons = _validate_t_geometry(metrics, rule)
    else:
        reasons = []

    ok = len(reasons) == 0

    return _build_result(
        label=normalized_label,
        required=True,
        ok=ok,
        metrics=metrics,
        reasons=reasons,
        expected_hands=expected_hands,
    )


def validate_static_gesture_from_captured_items(
    *,
    captured_items: list[dict],
    label: str,
    expected_hands: int,
) -> StaticGeometryValidationResult:
    normalized_label = str(label).upper()

    if normalized_label not in STATIC_GEOMETRY_RULES:
        return _build_result(
            label=normalized_label,
            required=False,
            ok=True,
            metrics={
                "frames_used": int(len(captured_items or [])),
                "valid_frames": 0,
                "valid_ratio": 0.0,
            },
            expected_hands=expected_hands,
        )

    frame_metrics = _extract_primary_metrics_per_frame(
        captured_items=captured_items or [],
        expected_hands=int(expected_hands),
    )

    metrics = aggregate_primary_metrics(frame_metrics)

    return validate_static_geometry_from_metrics(
        label=normalized_label,
        metrics=metrics,
        expected_hands=expected_hands,
    )
