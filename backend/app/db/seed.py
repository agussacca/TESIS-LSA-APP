from __future__ import annotations

from sqlalchemy.orm import Session

from app.db import models
from app.core.security import hash_password


CATEGORIAS_INICIALES = [
    ("Abecedario", "Categoría inicial para el aprendizaje del abecedario manual.", "/assets/categories/abecedario.png", 1),
    ("Comunicación Básica", "Señas básicas para iniciar interacciones cotidianas.", "/assets/categories/comunicacion.png", 2),
    ("Familia", "Vocabulario relacionado con integrantes de la familia.", "/assets/categories/familia.png", 3),
    ("Colores", "Señas correspondientes a colores básicos.", "/assets/categories/colores.png", 4),
    ("Números", "Señas correspondientes a números.", "/assets/categories/numeros.png", 5),
    ("Deportes", "Vocabulario relacionado con actividades deportivas.", "/assets/categories/deportes.png", 6),
    ("Provincias", "Señas vinculadas a provincias argentinas.", "/assets/categories/provincias.png", 7),
]

ABECEDARIO_LETRAS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "Ñ", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
]


def _asset_letra(letra: str) -> str:
    return "enie" if letra == "Ñ" else letra.lower()


SENIAS_INICIALES = [
    *[
        {
            "categoria": "Abecedario",
            "nombre": letra,
            "descripcion": f"Seña correspondiente a la letra {letra} del abecedario manual.",
            "imagen_url": f"/assets/signs/abecedario/{_asset_letra(letra)}.png",
            "video_url": f"/assets/videos/abecedario/{_asset_letra(letra)}.mp4",
            "orden": index + 1,
        }
        for index, letra in enumerate(ABECEDARIO_LETRAS)
    ],
    {"categoria": "Comunicación Básica", "nombre": "Hola", "descripcion": "Saludo básico utilizado para iniciar una conversación.", "imagen_url": "/assets/signs/comunicacion/hola.png", "video_url": "/assets/videos/comunicacion/hola.mp4", "orden": 1},
    {"categoria": "Comunicación Básica", "nombre": "Chau", "descripcion": "Saludo básico utilizado para terminar una conversación.", "imagen_url": "/assets/signs/comunicacion/chau.png", "video_url": "/assets/videos/comunicacion/chau.mp4", "orden": 2},
    {"categoria": "Comunicación Básica", "nombre": "Nombre", "descripcion": "Expresión utilizada para presentarse.", "imagen_url": "/assets/signs/comunicacion/nombre.png", "video_url": "/assets/videos/comunicacion/nombre.mp4", "orden": 3},
    {"categoria": "Comunicación Básica", "nombre": "Yo", "descripcion": "Yo.", "imagen_url": "/assets/signs/comunicacion/yo.png", "video_url": "/assets/videos/comunicacion/yo.mp4", "orden": 4},
    {"categoria": "Comunicación Básica", "nombre": "Buenos días", "descripcion": "Saludo utilizado durante la mañana.", "imagen_url": "/assets/signs/comunicacion/buenos_dias.png", "video_url": "/assets/videos/comunicacion/buenos_dias.mp4", "orden": 5},
    {"categoria": "Comunicación Básica", "nombre": "Buenas tardes", "descripcion": "Saludo utilizado durante la tarde.", "imagen_url": "/assets/signs/comunicacion/buenas_tardes.png", "video_url": "/assets/videos/comunicacion/buenas_tardes.mp4", "orden": 6},
    {"categoria": "Comunicación Básica", "nombre": "Buenas noches", "descripcion": "Saludo utilizado durante la noche.", "imagen_url": "/assets/signs/comunicacion/buenas_noches.png", "video_url": "/assets/videos/comunicacion/buenas_noches.mp4", "orden": 7},
    {"categoria": "Comunicación Básica", "nombre": "De nada", "descripcion": "De nada.", "imagen_url": "/assets/signs/comunicacion/de_nada.png", "video_url": "/assets/videos/comunicacion/de_nada.mp4", "orden": 8},
    {"categoria": "Comunicación Básica", "nombre": "Gracias", "descripcion": "Gracias.", "imagen_url": "/assets/signs/comunicacion/gracias.png", "video_url": "/assets/videos/comunicacion/gracias.mp4", "orden": 9},
    {"categoria": "Comunicación Básica", "nombre": "Saludar", "descripcion": "Saludo.", "imagen_url": "/assets/signs/comunicacion/saludar.png", "video_url": "/assets/videos/comunicacion/saludar.mp4", "orden": 10},

    {"categoria": "Familia", "nombre": "Mamá", "descripcion": "Seña correspondiente a mamá.", "imagen_url": "/assets/signs/familia/mama.png", "video_url": "/assets/videos/familia/mama.mp4", "orden": 1},
    {"categoria": "Familia", "nombre": "Papá", "descripcion": "Seña correspondiente a papá.", "imagen_url": "/assets/signs/familia/papa.png", "video_url": "/assets/videos/familia/papa.mp4", "orden": 2},
    {"categoria": "Familia", "nombre": "Hermano/a", "descripcion": "Seña correspondiente a hermano o hermana.", "imagen_url": "/assets/signs/familia/hermano-a.png", "video_url": "/assets/videos/familia/hermano-a.mp4", "orden": 3},
    {"categoria": "Familia", "nombre": "Abuelo/a", "descripcion": "Seña correspondiente a abuelo o abuela.", "imagen_url": "/assets/signs/familia/abuelo-a.png", "video_url": "/assets/videos/familia/abuelo-a.mp4", "orden": 4},
    {"categoria": "Familia", "nombre": "Esposo/a", "descripcion": "Seña correspondiente a esposo o esposa.", "imagen_url": "/assets/signs/familia/esposo-a.png", "video_url": "/assets/videos/familia/esposo-a.mp4", "orden": 5},
    {"categoria": "Familia", "nombre": "Hijo/a", "descripcion": "Seña correspondiente a hijo o hija.", "imagen_url": "/assets/signs/familia/hijo-a.png", "video_url": "/assets/videos/familia/hijo-a.mp4", "orden": 6},
    {"categoria": "Familia", "nombre": "Nieto/a", "descripcion": "Seña correspondiente a nieto o nieta.", "imagen_url": "/assets/signs/familia/nieto-a.png", "video_url": "/assets/videos/familia/nieto-a.mp4", "orden": 7},
    {"categoria": "Familia", "nombre": "Primo/a", "descripcion": "Seña correspondiente a primo o prima.", "imagen_url": "/assets/signs/familia/primo-a.png", "video_url": "/assets/videos/familia/primo-a.mp4", "orden": 8},
    {"categoria": "Familia", "nombre": "Sobrino/a", "descripcion": "Seña correspondiente a sobrino o sobrina.", "imagen_url": "/assets/signs/familia/sobrino-a.png", "video_url": "/assets/videos/familia/sobrino-a.mp4", "orden": 9},
    {"categoria": "Familia", "nombre": "Tia", "descripcion": "Seña correspondiente a tía.", "imagen_url": "/assets/signs/familia/tia.png", "video_url": "/assets/videos/familia/tia.mp4", "orden": 10},
    {"categoria": "Familia", "nombre": "Tio", "descripcion": "Seña correspondiente a tío.", "imagen_url": "/assets/signs/familia/tio.png", "video_url": "/assets/videos/familia/tio.mp4", "orden": 11},

    {"categoria": "Colores", "nombre": "Rojo", "descripcion": "Seña correspondiente al color rojo.", "imagen_url": "/assets/signs/colores/rojo.png", "video_url": "/assets/videos/colores/rojo.mp4", "orden": 1},
    {"categoria": "Colores", "nombre": "Azul", "descripcion": "Seña correspondiente al color azul.", "imagen_url": "/assets/signs/colores/azul.png", "video_url": "/assets/videos/colores/azul.mp4", "orden": 2},
    {"categoria": "Colores", "nombre": "Verde", "descripcion": "Seña correspondiente al color verde.", "imagen_url": "/assets/signs/colores/verde.png", "video_url": "/assets/videos/colores/verde.mp4", "orden": 3},

    *[
        {"categoria": "Números", "nombre": nombre, "descripcion": f"Seña correspondiente al número {numero}.", "imagen_url": f"/assets/signs/numeros/{numero}.png", "video_url": f"/assets/videos/numeros/{numero}.mp4", "orden": numero}
        for numero, nombre in [(0,"Cero"), (1, "Uno"), (2, "Dos"), (3, "Tres"), (4, "Cuatro"), (5, "Cinco"), (6, "Seis"),(7, "Siete"), (8, "Ocho"), (9, "Nueve"), (10, "Diez"),(20, "Veinte"), (30, "Treinta"), (40, "Cuarenta"), (50, "Cincuenta"), (60, "Sesenta"), (70, "Setenta"), (80, "Ochenta"), (90, "Noventa"), (100, "Cien")]
    ],

    {"categoria": "Deportes", "nombre": "Fútbol", "descripcion": "Seña correspondiente a fútbol.", "imagen_url": "/assets/signs/deportes/futbol.png", "video_url": "/assets/videos/deportes/futbol.mp4", "orden": 1},
    {"categoria": "Deportes", "nombre": "Básquet", "descripcion": "Seña correspondiente a básquet.", "imagen_url": "/assets/signs/deportes/basquet.png", "video_url": "/assets/videos/deportes/basquet.mp4", "orden": 2},
    {"categoria": "Deportes", "nombre": "Tenis", "descripcion": "Seña correspondiente a tenis.", "imagen_url": "/assets/signs/deportes/tenis.png", "video_url": "/assets/videos/deportes/tenis.mp4", "orden": 3},
    {"categoria": "Deportes", "nombre": "Atletismo", "descripcion": "Seña correspondiente a atletismo.", "imagen_url": "/assets/signs/deportes/atletismo.png", "video_url": "/assets/videos/deportes/atletismo.mp4", "orden": 4},
    {"categoria": "Deportes", "nombre": "Automovilismo", "descripcion": "Seña correspondiente a automovilismo.", "imagen_url": "/assets/signs/deportes/automovilismo.png", "video_url": "/assets/videos/deportes/automovilismo.mp4", "orden": 5},
    {"categoria": "Deportes", "nombre": "Bolos", "descripcion": "Seña correspondiente a bolos.", "imagen_url": "/assets/signs/deportes/bolos.png", "video_url": "/assets/videos/deportes/bolos.mp4", "orden": 6},
    {"categoria": "Deportes", "nombre": "Ciclismo", "descripcion": "Seña correspondiente a ciclismo.", "imagen_url": "/assets/signs/deportes/ciclismo.png", "video_url": "/assets/videos/deportes/ciclismo.mp4", "orden": 7},
    {"categoria": "Deportes", "nombre": "Golf", "descripcion": "Seña correspondiente a golf.", "imagen_url": "/assets/signs/deportes/golf.png", "video_url": "/assets/videos/deportes/golf.mp4", "orden": 8},
    {"categoria": "Deportes", "nombre": "Handball", "descripcion": "Seña correspondiente a handball.", "imagen_url": "/assets/signs/deportes/handball.png", "video_url": "/assets/videos/deportes/handball.mp4", "orden": 9},
    {"categoria": "Deportes", "nombre": "Natación", "descripcion": "Seña correspondiente a natación.", "imagen_url": "/assets/signs/deportes/natacion.png", "video_url": "/assets/videos/deportes/natacion.mp4", "orden": 10},
    {"categoria": "Deportes", "nombre": "Patines", "descripcion": "Seña correspondiente a patines.", "imagen_url": "/assets/signs/deportes/patines.png", "video_url": "/assets/videos/deportes/patines.mp4", "orden": 11},
    {"categoria": "Deportes", "nombre": "Surf", "descripcion": "Seña correspondiente a surf.", "imagen_url": "/assets/signs/deportes/surf.png", "video_url": "/assets/videos/deportes/surf.mp4", "orden": 12},
    {"categoria": "Deportes", "nombre": "Voley", "descripcion": "Seña correspondiente a voley.", "imagen_url": "/assets/signs/deportes/voley.png", "video_url": "/assets/videos/deportes/voley.mp4", "orden": 13},

    {"categoria": "Provincias", "nombre": "Salta", "descripcion": "Seña correspondiente a la provincia de Salta.", "imagen_url": "/assets/signs/provincias/salta.png", "video_url": "/assets/videos/provincias/salta.mp4", "orden": 1},
    {"categoria": "Provincias", "nombre": "Jujuy", "descripcion": "Seña correspondiente a la provincia de Jujuy.", "imagen_url": "/assets/signs/provincias/jujuy.png", "video_url": "/assets/videos/provincias/jujuy.mp4", "orden": 2},
    {"categoria": "Provincias", "nombre": "Tucumán", "descripcion": "Seña correspondiente a la provincia de Tucumán.", "imagen_url": "/assets/signs/provincias/tucuman.png", "video_url": "/assets/videos/provincias/tucuman.mp4", "orden": 3},
    {"categoria": "Provincias", "nombre": "Buenos Aires", "descripcion": "Seña correspondiente a la provincia de Buenos Aires.", "imagen_url": "/assets/signs/provincias/buenos_aires.png", "video_url": "/assets/videos/provincias/buenos_aires.mp4", "orden": 4},
    {"categoria": "Provincias", "nombre": "Catamarca", "descripcion": "Seña correspondiente a la provincia de Catamarca.", "imagen_url": "/assets/signs/provincias/catamarca.png", "video_url": "/assets/videos/provincias/catamarca.mp4", "orden": 5},
    {"categoria": "Provincias", "nombre": "Chaco", "descripcion": "Seña correspondiente a la provincia de Chaco.", "imagen_url": "/assets/signs/provincias/chaco.png", "video_url": "/assets/videos/provincias/chaco.mp4", "orden": 6},
    {"categoria": "Provincias", "nombre": "Chubut", "descripcion": "Seña correspondiente a la provincia de Chubut.", "imagen_url": "/assets/signs/provincias/chubut.png", "video_url": "/assets/videos/provincias/chubut.mp4", "orden": 7},
    {"categoria": "Provincias", "nombre": "Cordoba", "descripcion": "Seña correspondiente a la provincia de Cordoba.", "imagen_url": "/assets/signs/provincias/cordoba.png", "video_url": "/assets/videos/provincias/cordoba.mp4", "orden": 8},
    {"categoria": "Provincias", "nombre": "Corrientes", "descripcion": "Seña correspondiente a la provincia de Corrientes.", "imagen_url": "/assets/signs/provincias/corrientes.png", "video_url": "/assets/videos/provincias/corrientes.mp4", "orden": 9},
    {"categoria": "Provincias", "nombre": "Entre Rios", "descripcion": "Seña correspondiente a la provincia de Entre Rios.", "imagen_url": "/assets/signs/provincias/entre_rios.png", "video_url": "/assets/videos/provincias/entre_rios.mp4", "orden": 10},
    {"categoria": "Provincias", "nombre": "Mendoza", "descripcion": "Seña correspondiente a la provincia de Mendoza.", "imagen_url": "/assets/signs/provincias/mendoza.png", "video_url": "/assets/videos/provincias/mendoza.mp4", "orden": 11},
    {"categoria": "Provincias", "nombre": "La Pampa", "descripcion": "Seña correspondiente a la provincia de La Pampa.", "imagen_url": "/assets/signs/provincias/la_pampa.png", "video_url": "/assets/videos/provincias/la_pampa.mp4", "orden": 12},
    {"categoria": "Provincias", "nombre": "La Rioja", "descripcion": "Seña correspondiente a la provincia de La Rioja.", "imagen_url": "/assets/signs/provincias/la_rioja.png", "video_url": "/assets/videos/provincias/la_rioja.mp4", "orden": 13},
    {"categoria": "Provincias", "nombre": "Misiones", "descripcion": "Seña correspondiente a la provincia de Misiones.", "imagen_url": "/assets/signs/provincias/misiones.png", "video_url": "/assets/videos/provincias/misiones.mp4", "orden": 14},
    {"categoria": "Provincias", "nombre": "Neuquen", "descripcion": "Seña correspondiente a la provincia de Neuquen.", "imagen_url": "/assets/signs/provincias/neuquen.png", "video_url": "/assets/videos/provincias/neuquen.mp4", "orden": 15},
    {"categoria": "Provincias", "nombre": "Rio Negro", "descripcion": "Seña correspondiente a la provincia de Rio Negro.", "imagen_url": "/assets/signs/provincias/rio_negro.png", "video_url": "/assets/videos/provincias/rio_negro.mp4", "orden": 16},
    {"categoria": "Provincias", "nombre": "San Juan", "descripcion": "Seña correspondiente a la provincia de San Juan.", "imagen_url": "/assets/signs/provincias/san_juan.png", "video_url": "/assets/videos/provincias/san_juan.mp4", "orden": 17},
    {"categoria": "Provincias", "nombre": "San Luis", "descripcion": "Seña correspondiente a la provincia de San Luis.", "imagen_url": "/assets/signs/provincias/san_luis.png", "video_url": "/assets/videos/provincias/san_luis.mp4", "orden": 18},
    {"categoria": "Provincias", "nombre": "Santa Cruz", "descripcion": "Seña correspondiente a la provincia de Santa Cruz.", "imagen_url": "/assets/signs/provincias/santa_cruz.png", "video_url": "/assets/videos/provincias/santa_cruz.mp4", "orden": 19},
    {"categoria": "Provincias", "nombre": "Santa Fé", "descripcion": "Seña correspondiente a la provincia de Santa Fé.", "imagen_url": "/assets/signs/provincias/santa_fe.png", "video_url": "/assets/videos/provincias/santa_fe.mp4", "orden": 20},
    {"categoria": "Provincias", "nombre": "Santiago del Estero", "descripcion": "Seña correspondiente a la provincia de Santiago del Estero.", "imagen_url": "/assets/signs/provincias/santiago_del_estero.png", "video_url": "/assets/videos/provincias/santiago_del_estero.mp4", "orden": 21},
    {"categoria": "Provincias", "nombre": "Tierra del Fuego", "descripcion": "Seña correspondiente a la provincia de Tierra del Fuego.", "imagen_url": "/assets/signs/provincias/tierra_del_fuego.png", "video_url": "/assets/videos/provincias/tierra_del_fuego.mp4", "orden": 22},
    {"categoria": "Provincias", "nombre": "Formosa", "descripcion": "Seña correspondiente a la provincia de Formosa.", "imagen_url": "/assets/signs/provincias/formosa.png", "video_url": "/assets/videos/provincias/formosa.mp4", "orden": 23},
]

OBJETIVOS_INICIALES = [
    ("Aprendé 2 señas", "Aprender dos señas con cámara durante el día.", "DIARIA", 40),
    ("Completá 3 minijuegos", "Completar tres minijuegos durante el día.", "DIARIA", 60),
    ("Practicá con cámara", "Realizar al menos una práctica de seña del abecedario durante el día.", "DIARIA", 50),
    ("Aprendé 8 señas nuevas", "Aprender ocho señas nuevas durante la semana.", "SEMANAL", 120),
    ("Completá 10 rondas", "Completar diez rondas durante la semana.", "SEMANAL", 150),
    ("Lográ 3 rondas perfectas", "Lograr tres rondas perfectas durante la semana.", "SEMANAL", 180),
]

LOGROS_INICIALES = [
    ('Primer compromiso', 'Completar el primer objetivo diario.', '/assets/logros/primer_compromiso.png'),
    ('Rutina en marcha', 'Completar 10 objetivos diarios en total.', '/assets/logros/rutina_en_marcha.png'),
    ('Hábito firme', 'Completar 50 objetivos diarios en total.', '/assets/logros/habito_firme.png'),
    ('Meta semanal cumplida', 'Completar el primer objetivo semanal.', '/assets/logros/meta_semanal_cumplida.png'),
    ('Semana productiva', 'Completar 10 objetivos semanales en total.', '/assets/logros/semana_productiva.png'),
    ('Primer día activo', 'Completar al menos un objetivo diario en un día.', '/assets/logros/primer_dia_activo.png'),
    ('Tres días en marcha', 'Mantener una racha de 3 días.', '/assets/logros/tres_dias_en_marcha.png'),
    ('Semana imparable', 'Mantener una racha de 7 días.', '/assets/logros/semana_imparable.png'),
    ('Quince días de constancia', 'Mantener una racha de 15 días.', '/assets/logros/quince_dias_de_constancia.png'),
    ('Mes perfecto de práctica', 'Mantener una racha de 30 días.', '/assets/logros/mes_perfecto_de_practica.png'),
    ('Primer ascenso', 'Alcanzar el nivel 2.', '/assets/logros/primer_ascenso.png'),
    ('Aprendiz destacado', 'Alcanzar el nivel 5.', '/assets/logros/aprendiz_destacado.png'),
    ('Dominio en crecimiento', 'Alcanzar el nivel 10.', '/assets/logros/dominio_en_crecimiento.png'),
    ('Abecedario Conquistado', 'Aprender todas las señas de la categoría Abecedario.', '/assets/logros/abecedario_conquistado.png'),
    ('Maestría Alfabética', 'Completar 5 rondas de minijuegos de la categoría Abecedario sin cometer ningún error.', '/assets/logros/maestria_alfabetica.png'),
    ('Paleta Aprendida', 'Aprender todas las señas de la categoría Colores.', '/assets/logros/paleta_aprendida.png'),
    ('Maestría Cromática', 'Completar 5 rondas de minijuegos de la categoría Colores sin cometer ningún error.', '/assets/logros/maestria_cromatica.png'),
    ('Primeras Conversaciones', 'Aprender todas las señas de la categoría Comunicación Básica.', '/assets/logros/primeras_conversaciones.png'),
    ('Diálogo Impecable', 'Completar 5 rondas de minijuegos de la categoría Comunicación Básica sin cometer ningún error.', '/assets/logros/dialogo_impecable.png'),
    ('Espíritu Deportivo', 'Aprender todas las señas de la categoría Deportes.', '/assets/logros/espiritu_deportivo.png'),
    ('Jugada Perfecta', 'Completar 5 rondas de minijuegos de la categoría Deportes sin cometer ningún error.', '/assets/logros/jugada_perfecta.png'),
    ('Lazos Aprendidos', 'Aprender todas las señas de la categoría Familia.', '/assets/logros/lazos_aprendidos.png'),
    ('Lazos Firmes', 'Completar 5 rondas de minijuegos de la categoría Familia sin cometer ningún error.', '/assets/logros/lazos_firmes.png'),
    ('Números Dominados', 'Aprender todas las señas de la categoría Números.', '/assets/logros/numeros_dominados.png'),
    ('Cuenta Clara', 'Completar 5 rondas de minijuegos de la categoría Números sin cometer ningún error.', '/assets/logros/cuenta_clara.png'),
    ('Conocedor de Territorios', 'Aprender todas las señas de la categoría Provincias.', '/assets/logros/conocedor_de_territorios.png'),
    ('Como la Palma de Mi Mano', 'Completar 5 rondas de minijuegos de la categoría Provincias sin cometer ningún error.', '/assets/logros/como_la_palma_de_mi_mano.png'),
]

MARCOS_INICIALES = [
    ("Fuego", "/assets/marcos/fuego.png", 1, 1),
    ("Tierra", "/assets/marcos/tierra.png", 1, 2),
    ("Aire", "/assets/marcos/aire.png", 1, 3),
    ("Agua", "/assets/marcos/agua.png", 1, 4),
    ("Madera", "/assets/marcos/madera.png", 2, 5),
    ("Hierro", "/assets/marcos/hierro.png", 3, 6),
    ("Bronce", "/assets/marcos/bronce.png", 4, 7),
    ("Plata", "/assets/marcos/plata.png", 5, 8),
    ("Oro", "/assets/marcos/oro.png", 6, 9),
    ("Diamante", "/assets/marcos/diamante.png", 7, 10),
    ("Estelar", "/assets/marcos/estelar.png", 8, 11),
    ("Galáctico", "/assets/marcos/galactico.png", 9, 12),
    ("Fútbol", "/assets/marcos/futbol.png", None, 20),
    ("Tenis", "/assets/marcos/tenis.png", None, 21),
]

TITULOS_INICIALES = [
    ("Aprendiz constante", 1, 1),
    ("Explorador de señas", 2, 2),
    ("Practicante dedicado", 3, 3),
    ("Dominador del abecedario", 5, 4),
    ("Explorador experto", 7, 5),
    ("Maestro de señas", 10, 6),
]


def _upsert_por_nombre(db: Session, modelo, valores: dict):
    existente = db.query(modelo).filter(modelo.nombre == valores["nombre"]).first()
    if existente is None:
        objeto = modelo(**valores)
        db.add(objeto)
        db.flush()
        return objeto
    for clave, valor in valores.items():
        setattr(existente, clave, valor)
    db.flush()
    return existente


def _sembrar_categorias_y_senias(db: Session) -> None:
    categorias_por_nombre: dict[str, models.CategoriaAprendizaje] = {}
    for nombre, descripcion, imagen_portada_url, orden in CATEGORIAS_INICIALES:
        categoria = _upsert_por_nombre(
            db,
            models.CategoriaAprendizaje,
            {
                "nombre": nombre,
                "descripcion": descripcion,
                "imagen_portada_url": imagen_portada_url,
                "orden": orden,
            },
        )
        categorias_por_nombre[nombre] = categoria

    for item in SENIAS_INICIALES:
        categoria = categorias_por_nombre[item["categoria"]]
        existente = (
            db.query(models.Senia)
            .filter(
                models.Senia.categoria_id == categoria.id_categoria_aprendizaje,
                models.Senia.nombre == item["nombre"],
            )
            .first()
        )
        valores = {
            "categoria_id": categoria.id_categoria_aprendizaje,
            "nombre": item["nombre"],
            "descripcion": item["descripcion"],
            "imagen_url": item["imagen_url"],
            "video_url": item["video_url"],
            "orden": item["orden"],
        }
        if existente is None:
            db.add(models.Senia(**valores))
        else:
            for clave, valor in valores.items():
                setattr(existente, clave, valor)


def _sembrar_gamificacion(db: Session) -> None:
    for nombre, descripcion, periodicidad, xp_recompensa in OBJETIVOS_INICIALES:
        _upsert_por_nombre(
            db,
            models.Objetivo,
            {
                "nombre": nombre,
                "descripcion": descripcion,
                "periodicidad": periodicidad,
                "xp_recompensa": xp_recompensa,
            },
        )

    for nombre, descripcion, imagen_url in LOGROS_INICIALES:
        _upsert_por_nombre(
            db,
            models.Logro,
            {
                "nombre": nombre,
                "descripcion": descripcion,
                "imagen_url": imagen_url,
            },
        )

    for nombre, imagen_url, nivel_requerido, orden in MARCOS_INICIALES:
        _upsert_por_nombre(
            db,
            models.Marco,
            {
                "nombre": nombre,
                "imagen_url": imagen_url,
                "nivel_requerido": nivel_requerido,
                "orden": orden,
            },
        )

    for nombre, nivel_requerido, orden in TITULOS_INICIALES:
        _upsert_por_nombre(
            db,
            models.Titulo,
            {
                "nombre": nombre,
                "nivel_requerido": nivel_requerido,
                "orden": orden,
            },
        )


def _sembrar_usuario_demo(db: Session) -> None:
    usuario = db.query(models.Usuario).filter(models.Usuario.email == "juan@senapp.test").first()
    if usuario is None:
        usuario = models.Usuario(
            email="juan@senapp.test",
            password_hash=hash_password("password123"),
            nombre_visible="Juan González",
            foto_perfil_url=None,
        )
        db.add(usuario)
        db.flush()

    if usuario.password_hash == "demo" or not str(usuario.password_hash or "").startswith("pbkdf2_sha256$"):
        usuario.password_hash = hash_password("password123")

    if usuario.marco_equipado_id is None:
        marco = db.query(models.Marco).filter(models.Marco.nombre == "Fuego").first()
        if marco is not None:
            usuario.marco_equipado_id = marco.id_marco

    if usuario.titulo_equipado_id is None:
        titulo = db.query(models.Titulo).filter(models.Titulo.nombre == "Aprendiz constante").first()
        if titulo is not None:
            usuario.titulo_equipado_id = titulo.id_titulo

    progreso = db.query(models.ProgresoUsuario).filter(models.ProgresoUsuario.usuario_id == usuario.id_usuario).first()
    if progreso is None:
        db.add(models.ProgresoUsuario(usuario_id=usuario.id_usuario))


def sembrar_datos_iniciales(db: Session) -> None:
    _sembrar_categorias_y_senias(db)
    _sembrar_gamificacion(db)
    _sembrar_usuario_demo(db)
    db.commit()


# Alias para mantener compatibilidad con versiones previas de main.py.
def sembrar_objetivos_iniciales(db: Session) -> None:
    sembrar_datos_iniciales(db)
