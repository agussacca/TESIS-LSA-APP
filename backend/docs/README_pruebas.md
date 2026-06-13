# Pruebas automatizadas finales - SeñApp

Este paquete contiene una propuesta de pruebas automatizadas alineada con el DER final simplificado y con los casos seleccionados para la tesis.

## Criterio aplicado

- El modo invitado no se persiste en la base de datos.
- El usuario invitado puede practicar con cámara, pero sus intentos no se guardan.
- Los intentos de práctica guardan letra esperada, letra predicha y validación final del mecanismo completo.
- Una letra aceptada se calcula como: letra_esperada = letra_predicha y validado = true.
- Las palabras deletreadas exitosamente se registran en una tabla específica.
- Las rondas de minijuegos guardan categoría, cantidad total y correctas.
- La ronda perfecta se deriva como correctas = cantidad_minijuegos.
- La experiencia de rondas se calcula y se acumula en el progreso del usuario; no se persiste en la ronda.
- Los logros son independientes de marcos y títulos.
- Los marcos y títulos equipados se guardan en usuarios.
- La disponibilidad de marcos y títulos se calcula por nivel y, para marcos deportivos, por rondas perfectas en Deportes.

## Clasificación aplicada

- La prueba de no persistencia de práctica invitada se clasificó como integración porque verifica comportamiento de API y base temporal.
- El resumen inicial se mantiene como prueba unitaria porque evalúa una función de cálculo sin depender de datos persistidos.
- La prevención de recompensa duplicada por objetivo se mantiene como prueba unitaria porque evalúa una regla de elegibilidad.
- La prueba unitaria de sincronización de objetivos se eliminó por redundancia y quedó cubierta por pruebas de integración de sincronización.

## Ubicación sugerida

Copiar la carpeta `tests` dentro del directorio `backend/` del proyecto.

Copiar `reportes/generar_reporte_pruebas.py` dentro de `backend/reportes/` o en la ubicación usada para generar los reportes.

## Ejecución

Desde el directorio `backend/`:

```bash
pytest tests --junitxml=reportes/pytest_resultados.xml
python reportes/generar_reporte_pruebas.py reportes/pytest_resultados.xml
```

## Nota importante

Estas pruebas están diseñadas contra el contrato final acordado. Si se ejecutan sobre el backend previo al cambio de base de datos, es esperable que fallen parcialmente. En ese caso, los fallos indican las partes que todavía deben ajustarse al DER final.
