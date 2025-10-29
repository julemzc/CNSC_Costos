SELECT
RANK() OVER (PARTITION BY padre.id ORDER BY em.id) AS id,
em.id empleo_id,
em.identificador,
em.asignacion_salarial,
em.vigencia_salarial,
COALESCE(em.concurso_ascenso,FALSE) AS concurso_ascenso,
em.grado_nivel_id,
nivel.id AS nivelid,
nivel.nombre AS nivel,
grado_nivel.grado,
deno.nombre AS denominacion,
co.id AS conv_id,
trim(regexp_replace(co.nombre, E'[\\n\\r\\t,;|]+', ' ', 'g')) AS conv_nombre,
co.agno AS conv_agno,
co.estado AS conv_estado,
padre.id AS conv_padre_id,
padre.nombre AS conv_padre,
regexp_replace(en.nombre,'\t|\n','') AS entidad,
en.nit,
ten.nombre AS tipo_entidad,
nbc.reqs_estudio,
nbc_tecnico,
nbc_esp_tecnico,
nbc_tecnologico,
nbc_esp_tecnologico,
nbc_profesional,
nbc_esp_profesional,
nbc_maestria,
nbc_doctorado,
NULL departamento,
NULL municipio,
NULL codigo_dane,
vat.vacantes_opec,
vat.vacantes_municipios,
NULL vacantes
FROM {esquema}.empleo em
LEFT JOIN {esquema}.convocatoria co ON (em.convocatoria_id = co.id AND em.entidad_id IS NULL and co.estado IN ('A','P'))
LEFT JOIN {esquema}.convocatoria padre ON (co.convocatoria_padre_id = padre.id AND padre.estado IN ('A','P'))
INNER JOIN {esquema}.entidad en ON (co.entidad_id = en.id)
INNER JOIN (
  SELECT empleo_id, count(DISTINCT municipio_id) vacantes_municipios, 
  sum(cantidad) vacantes_opec
  FROM {esquema}.vacante 
  WHERE cantidad > 0
  GROUP BY 1
  ) vat ON (vat.empleo_id = em.id)
LEFT JOIN {esquema}.tipo_entidad ten on (en.tipo_entidad_id = ten.id)
LEFT JOIN {esquema}.grado_nivel on (em.grado_nivel_id = grado_nivel.id)
LEFT JOIN {esquema}.nivel on (grado_nivel.nivel_id = nivel.id)
LEFT JOIN {esquema}.denominacion deno on (em.denominacion_id=deno.id)
LEFT JOIN (
  SELECT
  empleo_id,
  json_agg(carrera) FILTER (WHERE nef = 4)::varchar AS nbc_tecnico,
  json_agg(carrera) FILTER (WHERE nef = 15)::varchar AS nbc_esp_tecnico,
  json_agg(carrera) FILTER (WHERE nef = 5)::varchar AS nbc_tecnologico,
  json_agg(carrera) FILTER (WHERE nef = 14)::varchar AS nbc_esp_tecnologico,
  json_agg(carrera) FILTER (WHERE nef = 6)::varchar AS nbc_profesional,
  json_agg(carrera) FILTER (WHERE nef = 7)::varchar AS nbc_esp_profesional,
  json_agg(carrera) FILTER (WHERE nef = 8)::varchar AS nbc_maestria,
  json_agg(carrera) FILTER (WHERE nef = 9)::varchar AS nbc_doctorado,
  '['||json_agg(DISTINCT estudio)::varchar||']' AS reqs_estudio
  FROM (
    SELECT 
    rm.empleo_id,
    nef.id nef,
    nef.nombre estudio,
    json_agg(et.nombre) carrera
    FROM {esquema}.requisito_minimo rm
    INNER JOIN {esquema}.criterio cr ON (rm.id = cr.id AND cr.tipo = 'RM')
    INNER JOIN {esquema}.criterio_item ci ON (cr.id = ci.criterio_id)
    INNER JOIN {esquema}.item_criterio_etiqueta ce ON (ci.id = ce.criterio_item_id)
    INNER JOIN {esquema}.etiqueta et ON (ce.etiqueta_id = et.id)
    INNER JOIN {esquema}.nivel_educacion_formal nef ON (et.nivel_educacion_formal = nef.id)
    WHERE et.tipo_etiqueta_id = 3
    GROUP BY 1,2,3
    ) sub GROUP BY 1
  ) nbc ON (em.id = nbc.empleo_id)
WHERE co.id in ({convocatoria_id}) OR padre.id in ({convocatoria_id})