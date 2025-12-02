SELECT
RANK() OVER (PARTITION BY NULL ORDER BY em.id, cr.codigo_dane) AS id,
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
NULL conv_id,
NULL conv_nombre,
NULL conv_agno,
NULL conv_estado,
NULL conv_padre_id,
NULL conv_padre,
regexp_replace(en.nombre,'\t|\n','') AS entidad,
en.nit,
ten.nombre AS tipo_entidad,
NULL reqs_estudio,
NULL nbc_tecnico,
NULL nbc_esp_tecnico,
NULL nbc_tecnologico,
NULL nbc_esp_tecnologico,
NULL nbc_profesional,
NULL nbc_esp_profesional,
NULL nbc_maestria,
NULL nbc_doctorado,
cr.departamento,
cr.municipio,
cr.codigo_dane,
vat.vacantes_opec,
vat.vacantes_municipios,
cr.vacantes
FROM {esquema}.empleo em
INNER JOIN {esquema}.entidad en ON (em.entidad_id = en.id)
INNER JOIN (
  SELECT empleo_id, d.nombre departamento, m.nombre municipio, m.codigo_dane, count(*) vacantes
  FROM {esquema}.cargo c
  LEFT JOIN {esquema}.municipio m ON (c.municipio_id = m.id)
  LEFT JOIN {esquema}.departamento d ON (m.departamento_id = d.id)
  GROUP BY 1,2,3,4
  ) cr ON (em.id = cr.empleo_id)
LEFT JOIN ( 
  SELECT empleo_id, count(DISTINCT municipio_id) vacantes_municipios,
  count(*) vacantes_opec
  FROM {esquema}.cargo c
  GROUP BY 1
  ) vat ON (vat.empleo_id = em.id)
LEFT JOIN {esquema}.tipo_entidad ten on (en.tipo_entidad_id = ten.id)
LEFT JOIN {esquema}.grado_nivel on (em.grado_nivel_id = grado_nivel.id)
LEFT JOIN {esquema}.nivel on (grado_nivel.nivel_id = nivel.id)
LEFT JOIN {esquema}.denominacion deno on (em.denominacion_id=deno.id)
WHERE em.id IN {convocatoria_id}
