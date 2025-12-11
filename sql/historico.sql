SELECT
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
co.tipo_proc_sele_id conv_proceso_id,
co.tipo_proceso conv_proceso,
padre.id AS conv_padre_id,
padre.nombre AS conv_padre,
trim(regexp_replace(en.nombre, E'[\\n\\r\\t,;|]+', ' ', 'g')) AS entidad,
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
va.departamento,
va.municipio,
va.codigo_dane,
vat.vacantes_opec,
vat.vacantes_municipios,
va.vacantes,
ins.inscritos,
ceil(va.vacantes*1.0 / vat.vacantes_opec * ins.inscritos) AS mun_inscritos,
vrm.aprobo_vrm,
ceil(va.vacantes*1.0 / vat.vacantes_opec * vrm.aprobo_vrm) AS mun_aprobo_vrm,
esc.aprobo_escritas,
ceil(va.vacantes*1.0 / vat.vacantes_opec * esc.aprobo_escritas) AS mun_aprobo_escritas,
pcd_inscritos,
va.vacantes*1.0 / vat.vacantes_opec * pcd_inscritos AS mun_pcd_inscritos,
pcd_psicosocial,
pcd_intelectual,
pcd_auditiva,
pcd_visual,
pcd_fisica,
pcd_multiple,
pcd_sordoceguera,
current_timestamp AS fecha_actualizacion
FROM {esquema}.empleo em
LEFT JOIN {esquema}.convocatoria co ON (em.convocatoria_id = co.id AND em.entidad_id IS NULL and co.estado IN ('A','P'))
LEFT JOIN {esquema}.convocatoria padre ON (co.convocatoria_padre_id = padre.id AND padre.estado IN ('A','P'))
INNER JOIN {esquema}.entidad en ON (co.entidad_id = en.id)
INNER JOIN (
  SELECT empleo_id, dept.nombre departamento, mun.nombre municipio, mun.codigo_dane, sum(cantidad) vacantes
  FROM {esquema}.vacante
  INNER JOIN {esquema}.municipio mun on (COALESCE(municipio_id,1999) = mun.id)
  INNER JOIN {esquema}.departamento dept on (dept.id=mun.departamento_id)
  WHERE cantidad > 0 AND empleo_id IS NOT NULL
  GROUP BY 1,2,3,4
  ) va ON (em.id = va.empleo_id)
INNER JOIN (
  SELECT empleo_id, count(DISTINCT municipio_id) vacantes_municipios, sum(cantidad) vacantes_opec 
  FROM {esquema}.vacante 
  WHERE cantidad > 0 GROUP BY 1
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
INNER JOIN (
  SELECT empleo_id, count(*) inscritos 
  FROM {esquema}.inscripcion_convocatoria ic 
  WHERE estado IN ('I') GROUP BY 1
  ) ins ON (em.id = ins.empleo_id)
LEFT JOIN (
  SELECT empleo_id, count(*) aprobo_vrm
  FROM {esquema}.inscripcion_convocatoria ic 
  INNER JOIN {esquema}.evalua_prueba ep on (ep.inscripcion_convocatoria_id=ic.id) 
  INNER JOIN {esquema}.prueba pr on (ep.prueba_id = pr.id)
  WHERE ic.estado IN ('I') AND ep.aprobo AND ep.evalua_origen_id IS NULL AND pr.tipo_prueba_id = 1 
  GROUP BY 1
  ) vrm ON (em.id = vrm.empleo_id)
LEFT JOIN (
  SELECT empleo_id, count(*) aprobo_escritas
  FROM (
    SELECT empleo_id, ic.id 
    FROM {esquema}.inscripcion_convocatoria ic 
    INNER JOIN {esquema}.evalua_prueba ep ON ep.inscripcion_convocatoria_id = ic.id
    INNER JOIN {esquema}.prueba pr ON pr.id = ep.prueba_id 
    INNER JOIN {esquema}.tipo_prueba tp ON pr.tipo_prueba_id = tp.id
    WHERE ic.estado IN ('I') AND tp.calificacion AND tp.tipo_etapa_id = 4 
    GROUP BY ic.empleo_id, ic.id HAVING COUNT(*) = COUNT(CASE WHEN ep.aprobo THEN 1 END)
    ) sub GROUP BY 1
  ) esc ON (em.id = esc.empleo_id)
LEFT JOIN (
  SELECT
  empleo_id,
  count(DISTINCT dc.ciudadano_id) pcd_inscritos,
  count(DISTINCT dc1.ciudadano_id) pcd_psicosocial,
  count(DISTINCT dc3.ciudadano_id) pcd_intelectual,
  count(DISTINCT dc4.ciudadano_id) pcd_auditiva,
  count(DISTINCT dc5.ciudadano_id) pcd_visual,
  count(DISTINCT dc6.ciudadano_id) pcd_fisica,
  count(DISTINCT dc7.ciudadano_id) pcd_multiple,
  count(DISTINCT dc8.ciudadano_id) pcd_sordoceguera
  FROM {esquema}.empleo em
  INNER JOIN {esquema}.inscripcion_convocatoria ic ON (em.id = ic.empleo_id AND ic.estado = 'I')
  INNER JOIN {esquema}.ciudadano c ON ic.ciudadano_id = c.id AND c.certificado_discapacidad
  INNER JOIN {esquema}.discapacidad_ciudadano dc ON ic.ciudadano_id = dc.ciudadano_id
  LEFT JOIN {esquema}.discapacidad_ciudadano dc1 ON ic.ciudadano_id = dc1.ciudadano_id AND dc1.discapacidad_id = 1
  LEFT JOIN {esquema}.discapacidad_ciudadano dc3 ON ic.ciudadano_id = dc3.ciudadano_id AND dc3.discapacidad_id = 3
  LEFT JOIN {esquema}.discapacidad_ciudadano dc4 ON ic.ciudadano_id = dc4.ciudadano_id AND dc4.discapacidad_id = 4
  LEFT JOIN {esquema}.discapacidad_ciudadano dc5 ON ic.ciudadano_id = dc5.ciudadano_id AND dc5.discapacidad_id = 5
  LEFT JOIN {esquema}.discapacidad_ciudadano dc6 ON ic.ciudadano_id = dc6.ciudadano_id AND dc6.discapacidad_id = 6
  LEFT JOIN {esquema}.discapacidad_ciudadano dc7 ON ic.ciudadano_id = dc7.ciudadano_id AND dc7.discapacidad_id = 7
  LEFT JOIN {esquema}.discapacidad_ciudadano dc8 ON ic.ciudadano_id = dc8.ciudadano_id AND dc8.discapacidad_id = 8
  GROUP BY 1
  ) pcd on em.id = pcd.empleo_id
WHERE co.id NOT IN (434453448, 434453496, 58430971, 249696787, 108234, 460524154, 216315919, 59252980, 53155286, 1599964, 1160535)
AND COALESCE(padre.id, -1) NOT IN (10000, 10010, 10110, 10220, 145543242, 434431366);