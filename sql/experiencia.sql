SELECT DISTINCT ON (empleo_id) *,
case when exp_profesional = 0 and exp_prof_relacionada = 0 and exp_laboral = 0 and exp_labo_relacionada = 0 and exp_relacionada = 0 then TRUE else FALSE end as sin_experiencia 
FROM (
SELECT
rm.empleo_id,
ci.criterio_id,
trim(regexp_replace(experiencia, E'[\\n\\r\\t,;|]+', ' ', 'g')) experiencia,
sum(CASE WHEN ce.etiqueta_id = 40001 THEN CASE WHEN cc.unidad_id = 12 THEN cc.valor::int ELSE CEIL(cc.valor::int / 12.0) END ELSE 0 END) AS exp_profesional,
sum(CASE WHEN ce.etiqueta_id = 40005 THEN CASE WHEN cc.unidad_id = 12 THEN cc.valor::int ELSE CEIL(cc.valor::int / 12.0) END ELSE 0 END) AS exp_prof_relacionada,
sum(CASE WHEN ce.etiqueta_id = 40003 THEN CASE WHEN cc.unidad_id = 12 THEN cc.valor::int ELSE CEIL(cc.valor::int / 12.0) END ELSE 0 END) AS exp_laboral,
sum(CASE WHEN ce.etiqueta_id = 40006 THEN CASE WHEN cc.unidad_id = 12 THEN cc.valor::int ELSE CEIL(cc.valor::int / 12.0) END ELSE 0 END) AS exp_labo_relacionada,
sum(CASE WHEN ce.etiqueta_id = 40002 THEN CASE WHEN cc.unidad_id = 12 THEN cc.valor::int ELSE CEIL(cc.valor::int / 12.0) END ELSE 0 END) AS exp_relacionada,
TRUE etiqueta
FROM public.requisito_minimo rm
INNER JOIN public.criterio cr ON (rm.id = cr.id AND cr.tipo = 'RM' AND rm.id NOT IN (318271909,756853788,456467562))
INNER JOIN public.criterio_item ci ON (cr.id = ci.criterio_id )
INNER JOIN public.item_criterio_etiqueta ce ON ci.id = ce.criterio_item_id AND ce.etiqueta_id IN (40001, 40005, 40003, 40006, 40002)
LEFT JOIN public.cantidad_criterio cc ON ce.id = cc.item_criterio_etiqueta_id
GROUP BY 1,2,3
UNION
SELECT
empleo_id,
criterio_id,
experiencia,
COALESCE(CEIL(CASE WHEN criterio = 'PROFESIONAL' THEN anios END), 0) p,
COALESCE(CEIL(CASE WHEN criterio = 'PROFESIONAL RELACIONADA' THEN anios END), 0) pr,
COALESCE(CEIL(CASE WHEN criterio = 'LABORAL' THEN anios END), 0) l,
COALESCE(CEIL(CASE WHEN criterio = 'LABORAL RELACIONADA' THEN anios END), 0) lr,
COALESCE(CEIL(CASE WHEN criterio = 'RELACIONADA' THEN anios END), 0) re,
FALSE et
FROM (
SELECT rm2.*,
CASE 
  WHEN criterio IS NULL THEN 0
  WHEN mes_num > 0 THEN mes_num
  WHEN anio_num > 0 THEN anio_num
  WHEN mes_letra > 0 THEN mes_letra
  WHEN anio_letra > 0 THEN anio_letra
  ELSE 0
  END anios
FROM (
  SELECT 
  DISTINCT ON (rm1.empleo_id) empleo_id,
  rm1.criterio_id,
  rm1.criterio,
  rm1.experiencia,
  COALESCE((regexp_match(REGEXP_REPLACE(experiencia, '\((\d+)\)', '\1', 'g'),'(\d+)\s*año','i'))[1]::int4,0) AS anio_num,
  CEIL(COALESCE((regexp_match(REGEXP_REPLACE(experiencia, '\((\d+)\)', '\1', 'g'),'(\d+)\s*mes','i'))[1]::int4,0)/12.0) AS mes_num,
  CASE WHEN experiencia LIKE '%año%' THEN num.valor ELSE 0 END anio_letra,
  CASE WHEN experiencia LIKE '%mes%' THEN CEIL(num.valor/12.0) ELSE 0 END mes_letra
  FROM (
    SELECT
    empleo_id,
    cr.id criterio_id,
    CASE WHEN cr.experiencia ILIKE '%RELACI%' THEN
      CASE WHEN cr.experiencia ILIKE '%PROFESIONAL%' THEN 'PROFESIONAL RELACIONADA'
           WHEN cr.experiencia ILIKE '%LABORAL%' THEN 'LABORAL RELACIONADA'
           ELSE 'RELACIONADA' END
    ELSE 
      CASE WHEN cr.experiencia ILIKE '%PROFESIONAL%' THEN 'PROFESIONAL'
           WHEN cr.experiencia ILIKE '%LABORAL%' THEN 'LABORAL'
           WHEN cr.experiencia ~* '(MES|AÑO)' THEN 'RELACIONADA' 
      END
    END criterio,
    CASE WHEN rm.empleo_id NOT IN (24078,33486,33589,64134,67066,79266,110816) AND (rm.empleo_id IN (1,8862,54073,22071,136791,1184,11107,24868,24869,65835,135084) OR trim(COALESCE(cr.experiencia, '')) IN ('','0','-')
      OR COALESCE(cr.experiencia, '') ~* '(sin exper|no requie|no se requie|n/a|ning|cero|00 mes|0000|no aplica|no exige|no se exige|sin requisito|o reporta|o requerida|No  Aplica|no especifica|No / Aplica)')
    THEN 'NO REQUIERE EXPERIENCIA'
    ELSE trim(REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(TRANSLATE(lower(experiencia),'áéíóú','aeiou'),'28 años|25 años|18 años|año 2005|160 horas',''), E'[\\n\\r\\t,;:°|().”•]+', ' ', 'g'),'  ',' ')) END AS experiencia
    FROM public.requisito_minimo rm
    LEFT JOIN public.criterio cr ON (rm.id = cr.id AND cr.tipo = 'RM')
    WHERE cr.id NOT IN (SELECT criterio_id FROM public.criterio_item)
    ORDER BY rm.empleo_id, cr.id DESC
  ) rm1
  LEFT JOIN ( VALUES ('un ',1), ('uno ',1), ('dos ',2), ('tres ',3), ('cuatro ',4), ('cinco ',5), ('seis ',6), ('siete ',7), ('ocho ',8), ('nueve ',9), ('diez ',10), ('once ',11), ('doce ',12), 
    ('trece ',13), ('catorce ',14), ('quince ',15), ('dieciseis ',16), ('diecisiete ',17), ('dieciocho ',18), ('diecinueve ',19), ('veinte ',20), ('veintiun ',21), ('veinte un ',21), ('veintidos ',22), 
    ('vienticuatro ',24), ('veinte cuatro ',24), ('veinticuatro',24), ('veinticinco ',25), ('veintiseis ',26),('veintisiete ',27), ('veintiocho ',28), ('veintinueve ',29), ('treinta ',30), 
    ('treinta y un',31), ('treinta y dos ',32), ('treinta y tres ',33), ('treinta y cuatro ',34), ('treinta y cinco ',35),('treinta y seis ',36), ('treinta y siete ',37), ('treinta y nueve ',39), 
    ('cuarenta ',40), ('cuarenta y dos ',42), ('cuarenta y ocho ',48), ('cincuenta y un ',51), ('sesenta ',60) ) num(letras, valor)
    ON (rm1.experiencia like '% '||num.letras||'%' OR rm1.experiencia like num.letras||'%')
  ORDER BY rm1.empleo_id, criterio_id DESC, valor DESC
  ) rm2
  ) rm3
) rm4