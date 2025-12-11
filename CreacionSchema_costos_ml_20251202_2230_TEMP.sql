-- CREACIÓN DEL ESQUEMA

-- DROP SCHEMA costos_ml;
CREATE SCHEMA costos_ml AUTHORIZATION postgres;

-- Permissions
GRANT ALL ON SCHEMA costos_ml TO postgres;
GRANT ALL ON SCHEMA costos_ml TO usrcosteo;
GRANT USAGE ON SCHEMA costos_ml TO lmontoya;
GRANT USAGE ON SCHEMA costos_ml TO jlmartinez;


-- SECUENCIAS

-- DROP SEQUENCE costos_ml.seq_adm_entrada;
CREATE SEQUENCE costos_ml.seq_adm_entrada
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9223372036854775807
	START 3
	CACHE 1
	NO CYCLE;
-- Permissions
ALTER SEQUENCE costos_ml.seq_adm_entrada OWNER TO usrcosteo;
GRANT ALL ON SEQUENCE costos_ml.seq_adm_entrada TO usrcosteo;


-- DROP SEQUENCE costos_ml.seq_adm_etapa;
CREATE SEQUENCE costos_ml.seq_adm_etapa
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9223372036854775807
	START 1
	CACHE 1
	NO CYCLE;
-- Permissions
ALTER SEQUENCE costos_ml.seq_adm_etapa OWNER TO usrcosteo;
GRANT ALL ON SEQUENCE costos_ml.seq_adm_etapa TO usrcosteo;


-- DROP SEQUENCE costos_ml.seq_adm_etapa_entrada;
CREATE SEQUENCE costos_ml.seq_adm_etapa_entrada
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9223372036854775807
	START 1
	CACHE 1
	NO CYCLE;
-- Permissions
ALTER SEQUENCE costos_ml.seq_adm_etapa_entrada OWNER TO usrcosteo;
GRANT ALL ON SEQUENCE costos_ml.seq_adm_etapa_entrada TO usrcosteo;


-- DROP SEQUENCE costos_ml.seq_adm_etapa_item;
CREATE SEQUENCE costos_ml.seq_adm_etapa_item
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9223372036854775807
	START 1
	CACHE 1
	NO CYCLE;
-- Permissions
ALTER SEQUENCE costos_ml.seq_adm_etapa_item OWNER TO usrcosteo;
GRANT ALL ON SEQUENCE costos_ml.seq_adm_etapa_item TO usrcosteo;


-- DROP SEQUENCE costos_ml.seq_adm_item;
CREATE SEQUENCE costos_ml.seq_adm_item
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9223372036854775807
	START 1
	CACHE 1
	NO CYCLE;
-- Permissions
ALTER SEQUENCE costos_ml.seq_adm_item OWNER TO usrcosteo;
GRANT ALL ON SEQUENCE costos_ml.seq_adm_item TO usrcosteo;


-- DROP SEQUENCE costos_ml.seq_adm_unidad;
CREATE SEQUENCE costos_ml.seq_adm_unidad
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9223372036854775807
	START 5
	CACHE 1
	NO CYCLE;
-- Permissions
ALTER SEQUENCE costos_ml.seq_adm_unidad OWNER TO usrcosteo;
GRANT ALL ON SEQUENCE costos_ml.seq_adm_unidad TO usrcosteo;


-- DROP SEQUENCE costos_ml.seq_grado_salario;

CREATE SEQUENCE costos_ml.seq_grado_salario
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9999999
	START 1101
	CACHE 1
	CYCLE;
-- Permissions
ALTER SEQUENCE costos_ml.seq_grado_salario OWNER TO usrcosteo;
GRANT ALL ON SEQUENCE costos_ml.seq_grado_salario TO usrcosteo;


-- DROP SEQUENCE costos_ml.seq_np_escenas;
CREATE SEQUENCE costos_ml.seq_np_escenas
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9999999
	START 1
	CACHE 1
	NO CYCLE;
-- Permissions
ALTER SEQUENCE costos_ml.seq_np_escenas OWNER TO usrcosteo;
GRANT ALL ON SEQUENCE costos_ml.seq_np_escenas TO usrcosteo;


-- DROP SEQUENCE costos_ml.seq_np_escenas_detalle;
CREATE SEQUENCE costos_ml.seq_np_escenas_detalle
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 999999
	START 1
	CACHE 1
	NO CYCLE;
-- Permissions
ALTER SEQUENCE costos_ml.seq_np_escenas_detalle OWNER TO usrcosteo;
GRANT ALL ON SEQUENCE costos_ml.seq_np_escenas_detalle TO usrcosteo;


-- DROP SEQUENCE costos_ml.seq_np_parametros;
CREATE SEQUENCE costos_ml.seq_np_parametros
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 999999
	START 1
	CACHE 1
	NO CYCLE;
-- Permissions
ALTER SEQUENCE costos_ml.seq_np_parametros OWNER TO usrcosteo;
GRANT ALL ON SEQUENCE costos_ml.seq_np_parametros TO usrcosteo;



-- DROP SEQUENCE costos_ml.seq_np_redneuronal;
CREATE SEQUENCE costos_ml.seq_np_redneuronal
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9223372036854775807
	START 1
	CACHE 1
	NO CYCLE;
-- Permissions
ALTER SEQUENCE costos_ml.seq_np_redneuronal OWNER TO usrcosteo;
GRANT ALL ON SEQUENCE costos_ml.seq_np_redneuronal TO usrcosteo;


-- DROP SEQUENCE costos_ml.seq_np_variables;
CREATE SEQUENCE costos_ml.seq_np_variables
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 999999
	START 1
	CACHE 1
	NO CYCLE;
-- Permissions
ALTER SEQUENCE costos_ml.seq_np_variables OWNER TO usrcosteo;
GRANT ALL ON SEQUENCE costos_ml.seq_np_variables TO usrcosteo;


-- DROP SEQUENCE costos_ml.seq_seleccion_conv;
CREATE SEQUENCE costos_ml.seq_seleccion_conv
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 999999
	START 1
	CACHE 1
	NO CYCLE;
-- Permissions
ALTER SEQUENCE costos_ml.seq_seleccion_conv OWNER TO usrcosteo;
GRANT ALL ON SEQUENCE costos_ml.seq_seleccion_conv TO usrcosteo;



--- MODELO DE COSTOS -----------

-- DROP TABLE costos_ml.componente_presupuesto;
CREATE TABLE costos_ml.componente_presupuesto (
	id uuid NOT NULL,
	nombre varchar(255) NOT NULL,
	total float8 NULL,
	id_presupuesto uuid NOT NULL,
	tipo int4 NULL,
	variable varchar(50) NULL,
	CONSTRAINT pk_componente_presupuesto PRIMARY KEY (id)
);

-- DROP TABLE costos_ml.componente_proyeccion;
CREATE TABLE costos_ml.componente_proyeccion (
	id uuid NOT NULL,
	nombre varchar(255) NOT NULL,
	total float8 NULL,
	id_costeo uuid NOT NULL,
	tipo int4 NOT NULL,
	variable varchar(50) NULL,
	CONSTRAINT pk_componente_proyeccion PRIMARY KEY (id)
);

-- DROP TABLE costos_ml.componente_proyeccion_costos_variable;
CREATE TABLE costos_ml.componente_proyeccion_costos_variable (
	componente_proyeccion_id uuid NOT NULL,
	costos_variable_id uuid NOT NULL,
	CONSTRAINT uk_componente_proyeccion_costos_variable UNIQUE (costos_variable_id)
);

-- DROP TABLE costos_ml.componente_proyeccion_datos_entrada;
CREATE TABLE costos_ml.componente_proyeccion_datos_entrada (
	componente_proyeccion_id uuid NOT NULL,
	datos_entrada_id uuid NOT NULL,
	CONSTRAINT uk_componente_proyeccion_datos_entrada UNIQUE (datos_entrada_id)
);

-- DROP TABLE costos_ml.componente_proyeccion_datos_entrada_formulados;
CREATE TABLE costos_ml.componente_proyeccion_datos_entrada_formulados (
	componente_proyeccion_id uuid NOT NULL,
	datos_entrada_formulados_id uuid NOT NULL,
	CONSTRAINT uk_componente_proyeccion_datos_entrada_formulados UNIQUE (datos_entrada_formulados_id)
);

-- DROP TABLE costos_ml.costeo_proyeccion;
CREATE TABLE costos_ml.costeo_proyeccion (
	id uuid NOT NULL,
	cobertura varchar(255) NULL,
	entidades_participantes int4 NULL,
	estado varchar(255) NULL,
	fecha date NULL,
	inscritos_asesor int4 NULL,
	inscritos_asistencial int4 NULL,
	inscritos_profesional int4 NULL,
	inscritos_tecnico int4 NULL,
	templeos int4 NULL,
	total float8 NULL,
	tpagosproyectados int4 NULL,
	vacantes int4 NULL,
	"version" int4 NOT NULL,
	id_proceso_seleccion int8 NOT NULL,
	tpagosreales int8 NULL,
	proyectados bool DEFAULT true NULL,
	inscritos_otros_nivel2 int4 DEFAULT 0 NULL,
	inscritos_otros_nivel1 int4 DEFAULT 0 NULL,
	templeos_experiencia int4 DEFAULT 0 NULL,
	templeos_sin_experiencia int4 DEFAULT 0 NULL,
	vacantes_experiencia int4 DEFAULT 0 NULL,
	vacantes_sin_experiencia int4 DEFAULT 0 NULL,
	ejecucion_id int4 DEFAULT 0 NULL,
	esceneario_id int4 DEFAULT 0 NULL,
	redneuronal_id int4 DEFAULT 0 NULL,
	CONSTRAINT pk_costeo_proyeccion PRIMARY KEY (id)
);

-- DROP TABLE costos_ml.estado_proyeccion;
CREATE TABLE costos_ml.estado_proyeccion (
	id uuid NOT NULL,
	id_costeo uuid NOT NULL,
	estado varchar(50) NOT NULL,
	fecha_cambio timestamp DEFAULT CURRENT_TIMESTAMP NOT NULL,
	usuario varchar(255) NULL,
	observacion text NULL,
	CONSTRAINT pk_estado_proyeccion PRIMARY KEY (id)
);
CREATE INDEX idx_estado_proyeccion_id_costeo ON costos_ml.estado_proyeccion USING btree (id_costeo);

-- DROP TABLE costos_ml.etapa;
CREATE TABLE costos_ml.etapa (
	id_etapa uuid NOT NULL,
	nombre bpchar(200) NOT NULL,
	fecha date NOT NULL,
	estado bool NOT NULL,
	tipo int2 NOT NULL,
	variable varchar(50) NULL,
	posicion int4 NULL,
	CONSTRAINT uk_constraint_variable_etapa UNIQUE (variable),
	CONSTRAINT pk_etapa PRIMARY KEY (id_etapa)
);

-- DROP TABLE costos_ml.etapa_item;
CREATE TABLE costos_ml.etapa_item (
	id_etapa_item uuid NOT NULL,
	id_etapa uuid NOT NULL,
	id_item uuid NOT NULL,
	CONSTRAINT pk_etapa_item PRIMARY KEY (id_etapa_item)
);

-- DROP TABLE costos_ml.item;
CREATE TABLE costos_ml.item (
	id_item uuid NOT NULL,
	tipo int4 NOT NULL,
	nombre bpchar(200) NOT NULL,
	unidad int4 NULL,
	porcentual bool NOT NULL,
	valor float8 NULL,
	fecha date NULL,
	variable varchar(50) NULL,
	formula varchar(255) NULL,
	urlrecurso varchar(255) NULL,
	posicion numeric NULL,
	CONSTRAINT uk_constraint_variable UNIQUE (variable),
	CONSTRAINT pk_item PRIMARY KEY (id_item)
);

-- DROP TABLE costos_ml.item_proyeccion;
CREATE TABLE costos_ml.item_proyeccion (
	id uuid NOT NULL,
	cantidad float8 NULL,
	formula varchar(255) NULL,
	nombre varchar(255) NOT NULL,
	unidad varchar(255) NULL,
	variable varchar(255) NULL,
	vtotal float8 NULL,
	vunitario float8 NULL,
	CONSTRAINT pk_item_proyeccion PRIMARY KEY (id)
);

-- DROP TABLE costos_ml.parametro;
CREATE TABLE costos_ml.parametro (
	id_parametro int4 NOT NULL,
	descripcion bpchar(50) NOT NULL,
	tipo bpchar(50) NOT NULL,
	CONSTRAINT pk_parametro PRIMARY KEY (id_parametro)
);

-- DROP TABLE costos_ml.presupuesto;
CREATE TABLE costos_ml.presupuesto (
	id uuid NOT NULL,
	cobertura varchar(255) NULL,
	entidades_participantes int4 NULL,
	estado varchar(255) NULL,
	fecha date NULL,
	inscritos_asesor int4 NULL,
	inscritos_asistencial int4 NULL,
	inscritos_profesional int4 NULL,
	inscritos_tecnico int4 NULL,
	justificacion varchar(255) NULL,
	porcentaje_financia_entidad float4 NULL,
	porcentaje_financia_pines float4 NULL,
	recaudo_participacion float8 NULL,
	templeos int4 NULL,
	total float8 NULL,
	total_por_inscritos float8 NULL,
	total_por_vacante float8 NULL,
	tpagosproyectados int4 NULL,
	vacantes int4 NULL,
	valor_entidad float8 NULL,
	valor_entidad_por_vacante float8 NULL,
	"version" int4 NOT NULL,
	id_costeo uuid NOT NULL,
	id_proceso_seleccion int8 NOT NULL,
	tpagosreales int4 NULL,
	urlrecurso varchar(255) NULL,
	urlresolucion varchar(255) NULL,
	CONSTRAINT pk_presupuesto PRIMARY KEY (id)
);

-- DROP TABLE costos_ml.procesoseleccion;
CREATE TABLE costos_ml.procesoseleccion (
	id int8 NOT NULL,
	nombre varchar(255) NOT NULL,
	CONSTRAINT pk_procesoseleccion PRIMARY KEY (id)
);

-- DROP TABLE costos_ml.procesoseleccion_presupuesto;
CREATE TABLE costos_ml.procesoseleccion_presupuesto (
	id int8 NOT NULL,
	nombre varchar(255) NOT NULL,
	CONSTRAINT pk_procesoseleccion_presupuesto PRIMARY KEY (id)
);

-- DROP TABLE costos_ml.seleccion_convocatoria;
CREATE TABLE costos_ml.seleccion_convocatoria (
	id int4 DEFAULT nextval('seq_seleccion_conv'::regclass) NOT NULL,
	convocatoria_id int4 NULL,
	nombre varchar(255) NULL,
	proyectar bool DEFAULT false NULL,
	ejecucion_id int4 NULL,
	usuario_creacion varchar(63) NULL,
	fecha_creacion timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	fecha_ejecucion timestamp NULL,
	CONSTRAINT pk_seleccion_convocatoria PRIMARY KEY (id)
);


------ MACHINE LEARNING -------------

-- DROP TABLE costos_ml.nn_empleo;
CREATE TABLE costos_ml.nn_empleo (
	ejecucion_id int4 NOT NULL,
	id int4 NOT NULL,
	mun_inscritos int4 NULL,
	mun_aprobo_vrm int4 NULL,
	mun_aprobo_escritas int4 NULL,
	empleo_id int4 NULL,
	concurso_ascenso bool NULL,
	identificador varchar(4) NULL,
	asignacion_salarial int8 NULL,
	vigencia_salarial int8 NULL,
	agno int4 NULL,
	smmlv float4 NULL,
	grado_nivel_id int8 NULL,
	nivelid int4 NULL,
	nivel varchar(32) NULL,
	grado varchar(4) NULL,
	denominacion varchar(128) NULL,
	conv_id int4 NULL,
	conv_nombre varchar(256) NULL,
	conv_agno int4 NULL,
	conv_estado varchar(1) NULL,
	conv_padre_id int4 NULL,
	conv_padre varchar(128) NULL,
	entidad varchar(256) NULL,
	nit varchar(16) NULL,
	tipo_entidad varchar(64) NULL,
	reqs_estudio varchar(128) NULL,
	nbc_tecnico varchar(2048) NULL,
	nbc_esp_tecnico varchar(256) NULL,
	nbc_tecnologico varchar(2048) NULL,
	nbc_esp_tecnologico varchar(1024) NULL,
	nbc_profesional varchar(2048) NULL,
	nbc_esp_profesional varchar(1024) NULL,
	nbc_maestria varchar(1024) NULL,
	nbc_doctorado varchar(256) NULL,
	departamento varchar(64) NULL,
	municipio varchar(32) NULL,
	codigo_dane varchar(16) NULL,
	mun_categoria varchar(1) NULL,
	vacantes_opec int4 NULL,
	vacantes_municipios int4 NULL,
	vacantes int8 NULL,
	criterio_id int4 NULL,
	experiencia varchar(1024) NULL,
	etiqueta bool DEFAULT false NULL,
	sin_experiencia bool NULL,
	exp_profesional int4 NULL,
	exp_prof_relacionada int4 NULL,
	exp_laboral int4 NULL,
	exp_labo_relacionada int4 NULL,
	exp_relacionada int4 NULL,
	fecha_creacion timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT pk_nn_empleo PRIMARY KEY (ejecucion_id, id)
);

-- DROP TABLE costos_ml.nn_proyeccion;
CREATE TABLE costos_ml.nn_proyeccion (
	id int8 NOT NULL,
	ejecucion_id int4 NULL,
	escenario_id int4 NULL,
	redneuronal_id int4 NULL,
	ascenso bool DEFAULT false NULL,
	nn_empleo_id int4 NULL,
	mun_inscritos int8 NULL,
	mun_aprobo_vrm int8 NULL,
	mun_aprobo_escritas int8 NULL,
	mun_pcd_inscritos int8 NULL,
	fecha_creacion timestamp NULL,
	CONSTRAINT pk_nn_proyeccion PRIMARY KEY (id)
);

-- DROP TABLE costos_ml.nn_simo;
CREATE TABLE costos_ml.nn_simo (
	empleo_id int4 NULL,
	concurso_ascenso bool NULL,
	identificador varchar(4) NULL,
	asignacion_salarial int8 NULL,
	vigencia_salarial int8 NULL,
	agno int8 NULL,
	smmlv float8 NULL,
	grado_nivel_id int8 NULL,
	nivelid int8 NULL,
	nivel varchar(32) NULL,
	grado int8 NULL,
	denominacion varchar(128) NULL,
	conv_id int8 NULL,
	conv_nombre varchar(256) NULL,
	conv_agno int8 NULL,
	conv_estado varchar(1) NULL,
	conv_proceso_id int4 NULL,
	conv_proceso varchar(64) NULL,
	conv_padre_id int8 NULL,
	conv_padre varchar(128) NULL,
	entidad varchar(256) NULL,
	nit varchar(16) NULL,
	tipo_entidad varchar(64) NULL,
	reqs_estudio varchar(128) NULL,
	nbc_tecnico varchar(2048) NULL,
	nbc_esp_tecnico varchar(128) NULL,
	nbc_tecnologico varchar(2048) NULL,
	nbc_esp_tecnologico varchar(1024) NULL,
	nbc_profesional varchar(2048) NULL,
	nbc_esp_profesional varchar(1024) NULL,
	nbc_maestria varchar(1024) NULL,
	nbc_doctorado varchar(64) NULL,
	departamento varchar(64) NULL,
	municipio varchar(32) NULL,
	codigo_dane varchar(16) NULL,
	vacantes_opec int4 NULL,
	vacantes_municipios int4 NULL,
	vacantes int4 NULL,
	inscritos int4 NULL,
	aprobo_vrm int4 NULL,
	aprobo_escritas int4 NULL,
	pcd_inscritos int8 NULL,
	mun_inscritos int4 NULL,
	mun_aprobo_vrm int4 NULL,
	mun_aprobo_escritas int4 NULL,
	mun_pcd_inscritos float8 NULL,
	pcd_psicosocial int8 NULL,
	pcd_intelectual int8 NULL,
	pcd_auditiva int8 NULL,
	pcd_visual int8 NULL,
	pcd_fisica int8 NULL,
	pcd_multiple int8 NULL,
	pcd_sordoceguera int8 NULL,
	mun_categoria varchar(1) NULL,
	criterio_id int8 NULL,
	etiqueta bool NULL,
	experiencia varchar(1024) NULL,
	sin_experiencia bool NULL,
	exp_profesional int4 NULL,
	exp_prof_relacionada int4 NULL,
	exp_laboral int4 NULL,
	exp_labo_relacionada int4 NULL,
	exp_relacionada int4 NULL,
	fecha_actualizacion timestamptz NULL
);

-- DROP TABLE costos_ml.nn_stats;
CREATE TABLE costos_ml.nn_stats (
	ejecucion_id int4 NOT NULL,
	escenario_id int4 NOT NULL,
	redneuronal_id int4 NOT NULL,
	ascenso bool DEFAULT false NOT NULL,
	r2 float8 NULL,
	mse float8 NULL,
	rmse float8 NULL,
	mae float8 NULL,
	porc_rm float8 NULL,
	porc_escritas float8 NULL,
	porc_pcd float8 NULL,
	elegida bool NULL,
	ranking int4 NULL,
	fecha_creacion timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT pk_nn_stats PRIMARY KEY (ejecucion_id, escenario_id, redneuronal_id, ascenso)
);

-- DROP TABLE costos_ml.np_ejecucion;
CREATE TABLE costos_ml.np_ejecucion (
	id int4 NOT NULL,
	hostname varchar(31) NULL,
	ip varchar(31) NULL,
	filtro varchar(3000) NULL,
	escenarios varchar(8000) NULL,
	variables varchar(1000) NULL,
	redneuronal varchar(2000) NULL,
	ruta varchar(256) NULL,
	fecha_creacion timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT pk_np_ejecucion PRIMARY KEY (id)
);

-- DROP TABLE costos_ml.np_escenas;
CREATE TABLE costos_ml.np_escenas (
	id int4 DEFAULT nextval('seq_np_escenas'::regclass) NOT NULL,
	nombre varchar(127) NULL,
	activo bool DEFAULT true NULL,
	divide_ascenso bool DEFAULT false NULL,
	fecha_creacion timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT pk_np_escenas PRIMARY KEY (id)
);


-- DROP TABLE costos_ml.np_escenas_detalle;
CREATE TABLE costos_ml.np_escenas_detalle (
	id int4 DEFAULT nextval('seq_np_escenas_detalle'::regclass) NOT NULL,
	escena_id int4 NULL,
	campo varchar(127) NULL,
	operador varchar(15) NULL,
	valor varchar(1023) NULL,
	concatena varchar(15) NULL,
	activo bool DEFAULT true NULL,
	percmin float4 DEFAULT 0 NULL,
	percmax float4 DEFAULT 100 NULL,
	fecha_creacion timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT pk_np_escenas_detalle PRIMARY KEY (id)
);


-- DROP TABLE costos_ml.np_parametros;
CREATE TABLE costos_ml.np_parametros (
	id int4 DEFAULT nextval('seq_np_parametros'::regclass) NOT NULL,
	tipo varchar(16) NOT NULL,
	valor varchar(128) NULL,
	activo bool DEFAULT true NULL,
	descripcion varchar(128) NULL,
	CONSTRAINT pk_np_parametros PRIMARY KEY (id)
);


-- DROP TABLE costos_ml.np_redneuronal;
CREATE TABLE costos_ml.np_redneuronal (
	id int4 DEFAULT nextval('seq_np_redneuronal'::regclass) NOT NULL,
	nombre varchar(120) NULL,
	neuronas int4 DEFAULT 0 NULL,
	learnrate float8 NULL,
	dropout float8 DEFAULT 0 NULL,
	epocas int4 NULL,
	batch int4 NULL,
	activo bool NULL,
	fecha_creacion timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT pk_np_redneuronal PRIMARY KEY (id)
);

-- DROP TABLE costos_ml.np_variables;
CREATE TABLE costos_ml.np_variables (
	id int4 DEFAULT nextval('seq_np_variables'::regclass) NOT NULL,
	nombre varchar(127) NOT NULL,
	tipo varchar(31) NULL,
	activo bool NULL,
	no_unico bool NULL,
	fecha_creacion timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT ck_np_variables_tipo CHECK (((tipo)::text = ANY (ARRAY[('num'::character varying)::text, ('str'::character varying)::text, ('bool'::character varying)::text, ('list'::character varying)::text]))),
	CONSTRAINT pk_np_variables PRIMARY KEY (id)
);



--- PARA HOMOLOGAR CON MDM -------------

-- DROP TABLE costos_ml.grado_nivel;
CREATE TABLE costos_ml.grado_nivel (
	id int4 NOT NULL,
	grado int4 NULL,
	nivel_id int4 NULL,
	CONSTRAINT pk_grado_nivel PRIMARY KEY (id)
);


-- DROP TABLE costos_ml.grado_salario;
CREATE TABLE costos_ml.grado_salario (
	id int4 DEFAULT nextval('seq_grado_salario'::regclass) NULL,
	grado_nivel int4 NOT NULL,
	agno int4 NOT NULL,
	salario int4 NULL,
	CONSTRAINT grado_salario_pkey PRIMARY KEY (grado_nivel, agno)
);

-- DROP TABLE costos_ml.lugar;
CREATE TABLE costos_ml.lugar (
	id int4 NOT NULL,
	nombre varchar(255) NULL,
	tipo_lugar_id int4 NULL,
	codigo varchar(15) NULL,
	id_padre int4 NULL,
	dpto_capital_id int4 NULL,
	municipio_categoria_id int4 NULL,
	municipio_pdet bool NULL,
	sigla varchar(4) NULL,
	latitud float4 NULL,
	longitud float4 NULL,
	estado bool NULL,
	old_lugar_id int4 NULL,
	old_lugar_id_padre int4 NULL,
	CONSTRAINT pk_lugar PRIMARY KEY (id)
);

-- DROP TABLE costos_ml.salario_minimo;
CREATE TABLE costos_ml.salario_minimo (
	agno int4 NOT NULL,
	agno_smmlv int4 NOT NULL,
	CONSTRAINT pk_salario_minimo PRIMARY KEY (agno)
);


-- CONSTRAINTS

ALTER TABLE costos_ml.componente_presupuesto ADD CONSTRAINT fk_componente_pres_presupuesto FOREIGN KEY (id_presupuesto) REFERENCES costos_ml.presupuesto(id);
ALTER TABLE costos_ml.componente_proyeccion ADD CONSTRAINT fk_componente_proy_costeo_proyeccion FOREIGN KEY (id_costeo) REFERENCES costos_ml.costeo_proyeccion(id);
ALTER TABLE costos_ml.componente_proyeccion_costos_variable ADD CONSTRAINT fk_componente_pcv_componente_proyeccion FOREIGN KEY (componente_proyeccion_id) REFERENCES costos_ml.componente_proyeccion(id);
ALTER TABLE costos_ml.componente_proyeccion_costos_variable ADD CONSTRAINT fk_componente_pcv_item_proyeccion FOREIGN KEY (costos_variable_id) REFERENCES costos_ml.item_proyeccion(id);
ALTER TABLE costos_ml.componente_proyeccion_datos_entrada ADD CONSTRAINT fk_componente_pde_componente_proyeccion FOREIGN KEY (componente_proyeccion_id) REFERENCES costos_ml.componente_proyeccion(id);
ALTER TABLE costos_ml.componente_proyeccion_datos_entrada ADD CONSTRAINT fk_componente_pde_item_proyeccion FOREIGN KEY (datos_entrada_id) REFERENCES costos_ml.item_proyeccion(id);
ALTER TABLE costos_ml.componente_proyeccion_datos_entrada_formulados ADD CONSTRAINT fk_componente_pdef_componente_proyeccion FOREIGN KEY (componente_proyeccion_id) REFERENCES costos_ml.componente_proyeccion(id);
ALTER TABLE costos_ml.componente_proyeccion_datos_entrada_formulados ADD CONSTRAINT fk_componente_pdef_item_proyeccion FOREIGN KEY (datos_entrada_formulados_id) REFERENCES costos_ml.item_proyeccion(id);
ALTER TABLE costos_ml.costeo_proyeccion ADD CONSTRAINT fk_costeo_proyeccion_procesoseleccion FOREIGN KEY (id_proceso_seleccion) REFERENCES costos_ml.procesoseleccion(id);
ALTER TABLE costos_ml.estado_proyeccion ADD CONSTRAINT fk_estado_proy_costeo_proyeccion FOREIGN KEY (id_costeo) REFERENCES costos_ml.costeo_proyeccion(id);
ALTER TABLE costos_ml.etapa_item ADD CONSTRAINT fk_etapa_item_etapa FOREIGN KEY (id_etapa) REFERENCES costos_ml.etapa(id_etapa);
ALTER TABLE costos_ml.etapa_item ADD CONSTRAINT fk_etapa_item_item FOREIGN KEY (id_item) REFERENCES costos_ml.item(id_item);
ALTER TABLE costos_ml.item ADD CONSTRAINT fk_item_tipo_parametro FOREIGN KEY (tipo) REFERENCES costos_ml.parametro(id_parametro);
ALTER TABLE costos_ml.item ADD CONSTRAINT fk_item_unidad_parametro FOREIGN KEY (unidad) REFERENCES costos_ml.parametro(id_parametro);
ALTER TABLE costos_ml.presupuesto ADD CONSTRAINT fk_presupuesto_costeo_proyeccion FOREIGN KEY (id_costeo) REFERENCES costos_ml.costeo_proyeccion(id);
ALTER TABLE costos_ml.presupuesto ADD CONSTRAINT fk_presupuesto_procesoseleccion_presupuesto FOREIGN KEY (id_proceso_seleccion) REFERENCES costos_ml.procesoseleccion_presupuesto(id);

ALTER TABLE costos_ml.grado_salario ADD CONSTRAINT fk_grado_salario_nivel FOREIGN KEY (grado_nivel) REFERENCES costos_ml.grado_nivel(id);
ALTER TABLE costos_ml.nn_empleo ADD CONSTRAINT fk_nn_empleo_np_ejecucion FOREIGN KEY (ejecucion_id) REFERENCES costos_ml.np_ejecucion(id);
ALTER TABLE costos_ml.nn_proyeccion ADD CONSTRAINT fk_nn_proyeccion_nn_empleo FOREIGN KEY (ejecucion_id,nn_empleo_id) REFERENCES costos_ml.nn_empleo(ejecucion_id,id);
ALTER TABLE costos_ml.nn_proyeccion ADD CONSTRAINT fk_nn_proyeccion_np_escenas FOREIGN KEY (escenario_id) REFERENCES costos_ml.np_escenas(id);
ALTER TABLE costos_ml.nn_proyeccion ADD CONSTRAINT fk_nn_proyeccion_np_redneuronal FOREIGN KEY (redneuronal_id) REFERENCES costos_ml.np_redneuronal(id);
ALTER TABLE costos_ml.nn_stats ADD CONSTRAINT fk_stats_ejecucion FOREIGN KEY (ejecucion_id) REFERENCES costos_ml.np_ejecucion(id);
ALTER TABLE costos_ml.nn_stats ADD CONSTRAINT fk_stats_escena FOREIGN KEY (escenario_id) REFERENCES costos_ml.np_escenas(id);
ALTER TABLE costos_ml.nn_stats ADD CONSTRAINT fk_stats_redneuronal FOREIGN KEY (redneuronal_id) REFERENCES costos_ml.np_redneuronal(id);
ALTER TABLE costos_ml.np_escenas_detalle ADD CONSTRAINT fk_np_escenas_detalle FOREIGN KEY (escena_id) REFERENCES costos_ml.np_escenas(id);
ALTER TABLE costos_ml.seleccion_convocatoria ADD CONSTRAINT fk_seleccion_conv_ejecucion FOREIGN KEY (ejecucion_id) REFERENCES costos_ml.np_ejecucion(id);


--- FUNCIONES -------- 

-- DROP FUNCTION costos_ml.eliminar_ejecuciones(_int4);
CREATE OR REPLACE FUNCTION costos_ml.eliminar_ejecuciones(ids integer[])
 RETURNS void
 LANGUAGE plpgsql
AS $function$
BEGIN
  UPDATE costos_ml.seleccion_convocatoria 
  SET ejecucion_id = NULL, proyectar = FALSE 
  WHERE ejecucion_id = ANY(ids);

  DELETE FROM costos_ml.nn_stats WHERE ejecucion_id = ANY(ids);
  DELETE FROM costos_ml.nn_proyeccion WHERE ejecucion_id = ANY(ids);
  DELETE FROM costos_ml.nn_empleo WHERE ejecucion_id = ANY(ids);
  DELETE FROM costos_ml.np_ejecucion WHERE id = ANY(ids);
END;
$function$
;
-- Permissions
ALTER FUNCTION costos_ml.eliminar_ejecuciones(_int4) OWNER TO usrcosteo;
GRANT ALL ON FUNCTION costos_ml.eliminar_ejecuciones(_int4) TO usrcosteo;



--- VISTAS ------

--DROP VIEW costos_ml.nn_simo_unico;
CREATE OR REPLACE VIEW costos_ml.nn_simo_unico
AS SELECT DISTINCT nn_simo.empleo_id,
    nn_simo.identificador,
    nn_simo.asignacion_salarial,
    nn_simo.vigencia_salarial,
    nn_simo.concurso_ascenso,
    nn_simo.grado_nivel_id,
    nn_simo.nivelid,
    nn_simo.nivel,
    nn_simo.grado,
    nn_simo.denominacion,
    nn_simo.conv_id,
    nn_simo.conv_nombre,
    nn_simo.conv_agno,
    nn_simo.conv_estado,
    nn_simo.conv_padre_id,
    nn_simo.conv_padre,
    nn_simo.entidad,
    nn_simo.nit,
    nn_simo.tipo_entidad,
    nn_simo.reqs_estudio,
    nn_simo.nbc_tecnico,
    nn_simo.nbc_esp_tecnico,
    nn_simo.nbc_tecnologico,
    nn_simo.nbc_esp_tecnologico,
    nn_simo.nbc_profesional,
    nn_simo.nbc_esp_profesional,
    nn_simo.nbc_maestria,
    nn_simo.nbc_doctorado,
    nn_simo.vacantes_opec,
    nn_simo.vacantes_municipios,
    nn_simo.inscritos,
    nn_simo.aprobo_vrm,
    nn_simo.aprobo_escritas,
    nn_simo.pcd_inscritos,
    nn_simo.pcd_psicosocial,
    nn_simo.pcd_intelectual,
    nn_simo.pcd_auditiva,
    nn_simo.pcd_visual,
    nn_simo.pcd_fisica,
    nn_simo.pcd_multiple,
    nn_simo.pcd_sordoceguera,
    nn_simo.criterio_id,
    nn_simo.experiencia,
    nn_simo.exp_profesional,
    nn_simo.exp_prof_relacionada,
    nn_simo.exp_laboral,
    nn_simo.exp_labo_relacionada,
    nn_simo.exp_relacionada,
    nn_simo.etiqueta,
    nn_simo.sin_experiencia,
    nn_simo.agno,
    nn_simo.smmlv
   FROM costos_ml.nn_simo;
-- Permissions
ALTER TABLE costos_ml.nn_simo_unico OWNER TO usrcosteo;
GRANT ALL ON TABLE costos_ml.nn_simo_unico TO usrcosteo;


--DROP VIEW costos_ml.vw_detalle_componentes;
CREATE OR REPLACE VIEW costos_ml.vw_detalle_componentes
AS SELECT pit.id_item,
    pit.tipo AS id_tipo,
    pptc.descripcion AS tipo,
    COALESCE(pit.unidad::character varying, ''::character varying) AS id_unidad,
    COALESCE(ppuc.descripcion, ''::bpchar) AS unidad,
    pit.nombre,
    pit.porcentual,
    COALESCE(pit.valor, 0::double precision) AS valor,
    pit.fecha,
    pit.variable,
	CASE
		WHEN pit.formula IS NULL THEN 'Sin formula registrada'::character varying
		ELSE pit.formula
	END AS formula,
	CASE
		WHEN pit.urlrecurso IS NULL THEN 'Sin Url registrada'::character varying
		ELSE pit.urlrecurso
	END AS url_recurso,
	CASE
		WHEN pec.nombre IS NULL THEN 'Sin Componente Asociado'::bpchar
		ELSE pec.nombre
	END AS componente,
	CASE
		WHEN pec.estado THEN 'Activa'::text
		WHEN pec.estado IS FALSE THEN 'No Activo'::text
		WHEN pec.estado IS NULL THEN 'No Aplica'::text
		ELSE NULL::text
	END AS estado
   FROM costos_ml.item pit
     LEFT JOIN costos_ml.etapa_item peic ON pit.id_item = peic.id_item
     LEFT JOIN costos_ml.etapa pec ON peic.id_etapa = pec.id_etapa
     LEFT JOIN costos_ml.parametro pptc ON pit.tipo = pptc.id_parametro
     LEFT JOIN costos_ml.parametro ppuc ON pit.unidad = ppuc.id_parametro
  ORDER BY pit.nombre;

-- Permissions
ALTER TABLE costos_ml.vw_detalle_componentes OWNER TO usrcosteo;
GRANT ALL ON TABLE costos_ml.vw_detalle_componentes TO usrcosteo;

--DROP VIEW costos_ml.vw_etapa_detalle;
CREATE OR REPLACE VIEW costos_ml.vw_etapa_detalle AS 
SELECT pec.id_etapa,
pec.tipo AS tipo_etapa,
pec.nombre AS nombre_etapa,
pec.fecha AS fecha_etapa,
pec.variable,
CASE pec.estado WHEN true THEN 'Activa'::text WHEN false THEN 'No Activo'::text ELSE NULL::text END AS activa,
pic.nombre AS tipo_item,
pptc.descripcion AS descripcion_tipo_item,
pic.unidad,
ppuc.descripcion AS descripcion_unidad,
pic.valor AS "Valor",
pic.variable AS variable_item,
pic.formula,
pic.urlrecurso AS url_recurso
FROM costos_ml.etapa pec
JOIN costos_ml.etapa_item peic ON pec.id_etapa = peic.id_etapa
JOIN costos_ml.item pic ON peic.id_item = peic.id_item
LEFT JOIN costos_ml.parametro pptc ON pic.tipo = pptc.id_parametro
LEFT JOIN costos_ml.parametro ppuc ON pic.unidad = ppuc.id_parametro
ORDER BY pec.tipo, pec.fecha;

-- Permissions
ALTER TABLE costos_ml.vw_etapa_detalle OWNER TO usrcosteo;
GRANT ALL ON TABLE costos_ml.vw_etapa_detalle TO usrcosteo;


-- DROP VIEW IF EXISTS costos_ml.vw_inscritos_nivel;

CREATE OR REPLACE VIEW costos_ml.vw_inscritos_nivel AS 
SELECT DISTINCT ON (st.ejecucion_id, tt.escenario_id, tt.redneuronal_id)
st.ejecucion_id,
sc.convocatoria_id,
sc.nombre AS convocatoria,
sc.id AS seleccion_conv_id,
tt.escenario_id,
tt.escenario,
tt.escenario_detalle,
tt.redneuronal_id,
redn.red,
redn.red_detalle,
CASE WHEN st.ranking = 1 THEN TRUE ELSE FALSE END AS elegida,
st.ranking,
tt.vacantes,
tt.inscritos,
tt.aprobo_vrm,
tt.aprobo_escritas,
tt.pcd_inscritos,
tt.vac_ascenso,
tt.ins_ascenso,
tt.vac_ingreso,
tt.ins_ingreso,
tt.vac_asesor,
tt.ins_asesor,
tt.vac_profesional,
tt.ins_profesional,
tt.vac_tecnico,
tt.ins_tecnico,
tt.vac_asistencial,
tt.ins_asistencial,
tt.vac_otros_nivel1,
tt.ins_otros_nivel1,
tt.vac_otros_nivel2,
tt.ins_otros_nivel2,
st.fecha_creacion
FROM costos_ml.nn_stats st
LEFT JOIN costos_ml.seleccion_convocatoria sc ON st.ejecucion_id = sc.ejecucion_id
LEFT JOIN (
 	SELECT 
 	js.ejecucion_id,
    js.datos ->> 'id'::text AS red_id,
    btrim(js.datos ->> 'nombre'::text) AS red,
    (((((((
    	CASE WHEN (js.datos ->> 'neuronas'::text) = '0'::text THEN 'ArqBinaria'::text
            ELSE 'neuronas:'::text || btrim(js.datos ->> 'neuronas'::text) END 
		|| '|lr:'::text) || btrim(js.datos ->> 'learnrate'::text)) || '|dropout:'::text) 
		|| btrim(js.datos ->> 'dropout'::text)) || '|epocas:'::text) || btrim(js.datos ->> 'epocas'::text)) 
		|| '|batch:'::text) || btrim(js.datos ->> 'batch'::text) AS red_detalle
    FROM (
       SELECT
       np_ejecucion.id AS ejecucion_id,
       json_array_elements(np_ejecucion.redneuronal::json) AS datos
       FROM costos_ml.np_ejecucion
       ) js
	) redn ON st.ejecucion_id = redn.ejecucion_id AND st.redneuronal_id = redn.red_id::integer
JOIN (
	SELECT 
	pr.ejecucion_id,
	esc.escenario,
	esc.escenario_detalle,
    pr.escenario_id,
    pr.redneuronal_id,
    COALESCE(sum(em.vacantes), 0::numeric) AS vacantes,
    COALESCE(sum(pr.mun_inscritos), 0::numeric) AS inscritos,
    COALESCE(sum(pr.mun_aprobo_vrm), 0::numeric) AS aprobo_vrm,
    COALESCE(sum(pr.mun_aprobo_escritas), 0::numeric) AS aprobo_escritas,
    COALESCE(sum(pr.mun_pcd_inscritos), 0::numeric) AS pcd_inscritos,
    COALESCE(sum( CASE WHEN em.concurso_ascenso THEN em.vacantes ELSE NULL::bigint END), 0::numeric) AS vac_ascenso,
    COALESCE(sum( CASE WHEN em.concurso_ascenso THEN pr.mun_inscritos ELSE NULL::bigint END), 0::numeric) AS ins_ascenso,
    COALESCE(sum( CASE WHEN NOT em.concurso_ascenso THEN em.vacantes ELSE NULL::bigint END), 0::numeric) AS vac_ingreso,
    COALESCE(sum( CASE WHEN NOT em.concurso_ascenso THEN pr.mun_inscritos ELSE NULL::bigint END), 0::numeric) AS ins_ingreso,
    COALESCE(sum( CASE WHEN em.nivelid = 2 THEN em.vacantes ELSE NULL::bigint END), 0::numeric) AS vac_asesor,
    COALESCE(sum( CASE WHEN em.nivelid = 2 THEN pr.mun_inscritos ELSE NULL::bigint END), 0::numeric) AS ins_asesor,
    COALESCE(sum( CASE WHEN em.nivelid = ANY (ARRAY[3, 15, 17]) THEN em.vacantes ELSE NULL::bigint END), 0::numeric) AS vac_profesional,
    COALESCE(sum( CASE WHEN em.nivelid = ANY (ARRAY[3, 15, 17]) THEN pr.mun_inscritos ELSE NULL::bigint END), 0::numeric) AS ins_profesional,
    COALESCE(sum( CASE WHEN em.nivelid = ANY (ARRAY[4, 19]) THEN em.vacantes ELSE NULL::bigint END), 0::numeric) AS vac_tecnico,
    COALESCE(sum( CASE WHEN em.nivelid = ANY (ARRAY[4, 19]) THEN pr.mun_inscritos ELSE NULL::bigint END), 0::numeric) AS ins_tecnico,
    COALESCE(sum( CASE WHEN em.nivelid = ANY (ARRAY[5, 13, 16]) THEN em.vacantes ELSE NULL::bigint END), 0::numeric) AS vac_asistencial,
    COALESCE(sum( CASE WHEN em.nivelid = ANY (ARRAY[5, 13, 16]) THEN pr.mun_inscritos ELSE NULL::bigint END), 0::numeric) AS ins_asistencial,
    COALESCE(sum( CASE WHEN em.nivelid = ANY (ARRAY[10, 21]) THEN em.vacantes ELSE NULL::bigint END), 0::numeric) AS vac_otros_nivel1,
    COALESCE(sum( CASE WHEN em.nivelid = ANY (ARRAY[10, 21]) THEN pr.mun_inscritos ELSE NULL::bigint END), 0::numeric) AS ins_otros_nivel1,
    COALESCE(sum( CASE WHEN em.nivelid = ANY (ARRAY[1, 6, 7, 9, 11, 12]) THEN em.vacantes ELSE NULL::bigint END), 0::numeric) AS vac_otros_nivel2,
    COALESCE(sum( CASE WHEN em.nivelid = ANY (ARRAY[1, 6, 7, 9, 11, 12]) THEN pr.mun_inscritos ELSE NULL::bigint END), 0::numeric) AS ins_otros_nivel2
    FROM costos_ml.nn_proyeccion pr
    LEFT JOIN (
			SELECT 
			e.ejecucion_id,
			e.escena AS escenario,
			e.escena_id,
			e.ascenso,
			string_agg( concat_ws( ' ', e.campo, e.operador, e.valor, e.concatena ),  ' ' ORDER BY e.concatena DESC, e.ord ) AS escenario_detalle
			FROM (
			    SELECT ne.id AS ejecucion_id, trim(r->>'nombre') AS escena, trim(r->>'escena')::int4 AS escena_id, 
			    trim(r->>'campo') AS campo, trim(r->>'operador') AS operador, trim(r->>'valor') AS valor, 
			    trim(r->>'concatena') AS concatena, trim(r->>'ord')::int4 AS ord,
			    COALESCE(trim(r->>'divide_ascenso')::bool,FALSE) AS ascenso
			    FROM costos_ml.np_ejecucion ne,
			    json_array_elements(ne.escenarios::json) AS r
			) e 
			GROUP BY 1,2,3,4
		) esc ON pr.ejecucion_id = esc.ejecucion_id AND pr.escenario_id = esc.escena_id
    JOIN costos_ml.nn_empleo em ON pr.nn_empleo_id = em.id AND pr.ejecucion_id = em.ejecucion_id AND CASE WHEN esc.ascenso THEN pr.ascenso = em.concurso_ascenso ELSE TRUE END
    GROUP BY pr.ejecucion_id, pr.escenario_id, pr.redneuronal_id, esc.escenario, esc.escenario_detalle
  ) tt ON st.ejecucion_id = tt.ejecucion_id AND st.escenario_id = tt.escenario_id AND st.redneuronal_id = tt.redneuronal_id
  ORDER BY st.ejecucion_id DESC, tt.escenario_id, tt.redneuronal_id, st.ranking;

-- Permissions
ALTER TABLE costos_ml.vw_inscritos_nivel OWNER TO usrcosteo;
GRANT ALL ON TABLE costos_ml.vw_inscritos_nivel TO usrcosteo;


--DROP VIEW costos_ml.municipio;
CREATE OR REPLACE VIEW costos_ml.municipio AS 
SELECT 
lp.id AS pais_id,
lp.codigo AS pais_codigo,
lp.nombre AS pais_nombre,
lp.sigla AS pais_sigla,
ld.id AS dpto_id,
ld.codigo AS dpto_codigo,
ld.nombre AS dpto_nombre,
lm.id,
lm.codigo,
lm.nombre,
'3-Municipio'::text AS tipo_lugar,
lm.latitud,
lm.longitud,
lm.municipio_pdet,
lm.municipio_categoria_id,
CASE WHEN lm.estado IS TRUE THEN 'Activo'::text WHEN lm.estado IS FALSE THEN 'Inactivo'::text ELSE 'Reg. Sin Estado'::text END AS estado
FROM costos_ml.lugar lp
LEFT JOIN costos_ml.lugar ld ON lp.id = ld.id_padre
LEFT JOIN costos_ml.lugar lm ON ld.id = lm.id_padre
WHERE lm.tipo_lugar_id = 3;
-- Permissions
ALTER TABLE costos_ml.municipio OWNER TO usrcosteo;
GRANT ALL ON TABLE costos_ml.municipio TO usrcosteo;

--DROP VIEW costos_ml.departamento;
CREATE OR REPLACE VIEW costos_ml.departamento AS 
SELECT 
de.id,
de.id_padre AS pais_id,
pa.nombre AS pais_nombre,
de.codigo,
de.nombre,
'2-Departamento'::text AS lugar_tipo,
de.dpto_capital_id,
mu.codigo AS capital_codigo,
mu.nombre AS capital_nombre,
CASE WHEN de.estado IS TRUE THEN 'Activo'::text WHEN de.estado IS FALSE THEN 'Inactivo'::text ELSE 'Reg. Sin Estado'::text END AS estado
FROM costos_ml.lugar de
LEFT JOIN costos_ml.lugar pa ON de.id_padre = pa.id
LEFT JOIN costos_ml.lugar mu ON de.dpto_capital_id = mu.id
WHERE de.tipo_lugar_id = 2
ORDER BY de.id;

-- Permissions
ALTER TABLE costos_ml.departamento OWNER TO usrcosteo;
GRANT ALL ON TABLE costos_ml.departamento TO usrcosteo;


--- PERMISOS -------------

ALTER TABLE costos_ml.componente_presupuesto OWNER TO usrcosteo;
ALTER TABLE costos_ml.componente_proyeccion OWNER TO usrcosteo;
ALTER TABLE costos_ml.componente_proyeccion_costos_variable OWNER TO usrcosteo;
ALTER TABLE costos_ml.componente_proyeccion_datos_entrada OWNER TO usrcosteo;
ALTER TABLE costos_ml.componente_proyeccion_datos_entrada_formulados OWNER TO usrcosteo;
ALTER TABLE costos_ml.costeo_proyeccion OWNER TO usrcosteo;
ALTER TABLE costos_ml.estado_proyeccion OWNER TO usrcosteo;
ALTER TABLE costos_ml.etapa OWNER TO usrcosteo;
ALTER TABLE costos_ml.etapa_item OWNER TO usrcosteo;
ALTER TABLE costos_ml.grado_nivel OWNER TO usrcosteo;
ALTER TABLE costos_ml.grado_salario OWNER TO usrcosteo;
ALTER TABLE costos_ml.item OWNER TO usrcosteo;
ALTER TABLE costos_ml.item_proyeccion OWNER TO usrcosteo;
ALTER TABLE costos_ml.lugar OWNER TO usrcosteo;
ALTER TABLE costos_ml.nn_empleo OWNER TO usrcosteo;
ALTER TABLE costos_ml.nn_proyeccion OWNER TO usrcosteo;
ALTER TABLE costos_ml.nn_simo OWNER TO usrcosteo;
ALTER TABLE costos_ml.nn_stats OWNER TO usrcosteo;
ALTER TABLE costos_ml.np_ejecucion OWNER TO usrcosteo;
ALTER TABLE costos_ml.np_escenas OWNER TO usrcosteo;
ALTER TABLE costos_ml.np_escenas_detalle OWNER TO usrcosteo;
ALTER TABLE costos_ml.np_parametros OWNER TO usrcosteo;
ALTER TABLE costos_ml.np_redneuronal OWNER TO usrcosteo;
ALTER TABLE costos_ml.np_variables OWNER TO usrcosteo;
ALTER TABLE costos_ml.parametro OWNER TO usrcosteo;
ALTER TABLE costos_ml.presupuesto OWNER TO usrcosteo;
ALTER TABLE costos_ml.procesoseleccion OWNER TO usrcosteo;
ALTER TABLE costos_ml.procesoseleccion_presupuesto OWNER TO usrcosteo;
ALTER TABLE costos_ml.salario_minimo OWNER TO usrcosteo;
ALTER TABLE costos_ml.seleccion_convocatoria OWNER TO usrcosteo;

GRANT SELECT ON TABLE costos_ml.componente_presupuesto TO lmontoya;
GRANT SELECT ON TABLE costos_ml.componente_proyeccion TO lmontoya;
GRANT SELECT ON TABLE costos_ml.componente_proyeccion_costos_variable TO lmontoya;
GRANT SELECT ON TABLE costos_ml.componente_proyeccion_datos_entrada TO lmontoya;
GRANT SELECT ON TABLE costos_ml.componente_proyeccion_datos_entrada_formulados TO lmontoya;
GRANT SELECT ON TABLE costos_ml.costeo_proyeccion TO lmontoya;
GRANT SELECT ON TABLE costos_ml.estado_proyeccion TO lmontoya;
GRANT SELECT ON TABLE costos_ml.etapa TO lmontoya;
GRANT SELECT ON TABLE costos_ml.etapa_item TO lmontoya;
GRANT SELECT ON TABLE costos_ml.grado_nivel TO lmontoya;
GRANT SELECT ON TABLE costos_ml.grado_salario TO lmontoya;
GRANT SELECT ON TABLE costos_ml.item TO lmontoya;
GRANT SELECT ON TABLE costos_ml.item_proyeccion TO lmontoya;
GRANT SELECT ON TABLE costos_ml.lugar TO lmontoya;
GRANT SELECT ON TABLE costos_ml.nn_empleo TO lmontoya;
GRANT SELECT ON TABLE costos_ml.nn_proyeccion TO lmontoya;
GRANT SELECT ON TABLE costos_ml.nn_simo TO lmontoya;
GRANT SELECT ON TABLE costos_ml.nn_stats TO lmontoya;
GRANT SELECT ON TABLE costos_ml.np_ejecucion TO lmontoya;
GRANT SELECT ON TABLE costos_ml.np_escenas TO lmontoya;
GRANT SELECT ON TABLE costos_ml.np_escenas_detalle TO lmontoya;
GRANT SELECT ON TABLE costos_ml.np_parametros TO lmontoya;
GRANT SELECT ON TABLE costos_ml.np_redneuronal TO lmontoya;
GRANT SELECT ON TABLE costos_ml.np_variables TO lmontoya;
GRANT SELECT ON TABLE costos_ml.parametro TO lmontoya;
GRANT SELECT ON TABLE costos_ml.presupuesto TO lmontoya;
GRANT SELECT ON TABLE costos_ml.procesoseleccion TO lmontoya;
GRANT SELECT ON TABLE costos_ml.procesoseleccion_presupuesto TO lmontoya;
GRANT SELECT ON TABLE costos_ml.salario_minimo TO lmontoya;
GRANT SELECT ON TABLE costos_ml.seleccion_convocatoria TO lmontoya;

GRANT SELECT ON TABLE costos_ml.componente_presupuesto TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.componente_proyeccion TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.componente_proyeccion_costos_variable TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.componente_proyeccion_datos_entrada TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.componente_proyeccion_datos_entrada_formulados TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.costeo_proyeccion TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.estado_proyeccion TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.etapa TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.etapa_item TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.grado_nivel TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.grado_salario TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.item TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.item_proyeccion TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.lugar TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.nn_empleo TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.nn_proyeccion TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.nn_simo TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.nn_stats TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.np_ejecucion TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.np_escenas TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.np_escenas_detalle TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.np_parametros TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.np_redneuronal TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.np_variables TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.parametro TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.presupuesto TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.procesoseleccion TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.procesoseleccion_presupuesto TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.salario_minimo TO jlmartinez;
GRANT SELECT ON TABLE costos_ml.seleccion_convocatoria TO jlmartinez;