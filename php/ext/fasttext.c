#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "libfasttext.h"
#include "php.h"
#include "php_ini.h"
#include "ext/standard/info.h"
#include "php_fasttext.h"
#include "main/SAPI.h"


#include "zend_exceptions.h"
#include "zend_interfaces.h"
#include "SAPI.h"

ZEND_DECLARE_MODULE_GLOBALS(fasttext)

/* {{{ PHP_INI
*/
PHP_INI_BEGIN()
	STD_PHP_INI_ENTRY("fasttext.model_dir",  NULL, PHP_INI_SYSTEM, OnUpdateString, model_dir, zend_fasttext_globals, fasttext_globals)
PHP_INI_END()
/* }}} */

/* Handlers */
static zend_object_handlers fasttext_object_handlers;

/* Class entries */
zend_class_entry *php_fasttext_sc_entry;


static void fasttext_vec_add_array(zval *retval, int idx, FTReal val) /* {{{ */
{
	zval tmp;
	zend_string *str = zend_strpprintf(0, "%f", val);
	ZVAL_STR(&tmp, str);
	add_index_zval(retval, idx, &tmp);
}
/* }}} */

static void fasttext_value_add_array(zval *retval, int idx, const char *label, const char *prob) /* {{{ */
{
	zval tmp;

	array_init(&tmp);
	add_assoc_string(&tmp, "label", (char *)label);
	add_assoc_string(&tmp, "prob", (char *)prob);

	add_index_zval(retval, idx, &tmp);
}
/* }}} */


/* {{{ proto void fasttext::__construct()
 */
PHP_METHOD(fasttext, __construct)
{
	php_fasttext_object *ft_obj;
	zval *object = getThis();

	ft_obj = Z_FASTTEXT_P(object);

	if (zend_parse_parameters_none() == FAILURE) {
		return;
	}

	ft_obj->fasttext = FastTextCreate();
}
/* }}} */

/* {{{ proto long fasttext::load(String filename)
 */
PHP_METHOD(fasttext, load)
{
	php_fasttext_object *ft_obj;
	zval *object = getThis();
	char *filename;
	size_t filename_len;
	zend_long res;

	ft_obj = Z_FASTTEXT_P(object);

	if (FAILURE == zend_parse_parameters_throw(ZEND_NUM_ARGS(), "s", &filename, &filename_len)) {
		return;
	}
	res = FastTextLoadModel(ft_obj->fasttext, filename);

	RETURN_LONG(res);
}
/* }}} */

/* {{{ proto long fasttext::getWordRows()
 */
PHP_METHOD(fasttext, getWordRows)
{
	php_fasttext_object *ft_obj;
	zval *object = getThis();
	zend_long res;

	ft_obj = Z_FASTTEXT_P(object);

	if (zend_parse_parameters_none() == FAILURE) {
		return;
	}
	res = FastTextWordRows(ft_obj->fasttext);

	RETURN_LONG(res);
}
/* }}} */

/* {{{ proto long fasttext::getWordRows()
 */
PHP_METHOD(fasttext, getLabelRows)
{
	php_fasttext_object *ft_obj;
	zval *object = getThis();
	zend_long res;

	ft_obj = Z_FASTTEXT_P(object);

	if (zend_parse_parameters_none() == FAILURE) {
		return;
	}
	res = FastTextLabelRows(ft_obj->fasttext);

	RETURN_LONG(res);
}
/* }}} */

/* {{{ proto long fasttext::getWordId(String word)
 */
PHP_METHOD(fasttext, getWordId)
{
	php_fasttext_object *ft_obj;
	zval *object = getThis();
	char *word;
	size_t word_len;
	zend_long id;

	ft_obj = Z_FASTTEXT_P(object);

	if (FAILURE == zend_parse_parameters_throw(ZEND_NUM_ARGS(), "s", &word, &word_len)) {
		return;
	}
	id = (zend_long)FastTextWordId(ft_obj->fasttext, (const char*)word);

	RETURN_LONG(id);
}
/* }}} */

/* {{{ proto long fasttext::getSubwordId(String word)
 */
PHP_METHOD(fasttext, getSubwordId)
{
	php_fasttext_object *ft_obj;
	zval *object = getThis();
	char *word;
	size_t word_len;
	zend_long id;

	ft_obj = Z_FASTTEXT_P(object);

	if (FAILURE == zend_parse_parameters_throw(ZEND_NUM_ARGS(), "s", &word, &word_len)) {
		return;
	}
	id = (zend_long)FastTextSubwordId(ft_obj->fasttext, (const char*)word);

	RETURN_LONG(id);
}
/* }}} */

/* {{{ proto mixed fasttext::getWord(int id])
 */
PHP_METHOD(fasttext, getWord)
{
	php_fasttext_object *ft_obj;
	zval *object = getThis();
	zend_long id;
	FTValues ft_vals;

	ft_obj = Z_FASTTEXT_P(object);

	if (FAILURE == zend_parse_parameters_throw(ZEND_NUM_ARGS(), "l", &id)) {
		return;
	}
	ft_vals = FastTextGetWord(ft_obj->fasttext, (int32_t)id);

	if (ft_vals->is_error) {
		ZVAL_STRING(&ft_obj->error, ft_vals->buff);
		FastTextValuesFree(ft_vals);
		RETURN_FALSE;
	}

	ZVAL_STRING(return_value, ft_vals->buff);
	FastTextValuesFree(ft_vals);
}
/* }}} */

/* {{{ proto mixed fasttext::getLabel(int id])
 */
PHP_METHOD(fasttext, getLabel)
{
	php_fasttext_object *ft_obj;
	zval *object = getThis();
	zend_long id;
	FTValues ft_vals;

	ft_obj = Z_FASTTEXT_P(object);

	if (FAILURE == zend_parse_parameters_throw(ZEND_NUM_ARGS(), "l", &id)) {
		return;
	}
	ft_vals = FastTextGetLabel(ft_obj->fasttext, (int32_t)id);

	if (ft_vals->is_error) {
		ZVAL_STRING(&ft_obj->error, ft_vals->buff);
		FastTextValuesFree(ft_vals);
		RETURN_FALSE;
	}

	ZVAL_STRING(return_value, ft_vals->buff);
	FastTextValuesFree(ft_vals);
}
/* }}} */

/* {{{ proto mixed fasttext::getWordVectors(String word)
 */
PHP_METHOD(fasttext, getWordVectors)
{
	php_fasttext_object *ft_obj;
	zval *object = getThis();
	char *word;
	size_t word_len;
	FTVectors ft_vecs;

	ft_obj = Z_FASTTEXT_P(object);

	if (FAILURE == zend_parse_parameters_throw(ZEND_NUM_ARGS(), "s", &word, &word_len)) {
		return;
	}
	ft_vecs = FastTextWordVectors(ft_obj->fasttext, (const char*)word);

	array_init(return_value);
	for (int64_t idx=0; idx<ft_vecs->size; idx++) {
		fasttext_vec_add_array(return_value, (int)idx, ft_vecs->vals[idx]);
	}
	FastTextVectorsFree(ft_vecs);
}
/* }}} */

/* {{{ proto mixed fasttext::getSubwordVector(String word)
 */
PHP_METHOD(fasttext, getSubwordVector)
{
	php_fasttext_object *ft_obj;
	zval *object = getThis();
	char *word;
	size_t word_len;
	FTVectors ft_vecs;

	ft_obj = Z_FASTTEXT_P(object);

	if (FAILURE == zend_parse_parameters_throw(ZEND_NUM_ARGS(), "s", &word, &word_len)) {
		return;
	}
	ft_vecs = FastTextSubwordVector(ft_obj->fasttext, (const char*)word);

	array_init(return_value);
	for (int64_t idx=0; idx<ft_vecs->size; idx++) {
		fasttext_vec_add_array(return_value, (int)idx, ft_vecs->vals[idx]);
	}
	FastTextVectorsFree(ft_vecs);
}
/* }}} */

/* {{{ proto mixed fasttext::getSentenceVectors(String sentence)
 */
PHP_METHOD(fasttext, getSentenceVectors)
{
	php_fasttext_object *ft_obj;
	zval *object = getThis();
	char *sentence;
	size_t sentence_len;
	FTVectors ft_vecs;

	ft_obj = Z_FASTTEXT_P(object);

	if (FAILURE == zend_parse_parameters_throw(ZEND_NUM_ARGS(), "s", &sentence, &sentence_len)) {
		return;
	}
	ft_vecs = FastTextSubwordVector(ft_obj->fasttext, (const char*)sentence);

	array_init(return_value);
	for (int64_t idx=0; idx<ft_vecs->size; idx++) {
		fasttext_vec_add_array(return_value, (int)idx, ft_vecs->vals[idx]);
	}
	FastTextVectorsFree(ft_vecs);
}
/* }}} */

/* {{{ proto mixed fasttext::getPredict(String word[, int k])
 */
PHP_METHOD(fasttext, getPredict)
{
	php_fasttext_object *ft_obj;
	zval *object = getThis();
	char *word;
	size_t word_len;
	zend_long k = 10;
	FTProbs ft_vals;

	ft_obj = Z_FASTTEXT_P(object);

	if (FAILURE == zend_parse_parameters_throw(ZEND_NUM_ARGS(), "s|l", &word, &word_len, &k)) {
		return;
	}
	ft_vals = FastTextPredict(ft_obj->fasttext, (const char*)word, (const int)k);

	if (ft_vals->is_error) {
		ZVAL_STRING(&ft_obj->error, ft_vals->buff);
		FastTextProbsFree(ft_vals);
		RETURN_FALSE;
	}

	array_init(return_value);
	for (int idx=0; idx<ft_vals->size; idx++) {
		fasttext_value_add_array(return_value, idx, ft_vals->labels[idx], ft_vals->probs[idx]);
	}
	FastTextProbsFree(ft_vals);
}
/* }}} */

/* {{{ proto mixed fasttext::getNN(String word[, int k])
 */
PHP_METHOD(fasttext, getNN)
{
	php_fasttext_object *ft_obj;
	zval *object = getThis();
	char *word;
	size_t word_len;
	zend_long k = 10;
	FTProbs ft_vals;

	ft_obj = Z_FASTTEXT_P(object);

	if (FAILURE == zend_parse_parameters_throw(ZEND_NUM_ARGS(), "s|l", &word, &word_len, &k)) {
		return;
	}
	ft_vals = FastTextNN(ft_obj->fasttext, (const char*)word, (const int)k);

	if (ft_vals->is_error) {
		ZVAL_STRING(&ft_obj->error, ft_vals->buff);
		FastTextProbsFree(ft_vals);
		RETURN_FALSE;
	}

	array_init(return_value);
	for (int idx=0; idx<ft_vals->size; idx++) {
		fasttext_value_add_array(return_value, idx, ft_vals->labels[idx], ft_vals->probs[idx]);
	}
	FastTextProbsFree(ft_vals);
}
/* }}} */

/* {{{ proto mixed fasttext::getAnalogies(String word[, int k])
 */
PHP_METHOD(fasttext, getAnalogies)
{
	php_fasttext_object *ft_obj;
	zval *object = getThis();
	char *word;
	size_t word_len;
	zend_long k = 10;
	FTProbs ft_vals;

	ft_obj = Z_FASTTEXT_P(object);

	if (FAILURE == zend_parse_parameters_throw(ZEND_NUM_ARGS(), "s|l", &word, &word_len, &k)) {
		return;
	}
	ft_vals = FastTextAnalogies(ft_obj->fasttext, (const char*)word, (const int)k);

	if (ft_vals->is_error) {
		ZVAL_STRING(&ft_obj->error, ft_vals->buff);
		FastTextProbsFree(ft_vals);
		RETURN_FALSE;
	}

	array_init(return_value);
	for (int idx=0; idx<ft_vals->size; idx++) {
		fasttext_value_add_array(return_value, idx, ft_vals->labels[idx], ft_vals->probs[idx]);
	}
	FastTextProbsFree(ft_vals);
}
/* }}} */

/* {{{ proto mixed fasttext::getNgramVectors(String word)
 */
PHP_METHOD(fasttext, getNgramVectors)
{
	php_fasttext_object *ft_obj;
	zval *object = getThis();
	char *word;
	size_t word_len;
	FTProbs ft_vals;

	ft_obj = Z_FASTTEXT_P(object);

	if (FAILURE == zend_parse_parameters_throw(ZEND_NUM_ARGS(), "s", &word, &word_len)) {
		return;
	}
	ft_vals = FastTextNgramVectors(ft_obj->fasttext, (const char*)word);

	if (ft_vals->is_error) {
		ZVAL_STRING(&ft_obj->error, ft_vals->buff);
		FastTextProbsFree(ft_vals);
		RETURN_FALSE;
	}

	array_init(return_value);
	for (int idx=0; idx<ft_vals->size; idx++) {
		fasttext_value_add_array(return_value, idx, ft_vals->labels[idx], ft_vals->probs[idx]);
	}
	FastTextProbsFree(ft_vals);
}
/* }}} */

/* {{{ proto string fasttext::lastErrorMsg()
 */
PHP_METHOD(fasttext, lastErrorMsg)
{
	php_fasttext_object *ft_obj;
	zval *object = getThis();

	ft_obj = Z_FASTTEXT_P(object);

	if (zend_parse_parameters_none() == FAILURE) {
		return;
	}
	RETURN_ZVAL(&ft_obj->error, 1, 0);
}
/* }}} */

/* {{{ arginfo */
ZEND_BEGIN_ARG_INFO(arginfo_fasttext_void, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_fasttext_load, 0, 0, 1)
	ZEND_ARG_INFO(0, fileformat)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_fasttext_word, 0, 0, 1)
	ZEND_ARG_INFO(0, word)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_fasttext_id, 0, 0, 1)
	ZEND_ARG_INFO(0, id)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_fasttext_wordk, 0, 0, 1)
	ZEND_ARG_INFO(0, word)
	ZEND_ARG_INFO(0, k)
ZEND_END_ARG_INFO()
/* }}} */


/* {{{ php_sfasttext_class_methods */
static zend_function_entry php_fasttext_class_methods[] = {
	PHP_ME(fasttext, __construct,		arginfo_fasttext_void,	ZEND_ACC_PUBLIC|ZEND_ACC_CTOR)
	PHP_ME(fasttext, load, 				arginfo_fasttext_load, 	ZEND_ACC_PUBLIC)
	PHP_ME(fasttext, getWordRows, 		arginfo_fasttext_void, 	ZEND_ACC_PUBLIC)
	PHP_ME(fasttext, getLabelRows, 		arginfo_fasttext_void, 	ZEND_ACC_PUBLIC)
	PHP_ME(fasttext, getWordId, 		arginfo_fasttext_word, 	ZEND_ACC_PUBLIC)
	PHP_ME(fasttext, getSubwordId, 		arginfo_fasttext_word,	ZEND_ACC_PUBLIC)
	PHP_ME(fasttext, getWord, 			arginfo_fasttext_id, 	ZEND_ACC_PUBLIC)
	PHP_ME(fasttext, getLabel, 			arginfo_fasttext_id, 	ZEND_ACC_PUBLIC)
	PHP_ME(fasttext, getWordVectors, 	arginfo_fasttext_word,	ZEND_ACC_PUBLIC)
	PHP_ME(fasttext, getSubwordVector, 	arginfo_fasttext_word,	ZEND_ACC_PUBLIC)
	PHP_ME(fasttext, getSentenceVectors,arginfo_fasttext_word,	ZEND_ACC_PUBLIC)
	PHP_ME(fasttext, getPredict, 		arginfo_fasttext_wordk, ZEND_ACC_PUBLIC)
	PHP_ME(fasttext, getNN,				arginfo_fasttext_wordk, ZEND_ACC_PUBLIC)
	PHP_ME(fasttext, getAnalogies, 		arginfo_fasttext_wordk, ZEND_ACC_PUBLIC)
	PHP_ME(fasttext, getNgramVectors,	arginfo_fasttext_word,	ZEND_ACC_PUBLIC)
	PHP_ME(fasttext, lastErrorMsg,		arginfo_fasttext_void,	ZEND_ACC_PUBLIC)

	PHP_FE_END
};
/* }}} */

static void php_fasttext_object_free_storage(zend_object *object) /* {{{ */
{
	php_fasttext_object *intern = php_fasttext_from_obj(object);

	if (!intern) {
		return;
	}

	if (intern->fasttext) {
		FastTextFree(intern->fasttext);
		intern->fasttext = NULL;
	}

	zend_object_std_dtor(&intern->zo);
}
/* }}} */

static zend_object *php_fasttext_object_new(zend_class_entry *class_type) /* {{{ */
{
	php_fasttext_object *intern;

	/* Allocate memory for it */
	intern = ecalloc(1, sizeof(php_fasttext_object) + zend_object_properties_size(class_type));

	zend_object_std_init(&intern->zo, class_type);
	object_properties_init(&intern->zo, class_type);

	intern->zo.handlers = &fasttext_object_handlers;

	return &intern->zo;
}
/* }}} */


/* {{{ PHP_MINIT_FUNCTION
*/
PHP_MINIT_FUNCTION(fasttext)
{
	zend_class_entry ce;

	memcpy(&fasttext_object_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));

	/* Register fastText Class */
	INIT_CLASS_ENTRY(ce, "fastText", php_fasttext_class_methods);
	ce.create_object = php_fasttext_object_new;
	fasttext_object_handlers.offset = XtOffsetOf(php_fasttext_object, zo);
	fasttext_object_handlers.clone_obj = NULL;
	fasttext_object_handlers.free_obj = php_fasttext_object_free_storage;
	php_fasttext_sc_entry = zend_register_internal_class(&ce);

	REGISTER_INI_ENTRIES();

	return SUCCESS;
}
/* }}} */

/* {{{ PHP_MSHUTDOWN_FUNCTION
*/
PHP_MSHUTDOWN_FUNCTION(fasttext)
{
	UNREGISTER_INI_ENTRIES();

	return SUCCESS;
}
/* }}} */

/* {{{ PHP_MINFO_FUNCTION
*/
PHP_MINFO_FUNCTION(fasttext)
{
	php_info_print_table_start();
	php_info_print_table_header(2, "fastText support", "enabled");
	php_info_print_table_row(2, "fastText module version", PHP_FASTTEXT_VERSION);
	php_info_print_table_row(2, "fastText Library", FastTextVersion());
	php_info_print_table_end();

	DISPLAY_INI_ENTRIES();
}
/* }}} */

/* {{{ PHP_GINIT_FUNCTION
*/
static PHP_GINIT_FUNCTION(fasttext)
{
	memset(fasttext_globals, 0, sizeof(*fasttext_globals));
}
/* }}} */

/* {{{ fasttext_module_entry
*/
zend_module_entry fasttext_module_entry = {
	STANDARD_MODULE_HEADER,
	"fasttext",
	NULL,
	PHP_MINIT(fasttext),
	PHP_MSHUTDOWN(fasttext),
	NULL,
	NULL,
	PHP_MINFO(fasttext),
	PHP_FASTTEXT_VERSION,
	PHP_MODULE_GLOBALS(fasttext),
	PHP_GINIT(fasttext),
	NULL,
	NULL,
	STANDARD_MODULE_PROPERTIES_EX
};
/* }}} */

#ifdef COMPILE_DL_FASTTEXT
#ifdef ZTS
ZEND_TSRMLS_CACHE_DEFINE()
#endif
ZEND_GET_MODULE(fasttext)
#endif

/*
 * Local variables:
 * tab-width: 4
 * c-basic-offset: 4
 * End:
 * vim600: noet sw=4 ts=4 fdm=marker
 * vim<600: noet sw=4 ts=4
 */
