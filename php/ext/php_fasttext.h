#ifndef PHP_FASTTEXT_H
#define PHP_FASTTEXT_H

#define PHP_FASTTEXT_VERSION	"0.1.0"

extern zend_module_entry fasttext_module_entry;
#define phpext_fasttext_ptr &fasttext_module_entry

ZEND_BEGIN_MODULE_GLOBALS(fasttext)
	char *model_dir;
ZEND_END_MODULE_GLOBALS(fasttext)

#ifdef ZTS
# define FASTTEXT_G(v) TSRMG(fasttext_globals_id, zend_fasttext_globals *, v)
# ifdef COMPILE_DL_FASTTEXT
ZEND_TSRMLS_CACHE_EXTERN()
# endif
#else
# define FASTTEXT_G(v) (fasttext_globals.v)
#endif

typedef struct {
    zend_object zo;
	zval error;
    FastTextHandle fasttext;
} php_fasttext_object;

static inline php_fasttext_object *php_fasttext_from_obj(zend_object *obj) {
	return (php_fasttext_object*)((char*)(obj) - XtOffsetOf(php_fasttext_object, zo));
}

#define Z_FASTTEXT_P(zv) php_fasttext_from_obj(Z_OBJ_P((zv)))


#endif  /* PHP_FASTTEXT_H */

/*
 * Local variables:
 * tab-width: 4
 * c-basic-offset: 4
 * indent-tabs-mode: t
 * End:
 */
