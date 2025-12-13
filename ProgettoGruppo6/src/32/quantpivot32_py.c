#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <numpy/arrayobject.h>
#include <xmmintrin.h>

#include "common.h"

#include "quantpivot32.c"

// Struttura per l'oggetto QuantPivot
typedef struct {
	// Espande a campi obbligatori che ogni oggetto Python deve avere
	PyObject_HEAD
	// Parametri
	params* input;
	// Salva i PyArrayObject
	PyArrayObject* DS_array;	// riferimento all'array dataset
	PyArrayObject* Q_array;		// riferimento all'array query
} QuantPivot32Object;

static void mm_free_destructor(PyObject* capsule) {
    void* ptr = PyCapsule_GetPointer(capsule, NULL);
    if (ptr != NULL) {
        _mm_free(ptr);
    }
}

// Deallocazione (pulizia memoria quando l'oggetto viene distrutto)
static void QuantPivot32_dealloc(QuantPivot32Object *self) {
	// Libera memoria allocata
	if (self->input->P != NULL)
		_mm_free(self->input->P);
	if (self->input->index != NULL)
		_mm_free(self->input->index);

    if (self->input->ds_plus != NULL) _mm_free(self->input->ds_plus);
    if (self->input->ds_minus != NULL) _mm_free(self->input->ds_minus);

	// Decrementa riferimenti agli array NumPy
	Py_XDECREF(self->DS_array);
	Py_XDECREF(self->Q_array);

	free(self->input);

	Py_TYPE(self)->tp_free((PyObject *)self);
}

static int QuantPivot32_init(QuantPivot32Object *self, PyObject *args, PyObject *kwargs) {
    self->DS_array = NULL;
    self->Q_array = NULL;

    self->input = (params*)calloc(1, sizeof(params));
    if (!self->input) {
        PyErr_NoMemory();
        return -1;
    }

    self->input->DS = NULL;
    self->input->P = NULL;
    self->input->h = -1;
    self->input->k = -1;
    self->input->x = -1;
    self->input->N = -1;
    self->input->D = -1;
    self->input->index = NULL;
    self->input->Q = NULL;
    self->input->nq = -1;
    self->input->id_nn = NULL;
    self->input->dist_nn = NULL;
    self->input->silent = 0;

    //i nuovi campi
    self->input->ds_plus = NULL;
    self->input->ds_minus = NULL;
    self->input->first_fit_call = false;

    return 0;
}

// Metodo fit
static PyObject* QuantPivot32_fit(QuantPivot32Object *self, PyObject *args, PyObject *kwargs) {
	PyArrayObject *ds_array;

	int h, x, silent = 1;

	static char *kwlist[] = {"dataset", "n_pivots", "quant_level", "silent", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!ii|i", kwlist,
									&PyArray_Type, &ds_array,
									&h, &x, &silent)) {
		return NULL;
	}

	// Verifica che sia un array NumPy valido
	if (PyArray_NDIM(ds_array) != 2) {
		PyErr_SetString(PyExc_ValueError, "Data must be a 2D array");
		return NULL;
	}

	// Verifica che sia float32
	if (PyArray_TYPE(ds_array) != NPY_FLOAT32) {
		PyErr_SetString(PyExc_TypeError, "Data must be float32");
		return NULL;
	}

	// Verifica che siano array contigui
	type* dataset = (type*)(PyArrayObject*)PyArray_DATA(ds_array);

	uintptr_t addr = (uintptr_t)dataset;
	int is_aligned = (addr % align == 0);

	if(!is_aligned){
		PyErr_SetString(PyExc_ValueError, "Input array (DS) not aligned");
		return NULL;
	}

	// Estrai dimensioni
	self->input->N = (int)PyArray_DIM(ds_array, 0);
	self->input->D = (int)PyArray_DIM(ds_array, 1);

	if (h <= 0 || h > self->input->N) {
		PyErr_SetString(PyExc_ValueError, "n_pivots (h) must be in [1..N]");
		return NULL;
	}
	if (x <= 0 || x > self->input->D) {
		PyErr_SetString(PyExc_ValueError, "quant_level (x) must be in [1..D]");
		return NULL;
	}


	// Estrae il numero di pivot
	self->input->h = h;

	// Estrae il livello di quantizzazione
	self->input->x = x;

	// Estrae il flag silent
	self->input->silent = silent;

	// Salva riferimento all'array con INCREF
	Py_INCREF(ds_array);
	Py_XDECREF(self->DS_array);
	self->DS_array = ds_array;

	self->input->DS = dataset;

	// ========================================= //
	fit(self->input);
	// ========================================= //

	// Restituisci self per permettere method chaining
	Py_INCREF(self);
	return (PyObject *)self;
}

// Metodo predict
static PyObject* QuantPivot32_predict(QuantPivot32Object *self, PyObject *args, PyObject *kwargs) {
	PyArrayObject* query_array;
	int k, silent = 0;

	static char* kwlist[] = {"query", "k", "silent", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!i|i", kwlist,
									&PyArray_Type, &query_array,
									&k, &silent))
		return NULL;

	// Verifica che fit sia stato chiamato
	if (self->input->index == NULL) {
		PyErr_SetString(PyExc_RuntimeError,
					"Model not fitted, call fit() before predict()");
		return NULL;
	}

	// Verifica che Q sia un array NumPy valido
	if (PyArray_NDIM(query_array) != 2) {
		PyErr_SetString(PyExc_ValueError, "Data must be a 2D array");
		return NULL;
	}

	// Verifica che sia float32
	if (PyArray_TYPE(query_array) != NPY_FLOAT32) {
		PyErr_SetString(PyExc_TypeError, "Data must be float32");
		return NULL;
	}

	// Verifica che siano array contigui
	type* query = (type*)(PyArrayObject*)PyArray_DATA(query_array);
	uintptr_t addr = (uintptr_t)query;
	int is_aligned = (addr % align == 0);

	if(!is_aligned){
		PyErr_SetString(PyExc_ValueError, "Query array (Q) not aligned");
		return NULL;
	}

	self->input->Q = query;

	// Estrai dimensioni
	self->input->nq = (int)PyArray_DIM(query_array, 0);

	int qD = (int)PyArray_DIM(query_array, 1);
	if (qD != self->input->D) {
		PyErr_SetString(PyExc_ValueError, "Query dimensionality must match dataset D");
		return NULL;
	}

	if (k <= 0 || k > self->input->N) {
		PyErr_SetString(PyExc_ValueError, "k must be in [1..N]");
		return NULL;
	}


	// Estrae il numero di K vicini
	self->input->k = k;

	// Estrae il flag silent
	self->input->silent = silent;

	self->input->id_nn = (int*) _mm_malloc(self->input->nq * self->input->k * sizeof(int), align);
	self->input->dist_nn = (type*) _mm_malloc(self->input->nq * self->input->k * sizeof(type), align);

	// ========================================= //
	predict(self->input);
	// ========================================= //

	npy_intp dims[2] = {self->input->nq, self->input->k};


	PyArrayObject* id_nn_array = (PyArrayObject*)PyArray_SimpleNewFromData(
		2,				// ndim
		dims,			// shape
		NPY_INT32,		// dtype
		self->input->id_nn		// data pointer (usa la memoria allineata)
	);
	// Crea un capsule per gestire la deallocazione
	PyObject* capsule_id = PyCapsule_New(self->input->id_nn, NULL, mm_free_destructor);

	// Associa il capsule all'array cosÃ¬ quando l'array viene distrutto,
	// la memoria allineata viene liberata
	PyArray_SetBaseObject(id_nn_array, capsule_id);

	PyArrayObject* dist_nn_array = (PyArrayObject*)PyArray_SimpleNewFromData(
		2,				// ndim
		dims,			// shape
		NPY_FLOAT32,	// dtype
		self->input->dist_nn	// data pointer (usa la memoria allineata)
	);
	// Crea un capsule per gestire la deallocazione
	PyObject* capsule_dist = PyCapsule_New(self->input->dist_nn, NULL, mm_free_destructor);

	// Associa il capsule all'array cosÃ¬ quando l'array viene distrutto,
	// la memoria allineata viene liberata
	PyArray_SetBaseObject(dist_nn_array, capsule_dist);

	// Restituisce una TUPLA con (ids, distances)
	PyObject* result = PyTuple_Pack(2,
									(PyObject*)id_nn_array,
									(PyObject*)dist_nn_array);

	Py_DECREF(id_nn_array);   // PyTuple_Pack ha fatto INCREF
	Py_DECREF(dist_nn_array);

    return result;
}

// Tabella dei metodi
static PyMethodDef QuantPivot32_methods[] = {
	{
		"fit",
		(PyCFunction)QuantPivot32_fit,
		METH_VARARGS | METH_KEYWORDS,
		"Build the index using data\n\n"
		"Parameters:\n"
		"  data: numpy array of shape (N, D)\n"
		"  n_pivots: number of pivots\n"
		"  x: quantization level\n"
		"  s: silent (default=False)\n"
		"\n"
		"Returns:\n"
		"  self"
	},
	{
		"predict",
		(PyCFunction)QuantPivot32_predict,
		METH_VARARGS | METH_KEYWORDS,
		"Query the index\n\n"
		"Parameters:\n"
		"  query: numpy array of shape (nq, D)\n"
		"  k: number of neighbors\n"
		"  s: silent (default=False)\n"
		"\n"
		"Returns:\n"
		"  numpy array of indices"
	},
	{NULL, NULL, 0, NULL}
};

// Definizione del tipo Python
static PyTypeObject QuantPivot32Type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	.tp_name = "gruppo6.quantpivot32.QuantPivot",
	.tp_doc = "QuantPivot 32-bit indexing and querying",
	.tp_basicsize = sizeof(QuantPivot32Object),
	.tp_itemsize = 0,
	.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
	.tp_new = PyType_GenericNew,
	.tp_init = (initproc)QuantPivot32_init,
	.tp_dealloc = (destructor)QuantPivot32_dealloc,
	.tp_methods = QuantPivot32_methods,
};


static struct PyModuleDef quantpivot32_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_quantpivot32",        // Nome del modulo C
    .m_doc = "Quantized Pivot Indexing and Querying (32bit)",  // Docstring
    .m_size = -1,                     // -1 significa che il modulo non mantiene stato
};

// Inizializzazione del modulo
PyMODINIT_FUNC PyInit__quantpivot32(void) {
	PyObject *m;

	// Prepara il tipo
	if (PyType_Ready(&QuantPivot32Type) < 0)
		return NULL;

	// Crea il modulo
	m = PyModule_Create(&quantpivot32_module);
	if (m == NULL)
		return NULL;

	// Aggiungi la classe al modulo
	Py_INCREF(&QuantPivot32Type);
	if (PyModule_AddObject(m, "QuantPivot", (PyObject *)&QuantPivot32Type) < 0) {
		Py_DECREF(&QuantPivot32Type);
		Py_DECREF(m);
		return NULL;
	}

	// Inizializza NumPy
	import_array();

	return m;
}