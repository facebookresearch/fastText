fasttext-predict
================

Python package for
`fasttext <https://github.com/facebookresearch/fastText>`__:

-  keep only the ``predict`` method, all other features are removed
-  the package does not depend on numpy
-  wheels for various architectures using GitHub workflows. The script
   is inspired by lxml build scripts.

Usage
-----

.. code:: python

   import fasttext
   model = fasttext.load_model('lid.176.ftz')
   result = model.predict('Fondant au chocolat et tarte aux myrtilles')
