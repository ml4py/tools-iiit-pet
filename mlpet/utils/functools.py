"""
This file is borrowed from the ML4PY project (github.com/ml4py/ml4py)!
"""


def constproperty(fn):
    """

    :param fn: A function or a callable object in general.
    :return: A value that the :py:class:`fn` returns.

    Refering a function or a callable object those return non-none values as constant variables can simply to use
    a module API. This decorator implements a wrapper that packs these callable objects and provides a values, which
    the objects return.

    >>> @constproperty
    ... def CONSTANT_ONE() -> int:
    ...     return 1
    >>> ml.print(CONSTANT_ONE)

    """

    if not callable(fn):
        raise AssertionError('Object must be callable.')

    return fn()


class lazy_import(object):
    """
    :param pkg_name: A module or package name.
    :type pkg_name: str
    This class implements a mechanism for lazy, or "*on demand*", importing packages or particular modules there.
    This practice is mainly used in web design and development to delay loading the application parts or initializing
    objects until they are really needed. There is called lazy loading. We essentially split an
    application into logical blocks, which import their dependencies at the points where they are called.
    This design pattern speeds up the first load and reduces overall weight as some blocks may never even be used.
    """

    def __init__(self, pkg_name: str):
        self._pkg_name = pkg_name
        self._mod = None

    def __getattr__(self, obj):
        import importlib

        if self._mod is None:
            self._mod = importlib.import_module(self._pkg_name)

        return getattr(self._mod, obj)


class optional_import(object):
    """
    :param pkg_name: A module or package name.
    :type pkg_name: str
    The lazy importer implementation :py:func:`lazy_import` is designed that a required module
    is mandatory. If the module does not exist in the python search paths, importing fails and an application crashes.
    This happens because the python interpreter does not have any native mechanism for checking an application
    dependencies before its starting. For loading an advanced or non-critical application modules, a proper way is to
    import them as optional features. This class implements such mechanism and prevents crashing an application when
    a module that we require is not installed in the search paths or implemented.
    """

    def __init__(self, pkg_name: str):
        self._pkg_name = pkg_name
        self._mod = None

    def __getattr__(self, obj):
        import importlib

        if self._mod is None:
            try:
                self._mod = importlib.import_module(self._pkg_name)
            except ImportError:
                return None

        try:
            attr = getattr(self._mod, obj)
        except AttributeError:
            attr = None

        return attr
