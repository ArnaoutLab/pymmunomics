class PymmunomicsBaseWarning(Warning):
    pass

class ArgumentWarning(PymmunomicsBaseWarning):
    pass

class AmbiguousValuesWarning(PymmunomicsBaseWarning):
    pass

class DivergingValuesWarning(PymmunomicsBaseWarning):
    pass

class PymmunomicsBaseError(Exception):
    pass

class InvalidArgumentError(PymmunomicsBaseError):
    pass

class NotImplementedError(PymmunomicsBaseError):
    pass
