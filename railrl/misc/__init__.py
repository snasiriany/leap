"""
Modules in here are self-contained modules, that depend only core libraries
like numpy and pythonplusplus. However, they should NOT depend on things that
are specific to railrl or rllab.

The only exception is when an external dependency is explicit in the name of
the module, e.g. rllab_util can depend on rllab. hyperopt can depend on
hyperopt, etc.
"""