from torch.nn.modules.module import Module
from collections import OrderedDict, Iterable, Mapping


class ModuleDict(Module):
    r"""Holds submodules in a dictionary.

    ModuleDict can be indexed like a regular Python dictionary, but modules it
    contains are properly registered, and will be visible by all Module methods.

    Arguments:
        modules (iterable, optional): a mapping (dictionary) of (string: module)
            or an iterable of key/value pairs of type (string, module)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.choices = nn.ModuleDict({
                        'conv': nn.Conv2d(10, 10, 3),
                        'pool': nn.MaxPool2d(3)
                })
                self.activations = nn.ModuleDict([
                        ['lrelu', nn.LeakyReLU()],
                        ['prelu', nn.PReLU()]
                ])

            def forward(self, x, choice, act):
                x = self.choices[choice](x)
                x = self.activations[act](x)
                return x
    """

    def __init__(self, modules=None):
        super(ModuleDict, self).__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key):
        return self._modules[key]

    def __setitem__(self, key, module):
        self.add_module(key, module)

    def __delitem__(self, key):
        del self._modules[key]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules)

    def __contains__(self, key):
        return key in self._modules

    def clear(self):
        """Remove all items from the ModuleDict.
        """
        self._modules.clear()

    def pop(self, key):
        r"""Remove key from the ModuleDict and return its module.

        Arguments:
            key (string): key to pop from the ModuleDict
        """
        v = self[key]
        del self[key]
        return v

    def keys(self):
        r"""Return an iterable of the ModuleDict keys.
        """
        return self._modules.keys()

    def items(self):
        r"""Return an iterable of the ModuleDict key/value pairs.
        """
        return self._modules.items()

    def values(self):
        r"""Return an iterable of the ModuleDict values.
        """
        return self._modules.values()

    def update(self, modules):
        r"""Update the ModuleDict with the key/value pairs from a mapping or
        an iterable, overwriting existing keys.

        Arguments:
            modules (iterable): a mapping (dictionary) of (string: :class:`~torch.nn.Module``) or
                an iterable of key/value pairs of type (string, :class:`~torch.nn.Module``)
        """
        if not isinstance(modules, Iterable):
            raise TypeError("ModuleDict.update should be called with an "
                            "iterable of key/value pairs, but got " +
                            type(modules).__name__)

        if isinstance(modules, Mapping):
            if isinstance(modules, OrderedDict):
                for key, module in modules.items():
                    self[key] = module
            else:
                for key, module in sorted(modules.items()):
                    self[key] = module
        else:
            for j, m in enumerate(modules):
                if not isinstance(m, Iterable):
                    raise TypeError("ModuleDict update sequence element "
                                    "#" + str(j) + " should be Iterable; is" +
                                    type(m).__name__)
                if not len(m) == 2:
                    raise ValueError("ModuleDict update sequence element "
                                     "#" + str(j) + " has length " + str(len(m)) +
                                     "; 2 is required")
                self[m[0]] = m[1]
