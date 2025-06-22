from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional, Sequence, Tuple


class Module:
    """Modules form a tree that store parameters and other
    submodules. They make up the basis of neural network stacks.

    Attributes
    ----------
        _modules : Storage of the child modules
        _parameters : Storage of the module's parameters
        training : Whether the module is in training mode or evaluation mode

    """

    _modules: Dict[str, Module]
    _parameters: Dict[str, Parameter]
    training: bool

    def __init__(self) -> None:
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self) -> Sequence[Module]:
        """Return the direct child modules of this module."""
        m: Dict[str, Module] = self.__dict__["_modules"]
        return list(m.values())

    def train(self) -> None:
        """Set the mode of this module and all descendent modules to `train`."""
        # TODO: Implement for Task 0.4.
        self.training = True
        nodes_to_process = self._modules.values()
        future_nodes = []
        while nodes_to_process:
            for node in nodes_to_process:
                node.training = True
                future_nodes.extend(node._modules.values())
            nodes_to_process = future_nodes
            future_nodes = []

    def eval(self) -> None:
        """Set the mode of this module and all descendent modules to `eval`."""
        # TODO: Implement for Task 0.4.
        self.training = False
        que = deque(self._modules.values())
        while que:
            node = que.popleft()
            node.training = False
            que.extend(node._modules.values())

    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        """Collect all the parameters of this module and its descendents.

        Returns
        -------
            The name and `Parameter` of each ancestor parameter.

        """
        # TODO: Implement for Task 0.4.
        raise NotImplementedError("Need to implement for Task 0.4")

    def parameters(self) -> Sequence[Parameter]:
        """Enumerate over all the parameters of this module and its descendents."""
        # TODO: Implement for Task 0.4.
        raise NotImplementedError("Need to implement for Task 0.4")

    def add_parameter(self, k: str, v: Any) -> Parameter:
        """Manually add a parameter. Useful helper for scalar parameters.

        Args:
        ----
            k: Local name of the parameter.
            v: Value for the parameter.

        Returns:
        -------
            Newly created parameter.

        """
        val = Parameter(v, k)
        self.__dict__["_parameters"][k] = val
        return val

    def __setattr__(self, key: str, val: Parameter) -> None:
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Any:
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        def _addindent(s_: str, numSpaces: int) -> str:
            s2 = s_.split("\n")
            if len(s2) == 1:
                return s_
            first = s2.pop(0)
            s2 = [(numSpaces * " ") + line for line in s2]
            s = "\n".join(s2)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Parameter:
    """A Parameter is a special container stored in a `Module`.

    It is designed to hold a `Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, x: Any, name: Optional[str] = None) -> None:
        self.value = x
        self.name = name
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def update(self, x: Any) -> None:
        """Update the parameter value."""
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self) -> str:
        return repr(self.value)

    def __str__(self) -> str:
        return str(self.value)


if __name__ == "__main__":
    # Test boilerplate for creating a module tree

    # Create leaf modules (no children)
    leaf1 = Module()
    leaf2 = Module()
    leaf3 = Module()
    leaf4 = Module()

    # Create intermediate modules
    intermediate1 = Module()
    intermediate1._modules["leaf1"] = leaf1
    intermediate1._modules["leaf2"] = leaf2

    intermediate2 = Module()
    intermediate2._modules["leaf3"] = leaf3
    intermediate2._modules["leaf4"] = leaf4

    # Create root module
    root = Module()
    root._modules["branch1"] = intermediate1
    root._modules["branch2"] = intermediate2

    # Test the tree structure
    print("Module tree structure:")
    print(f"Root has {len(root._modules)} children")
    print(f"Branch1 has {len(intermediate1._modules)} children")
    print(f"Branch2 has {len(intermediate2._modules)} children")

    # Test training mode before calling train()
    print("\nBefore calling train():")
    print(f"Root training: {root.training}")
    print(f"Branch1 training: {intermediate1.training}")
    print(f"Branch2 training: {intermediate2.training}")
    print(f"Leaf1 training: {leaf1.training}")
    print(f"Leaf2 training: {leaf2.training}")
    print(f"Leaf3 training: {leaf3.training}")
    print(f"Leaf4 training: {leaf4.training}")

    # Set some modules to eval mode to test
    intermediate1.training = False
    leaf1.training = False
    leaf3.training = False

    print("\nAfter setting some to False:")
    print(f"Root training: {root.training}")
    print(f"Branch1 training: {intermediate1.training}")
    print(f"Branch2 training: {intermediate2.training}")
    print(f"Leaf1 training: {leaf1.training}")
    print(f"Leaf2 training: {leaf2.training}")
    print(f"Leaf3 training: {leaf3.training}")
    print(f"Leaf4 training: {leaf4.training}")

    # Now test your train() method
    root.train()

    print("\nAfter calling root.train():")
    print(f"Root training: {root.training}")
    print(f"Branch1 training: {intermediate1.training}")
    print(f"Branch2 training: {intermediate2.training}")
    print(f"Leaf1 training: {leaf1.training}")
    print(f"Leaf2 training: {leaf2.training}")
    print(f"Leaf3 training: {leaf3.training}")
    print(f"Leaf4 training: {leaf4.training}")

    # All should be True if your implementation is correct
