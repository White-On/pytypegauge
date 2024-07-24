def function():
    return 1


def non_typed_function(a, b):
    return 1


def typed_function(a: int, b: int) -> int:
    return 1


def typed_function_with_default(a: int, b: int = 1) -> int:
    return 1


def multi_line_args(a: int, b: int) -> int:
    return 1


def multi_line_args_with_default(a: int, b: int = 1) -> int:
    return 1


def multi_line_args_with_default_and_annotation(a: int, b: int = 1) -> int:
    return 1


class TestClass:
    def method(self):
        return 1

    def non_typed_method(self, a, b):
        return 1

    def typed_method(self, a: int, b: int) -> int:
        return 1

    def typed_method_with_default(self, a: int, b: int = 1) -> int:
        return 1

    def multi_line_args(self, a: int, b: int) -> int:
        return 1

    def multi_line_args_with_default(self, a: int, b: int = 1) -> int:
        return 1

    def multi_line_args_with_default_and_annotation(self, a: int, b: int = 1) -> int:
        return 1
