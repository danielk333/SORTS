from .random_uniform_scans_controller import RandomUniformScansController


def setup_function():
    # a workaround to avoid pytest printings and test printings interleaved in the same line
    # https://github.com/pytest-dev/pytest/issues/8574#issuecomment-1806404215
    print()


def test_random_uniform_scan_points():
    controller = RandomUniformScansController(npoints=10)
    result = controller.generate()
    print(result)
