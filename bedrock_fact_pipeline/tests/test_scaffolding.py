import src
import src.generation
import src.verification
import src.data_report


def test_packages_importable():
    assert src and src.generation and src.verification and src.data_report
