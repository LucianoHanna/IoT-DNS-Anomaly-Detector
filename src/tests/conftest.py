def pytest_addoption(parser):
    parser.addoption(
        "--exp-id",
        action="store",
        default=None,
        help="Especifique o ID do experimento a ser testado (ex: exp_1)"
    )