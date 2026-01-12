from pdf2text.models.driver import Drivers


def test_get_drivers() -> None:
    drivers = Drivers.model_validate(
        {
            "drivers": [
                {
                    "name": "Alice",
                    "cars": [
                        {"model": "Tesla Model S", "engine": "Electric"},
                        {"model": "BMW i3", "engine": "Electric"},
                    ],
                },
                {
                    "name": "Bob",
                    "cars": [
                        {"model": "Ford Mustang", "engine": "V8"},
                    ],
                },
            ]
        }
    )
    assert len(drivers.drivers) == 2
    assert drivers.drivers[0].name == "Alice"
    assert len(drivers.drivers[0].cars) == 2
    assert drivers.drivers[1].name == "Bob"
    assert len(drivers.drivers[1].cars) == 1
    assert drivers.drivers[1].cars[0].model == "Ford Mustang"
    assert drivers.drivers[1].cars[0].engine == "V8"
