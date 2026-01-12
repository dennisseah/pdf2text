from pydantic import BaseModel


class Car(BaseModel):
    model: str
    engine: str


class Driver(BaseModel):
    name: str
    cars: list[Car]


class Drivers(BaseModel):
    drivers: list[Driver]
