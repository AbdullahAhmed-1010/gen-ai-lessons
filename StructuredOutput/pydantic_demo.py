from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str
    age: int = Field(ge=0, description="Age must be a non-negative number")
    country: str = "India"

person = Person(name="Abdullah", age=20)

print(person)
print(person.name)
print(person.model_dump())

try:
    invalid_person = Person(name="John Doe", age=-10)
except Exception as e:
    print("validation error:")
    print(e)