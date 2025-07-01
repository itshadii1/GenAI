from typing import TypedDict

class Person(TypedDict):

    name: str
    age: int

new_person : Person = {'name':'Hadi','age':22}
print(new_person)