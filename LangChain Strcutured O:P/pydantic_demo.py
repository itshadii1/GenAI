from pydantic import BaseModel, EmailStr, Field #can be used to validate email
from typing import Optional

class Student(BaseModel):
    name: str = 'Hadi'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt = 0, lt = 10, default=5, description="xyz")

new_student = {'age':22, 'email':'mahmoodhadi@gmail.com'}
Student = Student(**new_student)

print(Student)