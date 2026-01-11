from pydantic import BaseModel

class ChurnInput(BaseModel):
    RowNumber: int
    CustomerId: int
    Surname: str
    Geography: str
    Gender: str
    Tenure: int
    Age: int
    CreditScore: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
