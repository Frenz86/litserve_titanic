#pip install uvloop litserve
import joblib, numpy as np
import litserve as ls
import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional, Literal

class TitanicPassenger(BaseModel):
    Pclass: int = Field(..., ge=1, le=3, description="Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)")
    Sex: Literal['male', 'female'] = Field(..., description="Sex")
    Age: float = Field(..., ge=0, le=100, description="Age in years")
    SibSp: int = Field(..., ge=0, le=8, description="Number of Siblings/Spouses Aboard")
    Parch: int = Field(..., ge=0, le=6, description="Number of Parents/Children Aboard")
    Fare: float = Field(..., ge=0, le=600, description="Passenger Fare")
    Embarked: Literal['C', 'Q', 'S'] = Field(..., description="Port of Embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)")

class TitanicRequest(BaseModel):
    input: TitanicPassenger

class TitanicResponse(BaseModel):
    class_idx: int = Field(..., description="Survival prediction (0 = No, 1 = Yes)")
    probability: float = Field(..., ge=0, le=1, description="Probability of survival")

class XGBoostAPI(ls.LitAPI):
    def setup(self, device):
        self.model = joblib.load("titanic_pipe.pkl")
    
    def decode_request(self, request: TitanicRequest):
        passenger = request.input
        df = pd.DataFrame([{
                            'Pclass': passenger.Pclass,
                            'Sex': passenger.Sex,
                            'Age': passenger.Age,
                            'SibSp': passenger.SibSp,
                            'Parch': passenger.Parch,
                            'Fare': passenger.Fare,
                            'Embarked': passenger.Embarked
                        }])
        return df
    
    def predict(self, x):
        prediction = int(self.model.predict(x)[0])
        try:
            probability = float(self.model.predict_proba(x)[0][1])
        except:
            probability = float(prediction)
        return {"class": prediction, "probability": probability}
    
    def encode_response(self, output) -> TitanicResponse:
        return TitanicResponse(**output)

if __name__ == "__main__":
    api = XGBoostAPI()
    server = ls.LitServer(api, track_requests=True)
    server.run(port=8000)