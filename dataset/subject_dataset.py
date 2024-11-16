import torchio as tio
import pandas as pd
from typing import Callable, List

def subjects_from_csv(dataset_path: str, age=True, lambda_age: Callable = lambda x: x) -> List[tio.Subject]:
    subjects = []
    df = pd.read_csv(dataset_path)
    for index, row in df.iterrows():
        subject = tio.Subject(
            image=tio.ScalarImage(row['image']),
            label=tio.LabelMap(row['label'])
        )

        if age is True:
            age = lambda_age(row['age']) if lambda_age is None else row['age']
        else:
            if lambda_age is None:
                age_funct: lambda x : x
            subject = tio.Subject(
                image=tio.ScalarImage(row['image']),
                label=tio.LabelMap(row['label']),
                age=lambda_age(row['age'])
            )
        subjects.append(subject)
    return subjects


